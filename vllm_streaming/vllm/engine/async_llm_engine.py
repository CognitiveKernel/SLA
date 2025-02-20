import asyncio
import time
import numpy
import torch.nn.functional as F
from functools import partial
from typing import (Any, Dict, Iterable, List, Optional, Set, Tuple, Type,
                    Union, AsyncIterator)

from vllm.config import ModelConfig
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupOutput,
                           SequenceOutput, SequenceStatus)
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


class AsyncEngineDeadError(RuntimeError):
    pass


def _raise_exception_on_finish(task: asyncio.Task,
                               request_tracker: "RequestTracker") -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github.")
    try:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            raise AsyncEngineDeadError(
                msg + " See stack trace above for the actual cause.") from exc
        raise AsyncEngineDeadError(msg)
    except Exception as exc:
        request_tracker.propagate_exception(exc)
        raise exc


class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item: RequestOutput) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopIteration)
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if result is StopIteration:
            raise StopAsyncIteration
        elif isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = None

    def __contains__(self, item):
        return item in self._request_streams

    def init_event(self):
        self.new_requests_event = asyncio.Event()

    def propagate_exception(self,
                            exc: Exception,
                            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
        else:
            for stream in self._request_streams.values():
                stream.put(exc)

    def process_request_output(self,
                               request_output: RequestOutput,
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)

    def add_request(self, request_id: str,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))

        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        self.new_requests_event.clear()

        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        await self.new_requests_event.wait()


class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    async def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
        if scheduler_outputs.is_empty():
            return ignored
        # Execute the model.
        output = await self._run_workers_async(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )

        return self._process_model_outputs(output, scheduler_outputs) + ignored
    
    async def ck_step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
        if scheduler_outputs.is_empty():
            return ignored
        # print(seq_group_metadata_list[0].seq_data)
        # Execute the model.
        output, ck_meta_data = await self._ck_run_workers_async(
            "ck_execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        # print('ck_meta_data from ck_step_async', ck_meta_data)

        return self._ck_process_model_outputs(output, scheduler_outputs, ck_meta_data) + ignored

    async def _run_workers_async(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        coros = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                coros.append(
                    worker.execute_method.remote(method, *args, **kwargs))
            else:
                executor = getattr(worker, method)
                coros.append(asyncio.get_event_loop().run_in_executor(
                    None, partial(executor, *args, **kwargs)))

        all_outputs = await asyncio.gather(*coros)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output
    
    async def _run_workers_async_logits(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        coros = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                coros.append(
                    worker.execute_method.remote(method, *args, **kwargs))
            else:
                executor = getattr(worker, method)
                coros.append(asyncio.get_event_loop().run_in_executor(
                    None, partial(executor, *args, **kwargs)))

        all_outputs = await asyncio.gather(*coros)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert numpy.array_equal(output, other_output)
        return output
    
    async def _ck_run_workers_async(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        coros = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                coros.append(
                    worker.execute_method.remote(method, *args, **kwargs))
            else:
                executor = getattr(worker, method)
                coros.append(asyncio.get_event_loop().run_in_executor(
                    None, partial(executor, *args, **kwargs)))

        all_outputs = await asyncio.gather(*coros)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output[0] == other_output[0]
            
        return output[0], output[1]
    
    def add_request_for_logits(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
        """
        if arrival_time is None:
            arrival_time = time.monotonic()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                  arrival_time)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group_for_logits(seq_group)
        
    def process_parent_child_pairs(self, seq_group, parent_child_dict, target_parent_seq, mode='Normal'):
        local_child_seqs = []
        child_samples: List[SequenceOutput] = parent_child_dict[target_parent_seq.seq_id]
        # child_samples SequenceOutput; Usually a single token
        if len(child_samples) == 0:
            # This parent sequence has no children samples. Remove
            # the parent sequence from the sequence group since it will
            # not be used in the future iterations.
            target_parent_seq.status = SequenceStatus.FINISHED_ABORTED
            seq_group.remove(target_parent_seq.seq_id)
            self.scheduler.free_seq(target_parent_seq)
            return local_child_seqs
        for child_sample in child_samples[:-1]:
            new_child_seq_id = next(self.seq_counter)
            child = target_parent_seq.fork(new_child_seq_id)
            child.append_token_id(child_sample.output_token,
                                            child_sample.logprobs)
            local_child_seqs.append((child, target_parent_seq))
        # Continue the parent sequence for the last child sample.
        # We reuse the parent sequence here to reduce redundant memory
        # copies, especially when using non-beam search sampling methods.
        last_child_sample = child_samples[-1]
        if mode == 'Explore':
            # a new sequence needs to be constructed, and new subsequences continue to be explored later
            last_child_seq_id = next(self.seq_counter)
            last_child_seq = target_parent_seq.fork(last_child_seq_id)
            last_child_seq.append_token_id(last_child_sample.output_token,
                                            last_child_sample.logprobs)
            local_child_seqs.append((last_child_seq, target_parent_seq))
        elif mode == 'Normal':
            # Add the token to target_parent_seq
            target_parent_seq.append_token_id(last_child_sample.output_token,
                                        last_child_sample.logprobs)
            local_child_seqs.append((target_parent_seq, target_parent_seq))
        else:
            raise NotImplementedError
        for seq, parent in local_child_seqs:
            self._decode_sequence(seq, seq_group.sampling_params)
            self._check_stop(seq, seq_group.sampling_params)
            seq.update_ck_status(seq_group.sampling_params)
            if seq.seq_id != parent.seq_id:
                seq_group.add(seq)
                if not seq.is_finished():
                    self.scheduler.fork_seq(parent, seq)
            else:
                if seq.is_finished():
                    self.scheduler.free_seq(seq)
            
        if mode=='Explore':
            target_parent_seq.status = SequenceStatus.FINISHED_ABORTED
            self.scheduler.free_seq(target_parent_seq)
        return local_child_seqs
    
    def _ck_process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutput,
                                        ck_meta_data) -> None:
        
        # target_logit = ck_meta_data['target_logits']
        # positive_idx = 128128
        # positive_and_negative_logit = target_logit[:,positive_idx:positive_idx+2]
        # positive_logits_after_softmax = F.softmax(positive_and_negative_logit, dim=1)
        # positive_logits_after_softmax = positive_logits_after_softmax[:,0].cpu().tolist()
        sr_logits = ck_meta_data['sr_logits']
        sr_logits = sr_logits[...,0].cpu().tolist()

        prompt_logprobs = outputs.prompt_logprobs
        if prompt_logprobs is not None:
            seq_group.prompt_logprobs = prompt_logprobs
        # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        parent_seq_ids = [parent_seq.seq_id for parent_seq in parent_seqs]
        parent_child_dict = {
            parent_seq.seq_id: []
            for parent_seq in parent_seqs
        }
        for sample in samples:
            parent_child_dict[sample.parent_seq_id].append(sample)
        # print(f"parent_child_dict: {parent_child_dict}")
        # print(f"parent seq info:")
        # for tmp_parent in parent_seqs:
        #     print(f"seq_id: {tmp_parent.seq_id}, parent: {tmp_parent}")
        # print(f"{len(sr_logits)=}")
        # print(f"{sr_logits=}")
        # print(f"{len(parent_seqs)=}")
        # print(f"{parent_child_dict=}")
        # print(f"{parent_seq_ids=}")

        for i, parent in enumerate(parent_seqs):
            parent.append_ck_positive_logit(sr_logits[i])
            # parent.append_ck_positive_logit(positive_logits_after_softmax[i])
            # parent.update_ck_status(seq_group.sampling_params)

        # best_finished_seq_score = seq_group.get_best_finished_score()
        # print(f"{best_finished_seq_score=}")
        # parent_seqs_after_filtering = list()
        # for tmp_parent in parent_seqs:
        #     if best_finished_seq_score and \
        #         tmp_parent.get_ck_overall_score(seq_group.sampling_params.length_penalty) < best_finished_seq_score:
        #         tmp_parent.status = SequenceStatus.FINISHED_ABORTED
        #         self.scheduler.free_seq(tmp_parent)
        #     else:
        #         parent_seqs_after_filtering.append(tmp_parent)

        parent_seqs_after_filtering = parent_seqs
        # print(f"parent_seqs_after_filtering ids{[s.seq_id for s in parent_seqs_after_filtering]}")

        ck_status = parent_seqs[0].get_ck_status()
        # print(f"ck_status: {ck_status}")
        if seq_group.sampling_params.ck_mode == 'StreamingTreeSearch':
            # if ck_status == 0 or ck_status == 1: # Continuation generation
            if ck_status == 1:
                for parent in parent_seqs_after_filtering:
                    local_child_seqs = self.process_parent_child_pairs(seq_group, parent_child_dict, parent, mode='Normal')
            # elif ck_status == 2: # Branching and merge
            elif ck_status == 2 or ck_status == 0:
                incoming_seq_ids = []
                for parent in parent_seqs_after_filtering:
                    # Get a new quest subsequence and register it to seq_id_2_explorations
                    local_child_seqs = self.process_parent_child_pairs(seq_group, parent_child_dict, parent, mode='Explore')
                    for tmp_child_seq, tmp_parent_seq in local_child_seqs:
                        seq_group.add_exploration(tmp_parent_seq.seq_id, tmp_child_seq.seq_id)
                    incoming_seq_ids.append(parent.seq_id)
                if len(seq_group.waiting_seq_ids) == 0:
                    seq_group.waiting_seq_ids = incoming_seq_ids
                seqs_for_free = seq_group.check_and_merge_exploration()
                for tmp_seq in seqs_for_free:
                    tmp_seq.status = SequenceStatus.FINISHED_ABORTED
                    self.scheduler.free_seq(tmp_seq)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        return          

    def _ck_process_model_outputs(
            self, output: SamplerOutput,
            scheduler_outputs: SchedulerOutputs,
            ck_meta_data) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        for seq_group, outputs, tmp_ck_meta_data in zip(scheduled_seq_groups, output, ck_meta_data):
            self._ck_process_sequence_group_outputs(seq_group, outputs, tmp_ck_meta_data)

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in (scheduled_seq_groups +
                          scheduler_outputs.ignored_seq_groups):
            request_output = RequestOutput.ck_from_seq_group(seq_group)
            request_outputs.append(request_output)

        if self.log_stats:
            # Log the system stats.
            self._log_system_stats(scheduler_outputs.prompt_run,
                                   scheduler_outputs.num_batched_tokens)
            
        return request_outputs


class AsyncLLMEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.

    NOTE: For the comprehensive list of arguments, see `LLMEngine`.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args, *kwargs: Arguments for LLMEngine.
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 log_requests: bool = True,
                 max_log_len: Optional[int] = None,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.engine = self._init_engine(*args, **kwargs)

        self.background_loop = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded = None
        self.start_engine_loop = start_engine_loop
        self._request_tracker = RequestTracker()

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and not self.background_loop.done())

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        self._request_tracker.init_event()

        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish,
                    request_tracker=self._request_tracker))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def _init_engine(self, *args,
                     **kwargs) -> Union[_AsyncLLMEngine, "ray.ObjectRef"]:
        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            # FIXME(woosuk): This is a bit hacky. Be careful when changing the
            # order of the arguments.
            cache_config = args[1]
            parallel_config = args[2]
            if parallel_config.tensor_parallel_size == 1:
                num_gpus = cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            engine_class = ray.remote(num_gpus=num_gpus)(
                self._engine_class).remote
        return engine_class(*args, **kwargs)

    async def engine_step(self) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        new_requests, finished_requests = (
            self._request_tracker.get_new_and_finished_requests())

        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            # TODO: Maybe add add_request_batch to reduce Ray overhead
            if self.engine_use_ray:
                await self.engine.add_request.remote(**new_request)
            else:
                self.engine.add_request(**new_request)

        if finished_requests:
            await self._engine_abort(finished_requests)

        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()
        else:
            request_outputs = await self.engine.ck_step_async()

        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.log_requests)

        return len(request_outputs) > 0

    async def _engine_abort(self, request_ids: Iterable[str]):
        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_ids)
        else:
            self.engine.abort_request(request_ids)

    async def run_engine_loop(self):
        # Initialize the RequestTracker here so it uses the right event loop.
        has_requests_in_progress = False
        while True:
            if not has_requests_in_progress:
                await self._request_tracker.wait_for_new_requests()
            has_requests_in_progress = await self.engine_step()
            await asyncio.sleep(0)

    async def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> AsyncStream:
        if self.log_requests:
            shortened_prompt = prompt
            shortened_token_ids = prompt_token_ids
            if self.max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[:self.max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[:self.
                                                              max_log_len]
            logger.info(f"Received request {request_id}: "
                        f"prompt: {shortened_prompt!r}, "
                        f"sampling params: {sampling_params}, "
                        f"prompt token ids: {shortened_token_ids}.")

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        stream = self._request_tracker.add_request(
            request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time)

        return stream
    
    async def get_logits( # get ck reward logits
        self,
        prompt: Optional[str],
        target: Optional[str],
        sampling_params: SamplingParams,
        request_id: str,
    ) -> AsyncIterator[RequestOutput]:
        prompt_id = self.engine.tokenizer.encode(prompt)
        target_id = self.engine.tokenizer.encode(target)
        len_prompt_id = len(prompt_id)
        len_target_id = len(target_id)
        all_ids = prompt_id + target_id
        self.engine.add_request_for_logits(request_id=request_id, prompt=prompt+target, prompt_token_ids=all_ids, sampling_params=sampling_params)
        seq_group_metadata_list, scheduler_outputs = self.engine.scheduler.schedule_for_logits(request_id=request_id)
        logits = await self.engine._run_workers_async_logits(
            "get_logits",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        # print(f"{len(prompt_id)=}")
        # print(f"{len(target_id)=}")
        # print(f"{logits.shape=}")

        logits = logits.tolist()

        positive_logits = []
        for tmp_token_logits in logits[0][len_prompt_id:]:
            positive_logits.append(float(tmp_token_logits[0]))

        self.engine.scheduler.abort_seq_group(request_id)
        
        return positive_logits

    async def generate(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None
    ) -> AsyncIterator[RequestOutput]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.
        """
        # Preprocess the request.
        # This should not be used for logging, as it is monotonic time.
        arrival_time = time.monotonic()

        try:
            stream = await self.add_request(request_id,
                                            prompt,
                                            sampling_params,
                                            prompt_token_ids=prompt_token_ids,
                                            arrival_time=arrival_time)

            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id,
                                            verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()
        else:
            return self.engine.get_model_config()

    @classmethod
    def from_engine_args(cls,
                         engine_args: AsyncEngineArgs,
                         start_engine_loop: bool = True) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config, engine_args.engine_use_ray)
        # Create the async LLM engine.
        engine = cls(parallel_config.worker_use_ray,
                     engine_args.engine_use_ray,
                     *engine_configs,
                     distributed_init_method,
                     placement_group,
                     log_requests=not engine_args.disable_log_requests,
                     log_stats=not engine_args.disable_log_stats,
                     max_log_len=engine_args.max_log_len,
                     start_engine_loop=start_engine_loop)
        return engine
