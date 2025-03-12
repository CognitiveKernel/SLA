"""Sequence and its related classes."""
import copy
import enum
import math
from typing import Dict, List, Optional, Union

from vllm.block import LogicalTokenBlock
from vllm.sampling_params import SamplingParams

PromptLogprobs = List[Optional[Dict[int, float]]]
SampleLogprobs = List[Dict[int, float]]


class SequenceStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    WAITING_EXPLORATION = enum.auto()

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
            SequenceStatus.WAITING_EXPLORATION,
        ]

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> Union[str, None]:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SequenceData:
    """Data associated with a sequence.


    Args:
        prompt_token_ids: The token IDs of the prompt.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        prompt_token_ck_status: status of each token in all tokens: 0: prompt token, 1: continue token, 2: expand token, -1: not a ck sequence
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """

    def __init__(
        self,
        prompt_token_ids: List[int],
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.prompt_token_ck_status = [0] * len(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.cumulative_logprob = 0.0

    def append_token_id(self, token_id: int, logprob: float) -> None:
        self.output_token_ids.append(token_id)
        self.cumulative_logprob += logprob

    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids
    
    def get_token_ck_status_ids(self) -> List[int]:
        return self.prompt_token_ck_status

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"prompt_token_ck_status={self.prompt_token_ck_status}, "
                f"cumulative_logprob={self.cumulative_logprob})")


class Sequence:
    """Stores the data, status, and block information of a sequence.

    Args:
        seq_id: The ID of the sequence.
        prompt: The prompt of the sequence.
        prompt_token_ids: The token IDs of the prompt.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
    """

    def __init__(
        self,
        seq_id: int,
        prompt: str,
        prompt_token_ids: List[int],
        block_size: int,
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.block_size = block_size

        self.data = SequenceData(prompt_token_ids)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""

        self.logical_token_blocks: List[LogicalTokenBlock] = []
        # Initialize the logical token blocks with the prompt token ids.
        self._append_tokens_to_blocks(prompt_token_ids)
        self.status = SequenceStatus.WAITING

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        self.ck_positive_logits = []
        self.cumulative_ck_positive_logits = 0
        self.ck_status_record = []
        # Input + output tokens
        self.tokens: Optional[List[str]] = None
        self.ck_status = 0 # 0 means initialization, 1 means continue, 2 means expand
        self.cached_score = None

    def _append_logical_block(self) -> None:
        block = LogicalTokenBlock(
            block_number=len(self.logical_token_blocks),
            block_size=self.block_size,
        )
        self.logical_token_blocks.append(block)

    def _append_tokens_to_blocks(self, token_ids: List[int]) -> None:
        cursor = 0
        while cursor < len(token_ids):
            if not self.logical_token_blocks:
                self._append_logical_block()

            last_block = self.logical_token_blocks[-1]
            if last_block.is_full():
                self._append_logical_block()
                last_block = self.logical_token_blocks[-1]

            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append_tokens(token_ids[cursor:cursor +
                                               num_empty_slots])
            cursor += num_empty_slots
            
    def append_ck_positive_logit(self, positive_logit: int):
        self.ck_positive_logits.append(positive_logit)
        # accumulate the log of positive logits
        # self.cumulative_ck_positive_logits += math.log(positive_logit)
        self.cumulative_ck_positive_logits += positive_logit

    def get_ck_status(self):
        return self.ck_status
    
    def update_ck_status(self, sampling_params):
        # 0 means initialization, 1 means continue, 2 means expand
        continue_count = 0
        for tmp_ck_status_id in self.ck_status_record:
            if tmp_ck_status_id == 1:
                continue_count += 1
            else:
                continue_count = 0
        if continue_count == sampling_params.ck_k-1:
            next_state = 2
        elif continue_count < sampling_params.ck_k-1:
            next_state = 1
        else:
            raise NotImplementedError('continue_count should not be larger than ck_k-1')
        self.ck_status_record.append(next_state)
        self.ck_status = next_state
        self.data.prompt_token_ck_status.append(next_state)

    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, float],
    ) -> None:
        assert token_id in logprobs
        self._append_tokens_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        self.data.append_token_id(token_id, logprobs[token_id])

    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> List[int]:
        return self.data.get_token_ids()
    
    def get_token_ck_status_ids(self) -> List[int]:
        return self.data.get_token_ck_status_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> List[int]:
        return self.data.output_token_ids

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def get_beam_search_score(self,
                              length_penalty: float = 0.0,
                              seq_len: Optional[int] = None,
                              eos_token_id: Optional[int] = None) -> float:
        """Calculate the beam search score with length penalty.

        Adapted from

        https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
        """
        if seq_len is None:
            seq_len = self.get_len()
            # NOTE: HF implementation does not count the EOS token
            # towards the length, we align with that here for testing.
            if (eos_token_id is not None
                    and self.get_last_token_id() == eos_token_id):
                seq_len -= 1
        return self.get_cumulative_logprob() / (seq_len**length_penalty)

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)
    
    def is_aborted(self) -> bool:
        if self.status == SequenceStatus.FINISHED_ABORTED:
            return True
        else:
            return False
        
    def get_output_text(self):
        return self.output_text
    
    def set_as_aborted(self):
        self.status = SequenceStatus.FINISHED_ABORTED

    # def fork(self, new_seq_id: int) -> "Sequence":
    #     new_seq = copy.deepcopy(self)
    #     new_seq.seq_id = new_seq_id
    #     return new_seq
    
    def fork(self, new_seq_id: int) -> "Sequence":
        new_seq = Sequence(
            seq_id=new_seq_id,
            prompt=self.prompt,
            prompt_token_ids=self.data.prompt_token_ids.copy(),
            block_size=self.block_size
        )
        new_seq.data = SequenceData(self.data.prompt_token_ids.copy())
        new_seq.data.output_token_ids = self.data.output_token_ids.copy()
        new_seq.data.cumulative_logprob = self.data.cumulative_logprob
        new_seq.data.prompt_token_ck_status = self.data.prompt_token_ck_status.copy()
        new_seq.output_logprobs = copy.copy(self.output_logprobs)
        new_seq.output_text = self.output_text
        new_seq.logical_token_blocks = []
        for block in self.logical_token_blocks:
            new_block = LogicalTokenBlock(
                block_number=block.block_number,
                block_size=block.block_size
            )
            new_block.token_ids = block.token_ids.copy()
            new_block.num_tokens = block.num_tokens
            new_seq.logical_token_blocks.append(new_block)
        new_seq.status = self.status
        new_seq.prefix_offset = self.prefix_offset
        new_seq.read_offset = self.read_offset
        new_seq.ck_positive_logits = copy.copy(self.ck_positive_logits)
        new_seq.cumulative_ck_positive_logits = self.cumulative_ck_positive_logits
        new_seq.ck_status_record = copy.copy(self.ck_status_record)
        new_seq.tokens = self.tokens.copy() if self.tokens else None
        new_seq.ck_status = self.ck_status
        new_seq.cached_score = self.cached_score
        return new_seq
        
    def get_ck_overall_score(self, ck_length_penalty: float = 1.0):
        # seq_len = self.get_len()

        return sum(self.ck_positive_logits)/len(self.ck_positive_logits)

        # print("Using `get_cumulative_logprob` as ck_positive_logits")
        # return self.get_cumulative_logprob()/self.get_len()

    
    def get_info(self, ck_length_penalty=1.0) -> dict:
        return {'seq_id': self.seq_id,
                'output_text': self.output_text,
                'state': str(self.status),
                'log_prob': self.get_cumulative_logprob(),
                'ck_positive_logits': self.ck_positive_logits,
                'cumulative_ck_positive_logits': self.cumulative_ck_positive_logits,
                'overall_ck_score': self.get_ck_overall_score(ck_length_penalty=ck_length_penalty),
                'ck_status_record': self.ck_status_record,
                'ck_status': self.ck_status,
                }

    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"output_text={self.output_text},"
                f"ck_positive_logits={self.ck_positive_logits},"
                f"ck_status_record={self.ck_status_record},"
                f"cumulative_ck_positive_logits={self.cumulative_ck_positive_logits},"
                f"log_prob={self.get_cumulative_logprob()},"
                f"status={self.status.name}, "
                f"num_blocks={len(self.logical_token_blocks)})")


class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
    """

    def __init__(
        self,
        request_id: str,
        seqs: List[Sequence],
        sampling_params: SamplingParams,
        arrival_time: float,
    ) -> None:
        self.request_id = request_id
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        self.prompt_logprobs: Optional[PromptLogprobs] = None
        self.waiting_seq_ids = list()
        self.seq_id_2_explorations = dict()
        self.exploration_id_2_parent_ids = dict()
        self.confirmed_seq = None

    @property
    def prompt(self) -> str:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return next(iter(self.seqs_dict.values())).prompt

    @property
    def prompt_token_ids(self) -> List[int]:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return next(iter(self.seqs_dict.values())).data.prompt_token_ids

    def get_max_num_running_seqs(self) -> int:
        """The maximum number of sequences running in parallel in the remaining
        lifetime of the request."""
        if self.sampling_params.use_beam_search:
            # For beam search, maximally there will always be `best_of` beam
            # candidates running in the future.
            return self.sampling_params.best_of
        else:
            if self.sampling_params.ck_mode == 'StreamingCheck':
                return self.sampling_params.ck_n
            elif self.sampling_params.ck_mode == 'StreamingTreeSearch':
                # return self.sampling_params.ck_n * self.sampling_params.ck_d
                return self.sampling_params.ck_n ** self.sampling_params.ck_d
            else:
                if self.sampling_params.best_of > self.num_seqs():
                    # At prompt stage, the sequence group is not yet filled up
                    # and only have one sequence running. However, in the
                    # generation stage, we will have `best_of` sequences running.
                    return self.sampling_params.best_of
                # At sampling stages, return the number of actual sequences
                # that are not finished yet.
                return self.num_unfinished_seqs()

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        if status is None:
            return list(self.seqs_dict.values())
        else:
            return [
                seq for seq in self.seqs_dict.values() if seq.status == status
            ]
            
    def get_seqs_for_logits(
        self, ) -> List[Sequence]:
        return list(self.seqs_dict.values())

    def get_unfinished_seqs(self) -> List[Sequence]:
        return [
            seq for seq in self.seqs_dict.values() if not seq.is_finished()
        ]

    def get_finished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs_dict.values() if seq.is_finished()]

    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        return len(self.get_seqs(status))

    def num_unfinished_seqs(self) -> int:
        return len(self.get_unfinished_seqs())

    def num_finished_seqs(self) -> int:
        return len(self.get_finished_seqs())

    def find(self, seq_id: int) -> Sequence:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        return self.seqs_dict[seq_id]

    def add(self, seq: Sequence) -> None:
        if seq.seq_id in self.seqs_dict:
            raise ValueError(f"Sequence {seq.seq_id} already exists.")
        self.seqs_dict[seq.seq_id] = seq

    def remove(self, seq_id: int) -> None:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        del self.seqs_dict[seq_id]

    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.get_seqs())
    
    def get_ck_status(self):
        existing_sequences = self.get_seqs(status=SequenceStatus.RUNNING)
        assert len(existing_sequences) > 0
        return existing_sequences[0].get_ck_status()

    def get_ck_exploration_status(self):
        if len(self.waiting_seq_ids) == 0:
            return 1
        waiting_seq_token_length = len(self.find(self.waiting_seq_ids[0]).ck_status_record)
        exploration_seqs = self.get_seqs(status=SequenceStatus.RUNNING)
        exploration_seq_token_lengths = [len(tmp_seq.ck_status_record) for tmp_seq in exploration_seqs]
        if len(exploration_seqs) == 0:
            return 1
        exploration_seq_token_length = max(exploration_seq_token_lengths)
        if exploration_seq_token_length - waiting_seq_token_length < self.sampling_params.ck_k * self.sampling_params.ck_d:
            return 1
        else:
            return 2
        
    def add_exploration(self, parent_id, exploration_id):
        if parent_id not in self.seq_id_2_explorations:
            self.seq_id_2_explorations[parent_id] = []
        if exploration_id not in self.seq_id_2_explorations[parent_id]:
            self.seq_id_2_explorations[parent_id].append(exploration_id)
        self.exploration_id_2_parent_ids[exploration_id] = parent_id
        
    def get_parent_ck_overall_score(self, parent_id, ck_length_penalty):
        # If parent_id does not have a sub-exploration sequence, its ck score is the final score
        if parent_id not in self.seq_id_2_explorations:
            return [self.find(parent_id).get_ck_overall_score(ck_length_penalty)]
        # If there are subexploration sequences, iterate to get the ck scores of the deepest subexploration sequences and return a list of them
        total_scores = list()
        for tmp_child_id in self.seq_id_2_explorations[parent_id]:
            all_ck_scores = self.get_parent_ck_overall_score(tmp_child_id, ck_length_penalty)
            total_scores.extend(all_ck_scores)
        return total_scores
    
    def abort_sequence_and_explorations(self, seq_id):
        seqs_for_free = list()
        self.find(seq_id).status = SequenceStatus.FINISHED_ABORTED
        seqs_for_free.append(self.find(seq_id))
        if seq_id not in self.seq_id_2_explorations:
            return seqs_for_free
        else:
            for tmp_child_id in self.seq_id_2_explorations[seq_id]:
                seqs_for_free.extend(self.abort_sequence_and_explorations(tmp_child_id))
            return seqs_for_free

    def get_best_finished_score(self):
        finisehd_seqs = self.get_seqs(status=SequenceStatus.FINISHED_LENGTH_CAPPED) + self.get_seqs(status=SequenceStatus.FINISHED_STOPPED)
        if len(finisehd_seqs) > 0:
            sorting_key = lambda seq: seq.get_ck_overall_score(self.sampling_params.ck_length_penalty)
            sorted_seqs = sorted(finisehd_seqs, key=sorting_key, reverse=True)
            selected_seq = sorted_seqs[0]
            return selected_seq.get_ck_overall_score(self.sampling_params.ck_length_penalty)
        else:
            return None
        
    def check_and_merge_exploration(self):
        seqs_for_free = []
        ck_exploration_status = self.get_ck_exploration_status()
        if ck_exploration_status == 2:
            # We need to merge the exploration
            exploration_score = []
            for tmp_parent_seq_id in self.waiting_seq_ids:
                if self.sampling_params.ck_d > 0:
                    total_scores = self.get_parent_ck_overall_score(tmp_parent_seq_id, self.sampling_params.ck_length_penalty)
                else:
                    total_scores = [self.find(tmp_parent_seq_id).get_ck_overall_score(self.sampling_params.ck_length_penalty)]
                # For a sequence in waiting_seq_ids, its score is the maximum of the score of its deepest subexploration sequence
                # exploration_score.append((tmp_parent_seq_id, sum(total_scores)/len(total_scores)))
                exploration_score.append((tmp_parent_seq_id, max(total_scores)))
            # Keep the waiting_seq_ids with the highest score, set it to confirmed_seq, and release the other sequences
            sorted_parent_ids = sorted(exploration_score, key=lambda x: x[1], reverse=True)
            for tmp_aborted_parent_id in sorted_parent_ids[1:]:
                seqs_for_free.extend(self.abort_sequence_and_explorations(tmp_aborted_parent_id[0]))
            previous_confirmed_seq = self.confirmed_seq
            if previous_confirmed_seq is not None:
                previous_confirmed_seq.status = SequenceStatus.FINISHED_IGNORED
                seqs_for_free.append(previous_confirmed_seq)
            self.confirmed_seq = self.find(sorted_parent_ids[0][0])
            # If the selected confirmed_seq has already been explored, the next waiting_seq_ids is the direct sub-exploration sequence of confirmed_seq
            if sorted_parent_ids[0][0] in self.seq_id_2_explorations:
                self.waiting_seq_ids = self.seq_id_2_explorations[sorted_parent_ids[0][0]]
            else:
                # The selected parent has finished
                self.waiting_seq_ids = []
            return seqs_for_free
        else:
            return seqs_for_free

        return cotinue_seq_ids, seqs_for_free
        
    
    def get_all_info(self, ck_length_penalty=1.0):
        all_info = dict()
        all_info['seq_id_2_explorations'] = self.seq_id_2_explorations
        all_info['exploration_id_2_parent_ids'] = self.exploration_id_2_parent_ids
        all_info['waiting_seq_ids'] = self.waiting_seq_ids
        if self.confirmed_seq is not None:
            all_info['confirmed_seq'] = self.confirmed_seq.get_info(ck_length_penalty=ck_length_penalty)
        else:
            all_info['confirmed_seq'] = None
        all_info['seqs'] = []
        for tmp_seq in self.get_seqs():
            all_info['seqs'].append(tmp_seq.get_info(ck_length_penalty=ck_length_penalty))
        return all_info
    
    def get_best_finished_sequence_score(self):
        finished_sequences = self.get_seqs(status=SequenceStatus.FINISHED_LENGTH_CAPPED) + self.get_seqs(status=SequenceStatus.FINISHED_STOPPED)
        scores = [tmp_seq.get_ck_overall_score(self.sampling_params.ck_length_penalty) for tmp_seq in finished_sequences]
        if len(scores) == 0:
            return None
        return max(scores)

    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"confirmed_seq={self.confirmed_seq}, "
                f"num_waiting_seqs={len(self.waiting_seq_ids)}, "
                f"num_unfinished_seqs={self.num_unfinished_seqs()}, "
                f"num_finished_seqs={self.num_finished_seqs()}, "
                f"num_seqs={len(self.seqs_dict)})")


class SequenceGroupMetadata:
    """Metadata for a sequence group. Used to create `InputMetadata`.


    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. (Seq id -> list of physical block
            numbers)
    """

    def __init__(
        self,
        request_id: str,
        is_prompt: bool,
        seq_data: Dict[int, SequenceData],
        sampling_params: SamplingParams,
        block_tables: Dict[int, List[int]],
    ) -> None:
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = seq_data
        self.sampling_params = sampling_params
        self.block_tables = block_tables


class SequenceOutput:
    """The model output associated with a sequence.

    Args:
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """

    def __init__(
        self,
        parent_seq_id: int,
        output_token: int,
        logprobs: Dict[int, float],
    ) -> None:
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f"SequenceOutput(parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}, "
                f"logprobs={self.logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutput):
            raise NotImplementedError()
        return (self.parent_seq_id == other.parent_seq_id
                and self.output_token == other.output_token
                and self.logprobs == other.logprobs)


class SequenceGroupOutput:
    """The model output associated with a sequence group."""

    def __init__(
        self,
        samples: List[SequenceOutput],
        prompt_logprobs: Optional[PromptLogprobs],
    ) -> None:
        self.samples = samples
        self.prompt_logprobs = prompt_logprobs

    def __repr__(self) -> str:
        return (f"SequenceGroupOutput(samples={self.samples}, "
                f"prompt_logprobs={self.prompt_logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceGroupOutput):
            raise NotImplementedError()
        return (self.samples == other.samples
                and self.prompt_logprobs == other.prompt_logprobs)


# For each sequence group, we generate a list of SequenceOutput object,
# each of which contains one possible candidate for the next token.
SamplerOutput = List[SequenceGroupOutput]
