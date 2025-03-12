import argparse
import json
import ssl
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
# from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid
import time

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    print(f"original request: {request_dict}")
    time_anchor = time.time()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)
    print(f"time for setting up the generator: {time.time() - time_anchor}")
    time_anchor = time.time()
    

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        start_time = time.time()
        num_of_tokens = 0
        average_time = list()
        token_time = list()
        previous_time = time.time()
        async for request_output in results_generator:
            num_of_tokens += 1
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            # print(f"new_token: {text_outputs[0]}, time_usage: {time.time() - start_time}")
            average_time.append((time.time()-start_time)/num_of_tokens)
            token_time.append((time.time()-previous_time))
            previous_time = time.time()
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")
        print(f"average: {average_time} token: {token_time}")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)

@app.post("/ck_generate")
async def ck_generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    print("original request", request_dict)
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    return_all_info = request_dict.pop("return_all_info", False)
    # request_dict['ck_mode'] = 'StreamingCheck'
    request_dict['stop_token_ids'] = [128258]
    sampling_params = SamplingParams(**request_dict)
    print(f"sampling_params: {sampling_params}")
    
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        previous_text = ''
        start_time = time.time()
        async for request_output in results_generator: 
            # print("request_output.outputs", request_output.outputs)           
            if len(request_output.outputs) == 0:
                # We do not have response ready yet.
                continue
            elif len(request_output.outputs) == 1:
                current_text = request_output.outputs[0].text
                new_token = current_text.replace(previous_text, '')
                previous_text = current_text
                if len(new_token) > 0:
                    time_usage = time.time() - start_time
                    start_time = time.time()
                    print(f"{time.ctime()} new_token: |{repr(new_token)}|, time_usage: {time_usage:.2f}s")
                    if return_all_info:
                        # This will be very time consuming since `all_info` is large !
                        print("return all_info")
                        ret = {"new_token": new_token, "all_info": request_output.all_info}
                    else:
                        ret = {"new_token": new_token}
                    yield (json.dumps(ret) + "\0").encode("utf-8")
            else:
                raise ValueError(f"Unexpected number of outputs: {len(request_output.outputs)}")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [output.text for output in final_output.outputs]
    ret = {"text": text_outputs[0]}
    # print(ret)
    return JSONResponse(ret)

@app.post("/ck_check_reward_score")
async def ck_check_reward_score(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    # print(request_dict)
    prompt = request_dict.pop("prompt")
    response = request_dict.pop("response")
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    positive_logits = await engine.get_logits(prompt, response, sampling_params, request_id)

    
    ret = {"positive_logits": positive_logits}
    # print(ret)
    return JSONResponse(ret)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        # engine_args, usage_context=UsageContext.API_SERVER)
        engine_args)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
