import sys
import time
from tqdm import tqdm
import json
import os
import asyncio
import random
import string
import argparse
import shortuuid
from base_model_connection import BaseModelConnection

def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    # model args
    parser.add_argument('--ip', type=str, default="29.81.226.114:6667", help='IP address of the model server')
    parser.add_argument('--model_name', type=str, default="debug_model", help='Model name used to saved in the result file')
    parser.add_argument('--chat_template', type=str, default="ck")
    parser.add_argument('--backend', type=str, default="ck_vllm")

    # data args
    parser.add_argument('--benchmark', type=str, default="mt_bench")
    parser.add_argument('--data_path', type=str, default="", help='Path to the input data file')
    parser.add_argument('--save_path', type=str, default="", help='Path to save the output file')
    
    # genration args
    parser.add_argument('--ck_mode', type=str, default="StreamingTreeSearch")
    parser.add_argument('--ck_n', type=int, default=1)
    parser.add_argument('--ck_k', type=int, default=8)
    parser.add_argument('--ck_d', type=int, default=0)
    parser.add_argument('--max_tokens', type=int, default=2048)

    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num_beams', type=int, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--top_p', type=float, default=None)

    args = parser.parse_args()
    return args


async def get_output(message, backend='ck_vllm', **kwarg):
    if backend == 'ck_vllm':
        all_response = ""
        all_info = None
        async for tmp_response in current_connection.ck_generate_stream_eval(message, **kwarg):
            all_response += tmp_response["new_token"]
            # all_info = tmp_response["all_info"]
        return all_response, all_info
    elif backend == 'vllm':
        all_response = current_connection.get_response(message, **kwarg)
        all_info = None
        return all_response, all_info

async def get_ans_mtbench(args, data):
    answers = []
    for item in tqdm(data):
        message = []
        turn_responses = []
        for j in range(len(item["turns"])):
            message.append({'role':'user', 'content': item["turns"][j]})
            all_response, all_info = await get_output(message, 
                backend=args.backend,
                ck_mode=args.ck_mode,
                ck_n=args.ck_n,
                ck_k=args.ck_k,
                ck_d=args.ck_d,
                max_tokens=args.max_tokens,

                do_sample = args.do_sample,
                temperature = args.temperature,
                num_beams = args.num_beams,
                top_k = args.top_k,
                top_p = args.top_p,
            )
            message.append({'role':'assistant', 'content': all_response})
            turn_responses.append(all_response)
            print(repr(turn_responses))

        choices = [{"index": 0, "turns": turn_responses}]

        answers.append({
            'question_id': item['question_id'],
            'answer_id': shortuuid.uuid(),
            'model_id': args.model_name,
            'choices': choices,
            'tstamp': time.time()
        })
        
    return answers

async def get_ans_arenahard(args, data):
    answers = []
    for item in tqdm(data):
        message = []
        turn_responses = []
        for j in range(len(item["turns"])):
            message.append({'role':'user', 'content': item["turns"][j]['content']})
            all_response, all_info = await get_output(message,     
                backend=args.backend,
                ck_mode=args.ck_mode,
                ck_n=args.ck_n,
                ck_k=args.ck_k,
                ck_d=args.ck_d,
                max_tokens=args.max_tokens,

                do_sample = args.do_sample,
                temperature = args.temperature,
                num_beams = args.num_beams,
                top_k = args.top_k,
                top_p = args.top_p,
            )
            message.append({'role':'assistant', 'content': all_response})
            turn_responses.append({'content': all_response, 'token_len': -1})

        print(turn_responses)
        
        choices = [{"index": 0, "turns": turn_responses}]

        answers.append({
            'question_id': item['question_id'],
            'answer_id': shortuuid.uuid(),
            'model_id': args.model_name,
            'choices': choices,
            'tstamp': time.time()
        })
        
    return answers

async def get_ans_alpacaeval(args, data):
    answers = []
    for item in tqdm(data):
        message = []
        message.append({'role':'user', 'content': item["instruction"]})
        all_response, all_info = await get_output(message,  
            backend=args.backend,              
            ck_mode=args.ck_mode,
            ck_n=args.ck_n,
            ck_k=args.ck_k,
            ck_d=args.ck_d,
            max_tokens=args.max_tokens,

            do_sample = args.do_sample,
            temperature = args.temperature,
            num_beams = args.num_beams,
            top_k = args.top_k,
            top_p = args.top_p,
        )

        # print(all_response)

        answers.append({
            'instruction': item["instruction"],
            'dataset': item["dataset"],
            'output': all_response,
            'generator': args.model_name,
        })

    return answers


async def get_ans_ultrafeedback(args, data):
    answers = []
    for item in tqdm(data):

        print(len(item["prompt"].split()))

        message = []
        message.append({'role':'user', 'content': item["prompt"]})
        all_response, _ = await get_output(message,  
            backend=args.backend,              
            ck_mode=args.ck_mode,
            ck_n=args.ck_n,
            ck_k=args.ck_k,
            ck_d=args.ck_d,
            max_tokens=args.max_tokens,

            do_sample = args.do_sample,
            temperature = args.temperature,
            num_beams = args.num_beams,
            top_k = args.top_k,
            top_p = args.top_p,
        )
        
        print(repr(all_response))

        answers.append({
            'prompt_id': item["prompt_id"],
            'prompt': item["prompt"],
            'output': all_response,
            'generator': args.model_name,
        })

    return answers

if __name__ == '__main__':

    args = get_args()
    assert args.benchmark in ['mtbench', 'arenahard', 'alpacaeval','ultrafeedback']

    current_connection = BaseModelConnection(ip=args.ip, chat_template=args.chat_template)

    data = [json.loads(line) for line in open(args.data_path).readlines()]

    get_ans_func = {
        'mtbench': get_ans_mtbench,
        'arenahard': get_ans_arenahard,
        'alpacaeval': get_ans_alpacaeval,
        'ultrafeedback': get_ans_ultrafeedback,
    }[args.benchmark]

    answers = asyncio.run(get_ans_func(args, data))

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        f.writelines([
            json.dumps(item)+'\n' for item in answers
        ])
    print(f"Saved result to {args.save_path}")



