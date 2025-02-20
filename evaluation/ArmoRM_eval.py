import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import os
import argparse

def load_rm():
   path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
   print(f"Load model: {path}")
   model = AutoModelForSequenceClassification.from_pretrained(path, 
                                 device_map='cuda',
                                 trust_remote_code=True, torch_dtype='auto')
   tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
   return model, tokenizer

def get_rm_score(messages, model, tokenizer):
   input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
   with torch.no_grad():
      output = model(input_ids)
      preference_score = output.score.cpu().float() 
   return float(preference_score[0])


parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, default='arenahard')
parser.add_argument('--question_path', type=str, default='')
parser.add_argument('--model_answer_dir', type=str, default='')
parser.add_argument('--save_dir', type=str, default='')


args = parser.parse_args()

### -------------------- Score MT-Bench using ArmoRM --------------------
if args.benchmark in ['mt_bench']:
   model, tokenizer = load_rm()

   dir_path = args.save_dir
   model_answer_dir = args.model_answer_dir
   question_path = args.question_path
   
   questions = [json.loads(line) for line in open(question_path).readlines()]
   model_names = os.listdir(model_answer_dir)

   for name in model_names:

      # check
      if os.path.exists(os.path.join(dir_path, 'model_judgment', name)):
         continue

      # load model responses
      print(f"Load: {name}")
      path = os.path.join(model_answer_dir, name)
      responses = [json.loads(line) for line in open(path).readlines()]
      id2responses = {r['question_id']:r for r in responses}

      # get rm score
      data = []
      for i in tqdm(range(len(questions))):
         question = questions[i]
         response = id2responses[question['question_id']]
         rm_scores = []

         # turn 1
         message = [
            {'role': 'user', 'content': question['turns'][0]},
            {'role': 'assistant', 'content': response['choices'][0]['turns'][0]},
         ]
         rm_scores.append(get_rm_score(message, model, tokenizer))

         # turn 2
         message = [
            {'role': 'user', 'content': question['turns'][0]},
            {'role': 'assistant', 'content': response['choices'][0]['turns'][0]},
            {'role': 'user', 'content': question['turns'][1]},
            {'role': 'assistant', 'content': response['choices'][0]['turns'][1]},
         ]
         rm_scores.append(get_rm_score(message, model, tokenizer))

         response['ArmoRM_scores'] = rm_scores
         data.append(response)

      # save results
      print(f"Save: {name}")
      with open(os.path.join(dir_path, 'model_judgment', name), 'w') as f:
         f.writelines([json.dumps(item)+'\n' for item in data])

elif args.benchmark in ['arenahard']:

   ## -------------------- Score Arena-Hard using ArmoRM --------------------

   model, tokenizer = load_rm()

   dir_path = args.save_dir
   model_answer_dir = args.model_answer_dir
   question_path = args.question_path
   
   questions = [json.loads(line) for line in open(question_path).readlines()]
   model_names = os.listdir(model_answer_dir)

   for name in model_names:

      # check
      if os.path.exists(os.path.join(dir_path, 'model_judgment', name)):
         continue

      # load model responses
      print(f"Load: {name}")
      path = os.path.join(model_answer_dir, name)
      responses = [json.loads(line) for line in open(path).readlines()]
      id2responses = {r['question_id']:r for r in responses}

      # get rm score
      data = []
      for i in tqdm(range(len(questions))):
         question = questions[i]
         response = id2responses[question['question_id']]

         message = [
            {'role': 'user', 'content': question['turns'][0]['content']}, ## note the ['content'] for Arena-Hard
            {'role': 'assistant', 'content': response['choices'][0]['turns'][0]['content']},
         ]

         rm_score = get_rm_score(message, model, tokenizer)

         response['ArmoRM_score'] = rm_score
         data.append(response)

      # save results
      print(f"Save: {name}")
      with open(os.path.join(dir_path, 'model_judgment', name), 'w') as f:
         f.writelines([json.dumps(item)+'\n' for item in data])

elif args.benchmark in ['alpacaeval']:

   ## -------------------- Score Alpaca Eval using ArmoRM --------------------

   dir_path = args.save_dir
   model_answer_dir = args.model_answer_dir
   question_path = args.question_path

   model, tokenizer = load_rm()

   model_names = os.listdir(model_answer_dir)

   for name in model_names:

      # check
      if not os.path.isdir(os.path.join(model_answer_dir, name)):
         continue
      if os.path.exists(os.path.join(model_answer_dir, name, 'model_judgment.jsonl')):
         continue

      # load model responses
      print(f"Load: {name}")
      path = os.path.join(model_answer_dir, name, 'model_output.jsonl')
      responses = [json.loads(line) for line in open(path).readlines()]

      # get rm score
      data = []
      for i in tqdm(range(len(responses))):
         item = responses[i]
         question = item["instruction"]
         response = item["output"]

         message = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': response},
         ]

         rm_score = get_rm_score(message, model, tokenizer)

         item['ArmoRM_score'] = rm_score
         data.append(item)

      # save results
      print(f"Save: {name}")
      with open(os.path.join(dir_path, name, 'model_judgment.jsonl'), 'w') as f:
         f.writelines([json.dumps(item)+'\n' for item in data])


else:
   raise NotImplemented