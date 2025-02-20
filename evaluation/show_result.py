import json
import os
import os.path as osp
import numpy as np
from collections import defaultdict

mtbench_dir_path = "./mt_bench/model_judgment/"
arenahard_dir_path = "./arena-hard-v0.1/model_judgment/"
alpacaeval_dir_path = "./alpacaeval/results/"

baseline = 'Meta-Llama-3-8B-Instruct-greedy.jsonl'
compares = [
    'Meta-Llama-3-8B-Instruct-temperature_0.8.jsonl',
    'Meta-Llama-3-8B-Instruct-topp_0.9.jsonl',
    'Meta-Llama-3-8B-Instruct-topk_50.jsonl',
    'Meta-Llama-3-8B-Instruct-beam_4.jsonl',
]

collector = defaultdict(dict)

# mtbench
dir_path = mtbench_dir_path
for path_a in compares:
    if os.path.exists(osp.join(dir_path, path_a)):
        path_b = baseline
        data_a = [json.loads(line) for line in open(osp.join(dir_path, path_a))]
        data_b = [json.loads(line) for line in open(osp.join(dir_path, path_b))]
        scores_a = []
        scores_b = []
        for a, b in zip(data_a, data_b):
            assert a['question_id'] == b['question_id']
            scores_a += a['ArmoRM_scores']
            scores_b += b['ArmoRM_scores']
            
        win = np.mean(np.array(scores_a) > np.array(scores_b))*100
        tie = np.mean(np.array(scores_a) == np.array(scores_b))*100
        loss = np.mean(np.array(scores_a) < np.array(scores_b))*100
    
        key = path_a.replace(".jsonl", "")
        collector[key]['mtbench'] = {'win':win, 'tie':tie, 'loss':loss, 'equity':win+tie/2}
    else:
        key = path_a.replace(".jsonl", "")
        collector[key]['mtbench'] = {'win':0, 'tie':0, 'loss':0, 'equity':0}
        
# arenahard
dir_path = arenahard_dir_path
for path_a in compares:
    if os.path.exists(osp.join(dir_path, path_a)):
        path_b = baseline
        data_a = [json.loads(line) for line in open(osp.join(dir_path, path_a))]
        data_b = [json.loads(line) for line in open(osp.join(dir_path, path_b))]
        scores_a = []
        scores_b = []
        for a, b in zip(data_a, data_b):
            assert a['question_id'] == b['question_id']
            scores_a += [a['ArmoRM_score']]
            scores_b += [b['ArmoRM_score']]
    
        win = np.mean(np.array(scores_a) > np.array(scores_b))*100
        tie = np.mean(np.array(scores_a) == np.array(scores_b))*100
        loss = np.mean(np.array(scores_a) < np.array(scores_b))*100
    
        key = path_a.replace(".jsonl", "")
        collector[key]['arenahard'] = {'win':win, 'tie':tie, 'loss':loss, 'equity':win+tie/2}
    else:
        key = path_a.replace(".jsonl", "")
        collector[key]['arenahard'] = {'win':0, 'tie':0, 'loss':0, 'equity':0}
        
# alpacaeval
dir_path = alpacaeval_dir_path 
for path_a in compares:
    pa = path_a.replace(".jsonl", "")
    path_b = baseline.replace(".jsonl", "")

    if os.path.exists(osp.join(dir_path, pa, 'model_judgment.jsonl')):
        data_a = [json.loads(line) for line in open(osp.join(dir_path, pa, 'model_judgment.jsonl'))]
        data_b = [json.loads(line) for line in open(osp.join(dir_path, path_b, 'model_judgment.jsonl'))]
        scores_a = []
        scores_b = []
        for a, b in zip(data_a, data_b):
            assert a['instruction'] == b['instruction']
            scores_a += [a['ArmoRM_score']]
            scores_b += [b['ArmoRM_score']]
            
        win = np.mean(np.array(scores_a) > np.array(scores_b))*100
        tie = np.mean(np.array(scores_a) == np.array(scores_b))*100
        loss = np.mean(np.array(scores_a) < np.array(scores_b))*100
    
        key = path_a.replace(".jsonl", "")
        collector[key]['alpacaeval'] = {'win':win, 'tie':tie, 'loss':loss, 'equity':win+tie/2}
    
    else:
        key = path_a.replace(".jsonl", "")
        collector[key]['alpacaeval'] = {'win':0, 'tie':0, 'loss':0, 'equity':0}


for key in collector:
    s = f"{key:<70}"
    s += f"{collector[key]['mtbench']['win']:.1f} & {collector[key]['mtbench']['tie']:.1f} & {collector[key]['mtbench']['loss']:.1f} & "
    s += f"{collector[key]['arenahard']['win']:.1f} & {collector[key]['arenahard']['tie']:.1f} & {collector[key]['arenahard']['loss']:.1f} & "
    s += f"{collector[key]['alpacaeval']['win']:.1f} & {collector[key]['alpacaeval']['tie']:.1f} & {collector[key]['alpacaeval']['loss']:.1f} & "
    s += f"{(collector[key]['mtbench']['equity']+collector[key]['arenahard']['equity']+collector[key]['alpacaeval']['equity'])/3:.1f}"
    
    print(s)