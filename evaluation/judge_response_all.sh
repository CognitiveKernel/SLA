ProjectPath="<path>/<to>/<this>/<project>"


benchmark="mt_bench"
save_dir="${ProjectPath}/evaluation/mt_bench"
model_answer_dir="/${ProjectPath}/eval/FastChat/fastchat/llm_judge/data/mt_bench/model_answer"
question_path="/${ProjectPath}/eval/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
python ArmoRM_eval.py --benchmark ${benchmark} --question_path ${question_path} --model_answer_dir ${model_answer_dir} --save_dir ${save_dir}

benchmark="arenahard"
save_dir="/${ProjectPath}/Evaluation/arena-hard-v0.1"
model_answer_dir="/${ProjectPath}/eval/arena-hard-auto/data/arena-hard-v0.1/model_answer"
question_path="/${ProjectPath}/eval/arena-hard-auto/data/arena-hard-v0.1/question.jsonl"
python ArmoRM_eval.py --benchmark ${benchmark} --question_path ${question_path} --model_answer_dir ${model_answer_dir} --save_dir ${save_dir}

benchmark="alpacaeval"
save_dir="/${ProjectPath}/Evaluation/alpacaeval"
model_answer_dir="/${ProjectPath}/Evaluation/alpacaeval"
question_path="/${ProjectPath}/eval/alpacaeval/tatsu-lab-alpaca_eval.jsonl"
python ArmoRM_eval.py --benchmark ${benchmark} --question_path ${question_path} --model_answer_dir ${model_answer_dir} --save_dir ${save_dir}
