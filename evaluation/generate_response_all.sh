ProjectPath="<path>/<to>/<this>/<project>"

##### Generation Parameters
# host_ip: ip address and port number of the vllm service
# the_hosted_model: the model name
# ck_n: search width, i.e., we consider ck_n branches at each branch point 
# ck_k: step size, i.e., the number of tokens contained in each chunk
# ck_d: search depth, i.e., we use information with depth ck_d for branch selection
#####

host_ip="127.0.0.1:6667"
the_hosted_model="llama-3-8b-instruct-RT-TT256-L1-seqbt"

backend="ck_vllm"
chat_template="llama3"
#### --------------------- MTbench ---------------------

ck_n=1
ck_k=10
ck_d=0
model_name="${the_hosted_model}-${ck_n}-${ck_k}-${ck_d}"
cd ${ProjectPath}/code
python generate_response.py --benchmark mtbench --chat_template ${chat_template} --backend ${backend}  \
--ip ${host_ip} --model_name ${model_name} --ck_n ${ck_n} --ck_k ${ck_k} --ck_d ${ck_d} \
--data_path "${ProjectPath}/eval/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl" \
--save_path "${ProjectPath}/eval/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/${model_name}.jsonl"

ck_n=2
ck_k=10
ck_d=2
model_name="${the_hosted_model}-${ck_n}-${ck_k}-${ck_d}"
cd ${ProjectPath}/code
python generate_response.py --benchmark mtbench --chat_template ${chat_template} --backend ${backend}  \
--ip ${host_ip} --model_name ${model_name} --ck_n ${ck_n} --ck_k ${ck_k} --ck_d ${ck_d} \
--data_path "${ProjectPath}/eval/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl" \
--save_path "${ProjectPath}/eval/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/${model_name}.jsonl"


#### --------------------- ArenaHard ---------------------

ck_n=1
ck_k=10
ck_d=0
model_name="${the_hosted_model}-${ck_n}-${ck_k}-${ck_d}"
cd ${ProjectPath}/code
python generate_response.py --benchmark arenahard --chat_template ${chat_template} --backend ${backend}  \
--ip ${host_ip} --model_name ${model_name} --ck_n ${ck_n} --ck_k ${ck_k} --ck_d ${ck_d} \
--data_path "${ProjectPath}/eval/arena-hard-auto/data/arena-hard-v0.1/question.jsonl" \
--save_path "${ProjectPath}/eval/arena-hard-auto/data/arena-hard-v0.1/model_answer/${model_name}.jsonl" 

ck_n=2
ck_k=10
ck_d=2
model_name="${the_hosted_model}-${ck_n}-${ck_k}-${ck_d}"
cd ${ProjectPath}/code
python generate_response.py --benchmark arenahard --chat_template ${chat_template} --backend ${backend}  \
--ip ${host_ip} --model_name ${model_name} --ck_n ${ck_n} --ck_k ${ck_k} --ck_d ${ck_d} \
--data_path "${ProjectPath}/eval/arena-hard-auto/data/arena-hard-v0.1/question.jsonl" \
--save_path "${ProjectPath}/eval/arena-hard-auto/data/arena-hard-v0.1/model_answer/${model_name}.jsonl" 


#### --------------------- Alpacaeval ---------------------

ck_n=1
ck_k=10
ck_d=0
model_name="${the_hosted_model}-${ck_n}-${ck_k}-${ck_d}"
cd ${ProjectPath}/code
python generate_response.py --benchmark alpacaeval --chat_template ${chat_template} --backend ${backend} \
--ip ${host_ip} --model_name ${model_name} --ck_n ${ck_n} --ck_k ${ck_k} --ck_d ${ck_d} \
--data_path "${ProjectPath}/eval/alpacaeval/tatsu-lab-alpaca_eval.jsonl" \
--save_path "${ProjectPath}/eval/alpacaeval/results/${model_name}/model_output.jsonl"

ck_n=2
ck_k=10
ck_d=2
model_name="${the_hosted_model}-${ck_n}-${ck_k}-${ck_d}"
cd ${ProjectPath}/code
python generate_response.py --benchmark alpacaeval --chat_template ${chat_template} --backend ${backend} \
--ip ${host_ip} --model_name ${model_name} --ck_n ${ck_n} --ck_k ${ck_k} --ck_d ${ck_d} \
--data_path "${ProjectPath}/eval/alpacaeval/tatsu-lab-alpaca_eval.jsonl" \
--save_path "${ProjectPath}/eval/alpacaeval/results/${model_name}/model_output.jsonl"
