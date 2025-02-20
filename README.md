<h1 align="center">
Streaming Looking Ahead
</h1>

We introduce Streaming Looking Ahead (SLA), an efficient LLM Search Algorithm.

## Key Innovations

- **Unified Model**: Combines policy and reward models into a single model, thereby saving on the computational overhead that would otherwise be required to run an additional reward model proxy.
- **Token-Level Feedback**: Provides feedback at the token level, which facilitates the implementation of looking ahead and other search algorithms at this granular level, aiding in the discovery of better responses.
- **Integrated Search Algorithms**: Incorporates search algorithms within the efficient inference framework vLLM, further enhancing efficiency.

## Reward Transformer Training

### Step 1: Set up the Training Environment.

```bash
cd ./training
conda create -n train_env python=3.10 && conda activate train_env
pip install -r requirements.txt
```

### Step 2: Start Training.

You can modify training parameters in the YAML configuration file. The trained model will be saved in the directory specified by the `output_dir` parameter in the training configuration file.

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/RT_run.py RT_training_configs/default.yaml
```

Our training codebase is built upon the [SimPO repo](https://github.com/princeton-nlp/SimPO).

## Streaming Looking Ahead Inference

### Step 1: Set up the Inference Environment.

```bash
cd ./vllm_streaming
conda create -n vllm_streaming python=3.10 & conda activate vllm_streaming
pip install numpy==1.26.4 && pip install -e .
```

### Step 2: Launch the vllm service.

```bash
python vllm/entrypoints/api_server.py --host 0.0.0.0 --port 6667 --model "<path to the model>"
```

## Evaluation

### Step 1: Generate Responses.

```bash
cd ./evaluation
bash generate_response_all.sh
```

### Step 2: Judge the Responses with Reward Model.

```bash
bash judge_response_all.sh
```

### Step 3: Show the Evaluation Result.

```bash
python show_result.py
```

## Citation

If you find this repo helpful, please cite

```
@article{SLA,
  title={Streaming Looking Ahead with Token-level Self-reward},
  author={Zhang, Hongming and Hong, Ruixin and Yu, Dong},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

If you have any questions or suggestions, feel free to contact all above authors or open an issue. We will try to respond to them as soon as possible.
