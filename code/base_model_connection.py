import hmac
import os
import time
import uuid
import requests
import openai
import json
from jinja2 import Template

INFERENCE_SERVER_ENGINE = os.environ.get("INFERENCE_SERVER_ENGINE", "vLLM")

# STOPS = ["<|im_end|>", "<|im_start|>","<|start_header_id|>","<|end_header_id|>","<|eot_id|>"]
STOPS = ["<|im_end|>", "<|eot_id|>"]


class BaseModelConnection:
    def __init__(self, ip="30.207.99.138:8000", chat_template="ours"):
        "Initialize the base model connection"
        self.ip = ip
        self.chat_template = chat_template

    # def format_message(message: str, role: str, name: str = None) -> str:
    #     if role == "system" and name:
    #         return f"<|im_start|>{role} name={name}\n{message}<|im_end|>"
    #     else:
    #         return f"<|im_start|>{role}\n{message}<|im_end|>"

    def _generate_query(self, messages: list(), add_generation_prompt=True, remove_bos=False):
        """This function will generate the input messages to be the query we send to the base model.

        Args:
            messages (list): input messages
        """

        if self.chat_template == "ck":
            message_with_history = ""

            for tmp_round in messages:
                if tmp_round["role"] == "system":
                    if "name" in tmp_round:
                        message_with_history += f"<|im_start|>{tmp_round['role']} name={tmp_round['name']}\n{tmp_round['content']}<|im_end|>"
                    else:
                        message_with_history += f"<|im_start|>{tmp_round['role']}\n{tmp_round['content']}<|im_end|>"
                elif tmp_round["role"] == "user":
                    message_with_history += (
                        "<|im_start|>user\n" + tmp_round["content"] + "<|im_end|>"
                    )
                elif tmp_round["role"] == "assistant":
                    message_with_history += (
                        "<|im_start|>assistant\n" + tmp_round["content"] + "<|im_end|>"
                    )
                else:
                    raise NotImplementedError(
                        "Role {} is not implemented".format(tmp_round["role"])
                    )
            if add_generation_prompt:
                message_with_history += "<|im_start|>assistant\n"
            return message_with_history

        elif self.chat_template == "llama3":
            jinja_template="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
            template = Template(jinja_template)
            rendered_template = template.render(messages=messages, bos_token="<|begin_of_text|>", add_generation_prompt=add_generation_prompt)
            if remove_bos:
                rendered_template = rendered_template.replace("<|begin_of_text|>", "")
            return rendered_template

        else:
            raise NotImplmented

    def get_reward_score(self, messages: list()):
        query = self._generate_query(messages[:-1],add_generation_prompt=False)
        response = self._generate_query([messages[-1]],add_generation_prompt=False,remove_bos=True)
        # print(f"get_reward_score query: {repr(query)}")
        # print(f"get_reward_score response: {repr(response)}")
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": query,
            "response": response,
            "max_tokens": 1,
        }
        url = "http://" + self.ip + "/ck_check_reward_score"
        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            return response.json()["positive_logits"]
        elif response.status_code == 422:
            print("Failed:", response.status_code)
            return None
        else:
            print("Failed:", response.status_code)
            return None

    def get_response(self, messages: list(), max_tokens=256, **kwarg):
        query = self._generate_query(messages)
        headers = {"Content-Type": "application/json"}
        # print('query:', repr(query))
        if INFERENCE_SERVER_ENGINE == "tgi":
            data = {
                "inputs": query,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    #    "do_sample":True,
                    "stop": ["<|im_end|>"],
                },
            }
            url = "http://" + self.ip + "/generate"
            response = requests.post(url, headers=headers, data=json.dumps(data))

            if response.status_code == 200:
                return (
                    response.json()["generated_text"]
                    .replace("<|im_start|>", "")
                    .replace("<|im_end|>", "")
                )
            elif response.status_code == 422:
                print("Failed:", response.status_code)
                return "The input is too long. Please clean your history or try a shorter input."
            else:
                print("Failed:", response.status_code)
                return "Failed:" + str(response.status_code)

        elif INFERENCE_SERVER_ENGINE == "vLLM":
            # kwarg
            # greedy: do_sample=False, num_beams=1
            # temperature: do_sample=True, temperature=0.8
            # top-k: do_sample=True, top_k=50
            # top-p: do_sample=True, top_p=0.9
            # max_tokens

            do_sample = kwarg.get('do_sample', False)
            temperature = kwarg.get('temperature', 1.0)
            num_beams = kwarg.get('num_beams', None)
            top_k = kwarg.get('top_k', None)
            top_p = kwarg.get('top_p', None)
            max_tokens = kwarg.get('max_tokens', 1024)

            data = {
                "prompt": query,
                "stream": False,
                "max_tokens": max_tokens,
                "stop": STOPS,
            }

            # data['do_sample'] = do_sample # vLLM SamplingParameters do not support 'do_sample'
            data['temperature'] = temperature
            if num_beams is not None:
                data['best_of'] = num_beams # vLLM SamplingParameters use 'best_of'
            if top_k is not None:
                data['top_k'] = top_k
            if top_p is not None:
                data['top_p'] = top_p

            print("="*100, "Require with", data)

            url = "http://" + self.ip + "/generate"
            response = requests.post(
                url, headers=headers, data=json.dumps(data)
            )
            if response.status_code == 200:
                return json.loads(response.text)["text"][0].replace(query, "")
            elif response.status_code == 422:
                print("Failed:", response.status_code)
                return "The input is too long. Please clean your history or try a shorter input."
            else:
                print("Failed:", response.status_code)
                return "Failed:" + str(response.status_code)

        elif INFERENCE_SERVER_ENGINE == "vLLM-origin":
            data = {
                "prompt": query,
                "model": "my_model",
                "temperature": 0,
                "max_tokens": 512,
                "stop": ["<|im_end|>"],
            }
            url = "http://" + self.ip + "/v1/completions"
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()["choices"][0]["text"]
            elif response.status_code == 422:
                print("Failed:", response.status_code)
                return "The input is too long. Please clean your history or try a shorter input."
            else:
                print("Failed happened here:", response.status_code)
                return "Failed:" + str(response.status_code)
        else:
            raise NotImplementedError(
                "Inference server engine {} is not implemented".format(
                    INFERENCE_SERVER_ENGINE
                )
            )

    async def ck_generate_stream_eval(
        self, messages: list(), ck_mode=None, ck_k=8, ck_n=1, ck_d=0, max_tokens=256, temperature=1.0, **kwarg
    ):
        query = self._generate_query(messages)
        headers = {"Content-Type": "application/json"}
        if ck_mode is not None:
            data = {
                "prompt": query,
                "stream": True,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "ck_k": ck_k,
                "ck_n": ck_n,
                "ck_d": ck_d,
                "ck_mode": ck_mode,
                "ck_length_penalty": 1,
                "stop": STOPS,
            }
        else:
            data = {
                "prompt": query,
                "stream": True,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        print(data)
        url = "http://" + self.ip + "/ck_generate"
        response = requests.post(
            url, headers=headers, data=json.dumps(data), stream=True
        )
        if response.status_code == 200:
            try:
                buffer = ""

                for chunk in response.iter_content(chunk_size=1):

                    if chunk.endswith(b"\0"):
                        buffer += chunk.decode("utf-8")[:-1]
                        try:
                            json_data = json.loads(buffer)
                            yield json_data
                        except json.JSONDecodeError as e:
                            print(f"parsing error: {e}")
                        buffer = ""
                    else:
                        buffer += chunk.decode("utf-8")

            except json.JSONDecodeError as e:
                print(f"parsing error: {e}")
        elif response.status_code == 422:
            print("Failed:", response.status_code)
            yield "The input is too long. Please clean your history or try a shorter input."
        else:
            print("Failed:", response.status_code)
            yield "Failed:" + str(response.status_code)

 