
import pandas as pd
import time
import re
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
import torch

from math_verify import verify, parse
import random
import sys
sys.path.append("../")

from .dataset_loader import load_math, load_dapo, load_gsm8k, load_gsm8k_code, load_humeval, load_xsum, load_mbpp

def eval_math(model_name, dataset_type="gsm8k", samples=100, prompt="{prompt}", split="train", llm=None, lora_request=None):
    print("prompt in use:")
    print(prompt)

    if dataset_type == "gsm8k":
        train, test, reward_correct = load_gsm8k(prompt=prompt, sample=samples)
    elif dataset_type == "math":
        train, test, reward_correct = load_math(prompt=prompt, sample=samples)
    elif dataset_type == "gsm8k_code":
        train, test, reward_correct = load_gsm8k_code(prompt=prompt, sample=samples)
    elif dataset_type == "humeval":
        train, test, reward_correct = load_humeval(prompt=prompt, sample=samples)
    elif dataset_type == "mbpp":
        train, test, reward_correct = load_mbpp(prompt=prompt, sample=samples)
    elif dataset_type == "xsum":
        train, test, reward_correct = load_xsum(prompt=prompt, sample=samples)
    else:
        raise Exception(f"Dataset '{dataset_type}' not implemented")

    sampling_params = SamplingParams(n=1, temperature=0.9, top_p=1, top_k=50, max_tokens=8192, stop=["Q:",
                                                                                 "A:", "<end_of_turn>",
                                                                                 "<|im_end|>", "</s>", "<|eot_id|>"])

    def get_accuracy(llm, test, sampling_params):
        start_time = time.time()
        if lora_request:
            eval_outputs = llm.generate(test['prompt'], sampling_params, lora_request=lora_request)
        else:
            eval_outputs = llm.generate(test['prompt'], sampling_params)
        end_time = time.time()
        
        total_latency = end_time - start_time
        total_tokens = 0
        items = []
        acc = 0
        
        for index, output in enumerate(eval_outputs):
            # Record tokens generated for this output
            tokens_generated = len(output.outputs[0].token_ids)
            total_tokens += tokens_generated

            kwargs = {k: v for k, v in test[index].items() if k != 'answer'}
            result = reward_correct([output.outputs[0].text], [test['answer'][index] if 'answer' in test.features else None],
                                    **kwargs)[0]
            acc += result['correct']

            items.append({
                'prompt': test['prompt'][index], 
                'ground_truth': test['answer'][index] if 'answer' in test.features else None,
                'response': output.outputs[0].text,
                'finish_reason': output.outputs[0].finish_reason,
                'tokens_generated': tokens_generated,
                **result
            })
        
        avg_time_per_token = total_latency / total_tokens if total_tokens > 0 else 0
        
        return {
            'final_accuracy': acc / len(test), 
            'items': items,
            'total_latency': total_latency,
            'total_tokens': total_tokens,
            'avg_time_per_token': avg_time_per_token
        }

    print(train)

    results = []
    results = []
    if split == "train":
        dataset = train
    elif split == "test":
        dataset = test
    else:
        raise ValueError(f"Invalid split: {split}")

    if llm is None:
        llm = LLM(
                trust_remote_code=True,
                model= model_name,
                max_model_len=4096,
                gpu_memory_utilization=0.9
            )
        should_cleanup = True
    else:
        should_cleanup = False

    data = get_accuracy(llm, dataset, sampling_params)

    if should_cleanup:
        del llm
        cleanup_dist_env_and_memory()
        torch.cuda.empty_cache()

    return data
