import os
import re
import json
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets

from math_verify import verify, parse
from rouge_score import rouge_scorer

from .execution import check_correctness

def sample_dataset(dataset, sample_size, seed=1234):
    """
    Consistently samples a dataset.
    If sample_size is -1, it caps at 1000 items.
    If sample_size is larger than dataset length, returns full dataset.
    """
    if sample_size == -1:
        sample_size = 1000
    
    if sample_size is not None and sample_size > 0 and sample_size < len(dataset):
        return dataset.train_test_split(train_size=sample_size, shuffle=True, seed=seed)['train']
    return dataset

def load_dapo(sample=None):
    """Helper function for DAPO"""
    SYSTEM="""{prompt}"""

    def reward_correct(completions, answer, **kwargs):
        # check if the strings ends with </think><answer>[boxed answer]</answer>
        def check_format(s, gt):
            correct = None
            extracted_answer = parse(s)
            
            if verify(extracted_answer, parse(gt)): correct = True
            else: correct = False # extracted but incorrect then reward -0.5

            return {
                    "correct": correct,
                    "extracted_answer": "--".join([item for item in extracted_answer if type(item) == str])
                }

        return [check_format(c, gt) for c, gt in zip(completions, answer)]

    df = pd.read_csv("/scratch/ss13750/rl/dapo_14k.csv")
    train = Dataset.from_pandas(df)

    test = load_dataset("HuggingFaceH4/MATH-500", split="test")

    train = sample_dataset(train, sample, seed=42)
    test = sample_dataset(test, sample, seed=42)

    train = train.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": x["solution"],
        "level": x["level"]
        })

    test = test.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": x["solution"],
        "level": x["level"]
        })
    
    return train, test, reward_correct


def load_math(prompt, sample=None, diff_filter=False):
    """Helper function for MATH to load train, test, and reward function"""

    #SYSTEM="""{prompt}"""
    SYSTEM=prompt

    # SYSTEM = (
    #             "<|im_start|>system\n"
    #             "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
    #             "<|im_start|>user\n"
    #             "{prompt}"
    #             "<|im_end|>\n"
    #             "<|im_start|>assistant\n"
    #         )


    def reward_correct(completions, answer, **kwargs):
        # check if the strings ends with </think><answer>[boxed answer]</answer>
        def check_format(s, gt):
            correct = None
            extracted_answer = parse(s)
            
            if verify(extracted_answer, parse(gt)): correct = True
            else: correct= False # extracted but incorrect then reward -0.5

            return {
                    "correct": correct,
                    "extracted_answer": "--".join([item for item in extracted_answer if type(item) == str])
                }

        return [check_format(c, gt) for c, gt in zip(completions, answer)]

    train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    train = train.add_column("question_index", list(range(1, len(train) + 1)))
    test = load_dataset("HuggingFaceH4/MATH-500", split="test")

    train = sample_dataset(train, sample, seed=42)
    test = sample_dataset(test, sample, seed=42)

    train = train.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": x["solution"],
        "level": x["level"]
        })

    test = test.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": x["solution"],
        "level": x["level"]
        })

    
    train = train.remove_columns(["problem", "solution", "type"])
    test = test.remove_columns(["problem", "solution", "subject", "unique_id"])
    return train, test, reward_correct





def load_gsm8k(prompt, sample=None, diff_filter=False):
    """Helper function for gsm8k to load train, test, and reward function"""

    SYSTEM=prompt

    def reward_correct(completions, answer, **kwargs):
        # check if the strings ends with </think><answer>[boxed answer]</answer>
        def check_format(s, gt):
            correct = None
            extracted_answer = parse(s)
            
            if verify(extracted_answer, parse(gt)): correct = True
            else: correct = False # extracted but incorrect then reward -0.5

            return {
                    "correct": correct,
                    "extracted_answer": "--".join([item for item in extracted_answer if type(item) == str])
                }

        return [check_format(c, gt) for c, gt in zip(completions, answer)]

    data = load_dataset("openai/gsm8k", "main")
    train = data['train']
    test = data['test']

    train = sample_dataset(train, sample, seed=1234)
    test = sample_dataset(test, sample, seed=1234)

    train = train.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["question"]),
        "answer": x["answer"],
        })

    test = test.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["question"]),
        "answer": x["answer"],
        })

    return train, test, reward_correct


def load_gsm8k_code(prompt, sample=None):
    """Helper function to load HumnEval+ train, test datasets and the reward function.
    Reward function is built upon implementation from OpenAI: https://github.com/openai/human-eval"""
    SYSTEM=prompt
    
    data = load_dataset("openai/gsm8k", "main")
    train = data['train']
    test = data['test']

    def reward_correct(completions, answer, **kwargs):
        def extract_code(text):
            pattern = r"```python(.+?)```"
            match = re.search(pattern, text, re.DOTALL)
            if match: 
                print("1st extract worked")
                return match.group(1).strip()

            pattern = r"```(.+?)```"
            match = re.search(pattern, text, re.DOTALL)
            if match: 
                print("2nd extract worked")
                return match.group(1).strip()
            
            print("3rd extract")
            return text.strip()

        def calculate_reward(output, problem):
            s = extract_code(output)

            # answer format is correct now
            result = check_correctness(problem, s, timeout=30)    
            
            return result
        
        results = []
        for index in range(len(completions)):
            c = completions[index]
            c = "```python\ndef solution():" + c

            problem = {
                        'answer': answer[index]
                    }
            results.append(calculate_reward(c, problem))
        
        return results

    

    train = sample_dataset(train, sample, seed=1234)
    test = sample_dataset(test, sample, seed=1234)

    train = train.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["question"]),
        "answer": x["answer"].split("####")[-1].strip()
        })

    test = test.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["question"]),
        "answer": x["answer"].split("####")[-1].strip(),
    })

    return train, test, reward_correct




def load_humeval(prompt, sample=None):
    """Helper function to load HumnEval+ train, test datasets and the reward function.
    Reward function is built upon implementation from OpenAI: https://github.com/openai/human-eval"""
    SYSTEM=prompt
    
    train = load_dataset("evalplus/humanevalplus", split="test")
    train = train.rename_column("prompt", "question")

    test = sample_dataset(train, sample, seed=42)
    train = test

    def reward_correct(completions, _, **kwargs):
        def extract_code(text):
            pattern = r"```python(.+?)```"
            match = re.search(pattern, text, re.DOTALL)
            if match: 
                print("1st extract worked")
                return match.group(1).strip()

            pattern = r"```(.+?)```"
            match = re.search(pattern, text, re.DOTALL)
            if match: 
                print("2nd extract worked")
                return match.group(1).strip()
            
            print("3rd extract")
            return text

        def calculate_reward(output, problem):
            s = extract_code(output)

            # answer format is correct now
            result = check_correctness(problem, s, timeout=30)    
            
            return result
        
        results = []
        for index in range(len(completions)):
            c = completions[index]
            c = "```python\n" + kwargs['question'] + c
            print("evaluating...")
            
            print(c)
            
            problem = {
                        'prompt': kwargs['question'],
                        'entry_point': kwargs['entry_point'],
                        'test': kwargs['test']
                    }
            results.append(calculate_reward(c, problem))
        
        return results

    train = train.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["question"]),
        })

    test = test.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["question"]),
    })

    return train, test, reward_correct


def load_mbpp(prompt, sample=None):
    """Helper function to load MBPP+ train, test datasets and the reward function.
    MBPP+ is different from HumanEval+ as it doesn't provide function signatures.
    Instead, we use the problem description + first test assertion as context."""
    SYSTEM = prompt
    
    # Load MBPP+ dataset
    data = load_dataset("evalplus/mbppplus", split="test")
    
    test = sample_dataset(data, sample, seed=42)
    train = test  # Use same split for consistency
    
    def reward_correct(completions, _, **kwargs):
        def extract_code(text):
            # First try to extract from python code block
            pattern = r"```python(.+?)```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                print("1st extract worked")
                return match.group(1).strip()
            
            # Try without python keyword
            pattern = r"```(.+?)```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                print("2nd extract worked")
                return match.group(1).strip()
            
            # If no code block, return the entire text
            print("3rd extract - using full text")
            return text.strip()
        
        def calculate_reward(output, problem):
            code = extract_code(output)
            
            # For MBPP+, we construct a test program that includes:
            # 1. Test imports (e.g., import math)
            # 2. Generated code (the function)
            test_imports_str = "\n".join(problem['test_imports'])
            
            # We combine imports and code into the completion field
            # And put the assertions in the test field
            full_code = f"{test_imports_str}\n\n{code}"
            
            # The test assertions
            test_list_str = "\n".join(problem['test_list'])
            
            mbpp_problem = {
                'test': test_list_str,
                'entry_point': '' # No entry point needed as assertions are direct
            }
            
            # Use the safer check_correctness function which uses subprocesses and reliability guards
            result = check_correctness(mbpp_problem, full_code, timeout=30)
            
            return result
        
        results = []
        for index in range(len(completions)):
            c = completions[index]
            # Add code block wrapper if not present
            if "```python" not in c:
                c = "```python\n" + c
            
            print(f"Evaluating MBPP problem {kwargs.get('task_id', 'unknown')}...")
            
            problem = {
                'task_id': kwargs.get('task_id', ''),
                'test': kwargs.get('test', ''),
                'test_imports': kwargs.get('test_imports', []),
                'test_list': kwargs.get('test_list', [])
            }
            results.append(calculate_reward(c, problem))
        
        return results
    
    # Format prompts with problem description + first test assertion
    train = train.map(lambda x: {
        "prompt": SYSTEM.format(
            prompt=x["prompt"],
            test_assertion=x["test_list"][0] if x["test_list"] else ""
        ),
        "task_id": x["task_id"],
        "test_imports": x["test_imports"],
        "test_list": x["test_list"],
        "test": x.get("test", ""),
        "code": x["code"]  # Reference solution (not used in evaluation)
    })
    
    test = test.map(lambda x: {
        "prompt": SYSTEM.format(
            prompt=x["prompt"],
            test_assertion=x["test_list"][0] if x["test_list"] else ""
        ),
        "task_id": x["task_id"],
        "test_imports": x["test_imports"],
        "test_list": x["test_list"],
        "test": x.get("test", ""),
        "code": x["code"]
    })
    
    return train, test, reward_correct


def load_xsum(prompt, sample=None):
    """Helper function to load XSum dataset for summarization evaluation.
    XSum is an extreme summarization dataset where the summary is typically one sentence.
    """
    SYSTEM = prompt
    
    # Load XSum dataset - Prefer local file to avoid network/script issues
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets/xsum_local/0000.parquet")
    if os.path.exists(local_path):
        print(f"Loading XSum from local parquet: {local_path}")
        data = load_dataset("parquet", data_files={'validation': local_path}, split="validation")
    else:
        # Fallback to network load
        try:
            data = load_dataset("EdinburghNLP/xsum", split="validation", trust_remote_code=True)
        except Exception:
             data = load_dataset("parquet", data_files={'validation': "https://huggingface.co/datasets/EdinburghNLP/xsum/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"}, split="validation")
    
    # For XSum, we only use test split for evaluation
    train = data
    test = data
    
    train = sample_dataset(train, sample, seed=42)
    test = train  # Use same split for consistency
    
    def reward_correct(completions, answer, **kwargs):
        """
        Evaluate summarization quality using ROUGE metrics.
        Returns ROUGE-L F1 score as the 'correct' metric for compatibility.
        """
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        results = []
        for completion, reference in zip(completions, answer):
            scores = scorer.score(reference, completion)
            
            # Use ROUGE-L F1 as primary metric (compatible with 'correct' field)
            rougeL_f1 = scores['rougeL'].fmeasure
            
            results.append({
                'correct': rougeL_f1,  # Use ROUGE-L as correctness metric
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure,
                'extracted_answer': completion[:100]  # First 100 chars of summary
            })
        
        return results
    
    # Map dataset to use prompt template
    train = train.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["document"]),
        "answer": x["summary"]
    })
    
    test = test.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["document"]),
        "answer": x["summary"]
    })
    
    return train, test, reward_correct
    
