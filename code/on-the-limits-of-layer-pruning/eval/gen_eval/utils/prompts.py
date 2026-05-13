QWEN_INSTRUCT = """<|im_start|>system
Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""

OLMO_INSTRUCT = """<|endoftext|><|user|>
Please reason step by step, and put your final answer within \\boxed{{}}.
{prompt}
<|assistant|>
"""

OLMO3_INSTRUCT = """<|im_start|>system
You are a helpful function-calling AI assistant. 
You do not currently have access to any functions. <functions></functions><|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""


BASE_PROMPT="""Please reason step by step, and put your final answer within \\boxed{{}}.
Question: {prompt}
Answer: """

QWEN_CODE_PROMPT="""<|im_start|>system
Please solve the following problem with python code. Put the code inside a function named solution. ONLY give me the function as the output, don't attempt to solve the problem yourself.
```python
def solution():
    #Solve the problem
    ...
    return answer
```<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
```python
def solution():"""

QWEN_HUMEVAL_PROMPT="""<|im_start|>system
Please solve the following problem with python code. Keep it concise. Follow the prefix I provide and just finish the function inside ```python (code)```.
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant

Here is the final answer:

```python
{prompt}"""


QWEN_MBPP_PROMPT = """<|im_start|>system
Please solve the following programming problem with python code. Write a complete function that satisfies the requirements.<|im_end|>
<|im_start|>user
{prompt}
Your code should satisfy the following assertion:
{test_assertion}
<|im_end|>
<|im_start|>assistant
Here is a solution to this programming problem:
```python
"""



QWEN3_NOTHINK="""<|im_start|>system
Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n"""

DMATH_PROMPT="""<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \\boxed{{}}.\n\nUser: {prompt}\n\nAssistant:"""

LLAMA_PROMPT="""'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'"""

LLAMA_HUMEVAL_PROMPT="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nPlease solve the following problem with python code. Keep it concise. Follow the prefix I provide and just finish the function inside ```python (code)```.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n

Here is the final answer:

```python
{prompt}"""


LLAMA_MBPP_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024
Please solve the following programming problem with python code. Write a complete function that satisfies the requirements.<|eot_id|><|start_header_id|>user<|end_header_id|>
{prompt}
Your code should satisfy the following assertion:
{test_assertion}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Here is a solution to this programming problem:
```python
"""


SIMPLERL_COMPLEX="""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{prompt}
Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>
<|im_start|>assistant
"""

# XSum Summarization Prompts
XSUM_BASIC = """Summarize the following article in one sentence:

{prompt}

Summary:"""


QWEN_XSUM = """<|im_start|>system
Please provide a concise one-sentence summary of the following article.<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""

LLAMA_XSUM = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

Please provide a concise one-sentence summary of the following article.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

MISTRAL_INSTRUCT = """<s>[INST] Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{prompt}[/INST]"""

MISTRAL_MBPP = """<s>[INST] Please solve the following programming problem with python code. Write a complete function that satisfies the requirements.

{prompt}
Your code should satisfy the following assertion:
{test_assertion}[/INST] 
Here is a solution to this programming problem:
```python"""

MISTRAL_HUMEVAL = """<s>[INST] Please solve the following problem with python code. Put the code inside a function named solution. ONLY give me the function as the output, don't attempt to solve the problem yourself.\n\n{prompt}[/INST] 
Here is the final answer:

```python
{prompt}"""


MISTRAL_XSUM = """<s>[INST] Please provide a concise one-sentence summary of the following article.

{prompt}[/INST]"""





OLMO2_INSTRUCT = """<|endoftext|><|system|>
Please reason step by step, and put your final answer within \\boxed{{}}.
<|user|>
{prompt}
<|assistant|>
"""

OLMO2_HUMEVAL = """<|endoftext|><|system|>
Please solve the following problem with python code. Keep it concise. Follow the prefix I provide and just finish the function inside ```python (code)```.
<|user|>
{prompt}
<|assistant|>

Here is the final answer:

```python
{prompt}"""


OLMO2_XSUM = """<|endoftext|><|system|>
Please provide a concise one-sentence summary of the following article.
<|user|>
{prompt}
<|assistant|>
"""


DEEPSEEK_INSTRUCT = """"<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \\boxed{{}}.\n\nUser: {prompt}\n\nAssistant: """

DEEPSEEK_HUMEVAL = """"<｜begin▁of▁sentence｜>Please solve the following problem with python code. Keep it concise. Follow the prefix I provide and just finish the function inside ```python (code)```.\n\nUser: {prompt}\n\nAssistant: 

Here is the final answer:

```python
{prompt}"""

DEEPSEEK_XSUM = """"<｜begin▁of▁sentence｜>Please provide a concise one-sentence summary of the following article.\n\nUser: {prompt}\n\nAssistant: """


GEMMA_INSTRUCT = """<bos><start_of_turn>user
Please reason step by step, and put your final answer within \\boxed{{}}.

{prompt}<end_of_turn>
<start_of_turn>model"""

GEMMA_HUMEVAL = """<bos><start_of_turn>user
Please solve the following problem with python code. Keep it concise. Follow the prefix I provide and just finish the function inside ```python (code)```.

{prompt}<end_of_turn>
<start_of_turn>model

Here is the final answer:

```python
{prompt}"""

GEMMA_MBPP = """<|im_start|>user
Please solve the following programming problem with python code. Write a complete function that satisfies the requirements.

{prompt}
Your code should satisfy the following assertion:
{test_assertion}
<|im_end|>
<|im_start|>assistant
Here is a solution to this programming problem:
```python"""

GEMMA_XSUM = """<bos><start_of_turn>user
Please provide a concise one-sentence summary of the following article.

{prompt}<end_of_turn>
<start_of_turn>model"""