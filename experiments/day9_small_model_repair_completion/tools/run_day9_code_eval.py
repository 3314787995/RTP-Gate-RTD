#!/usr/bin/env python
import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


STOP_STRINGS = ["<end_of_turn>", "<|im_end|>", "</s>", "<|eot_id|>", "```text", "```"]


def require_under_root(path, root):
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Refusing path outside ROOT: {resolved} not under {root_resolved}")
    return resolved


def set_cache_config(model):
    model.config.use_cache = False
    if hasattr(model.config, "cache_implementation"):
        model.config.cache_implementation = None
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
        if hasattr(model.generation_config, "cache_implementation"):
            model.generation_config.cache_implementation = None


def sample_dataset(dataset, sample_size, seed=42):
    if sample_size <= 0 or sample_size >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(sample_size))


def humeval_prompt(question, variant):
    if variant == "prefix_continuation":
        return (
            "<bos><start_of_turn>user\n"
            "Complete the following Python function. Return only valid Python code; do not explain.\n\n"
            f"{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
            f"{question}"
        )
    if variant == "fenced_prefix":
        return (
            "<bos><start_of_turn>user\n"
            "Complete the following Python function. Return only the code inside the Python block.\n\n"
            f"```python\n{question}\n```<end_of_turn>\n"
            "<start_of_turn>model\n"
            f"```python\n{question}"
        )
    if variant == "plain_request":
        return (
            "<bos><start_of_turn>user\n"
            "Complete the following Python function. Return only one complete Python function; do not explain.\n\n"
            f"{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    raise ValueError(f"Unknown HumanEval variant: {variant}")


def mbpp_prompt(prompt, first_assertion):
    return (
        "<bos><start_of_turn>user\n"
        "Solve this Python programming problem. Return only valid Python code for the required function; do not explain.\n\n"
        f"{prompt}\n\n"
        f"Your code should satisfy this assertion:\n{first_assertion}<end_of_turn>\n"
        "<start_of_turn>model\n"
        "```python\n"
    )


def trim_stop_strings(text):
    best = len(text)
    for stop in STOP_STRINGS:
        idx = text.find(stop)
        if idx >= 0:
            best = min(best, idx)
    return text[:best].rstrip()


def extract_code(text):
    text = trim_stop_strings(text)
    fenced = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip("\n\r")
    fenced = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip("\n\r")
    text = re.sub(r"^\s*Here is (a|the).*?:\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip("\n\r")


def assemble_humeval_code(raw_completion, question, entry_point):
    body = extract_code(raw_completion)
    if f"def {entry_point}" in body:
        return body[body.find(f"def {entry_point}") :].strip()
    stripped_question = question.strip()
    if stripped_question and stripped_question in body:
        return body[body.find(stripped_question) :].strip()
    separator = "" if question.endswith(("\n", " ", "\t")) else "\n"
    return (question + separator + body).strip()


def bracket_balance(code):
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {v: k for k, v in pairs.items()}
    stack = []
    for ch in code:
        if ch in pairs:
            stack.append(ch)
        elif ch in closing:
            if not stack or stack[-1] != closing[ch]:
                return False
            stack.pop()
    return not stack


def classify_result(result, code):
    try:
        compile(code, "<day9_code_eval>", "exec")
    except SyntaxError:
        return "syntax"
    except IndentationError:
        return "syntax"
    raw = str(result)
    lowered = raw.lower()
    if result == "passed":
        return "passed"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "nameerror" in lowered or "not defined" in lowered or "is not defined" in lowered:
        return "not_defined"
    if "syntax" in lowered or "indent" in lowered or "eof" in lowered:
        return "syntax"
    if "assert" in lowered or raw.startswith("failed: "):
        return "runtime_or_assertion"
    return "other"


def load_task(task, samples, seed, cache_dir):
    if task == "humeval":
        data = load_dataset("evalplus/humanevalplus", split="test", cache_dir=cache_dir)
        data = data.rename_column("prompt", "question")
        return sample_dataset(data, samples, seed)
    if task == "mbpp":
        data = load_dataset("evalplus/mbppplus", split="test", cache_dir=cache_dir)
        return sample_dataset(data, samples, seed)
    raise ValueError(f"Unsupported code task: {task}")


def make_prompt(task, item, variant):
    if task == "humeval":
        return humeval_prompt(item["question"], variant)
    first_assertion = item["test_list"][0] if item.get("test_list") else ""
    return mbpp_prompt(item["prompt"], first_assertion)


def evaluate_code(task, item, completion, repo_root):
    sys.path.insert(0, str(repo_root / "eval" / "gen_eval"))
    from utils.execution import check_correctness

    if task == "humeval":
        code = assemble_humeval_code(completion, item["question"], item["entry_point"])
        problem = {"test": item["test"], "entry_point": item["entry_point"]}
    else:
        code = extract_code(completion)
        imports = "\n".join(item.get("test_imports") or [])
        tests = "\n".join(item.get("test_list") or [])
        code = f"{imports}\n\n{code}".strip()
        problem = {"test": tests, "entry_point": ""}
    result = check_correctness(problem, code, timeout=30)
    error_type = classify_result(result.get("result"), code)
    return {
        "correct": bool(result.get("correct")),
        "result": result.get("result"),
        "error_type": error_type,
        "bracket_balanced": bracket_balance(code),
        "extracted_code": code,
    }


def load_existing(out_json):
    if not out_json.exists():
        return None
    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)


def read_baseline_score(output_dir, task, variant, samples):
    baseline_path = Path(output_dir) / f"gemma_base_{task}_{variant}_{samples}.json"
    if not baseline_path.exists():
        return None
    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("status") == "done" and data.get("score") not in (None, ""):
            return float(data["score"])
    except Exception:
        return None
    return None


def write_failed(out_json, args, exc):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "status": "failed",
        "run_id": args.run_id,
        "task": args.task,
        "variant": args.variant,
        "samples": args.samples,
        "error": repr(exc),
        "traceback": traceback.format_exc()[-4000:],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--task", required=True, choices=["humeval", "mbpp"])
    parser.add_argument("--variant", default="assertion_guided")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        raise RuntimeError("Set HF_ALLOW_CODE_EVAL=1 before running code evaluation.")
    root = os.environ.get("ROOT")
    if not root:
        raise RuntimeError("ROOT is not set. Source env.sh before running.")
    root_path = Path(root).resolve()
    output_dir = require_under_root(args.output_dir, root_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = None
    if args.adapter_path and args.adapter_path.lower() not in ("none", "null"):
        adapter_path = require_under_root(args.adapter_path, root_path)
    repo_root = Path(args.repo_root) if args.repo_root else root_path / "code" / "on-the-limits-of-layer-pruning"
    cache_dir = os.environ.get("HF_HOME")

    out_json = output_dir / f"{args.run_id}_{args.task}_{args.variant}_{args.samples}.json"
    out_jsonl = output_dir / f"{args.run_id}_{args.task}_{args.variant}_{args.samples}.jsonl"
    existing = load_existing(out_json)
    if existing and existing.get("status") == "done":
        print(json.dumps({"status": "done", "output_path": str(out_json), "score": existing.get("score")}, indent=2))
        return

    try:
        dataset = load_task(args.task, args.samples, args.seed, cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=cache_dir,
            attn_implementation="eager",
        )
        set_cache_config(model)
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(adapter_path))
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        items = []
        if existing and existing.get("status") == "partial":
            items = list(existing.get("items") or [])
        completed = {int(x["index"]) for x in items if "index" in x}
        started = time.time()

        with open(out_jsonl, "a", encoding="utf-8") as jsonl:
            for index in range(len(dataset)):
                if index in completed:
                    continue
                item = dict(dataset[index])
                prompt = make_prompt(args.task, item, args.variant)
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
                encoded = {k: v.to(device) for k, v in encoded.items()}
                with torch.no_grad():
                    output_ids = model.generate(
                        **encoded,
                        do_sample=False,
                        temperature=0.0,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=False,
                    )
                new_ids = output_ids[0][encoded["input_ids"].shape[1] :]
                completion = tokenizer.decode(new_ids, skip_special_tokens=True)
                eval_result = evaluate_code(args.task, item, completion, repo_root)
                record = {
                    "index": index,
                    "task_id": item.get("task_id", item.get("task_id_plus", "")),
                    "entry_point": item.get("entry_point", ""),
                    "prompt": prompt,
                    "completion": completion,
                    "tokens_generated": int(new_ids.numel()),
                    **eval_result,
                }
                items.append(record)
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl.flush()
                partial = {
                    "status": "partial",
                    "run_id": args.run_id,
                    "task": args.task,
                    "variant": args.variant,
                    "samples": args.samples,
                    "completed_items": len(items),
                    "score": sum(1 for x in items if x["correct"]) / len(items),
                    "items": items,
                }
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(partial, f, indent=2, ensure_ascii=False)

        score = sum(1 for x in items if x["correct"]) / len(items) if items else 0.0
        baseline_score = 1.0 if args.run_id == "gemma_base" else read_baseline_score(output_dir, args.task, args.variant, args.samples)
        retention = None
        if baseline_score not in (None, 0):
            retention = float(score) / float(baseline_score)
        error_counts = {}
        for item in items:
            error_counts[item.get("error_type", "unknown")] = error_counts.get(item.get("error_type", "unknown"), 0) + 1
        data = {
            "status": "done",
            "run_id": args.run_id,
            "model_path": args.model_path,
            "adapter_path": str(adapter_path) if adapter_path is not None else None,
            "task": args.task,
            "variant": args.variant,
            "samples": args.samples,
            "metric": "pass_rate",
            "score": score,
            "baseline_score": baseline_score,
            "retention": retention,
            "completed_items": len(items),
            "error_type_counts": error_counts,
            "bracket_balanced_rate": sum(1 for x in items if x.get("bracket_balanced")) / len(items) if items else 0.0,
            "total_latency": time.time() - started,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "items": items,
            "jsonl_path": str(out_jsonl),
            "file": str(out_json),
            "diagnostic_only": True,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(json.dumps({"status": "done", "output_path": str(out_json), "score": score, "items": len(items)}, indent=2))
    except BaseException as exc:
        write_failed(out_json, args, exc)
        raise


if __name__ == "__main__":
    main()
