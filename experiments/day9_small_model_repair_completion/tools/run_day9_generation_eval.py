#!/usr/bin/env python
import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


GEMMA_INSTRUCT = """<bos><start_of_turn>user
Please reason step by step, and put your final answer within \\boxed{{}}.

{prompt}<end_of_turn>
<start_of_turn>model"""

GEMMA_XSUM = """<bos><start_of_turn>user
Please provide a concise one-sentence summary of the following article.

{prompt}<end_of_turn>
<start_of_turn>model"""

GEMMA_HUMEVAL = """<bos><start_of_turn>user
Complete the following Python function. Return only valid Python code. Do not explain.

```python
{prompt}
```<end_of_turn>
<start_of_turn>model
```python
{prompt}"""

STOP_STRINGS = ["Q:", "A:", "<end_of_turn>", "<|im_end|>", "</s>", "<|eot_id|>"]


def require_under_root(path, root):
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Refusing path outside ROOT: {resolved} not under {root_resolved}")
    return resolved


def parse_layers(raw):
    if raw is None or not str(raw).strip():
        return []
    return sorted({int(x.strip()) for x in str(raw).split(",") if x.strip()})


def set_cache_config(model):
    model.config.use_cache = False
    if hasattr(model.config, "cache_implementation"):
        model.config.cache_implementation = None
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
        if hasattr(model.generation_config, "cache_implementation"):
            model.generation_config.cache_implementation = None


def runtime_skip_layers(model, skip_layers):
    if not skip_layers:
        return {"enabled": False, "skip_layers": [], "kept_layers": list(range(len(model.model.layers)))}
    original_layers = list(model.model.layers)
    kept = [(idx, layer) for idx, layer in enumerate(original_layers) if idx not in set(skip_layers)]
    changed_sliding = []
    for new_idx, (old_idx, layer) in enumerate(kept):
        old_sliding = bool(getattr(layer, "is_sliding", old_idx % 2 == 0))
        new_sliding_if_resaved = new_idx % 2 == 0
        if old_sliding != new_sliding_if_resaved:
            changed_sliding.append(
                {
                    "old_layer": old_idx,
                    "new_position": new_idx,
                    "original_is_sliding": old_sliding,
                    "resaved_would_be_sliding": new_sliding_if_resaved,
                }
            )
    model.model.layers = torch.nn.ModuleList([layer for _, layer in kept])
    model.config.num_hidden_layers = len(kept)
    if hasattr(model.model, "config"):
        model.model.config.num_hidden_layers = len(kept)
    return {
        "enabled": True,
        "skip_layers": skip_layers,
        "kept_layers": [idx for idx, _ in kept],
        "changed_sliding_if_resaved": changed_sliding,
    }


def trim_stop_strings(text):
    best = len(text)
    for stop in STOP_STRINGS:
        idx = text.find(stop)
        if idx >= 0:
            best = min(best, idx)
    return text[:best].strip()


def load_task(repo_root, task, samples, split):
    eval_root = repo_root / "eval" / "gen_eval"
    sys.path.insert(0, str(eval_root))
    from utils.dataset_loader import load_gsm8k, load_humeval, load_xsum

    if task == "gsm8k":
        train, test, reward_correct = load_gsm8k(GEMMA_INSTRUCT, sample=samples)
        return train if split == "train" else test, reward_correct, GEMMA_INSTRUCT
    if task == "xsum":
        train, test, reward_correct = load_xsum(GEMMA_XSUM, sample=samples)
        return train if split == "train" else test, reward_correct, GEMMA_XSUM
    if task == "humeval":
        train, test, reward_correct = load_humeval(GEMMA_HUMEVAL, sample=samples)
        return train if split == "train" else test, reward_correct, GEMMA_HUMEVAL
    raise ValueError(f"Unsupported generation task: {task}")


def score_item(task, reward_correct, completion, item):
    answer = item.get("answer")
    kwargs = {k: v for k, v in item.items() if k not in ("prompt", "answer")}
    try:
        result = reward_correct([completion], [answer] if answer is not None else [None], **kwargs)[0]
        result["eval_error"] = None
        return result
    except BaseException as exc:
        return {"correct": False, "result": repr(exc), "eval_error": repr(exc)}


def humeval_syntax_diag(completion, item):
    text = trim_stop_strings(completion)
    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    body = match.group(1) if match else text
    prefix = item.get("question", "")
    raw = (prefix + body).strip()
    indented = prefix + "\n" + "\n".join(("    " + line if line.strip() else line) for line in body.splitlines())

    def check(code):
        try:
            compile(code, "<day8_humeval>", "exec")
            return True, "syntax_passed"
        except BaseException as exc:
            return False, f"{type(exc).__name__}: {exc}"

    raw_ok, raw_result = check(raw)
    indent_ok, indent_result = check(indented)
    return {
        "syntax_raw_correct": raw_ok,
        "syntax_raw_result": raw_result,
        "syntax_indent_correct": indent_ok,
        "syntax_indent_result": indent_result,
    }


def load_existing(out_json):
    if not out_json.exists():
        return None
    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)


def write_failed(out_json, args, exc):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "status": "failed",
        "run_id": args.run_id,
        "task": args.task,
        "samples": args.samples,
        "job_group": args.job_group,
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
    parser.add_argument("--task", required=True, choices=["gsm8k", "xsum", "humeval"])
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-group", required=True)
    parser.add_argument("--runtime-skip-layers", default="")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

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

    out_json = output_dir / f"{args.run_id}_{args.task}_{args.samples}.json"
    out_jsonl = output_dir / f"{args.run_id}_{args.task}_{args.samples}.jsonl"
    existing = load_existing(out_json)
    if existing and existing.get("status") == "done":
        print(json.dumps({"status": "done", "output_path": str(out_json), "score": existing.get("score")}, indent=2))
        return

    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        dataset, reward_correct, prompt_template = load_task(repo_root, args.task, args.samples, args.split)
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
        skip_layers = parse_layers(args.runtime_skip_layers)
        runtime_skip = runtime_skip_layers(model, skip_layers)
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(adapter_path))
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()

        items = []
        if existing and existing.get("status") == "partial":
            items = list(existing.get("items") or [])
        completed = {int(x["index"]) for x in items if "index" in x}
        started = time.time()
        total_tokens = sum(int(x.get("tokens_generated") or 0) for x in items)
        generation_latency = sum(float(x.get("latency") or 0.0) for x in items)
        do_sample = args.temperature > 0

        with open(out_jsonl, "a", encoding="utf-8") as jsonl:
            for index in range(len(dataset)):
                if index in completed:
                    continue
                item = dict(dataset[index])
                encoded = tokenizer(
                    item["prompt"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_input_tokens,
                )
                encoded = {k: v.to(model.device) for k, v in encoded.items()}
                input_len = int(encoded["input_ids"].shape[-1])
                generate_kwargs = {
                    **encoded,
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": do_sample,
                    "use_cache": False,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if do_sample:
                    generate_kwargs.update({"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k})
                item_started = time.time()
                with torch.no_grad():
                    output = model.generate(**generate_kwargs)
                latency = time.time() - item_started
                generated_ids = output[0][input_len:]
                completion = trim_stop_strings(tokenizer.decode(generated_ids, skip_special_tokens=True))
                tokens = int(generated_ids.shape[-1])
                total_tokens += tokens
                generation_latency += latency
                result = score_item(args.task, reward_correct, completion, item)
                if args.task == "humeval":
                    result.update(humeval_syntax_diag(completion, item))
                correct_value = result.get("correct", False)
                score_value = (1.0 if correct_value else 0.0) if isinstance(correct_value, bool) else float(correct_value)
                record = {
                    "index": index,
                    "prompt": item["prompt"],
                    "ground_truth": item.get("answer"),
                    "response": completion,
                    "tokens_generated": tokens,
                    "latency": latency,
                    "score_value": score_value,
                    **result,
                }
                items.append(record)
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl.flush()
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "status": "partial",
                            "run_id": args.run_id,
                            "job_group": args.job_group,
                            "task": args.task,
                            "samples": args.samples,
                            "completed_items": len(items),
                            "score": sum(x["score_value"] for x in items) / len(items),
                            "items": items,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

        score = sum(x["score_value"] for x in items) / len(items) if items else 0.0
        syntax_diag = None
        if args.task == "humeval":
            syntax_diag = {
                "syntax_raw_rate": sum(1 for x in items if x.get("syntax_raw_correct")) / len(items) if items else 0.0,
                "syntax_indent_rate": sum(1 for x in items if x.get("syntax_indent_correct")) / len(items) if items else 0.0,
            }
        data = {
            "status": "done",
            "run_id": args.run_id,
            "job_group": args.job_group,
            "model_path": args.model_path,
            "adapter_path": str(adapter_path) if adapter_path is not None else None,
            "task": args.task,
            "samples": args.samples,
            "split": args.split,
            "runtime_skip": runtime_skip,
            "prompt_template": prompt_template,
            "generation_config": {
                "max_new_tokens": args.max_new_tokens,
                "max_input_tokens": args.max_input_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "do_sample": do_sample,
                "use_cache": False,
                "seed": args.seed,
            },
            "score": score,
            "syntax_diagnostics": syntax_diag,
            "total_latency": time.time() - started,
            "generation_latency": generation_latency,
            "total_tokens": total_tokens,
            "avg_time_per_token": generation_latency / total_tokens if total_tokens else None,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "items": items,
            "jsonl_path": str(out_jsonl),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(json.dumps({"status": "done", "output_path": str(out_json), "score": score, "items": len(items)}, indent=2))
    except BaseException as exc:
        write_failed(out_json, args, exc)
        raise


if __name__ == "__main__":
    main()
