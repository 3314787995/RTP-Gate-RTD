#!/usr/bin/env python
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


STOP_STRINGS = ["Q:", "A:", "<end_of_turn>", "<|im_end|>", "</s>", "<|eot_id|>"]


def parse_layers(raw):
    if raw is None or not raw.strip():
        return []
    return sorted({int(x.strip()) for x in raw.split(",") if x.strip()})


def require_inside_root(path, root):
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Refusing to write outside ROOT: {resolved} not under {root_resolved}")
    return resolved


def configure_model_cache(model):
    model.config.use_cache = False
    if hasattr(model.config, "cache_implementation"):
        model.config.cache_implementation = None
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
        if hasattr(model.generation_config, "cache_implementation"):
            model.generation_config.cache_implementation = None


def trim_stop_strings(text):
    best = len(text)
    for stop in STOP_STRINGS:
        idx = text.find(stop)
        if idx >= 0:
            best = min(best, idx)
    return text[:best].strip()


def extract_python_code(text, prefix=""):
    match = re.search(r"```python(.*?)```", text, flags=re.DOTALL)
    if match:
        body = match.group(1)
    else:
        match = re.search(r"```(.*?)```", text, flags=re.DOTALL)
        body = match.group(1) if match else text
    return (prefix + body).strip()


def syntax_check_humeval(completion, item):
    prefix = item.get("question", "")
    code = extract_python_code(completion, prefix=prefix)
    try:
        compile(code, "<model_completion>", "exec")
        return {"correct": True, "result": "syntax_passed", "fallback": "syntax_only"}
    except SyntaxError as exc:
        return {"correct": False, "result": f"syntax_error: {exc}", "fallback": "syntax_only"}
    except Exception as exc:
        return {"correct": False, "result": f"compile_error: {exc}", "fallback": "syntax_only"}


def load_task(repo_root, task, prompt_key, samples, split):
    eval_root = repo_root / "eval" / "gen_eval"
    sys.path.insert(0, str(eval_root))
    from utils import prompts
    from utils.dataset_loader import load_gsm8k, load_humeval, load_xsum

    prompt_template = getattr(prompts, prompt_key)
    if task == "gsm8k":
        train, test, reward_correct = load_gsm8k(prompt_template, sample=samples)
    elif task == "humeval":
        train, test, reward_correct = load_humeval(prompt_template, sample=samples)
    elif task == "xsum":
        train, test, reward_correct = load_xsum(prompt_template, sample=samples)
    else:
        raise ValueError(f"Unsupported task: {task}")

    dataset = train if split == "train" else test
    return dataset, reward_correct, prompt_template


def score_item(task, reward_correct, completion, item, syntax_fallback=False):
    answer = item.get("answer")
    kwargs = {k: v for k, v in item.items() if k not in ("prompt", "answer")}
    try:
        result = reward_correct([completion], [answer] if answer is not None else [None], **kwargs)[0]
        result["fallback"] = None
        return result
    except Exception as exc:
        if task == "humeval" and syntax_fallback:
            result = syntax_check_humeval(completion, item)
            result["full_eval_error"] = repr(exc)
            return result
        return {"correct": False, "result": repr(exc), "fallback": "reward_exception"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", required=True, choices=["gsm8k", "humeval", "xsum"])
    parser.add_argument("--prompt-key", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prune-strategy", required=True)
    parser.add_argument("--pruned-layers", default="")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--syntax-fallback", action="store_true")
    args = parser.parse_args()

    root = os.environ.get("ROOT")
    if not root:
        raise RuntimeError("ROOT is not set. Source env.sh before running.")
    output_dir = require_inside_root(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root) if args.repo_root else Path(root) / "code" / "on-the-limits-of-layer-pruning"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pruned_layers = parse_layers(args.pruned_layers)
    dataset, reward_correct, prompt_template = load_task(
        repo_root=repo_root,
        task=args.task,
        prompt_key=args.prompt_key,
        samples=args.samples,
        split=args.split,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    configure_model_cache(model)
    model.eval()

    output_path = output_dir / f"{args.model_label}_{args.task}_{args.prompt_key}_{args.samples}.json"
    jsonl_path = output_dir / f"{args.model_label}_{args.task}_{args.prompt_key}_{args.samples}.jsonl"
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.temperature > 0,
        "use_cache": False,
        "seed": args.seed,
        "stop_strings": STOP_STRINGS,
    }

    started = time.time()
    items = []
    total_tokens = 0
    total_generation_latency = 0.0

    with open(jsonl_path, "w", encoding="utf-8") as jsonl:
        for index in range(len(dataset)):
            item = dataset[index]
            prompt = item["prompt"]
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            input_len = int(encoded["input_ids"].shape[-1])

            item_started = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    use_cache=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            item_latency = time.time() - item_started
            generated_ids = outputs[0][input_len:]
            completion = trim_stop_strings(tokenizer.decode(generated_ids, skip_special_tokens=True))
            tokens_generated = int(generated_ids.shape[-1])
            total_tokens += tokens_generated
            total_generation_latency += item_latency

            result = score_item(
                task=args.task,
                reward_correct=reward_correct,
                completion=completion,
                item=item,
                syntax_fallback=args.syntax_fallback,
            )
            correct_value = result.get("correct", False)
            if isinstance(correct_value, bool):
                score_value = 1.0 if correct_value else 0.0
            else:
                score_value = float(correct_value)

            record = {
                "index": index,
                "prompt": prompt,
                "ground_truth": item.get("answer"),
                "response": completion,
                "tokens_generated": tokens_generated,
                "latency": item_latency,
                "score_value": score_value,
                **result,
            }
            items.append(record)
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl.flush()

            partial = {
                "status": "partial",
                "model_label": args.model_label,
                "model_path": args.model_path,
                "task": args.task,
                "prompt_key": args.prompt_key,
                "samples": args.samples,
                "completed_items": len(items),
                "score": sum(x["score_value"] for x in items) / len(items),
                "items": items,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(partial, f, indent=2, ensure_ascii=False)

    score = sum(x["score_value"] for x in items) / len(items) if items else 0.0
    total_latency = time.time() - started
    output_data = {
        "status": "done",
        "model_label": args.model_label,
        "model_path": args.model_path,
        "task": args.task,
        "prompt_key": args.prompt_key,
        "prompt_template": prompt_template,
        "samples": args.samples,
        "split": args.split,
        "prune_strategy": args.prune_strategy,
        "pruned_layers": pruned_layers,
        "generation_config": generation_config,
        "score": score,
        "total_latency": total_latency,
        "generation_latency": total_generation_latency,
        "total_tokens": total_tokens,
        "avg_time_per_token": total_generation_latency / total_tokens if total_tokens else None,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "items": items,
        "jsonl_path": str(jsonl_path),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "output_path": str(output_path),
        "score": score,
        "items": len(items),
        "total_tokens": total_tokens,
        "avg_time_per_token": output_data["avg_time_per_token"],
    }, indent=2))


if __name__ == "__main__":
    main()
