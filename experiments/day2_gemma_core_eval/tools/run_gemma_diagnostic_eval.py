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


def runtime_skip_layers(model, skip_layers):
    if not skip_layers:
        return {"enabled": False, "kept_layers": list(range(len(model.model.layers))), "changed_sliding": []}
    original_layers = list(model.model.layers)
    kept = [(idx, layer) for idx, layer in enumerate(original_layers) if idx not in set(skip_layers)]
    changed_sliding = []
    for new_idx, (old_idx, layer) in enumerate(kept):
        # This deliberately preserves the module's original is_sliding/self_attn settings.
        old_sliding = bool(getattr(layer, "is_sliding", old_idx % 2 == 0))
        new_sliding_if_resaved = new_idx % 2 == 0
        if old_sliding != new_sliding_if_resaved:
            changed_sliding.append({
                "old_layer": old_idx,
                "new_position": new_idx,
                "original_is_sliding": old_sliding,
                "resaved_would_be_sliding": new_sliding_if_resaved,
            })
    model.model.layers = torch.nn.ModuleList([layer for _, layer in kept])
    # Leave each layer's original attention metadata intact. Only update config for reporting.
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
    return train if split == "train" else test, reward_correct, prompt_template


def extract_code_block_or_text(text):
    match = re.search(r"```python(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    # Drop common explanation sections for diagnosis.
    return re.split(r"\n\s*(\*\*Explanation|\bExplanation:|Let me know)", text, maxsplit=1)[0].strip()


def humeval_diagnostic_scores(completion, item):
    prefix = item.get("question", "")
    raw_body = extract_code_block_or_text(completion)
    joined_raw = (prefix + raw_body).strip()
    indented_body = "\n".join(("    " + line if line.strip() else line) for line in raw_body.splitlines())
    joined_indented = (prefix + indented_body).strip()

    def compile_status(code):
        try:
            compile(code, "<diagnostic_humeval>", "exec")
            return True, "syntax_passed"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    raw_ok, raw_result = compile_status(joined_raw)
    indent_ok, indent_result = compile_status(joined_indented)
    return {
        "raw_body_preview": raw_body[:500],
        "syntax_raw_correct": raw_ok,
        "syntax_raw_result": raw_result,
        "syntax_indent_correct": indent_ok,
        "syntax_indent_result": indent_result,
    }


def official_score(task, reward_correct, completion, item):
    answer = item.get("answer")
    kwargs = {k: v for k, v in item.items() if k not in ("prompt", "answer")}
    try:
        result = reward_correct([completion], [answer] if answer is not None else [None], **kwargs)[0]
        result["official_eval_error"] = None
        return result
    except Exception as exc:
        return {"correct": False, "result": repr(exc), "official_eval_error": repr(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", required=True, choices=["gsm8k", "humeval", "xsum"])
    parser.add_argument("--prompt-key", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--diagnostic-name", required=True)
    parser.add_argument("--runtime-skip-layers", default="")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
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

    dataset, reward_correct, prompt_template = load_task(repo_root, args.task, args.prompt_key, args.samples, args.split)

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
    skip_layers = parse_layers(args.runtime_skip_layers)
    runtime_skip_metadata = runtime_skip_layers(model, skip_layers)
    model.eval()

    out_json = output_dir / f"{args.diagnostic_name}_{args.model_label}_{args.task}_{args.samples}.json"
    out_jsonl = output_dir / f"{args.diagnostic_name}_{args.model_label}_{args.task}_{args.samples}.jsonl"

    do_sample = args.temperature > 0
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": do_sample,
        "use_cache": False,
        "seed": args.seed,
    }

    items = []
    total_tokens = 0
    generation_latency = 0.0
    started = time.time()
    with open(out_jsonl, "w", encoding="utf-8") as jsonl:
        for index in range(len(dataset)):
            item = dataset[index]
            encoded = tokenizer(
                item["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            input_len = int(encoded["input_ids"].shape[-1])
            kwargs = dict(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            if do_sample:
                kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            item_started = time.time()
            with torch.no_grad():
                output = model.generate(**kwargs)
            item_latency = time.time() - item_started
            generated_ids = output[0][input_len:]
            completion = trim_stop_strings(tokenizer.decode(generated_ids, skip_special_tokens=True))
            tokens = int(generated_ids.shape[-1])
            total_tokens += tokens
            generation_latency += item_latency

            official = official_score(args.task, reward_correct, completion, item)
            correct_value = official.get("correct", False)
            if isinstance(correct_value, bool):
                score_value = 1.0 if correct_value else 0.0
            else:
                score_value = float(correct_value)
            diagnostics = {}
            if args.task == "humeval":
                diagnostics = humeval_diagnostic_scores(completion, item)

            record = {
                "index": index,
                "prompt": item["prompt"],
                "ground_truth": item.get("answer"),
                "response": completion,
                "tokens_generated": tokens,
                "latency": item_latency,
                "score_value": score_value,
                **official,
                **diagnostics,
            }
            items.append(record)
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl.flush()
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({
                    "status": "partial",
                    "completed_items": len(items),
                    "score": sum(x["score_value"] for x in items) / len(items),
                    "items": items,
                }, f, indent=2, ensure_ascii=False)

    score = sum(x["score_value"] for x in items) / len(items) if items else 0.0
    humeval_diag = None
    if args.task == "humeval":
        humeval_diag = {
            "syntax_raw_rate": sum(1 for x in items if x.get("syntax_raw_correct")) / len(items) if items else 0.0,
            "syntax_indent_rate": sum(1 for x in items if x.get("syntax_indent_correct")) / len(items) if items else 0.0,
        }

    data = {
        "status": "done",
        "diagnostic_name": args.diagnostic_name,
        "model_label": args.model_label,
        "model_path": args.model_path,
        "task": args.task,
        "prompt_key": args.prompt_key,
        "prompt_template": prompt_template,
        "samples": args.samples,
        "split": args.split,
        "runtime_skip": runtime_skip_metadata,
        "generation_config": generation_config,
        "score": score,
        "total_latency": time.time() - started,
        "generation_latency": generation_latency,
        "total_tokens": total_tokens,
        "avg_time_per_token": generation_latency / total_tokens if total_tokens else None,
        "humeval_diagnostics": humeval_diag,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "items": items,
        "jsonl_path": str(out_jsonl),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "output_path": str(out_json),
        "score": score,
        "items": len(items),
        "total_tokens": total_tokens,
        "runtime_skip": runtime_skip_metadata,
        "humeval_diagnostics": humeval_diag,
    }, indent=2))


if __name__ == "__main__":
    main()
