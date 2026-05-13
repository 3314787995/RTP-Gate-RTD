#!/usr/bin/env python
import argparse
import json
import os
import random
import time
import traceback
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


TASKS = [
    "hellaswag",
    "piqa",
    "mmlu",
    "winogrande",
    "openbookqa",
    "arc_easy",
    "arc_challenge",
]


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


def sample_rows(dataset, samples, seed):
    if samples <= 0 or samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(samples))


def gemma_choice_prompt(question):
    return (
        "<bos><start_of_turn>user\n"
        "Choose the best answer. Reply with the answer text only.\n\n"
        f"{question}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def normalize_label(label):
    if isinstance(label, int):
        return label
    raw = str(label).strip()
    if raw.isdigit():
        return int(raw)
    if len(raw) == 1 and raw.isalpha():
        return ord(raw.upper()) - ord("A")
    return None


def load_mmlu(split, cache_dir):
    last_error = None
    candidates = [
        ("cais/mmlu", "all"),
        ("lukaemon/mmlu", "all"),
    ]
    for dataset_name, config in candidates:
        try:
            return load_dataset(dataset_name, config, split=split, cache_dir=cache_dir, trust_remote_code=True)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load MMLU dataset: {last_error}")


def load_first_available_dataset(candidates, split, cache_dir, label):
    last_error = None
    for dataset_name, config in candidates:
        try:
            kwargs = {
                "split": split,
                "cache_dir": cache_dir,
                "trust_remote_code": True,
            }
            if config is None:
                return load_dataset(dataset_name, **kwargs)
            return load_dataset(dataset_name, config, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load {label} dataset: {last_error}")


def normalize_record(task, row):
    row = dict(row)
    if task == "hellaswag":
        question = "Context:\n" + str(row.get("ctx") or row.get("ctx_a") or "").strip()
        choices = [str(x).strip() for x in row["endings"]]
        answer = int(row["label"])
        return question, choices, answer

    if task == "piqa":
        question = "Goal:\n" + str(row["goal"]).strip()
        choices = [str(row["sol1"]).strip(), str(row["sol2"]).strip()]
        return question, choices, int(row["label"])

    if task == "winogrande":
        sentence = str(row["sentence"]).strip()
        question = "Fill in the blank:\n" + sentence
        choices = [str(row["option1"]).strip(), str(row["option2"]).strip()]
        return question, choices, int(row["answer"]) - 1

    if task == "openbookqa":
        question = str(row["question_stem"]).strip()
        choices = [str(x).strip() for x in row["choices"]["text"]]
        labels = [str(x).strip() for x in row["choices"]["label"]]
        answer_key = str(row["answerKey"]).strip()
        answer = labels.index(answer_key) if answer_key in labels else normalize_label(answer_key)
        return question, choices, answer

    if task in ("arc_easy", "arc_challenge"):
        question = str(row["question"]).strip()
        choices = [str(x).strip() for x in row["choices"]["text"]]
        labels = [str(x).strip() for x in row["choices"]["label"]]
        answer_key = str(row["answerKey"]).strip()
        answer = labels.index(answer_key) if answer_key in labels else normalize_label(answer_key)
        return question, choices, answer

    if task == "mmlu":
        question = str(row["question"]).strip()
        choices = [str(x).strip() for x in row["choices"]]
        answer = normalize_label(row["answer"])
        return question, choices, answer

    raise ValueError(f"Unsupported classification task: {task}")


def load_task(task, split, samples, seed, cache_dir):
    if task == "hellaswag":
        ds = load_dataset("hellaswag", split="validation", cache_dir=cache_dir, trust_remote_code=True)
    elif task == "piqa":
        last_error = None
        for dataset_name in ("regisss/piqa", "ybisk/piqa", "piqa"):
            try:
                ds = load_dataset(dataset_name, split="validation", cache_dir=cache_dir, trust_remote_code=True)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"Could not load PIQA dataset: {last_error}")
    elif task == "winogrande":
        ds = load_dataset("winogrande", "winogrande_xl", split="validation", cache_dir=cache_dir, trust_remote_code=True)
    elif task == "openbookqa":
        ds = load_dataset("allenai/openbookqa", "main", split="validation", cache_dir=cache_dir, trust_remote_code=True)
    elif task == "arc_easy":
        ds = load_first_available_dataset(
            [("ai2_arc", "ARC-Easy"), ("allenai/ai2_arc", "ARC-Easy")],
            "validation",
            cache_dir,
            "ARC-Easy",
        )
    elif task == "arc_challenge":
        ds = load_first_available_dataset(
            [("ai2_arc", "ARC-Challenge"), ("allenai/ai2_arc", "ARC-Challenge")],
            "validation",
            cache_dir,
            "ARC-Challenge",
        )
    elif task == "mmlu":
        ds = load_mmlu(split, cache_dir)
    else:
        raise ValueError(f"Unsupported classification task: {task}")
    return sample_rows(ds, samples, seed)


def choice_logprob(model, tokenizer, prompt, choice, max_input_tokens):
    full_text = prompt + choice
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_input_tokens,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    prompt_len = min(len(prompt_ids), int(input_ids.shape[-1]))
    if prompt_len >= int(input_ids.shape[-1]):
        return {"sum_logprob": -1e30, "avg_logprob": -1e30, "tokens": 0}

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    start = max(prompt_len - 1, 0)
    continuation = token_log_probs[0, start:]
    if continuation.numel() == 0:
        return {"sum_logprob": -1e30, "avg_logprob": -1e30, "tokens": 0}
    total = float(continuation.sum().detach().cpu())
    count = int(continuation.numel())
    return {"sum_logprob": total, "avg_logprob": total / count, "tokens": count}


def load_existing(out_json):
    if not out_json.exists():
        return None
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("status") == "done":
        return data
    if data.get("status") == "partial":
        return data
    return data


def read_baseline_score(output_dir, task, samples):
    baseline_path = Path(output_dir) / f"gemma_base_{task}_{samples}.json"
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
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
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

        dataset = load_task(args.task, args.split, args.samples, args.seed, cache_dir)
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
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
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
                question, choices, answer = normalize_record(args.task, dataset[index])
                if answer is None or answer < 0 or answer >= len(choices):
                    raise ValueError(f"Bad answer index for task={args.task} index={index}: {answer}")
                prompt = gemma_choice_prompt(question)
                scored = [
                    {
                        "choice_index": choice_index,
                        "choice": choice,
                        **choice_logprob(model, tokenizer, prompt, choice, args.max_input_tokens),
                    }
                    for choice_index, choice in enumerate(choices)
                ]
                pred_norm = max(scored, key=lambda x: x["avg_logprob"])["choice_index"]
                pred_raw = max(scored, key=lambda x: x["sum_logprob"])["choice_index"]
                record = {
                    "index": index,
                    "question": question,
                    "choices": choices,
                    "answer_index": answer,
                    "prediction": pred_norm,
                    "prediction_raw": pred_raw,
                    "correct": pred_norm == answer,
                    "correct_raw": pred_raw == answer,
                    "scores": scored,
                }
                items.append(record)
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl.flush()
                partial = {
                    "status": "partial",
                    "run_id": args.run_id,
                    "model_path": args.model_path,
                    "adapter_path": str(adapter_path) if adapter_path is not None else None,
                    "task": args.task,
                    "samples": args.samples,
                    "completed_items": len(items),
                    "score": sum(1 for x in items if x["correct"]) / len(items),
                    "score_raw": sum(1 for x in items if x["correct_raw"]) / len(items),
                    "items": items,
                }
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(partial, f, indent=2, ensure_ascii=False)

        score = sum(1 for x in items if x["correct"]) / len(items) if items else 0.0
        score_raw = sum(1 for x in items if x["correct_raw"]) / len(items) if items else 0.0
        baseline_score = 1.0 if args.run_id == "gemma_base" else read_baseline_score(output_dir, args.task, args.samples)
        retention = None
        if baseline_score not in (None, 0):
            retention = float(score) / float(baseline_score)
        data = {
            "status": "done",
            "run_id": args.run_id,
            "model_path": args.model_path,
            "adapter_path": str(adapter_path) if adapter_path is not None else None,
            "task": args.task,
            "samples": args.samples,
            "split": args.split,
            "metric": "acc_norm",
            "score": score,
            "score_raw": score_raw,
            "baseline_score": baseline_score,
            "retention": retention,
            "completed_items": len(items),
            "total_latency": time.time() - started,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "items": items,
            "jsonl_path": str(out_jsonl),
            "file": str(out_json),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(json.dumps({"status": "done", "output_path": str(out_json), "score": score, "items": len(items)}, indent=2))
    except BaseException as exc:
        write_failed(out_json, args, exc)
        raise


if __name__ == "__main__":
    main()
