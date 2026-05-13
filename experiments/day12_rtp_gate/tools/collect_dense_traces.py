#!/usr/bin/env python
"""Collect dense Gemma GSM8K teacher-forced traces for RTP-Gate."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASELINE = "results/day5_technical_gap_closure/gemma_base_full_eval_gsm8k_1319.json"
DEFAULT_MODEL = "google/gemma-2-2b-it"


def require_under_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Refusing path outside ROOT: {resolved} not under {root_resolved}")
    return resolved


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_cache_config(model) -> None:
    model.config.use_cache = False
    if hasattr(model.config, "cache_implementation"):
        model.config.cache_implementation = None
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
        if hasattr(model.generation_config, "cache_implementation"):
            model.generation_config.cache_implementation = None


def clean_answer(answer) -> str:
    if answer is None:
        return ""
    return str(answer).strip().strip("*").strip("$").replace(",", "")


def find_answer_span(response: str, extracted_answer: str) -> tuple[int, int] | None:
    boxed = list(re.finditer(r"\\boxed\s*\{([^{}]*)\}", response))
    if boxed:
        match = boxed[-1]
        return match.start(), match.end()
    raw_answer = str(extracted_answer or "").strip().strip("*").strip("$")
    answer = clean_answer(extracted_answer)
    if answer or raw_answer:
        candidates = {raw_answer, answer, answer.replace(".0", "")}
        for candidate in sorted(candidates, key=len, reverse=True):
            if not candidate:
                continue
            matches = list(re.finditer(re.escape(candidate), response))
            if matches:
                match = matches[-1]
                return match.start(), match.end()
    if response.strip():
        end = len(response)
        return max(0, end - min(64, max(16, end // 5))), end
    return None


def token_text(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def build_masks(
    tokenizer,
    full_text: str,
    prompt: str,
    response: str,
    extracted_answer: str,
    response_token_ids: list[int],
    offsets,
) -> dict:
    n = len(response_token_ids)
    answer_mask = [0] * n
    span = find_answer_span(response, extracted_answer)
    if span:
        start, end = span
        global_start = len(prompt) + start
        global_end = len(prompt) + end
        if offsets:
            for i, (tok_start, tok_end) in enumerate(offsets):
                if tok_end > global_start and tok_start < global_end:
                    answer_mask[i] = 1
        if not any(answer_mask):
            tail = max(1, min(16, max(1, n // 5)))
            for i in range(max(0, n - tail), n):
                answer_mask[i] = 1
    elif n:
        tail = max(1, min(16, max(1, n // 5)))
        for i in range(max(0, n - tail), n):
            answer_mask[i] = 1

    math_mask = []
    math_re = re.compile(r"[0-9+\-*/=.$,%]")
    for token_id in response_token_ids:
        math_mask.append(1 if math_re.search(token_text(tokenizer, token_id)) else 0)

    late_start = int(math.floor(0.6 * n))
    late_mask = [1 if i >= late_start else 0 for i in range(n)]
    reason_mask = [1 if not answer_mask[i] else 0 for i in range(n)]
    return {
        "reason_mask": reason_mask,
        "answer_mask": answer_mask,
        "math_mask": math_mask,
        "late_mask": late_mask,
    }


def try_offsets(tokenizer, full_text: str, response_shift_positions: list[int]):
    try:
        encoded = tokenizer(full_text, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping")
        if not offsets:
            return None
        label_offsets = []
        for shift_pos in response_shift_positions:
            label_index = shift_pos + 1
            if label_index < len(offsets):
                label_offsets.append(tuple(offsets[label_index]))
            else:
                label_offsets.append((0, 0))
        return label_offsets
    except Exception:
        return None


def dense_forward(
    model,
    tokenizer,
    prompt: str,
    response: str,
    extracted_answer: str,
    top_k: int,
    max_seq_tokens: int,
) -> dict:
    full_text = prompt + response
    prompt_encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_tokens)
    full_encoded = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_seq_tokens)
    input_ids = full_encoded["input_ids"].to(model.device)
    attention_mask = full_encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    prompt_len = int(prompt_encoded["input_ids"].shape[-1])
    seq_len = int(input_ids.shape[-1])
    if seq_len <= prompt_len:
        raise ValueError("Full prompt+response encoding contains no response tokens.")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    response_positions = [i for i in range(labels.shape[-1]) if i + 1 >= prompt_len]
    if not response_positions:
        raise ValueError("No teacher-forced response positions found.")

    pos = torch.tensor(response_positions, device=model.device)
    selected_logits = logits.index_select(1, pos)[0].detach().cpu().float()
    selected_labels = labels.index_select(1, pos)[0].detach().cpu()
    del outputs, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_probs = torch.log_softmax(selected_logits, dim=-1)
    probs = torch.softmax(selected_logits, dim=-1)
    nll = -log_probs.gather(1, selected_labels[:, None]).squeeze(1)
    entropy = -(probs * log_probs).sum(dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)

    response_token_ids = [int(x) for x in selected_labels.detach().cpu().tolist()]
    offsets = try_offsets(tokenizer, full_text, response_positions)
    masks = build_masks(tokenizer, full_text, prompt, response, extracted_answer, response_token_ids, offsets)
    return {
        "prompt_token_count": prompt_len,
        "full_token_count": seq_len,
        "response_token_count": len(response_token_ids),
        "response_token_ids": response_token_ids,
        "dense_token_nll": [float(x) for x in nll.detach().cpu().tolist()],
        "dense_entropy": [float(x) for x in entropy.detach().cpu().tolist()],
        "dense_topk_token_ids": [[int(y) for y in row] for row in top_ids.detach().cpu().tolist()],
        "dense_topk_probs": [[float(y) for y in row] for row in top_probs.detach().cpu().tolist()],
        "masks": masks,
    }


def partition_for(offset: int, smoke: int, calibration: int, holdout: int) -> str | None:
    if offset < smoke:
        return "smoke"
    if offset < smoke + calibration:
        return "calibration"
    if offset < smoke + calibration + holdout:
        return "holdout"
    return None


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("ROOT", "/home/cike/hs/paper2_layer_pruning"))
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke-samples", type=int, default=50)
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--holdout-samples", type=int, default=100)
    parser.add_argument("--partition-mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-seq-tokens", type=int, default=2560)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    baseline_path = Path(args.baseline_json) if args.baseline_json else root / DEFAULT_BASELINE
    baseline_path = require_under_root(baseline_path, root)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "results" / "day12_rtp_gate" / "traces"
    output_dir = require_under_root(output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    smoke_n = args.smoke_samples
    calibration_n = 0 if args.partition_mode == "smoke" else args.calibration_samples
    holdout_n = 0 if args.partition_mode == "smoke" else args.holdout_samples
    total_needed = smoke_n + calibration_n + holdout_n

    baseline = load_json(baseline_path)
    correct_items = [item for item in baseline.get("items", []) if item.get("correct") is True]
    correct_items = correct_items[:total_needed]
    if len(correct_items) < total_needed:
        raise RuntimeError(f"Need {total_needed} dense-correct items, found {len(correct_items)}")

    cache_dir = os.environ.get("HF_HOME")
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
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    partitions = {"smoke": [], "calibration": [], "holdout": []}
    for offset, item in enumerate(correct_items):
        partition = partition_for(offset, smoke_n, calibration_n, holdout_n)
        if partition is None:
            continue
        trace = dense_forward(
            model,
            tokenizer,
            item["prompt"],
            item["response"],
            item.get("extracted_answer", ""),
            top_k=args.top_k,
            max_seq_tokens=args.max_seq_tokens,
        )
        record = {
            "trace_id": f"gsm8k_dense_{offset:04d}",
            "partition": partition,
            "source_index": item.get("index"),
            "prompt": item.get("prompt"),
            "response": item.get("response"),
            "ground_truth": item.get("ground_truth"),
            "extracted_answer": item.get("extracted_answer"),
            "source_correct": item.get("correct"),
            **trace,
        }
        partitions[partition].append(record)

    outputs = {}
    for name, rows in partitions.items():
        if not rows:
            continue
        path = output_dir / f"gsm8k_dense_traces_{name}.jsonl"
        write_jsonl(path, rows)
        outputs[name] = str(path)

    manifest = {
        "status": "done",
        "model_path": args.model_path,
        "baseline_json": str(baseline_path),
        "top_k": args.top_k,
        "max_seq_tokens": args.max_seq_tokens,
        "seed": args.seed,
        "partition_mode": args.partition_mode,
        "counts": {name: len(rows) for name, rows in partitions.items()},
        "outputs": outputs,
    }
    (output_dir / "gsm8k_dense_trace_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
