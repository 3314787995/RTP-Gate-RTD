#!/usr/bin/env python
"""Score RTP-Gate RTD components for one candidate layer skip set."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "google/gemma-2-2b-it"
RTD_WEIGHTS = {
    "normalized_delta_nll": 0.25,
    "topk_union_jsd": 0.18,
    "late_jsd_spike": 0.15,
    "late_entropy_spike": 0.15,
    "entropy_monotonicity_violation": 0.10,
    "math_token_delta_nll": 0.08,
    "answer_token_delta_nll": 0.05,
    "overconfidence_collapse": 0.04,
}


def require_under_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Refusing path outside ROOT: {resolved} not under {root_resolved}")
    return resolved


def parse_layers(raw) -> list[int]:
    if raw is None or not str(raw).strip():
        return []
    return sorted({int(x.strip()) for x in str(raw).split(",") if x.strip()})


def set_cache_config(model) -> None:
    model.config.use_cache = False
    if hasattr(model.config, "cache_implementation"):
        model.config.cache_implementation = None
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
        if hasattr(model.generation_config, "cache_implementation"):
            model.generation_config.cache_implementation = None


def runtime_skip_layers(model, skip_layers: list[int], preserve_layer_idx: bool = False) -> dict:
    original_layers = list(model.model.layers)
    bad_layers = [layer for layer in skip_layers if layer < 0 or layer >= len(original_layers)]
    if bad_layers:
        raise ValueError(f"Invalid layers for {len(original_layers)}-layer model: {bad_layers}")
    if not skip_layers:
        return {
            "enabled": False,
            "skip_layers": [],
            "kept_layers": list(range(len(original_layers))),
            "changed_sliding_if_resaved": [],
        }
    skip_set = set(skip_layers)
    kept = [(idx, layer) for idx, layer in enumerate(original_layers) if idx not in skip_set]
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
        if not preserve_layer_idx:
            if hasattr(layer, "layer_idx"):
                layer.layer_idx = new_idx
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = new_idx
    model.model.layers = torch.nn.ModuleList([layer for _, layer in kept])
    model.config.num_hidden_layers = len(kept)
    if hasattr(model.model, "config"):
        model.model.config.num_hidden_layers = len(kept)
    return {
        "enabled": True,
        "skip_layers": skip_layers,
        "kept_layers": [idx for idx, _ in kept],
        "changed_sliding_if_resaved": changed_sliding,
        "preserve_layer_idx": preserve_layer_idx,
    }


def load_traces(paths: list[Path], limit: int | None = None) -> list[dict]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                    if limit is not None and len(rows) >= limit:
                        return rows
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def masked(values: list[float], mask: list[int]) -> list[float]:
    return [value for value, keep in zip(values, mask) if keep]


def positive(values: list[float]) -> list[float]:
    return [max(0.0, value) for value in values]


def js_divergence(p: list[float], q: list[float]) -> float:
    eps = 1e-12
    total = 0.0
    for a, b in zip(p, q):
        a = max(float(a), eps)
        b = max(float(b), eps)
        m = 0.5 * (a + b)
        total += 0.5 * a * math.log(a / m) + 0.5 * b * math.log(b / m)
    return total


def topk_union_jsd(dense_ids, dense_probs, pruned_ids, pruned_probs) -> float:
    token_order = []
    seen = set()
    for token_id in list(dense_ids) + list(pruned_ids):
        token_id = int(token_id)
        if token_id not in seen:
            seen.add(token_id)
            token_order.append(token_id)
    dense_map = {int(t): float(p) for t, p in zip(dense_ids, dense_probs)}
    pruned_map = {int(t): float(p) for t, p in zip(pruned_ids, pruned_probs)}
    p = [dense_map.get(t, 0.0) for t in token_order]
    q = [pruned_map.get(t, 0.0) for t in token_order]
    p.append(max(0.0, 1.0 - sum(dense_map.values())))
    q.append(max(0.0, 1.0 - sum(pruned_map.values())))
    p_total = sum(p) or 1.0
    q_total = sum(q) or 1.0
    p = [x / p_total for x in p]
    q = [x / q_total for x in q]
    return js_divergence(p, q)


def teacher_forced_forward(model, tokenizer, trace: dict, top_k: int, max_seq_tokens: int) -> dict:
    full_text = trace["prompt"] + trace["response"]
    prompt_encoded = tokenizer(trace["prompt"], return_tensors="pt", truncation=True, max_length=max_seq_tokens)
    full_encoded = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_seq_tokens)
    input_ids = full_encoded["input_ids"].to(model.device)
    attention_mask = full_encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    prompt_len = int(prompt_encoded["input_ids"].shape[-1])
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    response_positions = [i for i in range(labels.shape[-1]) if i + 1 >= prompt_len]
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
    token_ids = [int(x) for x in selected_labels.detach().cpu().tolist()]
    trace_ids = [int(x) for x in trace.get("response_token_ids", [])]
    token_mismatch = token_ids != trace_ids[: len(token_ids)] or len(token_ids) != len(trace_ids)
    return {
        "token_ids": token_ids,
        "token_mismatch": token_mismatch,
        "nll": [float(x) for x in nll.detach().cpu().tolist()],
        "entropy": [float(x) for x in entropy.detach().cpu().tolist()],
        "topk_token_ids": [[int(y) for y in row] for row in top_ids.detach().cpu().tolist()],
        "topk_probs": [[float(y) for y in row] for row in top_probs.detach().cpu().tolist()],
    }


def trace_components(trace: dict, pruned: dict) -> dict:
    n = min(len(trace["dense_token_nll"]), len(pruned["nll"]))
    dense_nll = [float(x) for x in trace["dense_token_nll"][:n]]
    dense_entropy = [float(x) for x in trace["dense_entropy"][:n]]
    pruned_nll = pruned["nll"][:n]
    pruned_entropy = pruned["entropy"][:n]
    masks = trace.get("masks") or {}
    reason_mask = (masks.get("reason_mask") or [1] * n)[:n]
    answer_mask = (masks.get("answer_mask") or [0] * n)[:n]
    math_mask = (masks.get("math_mask") or [0] * n)[:n]
    late_mask = (masks.get("late_mask") or [0] * n)[:n]

    delta_nll = [float(p - d) for p, d in zip(pruned_nll, dense_nll)]
    positive_delta = positive(delta_nll)
    normalized_delta = [v / max(d, 1e-4) for v, d in zip(positive_delta, dense_nll)]
    delta_entropy = [float(p - d) for p, d in zip(pruned_entropy, dense_entropy)]

    jsd = []
    dense_top_ids = trace["dense_topk_token_ids"][:n]
    dense_top_probs = trace["dense_topk_probs"][:n]
    pruned_top_ids = pruned["topk_token_ids"][:n]
    pruned_top_probs = pruned["topk_probs"][:n]
    for i in range(n):
        limit = len(pruned_top_ids[i])
        jsd.append(
            topk_union_jsd(
                dense_top_ids[i][:limit],
                dense_top_probs[i][:limit],
                pruned_top_ids[i],
                pruned_top_probs[i],
            )
        )

    pruned_entropy_jumps = [
        max(0.0, pruned_entropy[i] - pruned_entropy[i - 1]) for i in range(1, len(pruned_entropy))
    ]
    entropy_monotonicity = mean(pruned_entropy_jumps)
    late_jsd = masked(jsd, late_mask)
    late_entropy_delta = positive(masked(delta_entropy, late_mask))
    math_reason_mask = [1 if math_mask[i] and not answer_mask[i] else 0 for i in range(n)]
    overconfidence_terms = []
    for d_nll, d_ent in zip(delta_nll, delta_entropy):
        if d_nll > 0 and d_ent < 0:
            overconfidence_terms.append(d_nll * abs(d_ent))

    components = {
        "normalized_delta_nll": mean(normalized_delta),
        "topk_union_jsd": mean(jsd),
        "late_jsd_spike": percentile(late_jsd, 0.95),
        "late_entropy_spike": percentile(late_entropy_delta, 0.95),
        "entropy_monotonicity_violation": entropy_monotonicity,
        "math_token_delta_nll": mean(masked(normalized_delta, math_reason_mask)),
        "answer_token_delta_nll": mean(masked(normalized_delta, answer_mask)),
        "overconfidence_collapse": mean(overconfidence_terms),
    }
    components["rtd"] = sum(RTD_WEIGHTS[name] * components[name] for name in RTD_WEIGHTS)
    components["token_count"] = n
    components["token_mismatch"] = bool(pruned.get("token_mismatch"))
    return components


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    fields = list(RTD_WEIGHTS) + ["rtd", "token_count"]
    return {field: mean([float(row.get(field, 0.0) or 0.0) for row in rows]) for field in fields}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("ROOT", "/home/cike/hs/paper2_layer_pruning"))
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--runtime-skip-layers", default="")
    parser.add_argument("--trace-jsonl", action="append", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-seq-tokens", type=int, default=2560)
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument("--preserve-layer-idx", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    trace_paths = [require_under_root(Path(p), root) for p in args.trace_jsonl]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "results" / "day12_rtp_gate" / "rtd_scores"
    output_dir = require_under_root(output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"{args.candidate_name}.json"
    out_csv = output_dir / f"{args.candidate_name}_items.csv"
    if out_json.exists() and not args.force:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        if data.get("status") == "done":
            print(json.dumps({"status": "done", "cached": True, "output": str(out_json)}, indent=2))
            return

    traces = load_traces(trace_paths, args.max_traces)
    skip_layers = parse_layers(args.runtime_skip_layers)
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
    runtime_skip = runtime_skip_layers(model, skip_layers, preserve_layer_idx=args.preserve_layer_idx)
    if args.adapter_path and args.adapter_path.lower() not in ("none", "null"):
        adapter_path = require_under_root(Path(args.adapter_path), root)
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required only when --adapter-path is provided.") from exc
        model = PeftModel.from_pretrained(model, str(adapter_path))
    else:
        adapter_path = None
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    item_rows = []
    for trace in traces:
        pruned = teacher_forced_forward(model, tokenizer, trace, top_k=args.top_k, max_seq_tokens=args.max_seq_tokens)
        comp = trace_components(trace, pruned)
        item_rows.append(
            {
                "candidate_name": args.candidate_name,
                "partition": trace.get("partition"),
                "trace_id": trace.get("trace_id"),
                "source_index": trace.get("source_index"),
                **comp,
            }
        )

    by_partition = {}
    for partition in sorted({row["partition"] for row in item_rows}):
        by_partition[partition] = aggregate([row for row in item_rows if row["partition"] == partition])
    payload = {
        "status": "done",
        "candidate_name": args.candidate_name,
        "model_path": args.model_path,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "runtime_skip_layers": skip_layers,
        "runtime_skip": runtime_skip,
        "top_k": args.top_k,
        "trace_paths": [str(p) for p in trace_paths],
        "trace_count": len(item_rows),
        "weights": RTD_WEIGHTS,
        "overall": aggregate(item_rows),
        "by_partition": by_partition,
        "item_csv": str(out_csv),
    }
    fields = ["candidate_name", "partition", "trace_id", "source_index"] + list(RTD_WEIGHTS) + [
        "rtd",
        "token_count",
        "token_mismatch",
    ]
    write_csv(out_csv, item_rows, fields)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
