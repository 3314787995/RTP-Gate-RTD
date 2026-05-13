#!/usr/bin/env python
import argparse
import json
from pathlib import Path


CANDIDATES = {
    "checkpoint_12000": "outputs/day2_7_paper_param_retrain/bi_alpaca_full_paperlr_seq2048/checkpoint-12000",
    "checkpoint_12500": "outputs/day2_7_paper_param_retrain/bi_alpaca_full_paperlr_seq2048/checkpoint-12500",
    "final_adapter": "outputs/day2_7_paper_param_retrain/bi_alpaca_full_paperlr_seq2048/final_adapter",
}


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_for(result_root, run_id, task, samples):
    path = result_root / "generation" / f"{run_id}_{task}_{samples}.json"
    if not path.exists():
        return None, None, str(path)
    data = load(path)
    if data.get("status") != "done":
        return None, data.get("status"), str(path)
    return data.get("score"), data.get("status"), str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/cike/hs/paper2_layer_pruning")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--gsm8k-max-abs-drop", type=float, default=0.05)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    result_root = root / "results" / "day9_small_model_repair_completion"
    rows = []
    for label, rel_adapter in CANDIDATES.items():
        run_id = f"bi_alpaca_{label}_valid"
        xsum_score, xsum_status, xsum_file = score_for(result_root, run_id, "xsum", args.samples)
        gsm_score, gsm_status, gsm_file = score_for(result_root, run_id, "gsm8k", args.samples)
        if xsum_score is None or gsm_score is None:
            raise RuntimeError(f"Validation result missing or incomplete for {label}: xsum={xsum_status} gsm8k={gsm_status}")
        rows.append(
            {
                "label": label,
                "adapter_path": str(root / rel_adapter),
                "xsum_score": float(xsum_score),
                "gsm8k_score": float(gsm_score),
                "xsum_file": xsum_file,
                "gsm8k_file": gsm_file,
            }
        )

    best_gsm8k = max(row["gsm8k_score"] for row in rows)
    gsm8k_floor = max(0.0, best_gsm8k - args.gsm8k_max_abs_drop)
    eligible = [row for row in rows if row["gsm8k_score"] >= gsm8k_floor]
    selected = max(eligible, key=lambda row: row["xsum_score"]) if eligible else max(rows, key=lambda row: row["xsum_score"])
    data = {
        "status": "done",
        "selection_rule": "choose highest validation XSUM among candidates with GSM8K no more than 0.05 absolute below the best validation GSM8K",
        "samples": args.samples,
        "best_validation_gsm8k": best_gsm8k,
        "gsm8k_floor": gsm8k_floor,
        "selected": selected,
        "candidates": rows,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(json.dumps({"status": "done", "selected": selected["label"], "adapter_path": selected["adapter_path"]}, indent=2))


if __name__ == "__main__":
    main()
