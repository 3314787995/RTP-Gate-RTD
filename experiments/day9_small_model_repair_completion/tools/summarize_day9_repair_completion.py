#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"status": "bad_json", "error": repr(exc), "file": str(path)}


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize_classification(result_root, report_root):
    rows = []
    baselines = {}
    for path in sorted((result_root / "classification").glob("*.json")):
        data = load_json(path)
        run_id = data.get("run_id", path.stem)
        task = data.get("task", "")
        score = data.get("score")
        if run_id == "gemma_base" and data.get("status") == "done" and score not in (None, ""):
            baselines[task] = float(score)
        rows.append(
            {
                "status": data.get("status"),
                "run_id": run_id,
                "task": task,
                "samples": data.get("samples"),
                "score": score,
                "score_raw": data.get("score_raw"),
                "baseline_score": data.get("baseline_score"),
                "retention": data.get("retention"),
                "adapter_path": data.get("adapter_path"),
                "file": str(path),
            }
        )
    for row in rows:
        if row["baseline_score"] in (None, ""):
            row["baseline_score"] = baselines.get(row["task"], "")
        if row["retention"] in (None, "") and row["baseline_score"] not in (None, "", 0) and row["score"] not in (None, ""):
            row["retention"] = float(row["score"]) / float(row["baseline_score"])
    write_csv(
        report_root / "classification_recovery_matrix.csv",
        rows,
        ["status", "run_id", "task", "samples", "score", "score_raw", "baseline_score", "retention", "adapter_path", "file"],
    )
    return rows


def summarize_generation(result_root, report_root):
    rows = []
    baselines = {}
    for path in sorted((result_root / "generation").glob("*.json")):
        data = load_json(path)
        run_id = data.get("run_id", path.stem)
        task = data.get("task", "")
        group = data.get("job_group", "")
        samples = data.get("samples")
        score = data.get("score")
        if run_id in ("early_stop_base", "selected_bi_alpaca_base") and score not in (None, ""):
            baselines[(group, task, samples)] = float(score)
        rows.append(
            {
                "status": data.get("status"),
                "job_group": group,
                "run_id": run_id,
                "task": task,
                "samples": samples,
                "score": score,
                "retention": data.get("retention", ""),
                "avg_tokens": avg_tokens(data),
                "adapter_path": data.get("adapter_path"),
                "file": str(path),
            }
        )
    for row in rows:
        base = baselines.get((row["job_group"], row["task"], row["samples"]))
        if row["retention"] in (None, "") and base not in (None, 0, "") and row["score"] not in (None, ""):
            row["retention"] = float(row["score"]) / float(base)
    write_csv(
        report_root / "bi_alpaca_xsum_early_stop.csv",
        [row for row in rows if row["job_group"] in ("bi_alpaca_early_stop", "bi_alpaca_selected_test")],
        ["status", "job_group", "run_id", "task", "samples", "score", "retention", "avg_tokens", "adapter_path", "file"],
    )
    return rows


def avg_tokens(data):
    items = data.get("items") or []
    if not items:
        return ""
    return sum(float(item.get("tokens_generated") or 0) for item in items) / len(items)


def summarize_code(result_root, report_root):
    rows = []
    for path in sorted((result_root / "code").glob("*.json")):
        data = load_json(path)
        rows.append(
            {
                "status": data.get("status"),
                "run_id": data.get("run_id", path.stem),
                "task": data.get("task"),
                "variant": data.get("variant"),
                "samples": data.get("samples"),
                "score": data.get("score"),
                "baseline_score": data.get("baseline_score"),
                "retention": data.get("retention"),
                "bracket_balanced_rate": data.get("bracket_balanced_rate"),
                "error_type_counts": json.dumps(data.get("error_type_counts") or {}, sort_keys=True),
                "diagnostic_only": data.get("diagnostic_only", True),
                "file": str(path),
            }
        )
    write_csv(
        report_root / "code_eval_repair.csv",
        rows,
        [
            "status",
            "run_id",
            "task",
            "variant",
            "samples",
            "score",
            "baseline_score",
            "retention",
            "bracket_balanced_rate",
            "error_type_counts",
            "diagnostic_only",
            "file",
        ],
    )
    return rows


def summarize_failure_modes(report_root, generation_rows, code_rows):
    rows = []
    for row in generation_rows:
        if row["task"] == "xsum":
            rows.append(
                {
                    "source": row["job_group"],
                    "run_id": row["run_id"],
                    "task": "xsum",
                    "metric": "avg_tokens",
                    "value": row["avg_tokens"],
                    "details": "length/truncation drift proxy",
                }
            )
        if row["task"] == "gsm8k":
            rows.append(
                {
                    "source": row["job_group"],
                    "run_id": row["run_id"],
                    "task": "gsm8k",
                    "metric": "score",
                    "value": row["score"],
                    "details": "arithmetic/reasoning aggregate; inspect item-level extracted_answer for misses",
                }
            )
    for row in code_rows:
        rows.append(
            {
                "source": "day9_code_eval_repair",
                "run_id": row["run_id"],
                "task": row["task"],
                "metric": "error_type_counts",
                "value": row["error_type_counts"],
                "details": f"bracket_balanced_rate={row.get('bracket_balanced_rate', '')}; diagnostic-only",
            }
        )
    write_csv(report_root / "failure_mode_quant.csv", rows, ["source", "run_id", "task", "metric", "value", "details"])
    return rows


def write_summary(report_root, class_rows, gen_rows, code_rows, failure_rows):
    counts = {}
    for row in class_rows + gen_rows + code_rows:
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    summary = {
        "classification_rows": len(class_rows),
        "generation_rows": len(gen_rows),
        "code_rows": len(code_rows),
        "failure_mode_rows": len(failure_rows),
        "status_counts": counts,
        "outputs": {
            "classification_recovery_matrix": str(report_root / "classification_recovery_matrix.csv"),
            "bi_alpaca_xsum_early_stop": str(report_root / "bi_alpaca_xsum_early_stop.csv"),
            "code_eval_repair": str(report_root / "code_eval_repair.csv"),
            "failure_mode_quant": str(report_root / "failure_mode_quant.csv"),
            "day9_repair_summary": str(report_root / "day9_repair_summary.csv"),
        },
        "caveat": "P100 fp16/transformers-compatible reproduction; code tasks remain diagnostic-only; repo paper version 2602.01997.",
    }
    with open(report_root / "day9_repair_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(report_root / "day9_repair_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in ("classification_rows", "generation_rows", "code_rows", "failure_mode_rows"):
            writer.writerow({"metric": key, "value": summary[key]})
        for status, count in sorted(counts.items()):
            writer.writerow({"metric": f"status_{status}", "value": count})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/cike/hs/paper2_layer_pruning")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result_root = root / "results" / "day9_small_model_repair_completion"
    report_root = root / "reports" / "day9_small_model_repair_completion"
    report_root.mkdir(parents=True, exist_ok=True)
    class_rows = summarize_classification(result_root, report_root)
    gen_rows = summarize_generation(result_root, report_root)
    code_rows = summarize_code(result_root, report_root)
    failure_rows = summarize_failure_modes(report_root, gen_rows, code_rows)
    write_summary(report_root, class_rows, gen_rows, code_rows, failure_rows)


if __name__ == "__main__":
    main()
