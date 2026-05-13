#!/usr/bin/env python
import argparse
import csv
import json
import os
from pathlib import Path


def require_inside_root(path, root):
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Refusing to write outside ROOT: {resolved} not under {root_resolved}")
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()

    root = os.environ.get("ROOT")
    if not root:
        raise RuntimeError("ROOT is not set. Source env.sh before running.")
    results_dir = require_inside_root(args.results_dir, root)
    reports_dir = require_inside_root(args.reports_dir, root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith("_summary.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "model_label" not in data or "task" not in data:
            continue
        task = data["task"]
        samples = data.get("samples")
        formal_samples = {"gsm8k": 20, "humeval": 5, "xsum": 20}
        if not args.include_smoke and samples != formal_samples.get(task):
            continue
        rows.append({
            "model_label": data["model_label"],
            "task": task,
            "samples": samples,
            "score": data.get("score"),
            "baseline_score": None,
            "retention": None,
            "result_json": str(path),
            "status": data.get("status"),
            "prune_strategy": data.get("prune_strategy"),
            "pruned_layers": ",".join(str(x) for x in data.get("pruned_layers", [])),
        })

    baseline_by_task = {
        row["task"]: row["score"]
        for row in rows
        if row["model_label"] == "gemma_base"
    }
    for row in rows:
        base = baseline_by_task.get(row["task"])
        row["baseline_score"] = base
        if base is not None and base != 0:
            row["retention"] = row["score"] / base if row["score"] is not None else None

    csv_path = reports_dir / "day2_gemma_core_eval_summary.csv"
    md_path = reports_dir / "day2_gemma_core_eval_summary.md"
    fields = [
        "model_label", "task", "samples", "score", "baseline_score",
        "retention", "result_json", "status", "prune_strategy", "pruned_layers"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Day 2 Gemma Core Eval Summary\n\n")
        f.write("| model_label | task | samples | score | baseline_score | retention | status |\n")
        f.write("|---|---|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['model_label']} | {row['task']} | {row['samples']} | "
                f"{fmt(row['score'])} | {fmt(row['baseline_score'])} | "
                f"{fmt(row['retention'])} | {row['status']} |\n"
            )

    summary_json = reports_dir / "day2_gemma_core_eval_summary.json"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "json": str(summary_json),
    }, indent=2))


if __name__ == "__main__":
    main()
