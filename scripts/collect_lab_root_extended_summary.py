#!/usr/bin/env python3
"""Collect phase-2 lab-root RTP-Gate k-curve and stability outputs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys


ROOT = Path("/root/hs/paper2_layer_pruning")
K_CURVE_RUN_IDS = [
    "rtp_gate_k2",
    "rtp_gate_k5",
    "reverse_2",
    "reverse_5",
    "iterative_proxy_k2",
    "iterative_proxy_k5",
]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing", "file": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unreadable", "file": str(path), "error": str(exc)}


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def runtime_skip_layers(payload: dict) -> list[int]:
    runtime_skip = payload.get("runtime_skip") or {}
    raw = runtime_skip.get("skip_layers")
    if isinstance(raw, list):
        return [int(value) for value in raw]
    raw = payload.get("runtime_skip_layers")
    if isinstance(raw, list):
        return [int(value) for value in raw]
    return []


def collect_k_curve(root: Path, samples: int) -> dict:
    output_dir = root / "results/day12_rtp_gate/raw_eval/generation_k_curve_20260514"
    manifest_path = output_dir.parent / f"generation_k_curve_20260514_manifest_{samples}.json"
    rows = []
    for run_id in K_CURVE_RUN_IDS:
        path = output_dir / f"{run_id}_gsm8k_{samples}.json"
        payload = read_json(path)
        rows.append(
            {
                "status": payload.get("status"),
                "run_id": payload.get("run_id") or run_id,
                "task": payload.get("task") or "gsm8k",
                "samples": payload.get("samples") or samples,
                "score": payload.get("score"),
                "runtime_skip_layers": runtime_skip_layers(payload),
                "model_path": payload.get("model_path"),
                "file": str(path),
            }
        )
    return {
        "manifest": read_json(manifest_path),
        "new_results": rows,
    }


def collect_stability(root: Path, seed: int) -> dict:
    output_root = root / f"results/day12_rtp_gate_stability/seed_{seed}"
    report_dir = root / "reports/day12_rtp_gate"
    return {
        "seed": seed,
        "manifest": read_json(output_root / "stability_manifest.json"),
        "trace_manifest": read_json(output_root / "traces/gsm8k_dense_trace_manifest.json"),
        "summary": read_csv_rows(report_dir / f"stability_seed_{seed}_summary.csv"),
        "selection_overlap": read_csv_rows(report_dir / f"stability_seed_{seed}_selection_overlap.csv"),
        "single_layer_rtd": read_csv_rows(report_dir / f"stability_seed_{seed}_single_layer_rtd.csv"),
        "report_md": str(report_dir / f"stability_seed_{seed}_report.md"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()

    root = Path(args.root)
    beijing = timezone(timedelta(hours=8))
    payload = {
        "snapshot": {
            "server": "lab-root",
            "root": str(root),
            "snapshot_time_beijing": datetime.now(beijing).strftime("%Y-%m-%d %H:%M"),
            "phase": "extended k-curve and seed stability",
        },
        "k_curve": collect_k_curve(root, args.samples),
        "stability": collect_stability(root, args.seed),
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
