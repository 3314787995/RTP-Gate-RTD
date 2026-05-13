#!/usr/bin/env python
"""Set-level greedy layer selection for RTP-Gate."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TOTAL_LAYERS = 26


def require_under_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Refusing path outside ROOT: {resolved} not under {root_resolved}")
    return resolved


def parse_int_list(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def layer_kind(layer: int) -> str:
    return "local_sliding" if layer % 2 == 0 else "global"


def structure_penalty(layers: list[int], total_layers: int = TOTAL_LAYERS) -> tuple[float, dict]:
    if not layers:
        return 0.0, {
            "adjacent_pairs": 0,
            "tail_adjacent_pairs": 0,
            "local_removed": 0,
            "global_removed": 0,
            "local_global_imbalance": 0.0,
            "early_guard": 0.0,
        }
    ordered = sorted(layers)
    adjacent_pairs = sum(1 for a, b in zip(ordered, ordered[1:]) if b == a + 1)
    tail_adjacent_pairs = sum(1 for a, b in zip(ordered, ordered[1:]) if b == a + 1 and a >= total_layers - 6)
    local_removed = sum(1 for x in layers if layer_kind(x) == "local_sliding")
    global_removed = len(layers) - local_removed
    total_local = (total_layers + 1) // 2
    total_global = total_layers // 2
    imbalance = abs(local_removed / total_local - global_removed / total_global)
    early_guard = 0.0
    if 0 in layers:
        early_guard += 0.25
    early_guard += 0.04 * sum(1 for x in layers if 1 <= x <= 4)
    penalty = 0.04 * adjacent_pairs + 0.06 * tail_adjacent_pairs + 0.10 * imbalance + early_guard
    return penalty, {
        "adjacent_pairs": adjacent_pairs,
        "tail_adjacent_pairs": tail_adjacent_pairs,
        "local_removed": local_removed,
        "global_removed": global_removed,
        "local_global_imbalance": imbalance,
        "early_guard": early_guard,
    }


def load_score(path: Path, partition: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if partition != "overall":
        return data.get("by_partition", {}).get(partition) or data.get("overall", {})
    return data.get("overall", {})


def candidate_name(prefix: str, layers: list[int]) -> str:
    suffix = "_".join(str(x) for x in sorted(layers)) or "none"
    return f"{prefix}_drop_{suffix}"


def ensure_score(
    root: Path,
    score_script: Path,
    trace_jsonl: list[Path],
    output_dir: Path,
    model_path: str,
    layers: list[int],
    name: str,
    top_k: int,
    max_traces: int | None,
    force: bool,
) -> Path:
    out_json = output_dir / f"{name}.json"
    if out_json.exists() and not force:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        if data.get("status") == "done":
            return out_json
    cmd = [
        sys.executable,
        str(score_script),
        "--root",
        str(root),
        "--candidate-name",
        name,
        "--model-path",
        model_path,
        "--runtime-skip-layers",
        ",".join(str(x) for x in sorted(layers)),
        "--output-dir",
        str(output_dir),
        "--top-k",
        str(top_k),
    ]
    for path in trace_jsonl:
        cmd.extend(["--trace-jsonl", str(path)])
    if max_traces is not None:
        cmd.extend(["--max-traces", str(max_traces)])
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True)
    return out_json


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_greedy(args) -> dict:
    root = Path(args.root).resolve()
    output_dir = require_under_root(Path(args.output_dir).resolve() if args.output_dir else root / "results" / "day12_rtp_gate" / "rtd_scores", root)
    selection_dir = require_under_root(
        Path(args.selection_dir).resolve() if args.selection_dir else root / "results" / "day12_rtp_gate" / "rtp_gate_selection",
        root,
    )
    selection_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_script = Path(args.score_script).resolve() if args.score_script else Path(__file__).with_name("score_rtd_candidates.py")
    trace_jsonl = [require_under_root(Path(p), root) for p in args.trace_jsonl]
    target_ks = sorted(set(parse_int_list(args.target_k)))
    max_k = max(target_ks)
    blocked_layers = set(parse_int_list(args.block_layers))
    chosen: list[int] = []
    all_layers = [i for i in range(args.total_layers) if i not in blocked_layers]
    step_rows = []
    selected_rows = []

    for step in range(1, max_k + 1):
        candidates = []
        for layer in all_layers:
            if layer in chosen:
                continue
            layers = sorted(chosen + [layer])
            name = candidate_name(args.candidate_prefix, layers)
            score_path = ensure_score(
                root,
                score_script,
                trace_jsonl,
                output_dir,
                args.model_path,
                layers,
                name,
                args.top_k,
                args.max_traces,
                args.force_scores,
            )
            score = load_score(score_path, args.selection_partition)
            rtd = float(score.get("rtd", 0.0) or 0.0)
            penalty, structure = structure_penalty(layers, total_layers=args.total_layers)
            selection_score = rtd + (penalty if args.structure_aware else 0.0)
            row = {
                "step": step,
                "candidate_layer": layer,
                "layers": ",".join(str(x) for x in layers),
                "candidate_name": name,
                "rtd": rtd,
                "structure_penalty": penalty,
                "selection_score": selection_score,
                **structure,
                "score_file": str(score_path),
            }
            candidates.append(row)
            step_rows.append(row)
        best = min(candidates, key=lambda r: (r["selection_score"], r["rtd"], r["candidate_layer"]))
        if args.stop_rtd is not None and float(best["rtd"]) > args.stop_rtd:
            selected_rows.append(
                {
                    "label": f"{args.candidate_prefix}_stopped_before_k{step}",
                    "k": step - 1,
                    "layers": ",".join(str(x) for x in sorted(chosen)),
                    "ordered_layers": ",".join(str(x) for x in chosen),
                    "stop_reason": f"best_rtd>{args.stop_rtd}",
                    "best_candidate_layer": best["candidate_layer"],
                    "best_rtd": best["rtd"],
                    "best_selection_score": best["selection_score"],
                }
            )
            break
        chosen.append(int(best["candidate_layer"]))
        if step in target_ks:
            selected_rows.append(
                {
                    "label": f"{args.candidate_prefix}_k{step}",
                    "k": step,
                    "layers": ",".join(str(x) for x in sorted(chosen)),
                    "ordered_layers": ",".join(str(x) for x in chosen),
                    "stop_reason": "",
                    "best_candidate_layer": best["candidate_layer"],
                    "best_rtd": best["rtd"],
                    "best_selection_score": best["selection_score"],
                }
            )

    payload = {
        "status": "done",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_prefix": args.candidate_prefix,
        "structure_aware": args.structure_aware,
        "selection_partition": args.selection_partition,
        "target_k": target_ks,
        "block_layers": sorted(blocked_layers),
        "selected": selected_rows,
        "step_rows_csv": str(selection_dir / f"{args.candidate_prefix}_greedy_steps.csv"),
        "selected_csv": str(selection_dir / f"{args.candidate_prefix}_selected.csv"),
    }
    write_csv(
        selection_dir / f"{args.candidate_prefix}_greedy_steps.csv",
        step_rows,
        [
            "step",
            "candidate_layer",
            "layers",
            "candidate_name",
            "rtd",
            "structure_penalty",
            "selection_score",
            "adjacent_pairs",
            "tail_adjacent_pairs",
            "local_removed",
            "global_removed",
            "local_global_imbalance",
            "early_guard",
            "score_file",
        ],
    )
    write_csv(
        selection_dir / f"{args.candidate_prefix}_selected.csv",
        selected_rows,
        [
            "label",
            "k",
            "layers",
            "ordered_layers",
            "stop_reason",
            "best_candidate_layer",
            "best_rtd",
            "best_selection_score",
        ],
    )
    (selection_dir / f"{args.candidate_prefix}_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("ROOT", "/home/cike/hs/paper2_layer_pruning"))
    parser.add_argument("--trace-jsonl", action="append", required=True)
    parser.add_argument("--score-script", default=None)
    parser.add_argument("--model-path", default="google/gemma-2-2b-it")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--selection-dir", default=None)
    parser.add_argument("--candidate-prefix", default="rtp_gate")
    parser.add_argument("--target-k", default="2,3,5")
    parser.add_argument("--total-layers", type=int, default=TOTAL_LAYERS)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument("--selection-partition", default="overall")
    parser.add_argument("--structure-aware", action="store_true")
    parser.add_argument("--block-layers", default="")
    parser.add_argument("--stop-rtd", type=float, default=None)
    parser.add_argument("--force-scores", action="store_true")
    args = parser.parse_args()
    payload = run_greedy(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

