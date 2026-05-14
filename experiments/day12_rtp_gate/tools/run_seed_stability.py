#!/usr/bin/env python
"""Run a seed-shuffled RTP-Gate RTD/selection stability check."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from queue import Queue
import subprocess

try:
    from lab_root_rtp_gate_jobs import build_score_command, remote_path
    from select_rtp_gate_layers import structure_penalty
except ModuleNotFoundError:
    from .lab_root_rtp_gate_jobs import build_score_command, remote_path
    from .select_rtp_gate_layers import structure_penalty


ROOT = Path("/root/hs/paper2_layer_pruning")
MODEL_PATH = ROOT / "cache/huggingface/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_gpus(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def env_for(root: Path, model_path: Path, gpu: str) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "ROOT": str(root),
            "HF_HOME": str(root / "cache/huggingface"),
            "HF_DATASETS_CACHE": str(root / "cache/huggingface/datasets"),
            "TRANSFORMERS_CACHE": str(root / "cache/huggingface"),
            "HF_ENDPOINT": env.get("HF_ENDPOINT", "https://hf-mirror.com"),
            "MODEL_PATH": str(model_path),
            "PYTHONPATH": f"{root}:{root / 'code/on-the-limits-of-layer-pruning'}:{env.get('PYTHONPATH', '')}",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu,
        }
    )
    return env


def run_logged(name: str, command: list[str], root: Path, model_path: Path, gpu: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / f"{name}.status.json"
    log_path = log_dir / f"{name}.log"
    payload = {"status": "running", "name": name, "gpu": gpu, "command": command, "started_at_utc": now()}
    write_json(status_path, payload)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[{now()}] START {name} gpu={gpu}\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=str(root), env=env_for(root, model_path, gpu), stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[{now()}] END {name} return_code={result.returncode}\n")
    payload.update(
        {
            "status": "done" if result.returncode == 0 else "failed",
            "return_code": result.returncode,
            "finished_at_utc": now(),
            "log": str(log_path),
        }
    )
    write_json(status_path, payload)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def score_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "done"
    except Exception:
        return False


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            result[ordered[k][0]] = rank
        i = j + 1
    return result


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 2:
        return None
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    denom = math.sqrt(sum(x * x for x in da) * sum(x * x for x in db))
    if denom == 0:
        return None
    return sum(x * y for x, y in zip(da, db)) / denom


def spearman(a: list[float], b: list[float]) -> float | None:
    return pearson(ranks(a), ranks(b))


def layer_set(raw: str) -> set[int]:
    return {int(x.strip()) for x in str(raw or "").split(",") if x.strip()}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def candidate_name_for_layers(layers: tuple[int, ...]) -> str:
    suffix = "_".join(str(x) for x in layers) or "none"
    return f"rtp_gate_stability_drop_{suffix}"


def score_path_for_layers(score_dir: Path, layers: tuple[int, ...]) -> Path:
    if len(layers) == 1:
        single_path = score_dir / f"single_layer_{layers[0]}.json"
        if score_done(single_path):
            return single_path
    return score_dir / f"{candidate_name_for_layers(layers)}.json"


def load_partition_rtd(path: Path, partition: str = "calibration") -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if partition != "overall":
        score = (payload.get("by_partition") or {}).get(partition) or payload.get("overall") or {}
    else:
        score = payload.get("overall") or {}
    return float(score.get("rtd", 0.0) or 0.0)


def ensure_layer_set_scores(
    *,
    root: Path,
    model_path: Path,
    score_dir: Path,
    trace_paths: list[Path],
    layer_sets: set[tuple[int, ...]],
    top_k: int,
    max_seq_tokens: int,
    force: bool,
    gpus: list[str],
    log_dir: Path,
) -> None:
    missing = [layers for layers in sorted(layer_sets) if force or not score_done(score_path_for_layers(score_dir, layers))]
    if not missing:
        return
    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def score_one(layers: tuple[int, ...]) -> dict:
        gpu = gpu_queue.get()
        try:
            name = candidate_name_for_layers(layers)
            command = build_score_command(
                python_bin=str(root / "envs/rtp-gate-2080ti/bin/python"),
                root=root,
                model_path=model_path,
                candidate_name=name,
                layers=list(layers),
                trace_paths=trace_paths,
                output_dir=score_dir,
                top_k=top_k,
                max_seq_tokens=max_seq_tokens,
                force=force,
            )
            run_logged(f"score_{name}", command, root, model_path, gpu, log_dir)
            return {"status": "done", "candidate_name": name, "gpu": gpu}
        finally:
            gpu_queue.put(gpu)

    with ThreadPoolExecutor(max_workers=min(len(gpus), len(missing))) as pool:
        for future in as_completed([pool.submit(score_one, layers) for layers in missing]):
            print(json.dumps(future.result(), ensure_ascii=False), flush=True)


def write_selection_outputs(
    selection_dir: Path,
    prefix: str,
    structure_aware: bool,
    target_ks: list[int],
    step_rows: list[dict],
    selected_rows: list[dict],
) -> None:
    write_csv(
        selection_dir / f"{prefix}_greedy_steps.csv",
        step_rows,
        [
            "step",
            "candidate_layer",
            "layers",
            "candidate_name",
            "score_candidate_name",
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
        selection_dir / f"{prefix}_selected.csv",
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
    payload = {
        "status": "done",
        "created_at_utc": now(),
        "candidate_prefix": prefix,
        "structure_aware": structure_aware,
        "selection_partition": "calibration",
        "target_k": target_ks,
        "selected": selected_rows,
        "step_rows_csv": str(selection_dir / f"{prefix}_greedy_steps.csv"),
        "selected_csv": str(selection_dir / f"{prefix}_selected.csv"),
        "score_cache": "shared rtp_gate_stability_drop_* scores; single-layer scores reused from single_layer_*",
    }
    write_json(selection_dir / f"{prefix}_selection.json", payload)


def run_shared_greedy_selection(
    *,
    root: Path,
    model_path: Path,
    score_dir: Path,
    selection_dir: Path,
    trace_paths: list[Path],
    gpus: list[str],
    log_dir: Path,
    target_ks: list[int],
    top_k: int,
    max_seq_tokens: int,
    force: bool,
    total_layers: int = 26,
) -> None:
    selection_dir.mkdir(parents=True, exist_ok=True)
    families = [
        {"prefix": "rtp_gate_pure", "structure_aware": False, "chosen": [], "step_rows": [], "selected_rows": []},
        {"prefix": "rtp_gate_structure", "structure_aware": True, "chosen": [], "step_rows": [], "selected_rows": []},
    ]
    all_layers = list(range(total_layers))
    max_k = max(target_ks)
    for step in range(1, max_k + 1):
        needed: set[tuple[int, ...]] = set()
        for family in families:
            chosen = family["chosen"]
            for layer in all_layers:
                if layer not in chosen:
                    needed.add(tuple(sorted([*chosen, layer])))
        ensure_layer_set_scores(
            root=root,
            model_path=model_path,
            score_dir=score_dir,
            trace_paths=trace_paths,
            layer_sets=needed,
            top_k=top_k,
            max_seq_tokens=max_seq_tokens,
            force=force,
            gpus=gpus,
            log_dir=log_dir,
        )
        for family in families:
            chosen = family["chosen"]
            candidates = []
            for layer in all_layers:
                if layer in chosen:
                    continue
                layers = tuple(sorted([*chosen, layer]))
                score_path = score_path_for_layers(score_dir, layers)
                rtd = load_partition_rtd(score_path, "calibration")
                penalty, structure = structure_penalty(list(layers), total_layers=total_layers)
                selection_score = rtd + (penalty if family["structure_aware"] else 0.0)
                prefix_name = f"{family['prefix']}_drop_{'_'.join(str(x) for x in layers)}"
                row = {
                    "step": step,
                    "candidate_layer": layer,
                    "layers": ",".join(str(x) for x in layers),
                    "candidate_name": prefix_name,
                    "score_candidate_name": score_path.stem,
                    "rtd": rtd,
                    "structure_penalty": penalty,
                    "selection_score": selection_score,
                    **structure,
                    "score_file": str(score_path),
                }
                candidates.append(row)
                family["step_rows"].append(row)
            best = min(candidates, key=lambda row: (row["selection_score"], row["rtd"], row["candidate_layer"]))
            chosen.append(int(best["candidate_layer"]))
            if step in target_ks:
                family["selected_rows"].append(
                    {
                        "label": f"{family['prefix']}_k{step}",
                        "k": step,
                        "layers": ",".join(str(x) for x in sorted(chosen)),
                        "ordered_layers": ",".join(str(x) for x in chosen),
                        "stop_reason": "",
                        "best_candidate_layer": best["candidate_layer"],
                        "best_rtd": best["rtd"],
                        "best_selection_score": best["selection_score"],
                    }
                )
    for family in families:
        write_selection_outputs(
            selection_dir,
            str(family["prefix"]),
            bool(family["structure_aware"]),
            target_ks,
            list(family["step_rows"]),
            list(family["selected_rows"]),
        )


def summarize_stability(root: Path, output_root: Path, seed: int, report_dir: Path) -> dict:
    base_score_dir = root / "results/day12_rtp_gate/rtd_scores"
    new_score_dir = output_root / "rtd_scores"
    pairs = []
    for layer in range(26):
        base_path = base_score_dir / f"single_layer_{layer}.json"
        new_path = new_score_dir / f"single_layer_{layer}.json"
        if not base_path.exists() or not new_path.exists():
            continue
        base = json.loads(base_path.read_text(encoding="utf-8"))
        new = json.loads(new_path.read_text(encoding="utf-8"))
        base_rtd = ((base.get("by_partition") or {}).get("calibration") or {}).get("rtd")
        new_rtd = ((new.get("by_partition") or {}).get("calibration") or {}).get("rtd")
        if base_rtd is None or new_rtd is None:
            continue
        pairs.append({"layer": layer, "base_calibration_rtd": float(base_rtd), "seed_calibration_rtd": float(new_rtd)})

    base_values = [row["base_calibration_rtd"] for row in pairs]
    seed_values = [row["seed_calibration_rtd"] for row in pairs]
    rho = spearman(base_values, seed_values)
    base_top5 = {row["layer"] for row in sorted(pairs, key=lambda row: row["base_calibration_rtd"], reverse=True)[:5]}
    seed_top5 = {row["layer"] for row in sorted(pairs, key=lambda row: row["seed_calibration_rtd"], reverse=True)[:5]}

    comparison_rows = []
    for family in ("rtp_gate_pure", "rtp_gate_structure"):
        base_rows = {row["label"]: row for row in read_csv_rows(root / "results/day12_rtp_gate/rtp_gate_selection" / f"{family}_selected.csv")}
        new_rows = {row["label"]: row for row in read_csv_rows(output_root / "rtp_gate_selection" / f"{family}_selected.csv")}
        for k in (2, 3, 5):
            label = f"{family}_k{k}"
            base_set = layer_set((base_rows.get(label) or {}).get("layers"))
            new_set = layer_set((new_rows.get(label) or {}).get("layers"))
            comparison_rows.append(
                {
                    "family": family,
                    "k": k,
                    "base_layers": ",".join(str(x) for x in sorted(base_set)),
                    "seed_layers": ",".join(str(x) for x in sorted(new_set)),
                    "jaccard": f"{jaccard(base_set, new_set):.6f}",
                }
            )

    report_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = report_dir / f"stability_seed_{seed}_summary.csv"
    write_csv(
        summary_csv,
        [
            {
                "seed": seed,
                "single_layer_count": len(pairs),
                "single_layer_spearman": "" if rho is None else f"{rho:.6f}",
                "base_top5_layers": ",".join(str(x) for x in sorted(base_top5)),
                "seed_top5_layers": ",".join(str(x) for x in sorted(seed_top5)),
                "top5_overlap": len(base_top5 & seed_top5),
            }
        ],
        ["seed", "single_layer_count", "single_layer_spearman", "base_top5_layers", "seed_top5_layers", "top5_overlap"],
    )
    selection_csv = report_dir / f"stability_seed_{seed}_selection_overlap.csv"
    write_csv(selection_csv, comparison_rows, ["family", "k", "base_layers", "seed_layers", "jaccard"])
    layer_csv = report_dir / f"stability_seed_{seed}_single_layer_rtd.csv"
    write_csv(layer_csv, pairs, ["layer", "base_calibration_rtd", "seed_calibration_rtd"])

    report_md = report_dir / f"stability_seed_{seed}_report.md"
    selection_lines = "\n".join(
        f"| {row['family']} | {row['k']} | {row['base_layers']} | {row['seed_layers']} | {row['jaccard']} |"
        for row in comparison_rows
    )
    report_md.write_text(
        "\n".join(
            [
                f"# RTP-Gate seed {seed} stability report",
                "",
                f"Generated: {now()}",
                "",
                "## Summary",
                "",
                f"- Single-layer pairs: {len(pairs)}",
                f"- Single-layer calibration RTD Spearman: {'' if rho is None else f'{rho:.6f}'}",
                f"- Base top-5 risky layers: {','.join(str(x) for x in sorted(base_top5))}",
                f"- Seed top-5 risky layers: {','.join(str(x) for x in sorted(seed_top5))}",
                f"- Top-5 overlap: {len(base_top5 & seed_top5)}/5",
                "",
                "## RTP-Gate selection overlap",
                "",
                "| family | k | base layers | seed layers | Jaccard |",
                "|---|---:|---|---|---:|",
                selection_lines,
                "",
                "RTD/selection stability is diagnostic only; no raw GSM8K was run for this seed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "summary_csv": str(summary_csv),
        "selection_csv": str(selection_csv),
        "single_layer_csv": str(layer_csv),
        "report_md": str(report_md),
        "single_layer_spearman": rho,
        "top5_overlap": len(base_top5 & seed_top5),
        "selection_rows": comparison_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--seed", type=int, default=5678)
    parser.add_argument("--smoke-samples", type=int, default=50)
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--holdout-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-seq-tokens", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    model_path = Path(args.model_path)
    output_root = Path(args.output_root) if args.output_root else root / f"results/day12_rtp_gate_stability/seed_{args.seed}"
    report_dir = Path(args.report_dir) if args.report_dir else root / "reports/day12_rtp_gate"
    traces = output_root / "traces"
    score_dir = output_root / "rtd_scores"
    selection_dir = output_root / "rtp_gate_selection"
    log_dir = output_root / "logs"
    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("At least one GPU id is required.")
    for path in (traces, score_dir, selection_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "stability_manifest.json"
    write_json(
        manifest_path,
        {
            "status": "running",
            "created_at_utc": now(),
            "seed": args.seed,
            "output_root": str(output_root),
            "gpus": gpus,
            "counts": {"smoke": args.smoke_samples, "calibration": args.calibration_samples, "holdout": args.holdout_samples},
        },
    )

    trace_manifest = traces / "gsm8k_dense_trace_manifest.json"
    if args.force or not trace_manifest.exists():
        collect_cmd = [
            str(root / "envs/rtp-gate-2080ti/bin/python"),
            str(root / "experiments/day12_rtp_gate/tools/collect_dense_traces.py"),
            "--root",
            str(root),
            "--model-path",
            str(model_path),
            "--baseline-json",
            str(root / "results/day5_technical_gap_closure/gemma_base_full_eval_gsm8k_1319.json"),
            "--output-dir",
            str(traces),
            "--partition-mode",
            "full",
            "--smoke-samples",
            str(args.smoke_samples),
            "--calibration-samples",
            str(args.calibration_samples),
            "--holdout-samples",
            str(args.holdout_samples),
            "--top-k",
            str(args.top_k),
            "--max-seq-tokens",
            str(args.max_seq_tokens),
            "--seed",
            str(args.seed),
            "--shuffle-correct-items",
        ]
        run_logged("collect_dense_traces", collect_cmd, root, model_path, gpus[0], log_dir)

    trace_paths = [traces / "gsm8k_dense_traces_calibration.jsonl", traces / "gsm8k_dense_traces_holdout.jsonl"]
    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def score_layer(layer: int) -> dict:
        name = f"single_layer_{layer}"
        output = score_dir / f"{name}.json"
        if score_done(output) and not args.force:
            return {"status": "skipped_done", "name": name}
        gpu = gpu_queue.get()
        try:
            command = build_score_command(
                python_bin=str(root / "envs/rtp-gate-2080ti/bin/python"),
                root=root,
                model_path=model_path,
                candidate_name=name,
                layers=[layer],
                trace_paths=trace_paths,
                output_dir=score_dir,
                top_k=args.top_k,
                max_seq_tokens=args.max_seq_tokens,
                force=args.force,
            )
            run_logged(f"score_{name}", command, root, model_path, gpu, log_dir)
            return {"status": "done", "name": name, "gpu": gpu}
        finally:
            gpu_queue.put(gpu)

    with ThreadPoolExecutor(max_workers=min(len(gpus), 26)) as pool:
        for future in as_completed([pool.submit(score_layer, layer) for layer in range(26)]):
            print(json.dumps(future.result(), ensure_ascii=False), flush=True)

    run_shared_greedy_selection(
        root=root,
        model_path=model_path,
        score_dir=score_dir,
        selection_dir=selection_dir,
        trace_paths=trace_paths,
        gpus=gpus,
        log_dir=log_dir,
        target_ks=[2, 3, 5],
        top_k=args.top_k,
        max_seq_tokens=args.max_seq_tokens,
        force=args.force,
    )

    summary = summarize_stability(root, output_root, args.seed, report_dir)
    final = json.loads(manifest_path.read_text(encoding="utf-8"))
    final.update({"status": "done", "finished_at_utc": now(), "summary": summary})
    write_json(manifest_path, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
