#!/usr/bin/env python
"""Shared job definitions for lab-root RTP-Gate reruns."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


GENERATION_TASKS = ("gsm8k", "xsum")
CLASSIFICATION_TASKS = (
    "hellaswag",
    "piqa",
    "mmlu",
    "winogrande",
    "openbookqa",
    "arc_easy",
    "arc_challenge",
)


@dataclass(frozen=True)
class RawEvalCandidate:
    name: str
    layers: list[int]
    source_candidate_name: str = ""
    calibration_rtd: float | None = None


def remote_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def known_multi_layer_candidates() -> dict[str, list[int]]:
    return {
        "reverse_2": [24, 25],
        "reverse_3": [23, 24, 25],
        "reverse_5": [21, 22, 23, 24, 25],
        "reverse_6": [20, 21, 22, 23, 24, 25],
        "bi_2": [23, 24],
        "bi_3": [2, 23, 24],
        "bi_5": [2, 20, 21, 23, 24],
        "bi_6": [2, 11, 20, 21, 23, 24],
        "iterative_proxy_k2": [24, 25],
        "iterative_proxy_k3": [1, 24, 25],
        "iterative_proxy_k5": [1, 21, 22, 24, 25],
        "iterative_proxy_k6": [1, 21, 22, 23, 24, 25],
    }


def layer_csv(layers: list[int]) -> str:
    return ",".join(str(layer) for layer in sorted(layers))


def build_score_command(
    *,
    python_bin: str,
    root: Path,
    model_path: Path,
    candidate_name: str,
    layers: list[int],
    trace_paths: list[Path],
    output_dir: Path,
    top_k: int,
    max_seq_tokens: int,
    force: bool = False,
) -> list[str]:
    command = [
        python_bin,
        remote_path(root / "experiments" / "day12_rtp_gate" / "tools" / "score_rtd_candidates.py"),
        "--root",
        remote_path(root),
        "--model-path",
        remote_path(model_path),
        "--candidate-name",
        candidate_name,
        "--runtime-skip-layers",
        layer_csv(layers),
        "--output-dir",
        remote_path(output_dir),
        "--top-k",
        str(top_k),
        "--max-seq-tokens",
        str(max_seq_tokens),
    ]
    for trace_path in trace_paths:
        command.extend(["--trace-jsonl", remote_path(trace_path)])
    if force:
        command.append("--force")
    return command


def build_generation_command(
    *,
    python_bin: str,
    root: Path,
    model_path: Path,
    candidate: RawEvalCandidate,
    task: str,
    samples: int,
    output_dir: Path,
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[str]:
    return [
        python_bin,
        remote_path(root / "experiments" / "day9_small_model_repair_completion" / "tools" / "run_day9_generation_eval.py"),
        "--run-id",
        candidate.name,
        "--model-path",
        remote_path(model_path),
        "--task",
        task,
        "--samples",
        str(samples),
        "--output-dir",
        remote_path(output_dir),
        "--job-group",
        "day12_rtp_gate_raw_eval",
        "--runtime-skip-layers",
        layer_csv(candidate.layers),
        "--repo-root",
        remote_path(root / "code" / "on-the-limits-of-layer-pruning"),
        "--max-input-tokens",
        str(max_input_tokens),
        "--max-new-tokens",
        str(max_new_tokens),
    ]


def build_classification_command(
    *,
    python_bin: str,
    root: Path,
    model_path: Path,
    candidate: RawEvalCandidate,
    task: str,
    samples: int,
    output_dir: Path,
    max_input_tokens: int,
) -> list[str]:
    return [
        python_bin,
        remote_path(root / "experiments" / "day12_rtp_gate" / "tools" / "run_day12_classification_eval.py"),
        "--run-id",
        candidate.name,
        "--model-path",
        remote_path(model_path),
        "--task",
        task,
        "--samples",
        str(samples),
        "--output-dir",
        remote_path(output_dir),
        "--runtime-skip-layers",
        layer_csv(candidate.layers),
        "--max-input-tokens",
        str(max_input_tokens),
    ]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _calibration_rtd(payload: dict) -> float | None:
    value = (payload.get("by_partition", {}).get("calibration") or {}).get("rtd")
    if value is None:
        value = (payload.get("overall") or {}).get("rtd")
    return None if value is None else float(value)


def select_risky_k5(score_dir: Path) -> RawEvalCandidate:
    best: tuple[float, str, list[int]] | None = None
    for path in sorted(score_dir.glob("*.json")):
        payload = _load_json(path)
        if payload.get("status") != "done":
            continue
        layers = [int(layer) for layer in payload.get("runtime_skip_layers") or []]
        if len(layers) != 5:
            continue
        rtd = _calibration_rtd(payload)
        if rtd is None:
            continue
        name = str(payload.get("candidate_name") or path.stem)
        if best is None or rtd > best[0]:
            best = (rtd, name, sorted(layers))
    if best is None:
        raise FileNotFoundError(f"No done k=5 RTD score files with calibration RTD found in {score_dir}")
    return RawEvalCandidate(
        name="risky_k5",
        layers=best[2],
        source_candidate_name=best[1],
        calibration_rtd=best[0],
    )


def selected_layers_csv(path: Path, label: str) -> list[int]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("label") == label:
                raw = row.get("layers") or ""
                return [int(x.strip()) for x in raw.split(",") if x.strip()]
    raise KeyError(f"Selection label not found in {path}: {label}")


def raw_eval_candidates(score_dir: Path, selection_dir: Path) -> list[RawEvalCandidate]:
    return [
        RawEvalCandidate("dense_base", []),
        RawEvalCandidate("reverse_3", known_multi_layer_candidates()["reverse_3"]),
        RawEvalCandidate("iterative_proxy_k3", known_multi_layer_candidates()["iterative_proxy_k3"]),
        RawEvalCandidate(
            "rtp_gate_pure_k3",
            selected_layers_csv(selection_dir / "rtp_gate_pure_selected.csv", "rtp_gate_pure_k3"),
        ),
        RawEvalCandidate(
            "rtp_gate_structure_k3",
            selected_layers_csv(selection_dir / "rtp_gate_structure_selected.csv", "rtp_gate_structure_k3"),
        ),
        select_risky_k5(score_dir),
    ]
