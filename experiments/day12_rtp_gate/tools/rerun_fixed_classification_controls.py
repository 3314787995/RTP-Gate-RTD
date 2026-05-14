#!/usr/bin/env python
"""Rerun Day12 classification controls after runtime-skip evaluator fixes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys

ROOT = Path("/root/hs/paper2_layer_pruning")
DEFAULT_MODEL = (
    ROOT
    / "cache/huggingface/models--google--gemma-2-2b-it/"
    / "snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_gpus(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def is_done_with_runtime_skip(path: Path, expected_layers: list[int]) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("status") != "done":
        return False
    runtime_skip = payload.get("runtime_skip")
    if not isinstance(runtime_skip, dict):
        return False
    return list(runtime_skip.get("skip_layers") or []) == sorted(expected_layers)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--gpus", default="0,1,3,4,5")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "results/day12_rtp_gate/raw_eval/classification_runtime_skip_fixed_20260514"
    log_dir = Path(args.log_dir) if args.log_dir else root / "logs/day12_rtp_gate/classification_runtime_skip_fixed_20260514"
    score_dir = root / "results/day12_rtp_gate/rtd_scores"
    selection_dir = root / "results/day12_rtp_gate/rtp_gate_selection"
    python_bin = str(root / "envs/rtp-gate-2080ti/bin/python")
    model_path = Path(args.model_path)
    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("At least one GPU id is required.")

    sys.path.insert(0, str(root))
    from experiments.day12_rtp_gate.tools.lab_root_rtp_gate_jobs import (  # noqa: E402
        CLASSIFICATION_TASKS,
        build_classification_command,
        raw_eval_candidates,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    candidates = raw_eval_candidates(score_dir, selection_dir)
    jobs = []
    for candidate in candidates:
        for task in CLASSIFICATION_TASKS:
            output = output_dir / f"{candidate.name}_{task}_{args.samples}.json"
            command = build_classification_command(
                python_bin=python_bin,
                root=root,
                model_path=model_path,
                candidate=candidate,
                task=task,
                samples=args.samples,
                output_dir=output_dir,
                max_input_tokens=args.max_input_tokens,
            )
            jobs.append((candidate.name, candidate.layers, task, output, command))

    manifest_path = output_dir.parent / "classification_runtime_skip_fixed_20260514_manifest.json"
    manifest = {
        "status": "running",
        "created_at_utc": now(),
        "output_dir": str(output_dir),
        "log_dir": str(log_dir),
        "gpus": gpus,
        "samples": args.samples,
        "candidates": [candidate.__dict__ for candidate in candidates],
        "tasks": list(CLASSIFICATION_TASKS),
        "job_count": len(jobs),
        "note": "Rerun after fixing Day12 classification runtime-skip wrapper.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def run_one(spec: tuple[str, list[int], str, Path, list[str]]) -> dict:
        candidate_name, layers, task, output, command = spec
        name = f"fixed_cls_{candidate_name}_{task}"
        status_path = log_dir / f"{name}.status.json"
        log_path = log_dir / f"{name}.log"
        if is_done_with_runtime_skip(output, layers):
            payload = {
                "status": "skipped_done",
                "name": name,
                "output": str(output),
                "expected_layers": sorted(layers),
                "finished_at_utc": now(),
            }
            status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return payload
        gpu = gpu_queue.get()
        try:
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
            payload = {
                "status": "running",
                "name": name,
                "gpu": gpu,
                "task": task,
                "candidate": candidate_name,
                "expected_layers": sorted(layers),
                "command": command,
                "output": str(output),
                "started_at_utc": now(),
            }
            status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"[{now()}] START {name} gpu={gpu}\n")
                log.write(" ".join(command) + "\n")
                log.flush()
                result = subprocess.run(command, cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT)
                log.write(f"[{now()}] END {name} return_code={result.returncode}\n")
            payload.update(
                {
                    "status": "done" if result.returncode == 0 else "failed",
                    "return_code": result.returncode,
                    "finished_at_utc": now(),
                    "log": str(log_path),
                }
            )
            status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return payload
        finally:
            gpu_queue.put(gpu)

    failures = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result.get("status") == "failed":
                failures.append(result)

    done = 0
    bad_metadata = []
    for candidate in candidates:
        for task in CLASSIFICATION_TASKS:
            path = output_dir / f"{candidate.name}_{task}_{args.samples}.json"
            if is_done_with_runtime_skip(path, candidate.layers):
                done += 1
            else:
                bad_metadata.append(str(path))

    manifest.update(
        {
            "status": "failed" if failures or bad_metadata else "done",
            "finished_at_utc": now(),
            "done_count": done,
            "failures": failures,
            "bad_metadata": bad_metadata,
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if failures or bad_metadata else 0


if __name__ == "__main__":
    raise SystemExit(main())
