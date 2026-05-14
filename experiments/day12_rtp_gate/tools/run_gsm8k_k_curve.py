#!/usr/bin/env python
"""Run GSM8K k=2/k=5 raw eval jobs for the RTP-Gate k curve."""
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

try:
    from lab_root_rtp_gate_jobs import RawEvalCandidate, build_generation_command
except ModuleNotFoundError:
    from .lab_root_rtp_gate_jobs import RawEvalCandidate, build_generation_command


ROOT = Path("/root/hs/paper2_layer_pruning")
MODEL_PATH = ROOT / "cache/huggingface/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_gpus(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def default_candidates() -> list[RawEvalCandidate]:
    return [
        RawEvalCandidate("rtp_gate_k2", [1, 24]),
        RawEvalCandidate("rtp_gate_k5", [1, 9, 10, 19, 24]),
        RawEvalCandidate("reverse_2", [24, 25]),
        RawEvalCandidate("reverse_5", [21, 22, 23, 24, 25]),
        RawEvalCandidate("iterative_proxy_k2", [24, 25]),
        RawEvalCandidate("iterative_proxy_k5", [1, 21, 22, 24, 25]),
    ]


def done_with_expected_layers(path: Path, expected_layers: list[int]) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("status") != "done":
        return False
    runtime_skip = payload.get("runtime_skip") or {}
    return list(runtime_skip.get("skip_layers") or []) == sorted(expected_layers)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    root = Path(args.root)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir) if args.output_dir else root / "results/day12_rtp_gate/raw_eval/generation_k_curve_20260514"
    log_dir = Path(args.log_dir) if args.log_dir else root / "logs/day12_rtp_gate/generation_k_curve_20260514"
    python_bin = str(root / "envs/rtp-gate-2080ti/bin/python")
    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("At least one GPU id is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    specs = []
    for candidate in default_candidates():
        output = output_dir / f"{candidate.name}_gsm8k_{args.samples}.json"
        command = build_generation_command(
            python_bin=python_bin,
            root=root,
            model_path=model_path,
            candidate=candidate,
            task="gsm8k",
            samples=args.samples,
            output_dir=output_dir,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        specs.append((candidate, output, command))

    manifest_path = output_dir.parent / f"generation_k_curve_20260514_manifest_{args.samples}.json"
    write_json(
        manifest_path,
        {
            "status": "running",
            "created_at_utc": now(),
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "samples": args.samples,
            "gpus": gpus,
            "candidates": [candidate.__dict__ for candidate, _, _ in specs],
        },
    )

    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    def run_one(spec: tuple[RawEvalCandidate, Path, list[str]]) -> dict:
        candidate, output, command = spec
        name = f"kcurve_{candidate.name}_gsm8k_{args.samples}"
        status_path = log_dir / f"{name}.status.json"
        log_path = log_dir / f"{name}.log"
        if done_with_expected_layers(output, candidate.layers):
            payload = {"status": "skipped_done", "name": name, "output": str(output), "finished_at_utc": now()}
            write_json(status_path, payload)
            return payload
        gpu = gpu_queue.get()
        try:
            payload = {
                "status": "running",
                "name": name,
                "gpu": gpu,
                "candidate": candidate.__dict__,
                "command": command,
                "output": str(output),
                "started_at_utc": now(),
            }
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
            return payload
        finally:
            gpu_queue.put(gpu)

    failures = []
    with ThreadPoolExecutor(max_workers=min(len(gpus), len(specs))) as pool:
        for future in as_completed([pool.submit(run_one, spec) for spec in specs]):
            result = future.result()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result.get("status") == "failed":
                failures.append(result)

    done_count = 0
    for candidate, output, _ in specs:
        if done_with_expected_layers(output, candidate.layers):
            done_count += 1
    final = json.loads(manifest_path.read_text(encoding="utf-8"))
    final.update(
        {
            "status": "done" if done_count == len(specs) and not failures else "failed",
            "finished_at_utc": now(),
            "done_count": done_count,
            "job_count": len(specs),
            "failures": failures,
        }
    )
    write_json(manifest_path, final)
    return 0 if final["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
