#!/usr/bin/env python
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys

ROOT = Path('/root/hs/paper2_layer_pruning')
PYTHON_BIN = str(ROOT / 'envs/rtp-gate-2080ti/bin/python')
MODEL_PATH = Path(os.environ.get('MODEL_PATH', ROOT / 'cache/huggingface/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8'))
LOG_DIR = ROOT / 'logs/day12_rtp_gate'
RAW_GEN = ROOT / 'results/day12_rtp_gate/raw_eval/generation'
RAW_CLS = ROOT / 'results/day12_rtp_gate/raw_eval/classification'
LOG_DIR.mkdir(parents=True, exist_ok=True)
RAW_GEN.mkdir(parents=True, exist_ok=True)
RAW_CLS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from experiments.day12_rtp_gate.tools.lab_root_rtp_gate_jobs import (  # noqa: E402
    CLASSIFICATION_TASKS,
    GENERATION_TASKS,
    RawEvalCandidate,
    build_classification_command,
    build_generation_command,
    known_multi_layer_candidates,
)

CANDIDATES = [
    RawEvalCandidate('dense_base', []),
    RawEvalCandidate('reverse_3', known_multi_layer_candidates()['reverse_3']),
    RawEvalCandidate('iterative_proxy_k3', known_multi_layer_candidates()['iterative_proxy_k3']),
]
GPUS = ['2', '3', '4', '5']


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('status') == 'done'
    except Exception:
        return False


def jobs() -> list[tuple[str, list[str], Path, str]]:
    out = []
    for candidate in CANDIDATES:
        for task in GENERATION_TASKS:
            output = RAW_GEN / f'{candidate.name}_{task}_500.json'
            command = build_generation_command(
                python_bin=PYTHON_BIN,
                root=ROOT,
                model_path=MODEL_PATH,
                candidate=candidate,
                task=task,
                samples=500,
                output_dir=RAW_GEN,
                max_input_tokens=2048,
                max_new_tokens=512,
            )
            out.append((f'prefill_gen_{candidate.name}_{task}', command, output, 'generation'))
        for task in CLASSIFICATION_TASKS:
            output = RAW_CLS / f'{candidate.name}_{task}_200.json'
            command = build_classification_command(
                python_bin=PYTHON_BIN,
                root=ROOT,
                model_path=MODEL_PATH,
                candidate=candidate,
                task=task,
                samples=200,
                output_dir=RAW_CLS,
                max_input_tokens=2048,
            )
            out.append((f'prefill_cls_{candidate.name}_{task}', command, output, 'classification'))
    return out


def run_one(gpu: str, spec: tuple[str, list[str], Path, str]) -> dict:
    name, command, output, kind = spec
    log_path = LOG_DIR / f'{name}.log'
    status_path = LOG_DIR / f'{name}.status.json'
    if is_done(output):
        payload = {'status': 'skipped_done', 'name': name, 'output': str(output), 'finished_at_utc': now()}
        status_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        return payload
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu
    env['TOKENIZERS_PARALLELISM'] = 'false'
    payload = {'status': 'running', 'name': name, 'gpu': gpu, 'kind': kind, 'command': command, 'started_at_utc': now(), 'output': str(output)}
    status_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    with log_path.open('w', encoding='utf-8') as log:
        log.write(f'[{now()}] START {name} gpu={gpu}\n')
        log.write(' '.join(command) + '\n')
        log.flush()
        result = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f'[{now()}] END {name} return_code={result.returncode}\n')
    payload.update({'status': 'done' if result.returncode == 0 else 'failed', 'return_code': result.returncode, 'finished_at_utc': now(), 'log': str(log_path)})
    status_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return payload


def main() -> int:
    all_jobs = jobs()
    manifest = {
        'status': 'running',
        'created_at_utc': now(),
        'gpus': GPUS,
        'candidates': [candidate.__dict__ for candidate in CANDIDATES],
        'job_count': len(all_jobs),
    }
    manifest_path = ROOT / 'results/day12_rtp_gate/raw_eval/prefill_fixed_raw_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    failures = []
    gpu_queue: Queue[str] = Queue()
    for gpu in GPUS:
        gpu_queue.put(gpu)

    def run_with_available_gpu(spec: tuple[str, list[str], Path, str]) -> dict:
        gpu = gpu_queue.get()
        try:
            return run_one(gpu, spec)
        finally:
            gpu_queue.put(gpu)

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = {pool.submit(run_with_available_gpu, spec): spec[0] for spec in all_jobs}
        for future in as_completed(futures):
            result = future.result()
            if result.get('status') == 'failed':
                failures.append(result)
    manifest.update({'status': 'done' if not failures else 'partial_failed', 'finished_at_utc': now(), 'failures': failures})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return 1 if failures else 0

if __name__ == '__main__':
    raise SystemExit(main())
