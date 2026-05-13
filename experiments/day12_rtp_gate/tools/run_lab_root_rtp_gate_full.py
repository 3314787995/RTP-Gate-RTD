#!/usr/bin/env python
"""Run a resumable RTP-Gate rerun on lab-root."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from lab_root_rtp_gate_jobs import (
    CLASSIFICATION_TASKS,
    GENERATION_TASKS,
    RawEvalCandidate,
    build_classification_command,
    build_generation_command,
    build_score_command,
    known_multi_layer_candidates,
    layer_csv,
    raw_eval_candidates,
    remote_path,
    selected_layers_csv,
    select_risky_k5,
)


DEFAULT_ROOT = Path("/root/hs/paper2_layer_pruning")
DEFAULT_MODEL_SNAPSHOT = (
    "cache/huggingface/models--google--gemma-2-2b-it/"
    "snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_gpu_ids(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Runner:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.root)
        self.python_bin = str(args.python_bin).replace("\\", "/")
        self.model_path = Path(args.model_path) if args.model_path else self.root / DEFAULT_MODEL_SNAPSHOT
        self.results = self.root / "results" / "day12_rtp_gate"
        self.reports = self.root / "reports" / "day12_rtp_gate"
        self.logs = self.root / "logs" / "day12_rtp_gate"
        self.traces = self.results / "traces"
        self.score_dir = self.results / "rtd_scores"
        self.selection_dir = self.results / "rtp_gate_selection"
        self.raw_generation_dir = self.results / "raw_eval" / "generation"
        self.raw_classification_dir = self.results / "raw_eval" / "classification"
        self.saved_dir = self.results / "saved_model_consistency"
        self.model_dir = self.root / "models" / "day12_rtp_gate"
        self.gpu_ids = parse_gpu_ids(args.gpu_ids)

    def env(self, gpu: str | None = None) -> dict:
        env = os.environ.copy()
        env["ROOT"] = remote_path(self.root)
        env["HF_HOME"] = remote_path(self.root / "cache" / "huggingface")
        env["HF_DATASETS_CACHE"] = remote_path(self.root / "cache" / "huggingface" / "datasets")
        env["HF_ENDPOINT"] = env.get("HF_ENDPOINT", "https://hf-mirror.com")
        env["MODEL_PATH"] = remote_path(self.model_path)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return env

    def ensure_dirs(self) -> None:
        for path in [
            self.traces,
            self.score_dir,
            self.selection_dir,
            self.raw_generation_dir,
            self.raw_classification_dir,
            self.saved_dir,
            self.reports,
            self.logs,
            self.model_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, command: list[str], gpu: str | None = None) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        log_path = self.logs / f"{name}.log"
        metadata_path = self.logs / f"{name}.status.json"
        payload = {"status": "running", "name": name, "gpu": gpu, "command": command, "started_at_utc": now()}
        write_json(metadata_path, payload)
        if self.args.dry_run:
            print(json.dumps({"dry_run": True, "name": name, "gpu": gpu, "command": command}, ensure_ascii=False))
            payload.update({"status": "dry_run", "finished_at_utc": now()})
            write_json(metadata_path, payload)
            return
        with log_path.open("ab") as log:
            log.write((f"\n[{now()}] START {name} gpu={gpu}\n").encode("utf-8"))
            log.write((" ".join(command) + "\n").encode("utf-8"))
            result = subprocess.run(command, env=self.env(gpu), stdout=log, stderr=subprocess.STDOUT)
            log.write((f"[{now()}] END {name} return_code={result.returncode}\n").encode("utf-8"))
        payload.update({"status": "done" if result.returncode == 0 else "failed", "return_code": result.returncode, "finished_at_utc": now(), "log": remote_path(log_path)})
        write_json(metadata_path, payload)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)

    def run_parallel(self, jobs: list[tuple[str, list[str]]]) -> None:
        if not jobs:
            return
        workers = max(1, min(len(self.gpu_ids), self.args.max_workers, len(jobs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for idx, (name, command) in enumerate(jobs):
                gpu = self.gpu_ids[idx % len(self.gpu_ids)]
                futures.append(pool.submit(self.run, name, command, gpu))
            for future in as_completed(futures):
                future.result()

    def trace_paths(self) -> list[Path]:
        return [
            self.traces / "gsm8k_dense_traces_calibration.jsonl",
            self.traces / "gsm8k_dense_traces_holdout.jsonl",
        ]

    def stage_smoke(self) -> None:
        collect = [
            self.python_bin,
            remote_path(self.root / "experiments" / "day12_rtp_gate" / "tools" / "collect_dense_traces.py"),
            "--root",
            remote_path(self.root),
            "--model-path",
            remote_path(self.model_path),
            "--baseline-json",
            remote_path(self.root / "results" / "day5_technical_gap_closure" / "gemma_base_full_eval_gsm8k_1319.json"),
            "--output-dir",
            remote_path(self.traces),
            "--partition-mode",
            "smoke",
            "--top-k",
            str(self.args.top_k),
            "--max-seq-tokens",
            str(self.args.max_seq_tokens),
        ]
        self.run("smoke_collect_dense_traces", collect, self.gpu_ids[0])
        score = build_score_command(
            python_bin=self.python_bin,
            root=self.root,
            model_path=self.model_path,
            candidate_name="smoke_drop_layer_24",
            layers=[24],
            trace_paths=[self.traces / "gsm8k_dense_traces_smoke.jsonl"],
            output_dir=self.score_dir,
            top_k=self.args.top_k,
            max_seq_tokens=self.args.max_seq_tokens,
            force=self.args.force,
        )
        self.run("smoke_score_drop_layer_24", score, self.gpu_ids[0])
        self.stage_report()

    def stage_full_traces(self) -> None:
        command = [
            self.python_bin,
            remote_path(self.root / "experiments" / "day12_rtp_gate" / "tools" / "collect_dense_traces.py"),
            "--root",
            remote_path(self.root),
            "--model-path",
            remote_path(self.model_path),
            "--baseline-json",
            remote_path(self.root / "results" / "day5_technical_gap_closure" / "gemma_base_full_eval_gsm8k_1319.json"),
            "--output-dir",
            remote_path(self.traces),
            "--partition-mode",
            "full",
            "--top-k",
            str(self.args.top_k),
            "--max-seq-tokens",
            str(self.args.max_seq_tokens),
        ]
        self.run("full_collect_dense_traces", command, self.gpu_ids[0])

    def stage_single(self) -> None:
        jobs = []
        for layer in range(26):
            jobs.append(
                (
                    f"score_single_layer_{layer}",
                    build_score_command(
                        python_bin=self.python_bin,
                        root=self.root,
                        model_path=self.model_path,
                        candidate_name=f"single_layer_{layer}",
                        layers=[layer],
                        trace_paths=self.trace_paths(),
                        output_dir=self.score_dir,
                        top_k=self.args.top_k,
                        max_seq_tokens=self.args.max_seq_tokens,
                        force=self.args.force,
                    ),
                )
            )
        self.run_parallel(jobs)
        self.stage_report()

    def stage_multi(self) -> None:
        jobs = []
        for name, layers in known_multi_layer_candidates().items():
            jobs.append(
                (
                    f"score_{name}",
                    build_score_command(
                        python_bin=self.python_bin,
                        root=self.root,
                        model_path=self.model_path,
                        candidate_name=name,
                        layers=layers,
                        trace_paths=self.trace_paths(),
                        output_dir=self.score_dir,
                        top_k=self.args.top_k,
                        max_seq_tokens=self.args.max_seq_tokens,
                        force=self.args.force,
                    ),
                )
            )
        self.run_parallel(jobs)
        self.stage_report()

    def stage_select(self) -> None:
        jobs = []
        for prefix, extra in [
            ("rtp_gate_pure", []),
            ("rtp_gate_structure", ["--structure-aware"]),
        ]:
            command = [
                self.python_bin,
                remote_path(self.root / "experiments" / "day12_rtp_gate" / "tools" / "select_rtp_gate_layers.py"),
                "--root",
                remote_path(self.root),
                "--model-path",
                remote_path(self.model_path),
                "--candidate-prefix",
                prefix,
                "--target-k",
                "2,3,5",
                "--selection-partition",
                "calibration",
                "--output-dir",
                remote_path(self.score_dir),
                "--selection-dir",
                remote_path(self.selection_dir),
                "--top-k",
                str(self.args.top_k),
                "--score-script",
                remote_path(self.root / "experiments" / "day12_rtp_gate" / "tools" / "score_rtd_candidates.py"),
            ]
            for trace_path in self.trace_paths():
                command.extend(["--trace-jsonl", remote_path(trace_path)])
            command.extend(extra)
            if self.args.force:
                command.append("--force-scores")
            jobs.append((f"select_{prefix}", command))
        self.run_parallel(jobs)
        self.stage_report()

    def stage_consistency(self) -> None:
        structure_layers = selected_layers_csv(self.selection_dir / "rtp_gate_structure_selected.csv", "rtp_gate_structure_k3")
        specs = [
            ("saved_single_layer_24", [24]),
            ("saved_rtp_gate_structure_k3", structure_layers),
        ]
        for label, layers in specs:
            out_dir = self.model_dir / label
            if not (out_dir / "config.json").exists() or self.args.force:
                command = [
                    self.python_bin,
                    remote_path(self.root / "experiments" / "day2_gemma_core_eval" / "tools" / "make_gemma_pruned_model.py"),
                    "--model-name",
                    remote_path(self.model_path),
                    "--output-dir",
                    remote_path(out_dir),
                    "--label",
                    label,
                    "--layers-to-remove",
                    layer_csv(layers),
                    "--overwrite",
                ]
                self.run(f"make_model_{label}", command, self.gpu_ids[0])
            score = build_score_command(
                python_bin=self.python_bin,
                root=self.root,
                model_path=out_dir,
                candidate_name=label,
                layers=[],
                trace_paths=self.trace_paths(),
                output_dir=self.saved_dir,
                top_k=self.args.top_k,
                max_seq_tokens=self.args.max_seq_tokens,
                force=self.args.force,
            )
            self.run(f"score_{label}", score, self.gpu_ids[0])

    def stage_raw(self) -> None:
        candidates = raw_eval_candidates(self.score_dir, self.selection_dir)
        jobs = []
        for candidate in candidates:
            for task in GENERATION_TASKS:
                jobs.append(
                    (
                        f"raw_gen_{candidate.name}_{task}",
                        build_generation_command(
                            python_bin=self.python_bin,
                            root=self.root,
                            model_path=self.model_path,
                            candidate=candidate,
                            task=task,
                            samples=500,
                            output_dir=self.raw_generation_dir,
                            max_input_tokens=self.args.max_seq_tokens,
                            max_new_tokens=512,
                        ),
                    )
                )
            for task in CLASSIFICATION_TASKS:
                jobs.append(
                    (
                        f"raw_cls_{candidate.name}_{task}",
                        build_classification_command(
                            python_bin=self.python_bin,
                            root=self.root,
                            model_path=self.model_path,
                            candidate=candidate,
                            task=task,
                            samples=200,
                            output_dir=self.raw_classification_dir,
                            max_input_tokens=self.args.max_seq_tokens,
                        ),
                    )
                )
        manifest = {
            "status": "planned",
            "created_at_utc": now(),
            "candidates": [candidate.__dict__ for candidate in candidates],
            "generation_tasks": list(GENERATION_TASKS),
            "classification_tasks": list(CLASSIFICATION_TASKS),
        }
        write_json(self.results / "raw_eval" / "raw_eval_manifest.json", manifest)
        self.run_parallel(jobs)

    def stage_report(self) -> None:
        command = [
            self.python_bin,
            remote_path(self.root / "experiments" / "day12_rtp_gate" / "tools" / "summarize_rtp_gate.py"),
            "--root",
            remote_path(self.root),
            "--score-dir",
            remote_path(self.score_dir),
            "--report-dir",
            remote_path(self.reports),
        ]
        self.run("summarize_rtp_gate", command, self.gpu_ids[0] if self.gpu_ids else None)
        self.write_lab_root_report()

    def write_lab_root_report(self) -> None:
        summary_path = self.reports / "day12_rtp_gate_summary.json"
        status_path = self.reports / "day12_rtp_gate_status.md"
        if not summary_path.exists():
            return
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rtd_metric = next((row for row in summary.get("component_metrics", []) if row.get("score_field") == "rtd"), {})
        lines = [
            "# Lab-root RTP-Gate Rerun Report",
            "",
            f"Generated: {now()}",
            "",
            "## Status",
            "",
            f"- scored candidates: {summary.get('rtd_rows')}",
            f"- single-layer rows with retention: {summary.get('single_layer_with_retention')}",
            f"- RTD Spearman abs: {rtd_metric.get('spearman_abs', '')}",
            f"- RTD risky AUROC: {rtd_metric.get('auroc_risky', '')}",
            "",
            "## Lab-root Rules",
            "",
            "- Formal lab-root outputs are under `/root/hs/paper2_layer_pruning`.",
            "- Old lab-cike RTP-Gate results are reference-only and are not merged into these metrics.",
            "- RTP-Gate is reported as a pruning-risk diagnostic/gate, not as a universal optimal pruning algorithm.",
            "",
            "## Files",
            "",
            f"- Summary: `{remote_path(summary_path)}`",
            f"- Status: `{remote_path(status_path)}`",
            f"- Scores: `{remote_path(self.reports / 'rtd_scores.csv')}`",
            f"- Raw eval manifest: `{remote_path(self.results / 'raw_eval' / 'raw_eval_manifest.json')}`",
        ]
        (self.reports / "lab_root_rtp_gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run_stages(self) -> None:
        self.ensure_dirs()
        stages = {
            "smoke": self.stage_smoke,
            "full_traces": self.stage_full_traces,
            "single": self.stage_single,
            "multi": self.stage_multi,
            "select": self.stage_select,
            "consistency": self.stage_consistency,
            "raw": self.stage_raw,
            "report": self.stage_report,
        }
        if self.args.stage == "all":
            order = ["smoke", "full_traces", "single", "multi", "select", "consistency", "raw", "report"]
        else:
            order = [self.args.stage]
        for stage in order:
            stages[stage]()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=remote_path(DEFAULT_ROOT))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--python-bin", default=remote_path(DEFAULT_ROOT / "envs" / "rtp-gate-2080ti" / "bin" / "python"))
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--stage", choices=["all", "smoke", "full_traces", "single", "multi", "select", "consistency", "raw", "report"], default="all")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-seq-tokens", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    Runner(args).run_stages()


if __name__ == "__main__":
    main()
