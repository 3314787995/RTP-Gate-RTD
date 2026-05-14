#!/usr/bin/env python
"""Day12 classification evaluator with runtime layer skipping.

The implementation reuses the Day9 classification evaluator and injects the
same runtime-skip behavior used by the generation evaluator.
"""
from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys


def parse_layers(raw: str) -> list[int]:
    if raw is None or not str(raw).strip():
        return []
    return sorted({int(x.strip()) for x in str(raw).split(",") if x.strip()})


def strip_runtime_skip_arg(argv: list[str]) -> tuple[list[str], list[int]]:
    cleaned = []
    layers = []
    i = 0
    while i < len(argv):
        if argv[i] == "--runtime-skip-layers":
            if i + 1 >= len(argv):
                raise SystemExit("--runtime-skip-layers requires a value")
            layers = parse_layers(argv[i + 1])
            i += 2
            continue
        cleaned.append(argv[i])
        i += 1
    return cleaned, layers


def arg_value(argv: list[str], name: str, default: str | None = None) -> str | None:
    if name not in argv:
        return default
    idx = argv.index(name)
    if idx + 1 >= len(argv):
        return default
    return argv[idx + 1]


def runtime_skip_layers(model, skip_layers: list[int]) -> dict:
    if not skip_layers:
        return {"enabled": False, "skip_layers": [], "kept_layers": list(range(len(model.model.layers)))}
    original_layers = list(model.model.layers)
    bad_layers = [layer for layer in skip_layers if layer < 0 or layer >= len(original_layers)]
    if bad_layers:
        raise ValueError(f"Invalid layers for {len(original_layers)}-layer model: {bad_layers}")
    kept = [(idx, layer) for idx, layer in enumerate(original_layers) if idx not in set(skip_layers)]
    changed_sliding = []
    for new_idx, (old_idx, layer) in enumerate(kept):
        old_sliding = bool(getattr(layer, "is_sliding", old_idx % 2 == 0))
        new_sliding_if_resaved = new_idx % 2 == 0
        if old_sliding != new_sliding_if_resaved:
            changed_sliding.append(
                {
                    "old_layer": old_idx,
                    "new_position": new_idx,
                    "original_is_sliding": old_sliding,
                    "resaved_would_be_sliding": new_sliding_if_resaved,
                }
            )
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_idx
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx
    import torch

    model.model.layers = torch.nn.ModuleList([layer for _, layer in kept])
    model.config.num_hidden_layers = len(kept)
    if hasattr(model.model, "config"):
        model.model.config.num_hidden_layers = len(kept)
    return {
        "enabled": True,
        "skip_layers": skip_layers,
        "kept_layers": [idx for idx, _ in kept],
        "changed_sliding_if_resaved": changed_sliding,
    }


def update_output_metadata(argv: list[str], runtime_skip: dict) -> None:
    output_dir = arg_value(argv, "--output-dir")
    run_id = arg_value(argv, "--run-id")
    task = arg_value(argv, "--task")
    samples = arg_value(argv, "--samples", "200")
    if not output_dir or not run_id or not task:
        return
    out_json = Path(output_dir) / f"{run_id}_{task}_{samples}.json"
    if not out_json.exists():
        return
    data = json.loads(out_json.read_text(encoding="utf-8"))
    data["runtime_skip"] = runtime_skip
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assert_runtime_skip_was_applied(argv: list[str], skip_layers: list[int], runtime_skip: dict) -> None:
    if not skip_layers or runtime_skip:
        return
    output_dir = arg_value(argv, "--output-dir")
    run_id = arg_value(argv, "--run-id")
    task = arg_value(argv, "--task")
    samples = arg_value(argv, "--samples", "200")
    out_json = Path(output_dir) / f"{run_id}_{task}_{samples}.json" if output_dir and run_id and task else None
    existing_runtime_skip = None
    if out_json and out_json.exists():
        existing_runtime_skip = json.loads(out_json.read_text(encoding="utf-8")).get("runtime_skip")
    if existing_runtime_skip and existing_runtime_skip.get("skip_layers") == skip_layers:
        return
    raise RuntimeError(
        "Runtime skip layers were requested, but the Day9 classification evaluator did not load a model. "
        "This usually means it returned an existing output JSON before the Day12 wrapper could apply layer "
        "skipping. Remove the stale output or use a fresh output directory before rerunning."
    )


def main() -> None:
    cleaned_argv, skip_layers = strip_runtime_skip_arg(sys.argv)
    experiments_dir = Path(__file__).resolve().parents[2]
    target = experiments_dir / "day9_small_model_repair_completion" / "tools" / "run_day9_classification_eval.py"
    if not target.exists():
        raise SystemExit(f"Missing Day9 classification evaluator: {target}")

    module_globals = runpy.run_path(str(target), run_name="day9_classification_eval")
    day9_globals = module_globals["main"].__globals__
    original_auto_model = day9_globals["AutoModelForCausalLM"]
    runtime_skip_state: dict = {}

    class RuntimeSkipAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model = original_auto_model.from_pretrained(*args, **kwargs)
            runtime_skip_state.update(runtime_skip_layers(model, skip_layers))
            return model

    day9_globals["AutoModelForCausalLM"] = RuntimeSkipAutoModel
    old_argv = sys.argv
    try:
        sys.argv = cleaned_argv
        module_globals["main"]()
    finally:
        sys.argv = old_argv
    assert_runtime_skip_was_applied(cleaned_argv, skip_layers, runtime_skip_state)
    update_output_metadata(cleaned_argv, runtime_skip_state)


if __name__ == "__main__":
    main()
