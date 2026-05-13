#!/usr/bin/env python
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_layers(raw):
    if not raw.strip():
        return []
    return sorted({int(x.strip()) for x in raw.split(",") if x.strip()})


def require_inside_root(path, root):
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Refusing to write outside ROOT: {resolved} not under {root_resolved}")
    return resolved


def clear_generation_cache_settings(model):
    model.config.use_cache = False
    if hasattr(model.config, "cache_implementation"):
        model.config.cache_implementation = None
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False
        if hasattr(model.generation_config, "cache_implementation"):
            model.generation_config.cache_implementation = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--layers-to-remove", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = os.environ.get("ROOT")
    if not root:
        raise RuntimeError("ROOT is not set. Source env.sh before running.")

    output_dir = require_inside_root(args.output_dir, root)
    layers_to_remove = parse_layers(args.layers_to_remove)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    total_layers = int(config.num_hidden_layers)
    if not layers_to_remove:
        raise ValueError("This script is for pruned models; layers-to-remove cannot be empty.")
    bad_layers = [i for i in layers_to_remove if i < 0 or i >= total_layers]
    if bad_layers:
        raise ValueError(f"Invalid layers for {total_layers}-layer model: {bad_layers}")

    print(json.dumps({
        "event": "load_start",
        "model_name": args.model_name,
        "label": args.label,
        "total_layers": total_layers,
        "layers_to_remove": layers_to_remove,
        "output_dir": str(output_dir),
    }, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    clear_generation_cache_settings(model)

    layers = model.model.layers
    kept_layers = [layer for idx, layer in enumerate(layers) if idx not in set(layers_to_remove)]
    model.model.layers = torch.nn.ModuleList(kept_layers)
    model.config.num_hidden_layers = len(kept_layers)
    if hasattr(model.model, "config"):
        model.model.config.num_hidden_layers = len(kept_layers)

    for new_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_idx
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output_dir)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(output_dir)

    metadata = {
        "label": args.label,
        "source_model": args.model_name,
        "total_layers_before": total_layers,
        "layers_to_remove": layers_to_remove,
        "total_layers_after": len(kept_layers),
        "prune_ratio": len(layers_to_remove) / total_layers,
        "layer_indexing": "0-indexed",
        "method": "direct ModuleList deletion; no mergekit",
        "use_cache": False,
    }
    with open(output_dir / "prune_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    reloaded = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    print(json.dumps({
        "event": "save_done",
        "output_dir": str(output_dir),
        "expected_layers": len(kept_layers),
        "reloaded_layers": int(reloaded.num_hidden_layers),
        "metadata": metadata,
    }, indent=2))

    if int(reloaded.num_hidden_layers) != len(kept_layers):
        raise RuntimeError("Reloaded config layer count does not match pruned layer count.")


if __name__ == "__main__":
    main()
