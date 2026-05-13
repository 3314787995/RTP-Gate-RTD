import argparse
import os
os.environ["VLLM_USE_V1"] = "0"
import pandas as pd
import sys
import json

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import prompts
from utils.evaluate import eval_math
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model to evaluate (comma-separated for multiple)")
    parser.add_argument("--base_model", type=str, default=None, help="Base model path (required if model_name is a LoRA adapter)")
    parser.add_argument("--dataset_type", type=str, default="gsm8k", help="Type of dataset to evaluate on (e.g., gsm8k, math)")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--prompt", type=str, default="QWEN_INSTRUCT", help="Prompt key from utils.prompts")
    parser.add_argument("--split", type=str, default="train", help="Split to evaluate on (train or test)")
    parser.add_argument("--latency_stats_file", type=str, default="eval_results/latency_stats.json", help="Path to save latency statistics JSON")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use (tensor parallel size)")
    
    args = parser.parse_args()
    
    # Parse prompts
    prompt_keys = [p.strip() for p in args.prompt.split(',')]
    dataset_types = [d.strip() for d in args.dataset_type.split(',') if d.strip()]
    
    # Map dataset to prompt key
    dataset_to_prompt_key = {}
    if len(prompt_keys) == 1:
        # Broadcast single prompt to all datasets
        for ds in dataset_types:
            dataset_to_prompt_key[ds] = prompt_keys[0]
    elif len(prompt_keys) == len(dataset_types):
        # 1-to-1 mapping
        for ds, pk in zip(dataset_types, prompt_keys):
            dataset_to_prompt_key[ds] = pk
    else:
        raise ValueError(f"Number of prompts ({len(prompt_keys)}) must match number of datasets ({len(dataset_types)}) or be 1.")

    print(f"Evaluating models: {args.model_name}")
    print(f"Datasets: {args.dataset_type}")
    print(f"Samples: {args.samples}")
    print(f"Prompt Mapping: {dataset_to_prompt_key}")

    # Run evaluation
    samples = args.samples
    
    model_names = [m.strip() for m in args.model_name.split(',') if m.strip()]
    base_models = [b.strip() for b in args.base_model.split(',') if b.strip()] if args.base_model else []
    
    # Map model to base model
    model_to_base = {}
    if not base_models:
        for m in model_names:
            model_to_base[m] = m
    elif len(base_models) == 1:
        for m in model_names:
            model_to_base[m] = base_models[0]
    elif len(base_models) == len(model_names):
        for m, b in zip(model_names, base_models):
            model_to_base[m] = b
    else:
        raise ValueError(f"Number of base models ({len(base_models)}) must be 1 or match number of models ({len(model_names)}).")

    overall_stats = {}
    results_dir = "eval_results"
    os.makedirs(results_dir, exist_ok=True)
    
    for model_path in model_names:
        if not model_path: continue
        
        # Extract parent folder and checkpoint name for unique output naming
        path_parts = model_path.strip("/").split("/")
        if len(path_parts) >= 2:
            model_name_clean = f"{path_parts[-2]}_{path_parts[-1]}"
        else:
            model_name_clean = path_parts[-1]
            
        print(f"\n{'='*20}\nEvaluating Model: {model_name_clean}\n{'='*20}")
        
        model_stats = {
            "total_latency": 0,
            "total_tokens": 0,
            "datasets": {}
        }
        
        try:
            print("Initializing LLM...")
            
            # Check if this is a LoRA adapter checkpoint
            is_lora_adapter = False
            adapter_path = None
            base_model_path = model_to_base[model_path]
            
            # Check if adapter_config.json exists (indicates LoRA checkpoint)
            if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                is_lora_adapter = True
                adapter_path = model_path
                print(f"Detected LoRA adapter at: {adapter_path}")
                print(f"Using base model: {base_model_path}")
            
            # Initialize LLM with or without LoRA
            if is_lora_adapter:
                llm = LLM(
                    trust_remote_code=True,
                    model=base_model_path,
                    max_model_len=8192,
                    gpu_memory_utilization=0.9,
                    enable_lora=True,
                    max_lora_rank=64,
                    tensor_parallel_size=args.tensor_parallel_size,
                )
                # Create LoRA request for this adapter
                lora_request = LoRARequest(
                    lora_name=model_name_clean,
                    lora_int_id=1,
                    lora_path=adapter_path
                )
            else:
                llm = LLM(
                    trust_remote_code=True,
                    model=model_path,
                    max_model_len=8192,
                    gpu_memory_utilization=0.9,
                    tensor_parallel_size=args.tensor_parallel_size,
                )
            
            for dataset_type in dataset_types:
                print(f"Evaluating on dataset: {dataset_type}")
                
                # Get specific prompt for this dataset
                current_prompt_key = dataset_to_prompt_key[dataset_type]
                try:
                    prompt_template = getattr(prompts, current_prompt_key)
                except AttributeError:
                    print(f"Warning: Prompt key '{current_prompt_key}' not found in utils.prompts. Using key as prompt string.")
                    prompt_template = current_prompt_key
                
                try:
                    data = eval_math(
                        model_name=model_path,
                        dataset_type=dataset_type,
                        samples=samples,
                        prompt=prompt_template,
                        split=args.split,
                        llm=llm,
                        lora_request=lora_request if is_lora_adapter else None
                    )
                    
                    # Accumulate model-level stats
                    model_stats["total_latency"] += data.get("total_latency", 0)
                    model_stats["total_tokens"] += data.get("total_tokens", 0)
                    
                    # Store dataset-specific stats
                    model_stats["datasets"][dataset_type] = {
                        "accuracy": data.get("final_accuracy"),
                        "latency": data.get("total_latency"),
                        "tokens": data.get("total_tokens"),
                        "avg_time_per_token": data.get("avg_time_per_token")
                    }
                    
                    # Construct filename and save JSON
                    # Include prompt key in filename to differentiate
                    filename = f"{model_name_clean}_{dataset_type}_{current_prompt_key}_eval.json"
                    output_path = os.path.join(results_dir, filename)
                    
                    output_data = {
                        "model_name": model_name_clean,
                        "dataset_type": dataset_type,
                        "prompt_template": current_prompt_key,
                        "final_accuracy": data.get("final_accuracy"),
                        "total_latency": data.get("total_latency"),
                        "total_tokens": data.get("total_tokens"),
                        "avg_time_per_token": data.get("avg_time_per_token"),
                        "items": data.get("items", [])
                    }
                    
                    with open(output_path, 'w') as f:
                        json.dump(output_data, f, indent=2)
                        
                    print(f"Results saved to {output_path}")
                    print(f"Final Accuracy for {dataset_type}: {data.get('final_accuracy', 'N/A')}")
                        
                except Exception as e:
                    print(f"Error evaluating on {dataset_type}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Calculate final model-level TPOT
            if model_stats["total_tokens"] > 0:
                model_stats["avg_time_per_token"] = model_stats["total_latency"] / model_stats["total_tokens"]
            else:
                model_stats["avg_time_per_token"] = 0
            
            overall_stats[model_name_clean] = model_stats
            
            # Cleanup LLM after all datasets for this model are done
            import torch
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            del llm
            cleanup_dist_env_and_memory()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to initialize or evaluate model {model_path}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save overall statistics to JSON
    with open(args.latency_stats_file, 'w') as f:
        json.dump(overall_stats, f, indent=4)
    print(f"\nLatency statistics saved to {args.latency_stats_file}")

    # Print summary table
    print(f"\n{'='*30}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'='*30}\n")
    
    summary_data = []
    for model_name, stats in overall_stats.items():
        for dataset_name, d_stats in stats.get("datasets", {}).items():
            summary_data.append({
                "Model": model_name,
                "Dataset": dataset_name,
                "Accuracy": f"{d_stats['accuracy']*100:.2f}%" if d_stats['accuracy'] is not None else "N/A",
                "TPOT (ms)": f"{d_stats['avg_time_per_token']*1000:.2f}" if d_stats['avg_time_per_token'] else "N/A",
                "Total Tokens": d_stats.get('tokens', 0)
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        # Use simple string representation for the table
        print(df.to_string(index=False))
        print(f"\n{'='*30}")
    else:
        print("No evaluation results were collected.")

if __name__ == "__main__":
    main()
