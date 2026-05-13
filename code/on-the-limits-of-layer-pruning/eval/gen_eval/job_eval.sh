#!/bin/bash

# shellcheck disable=SC2206
#SBATCH -p cs
#SBATCH -p nvidia
##SBATCH -q nvidia-xxl
#SBATCH --gres=gpu:1
##SBATCH --constraint=80g
#SBATCH --job-name=base_llama
#SBATCH --output=base_llama.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --tasks-per-node=1
#SBATCH --time=1-00:00:00

# Evaluation Configuration
# IMPORTANT: Manually sync #SBATCH --gres=gpu:1 with GPUS=1 below
GPUS=1
# SPECIFY THE MODEL EITHER A FULL MODEL OR ADAPTER
MODELS="meta-llama/Llama-3.1-8B-Instruct"
# SPECIFY THE BASE MODEL IF MODEL IS ADAPTER
BASES=""
DATASETS="gsm8k,math,humeval,mbpp,xsum"
PROMPTS="LLAMA_PROMPT,LLAMA_PROMPT,LLAMA_HUMEVAL_PROMPT,LLAMA_MBPP_PROMPT,LLAMA_XSUM"

python eval_script.py \
    --model_name "$MODELS" \
    --base_model "$BASES" \
    --dataset_type "$DATASETS" \
    --samples -1 \
    --prompt "$PROMPTS" \
    --split "test" \
    --tensor_parallel_size $GPUS
