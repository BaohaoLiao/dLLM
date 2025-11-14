#!/bin/bash

cd /data/chatgpt-lvs-h100/data/baliao/dlm/01_sft/dLLM/sft/LLaMA-Factory

save_dir="/mnt/nushare2/data/baliao/dlm/01_sft/SFT_100k_AR_Qwen3-1.7B"
mkdir -p ${save_dir}/logs

export WANDB_PROJECT="dLLM"
export WANDB_API_KEY=
export WANDB_ENTITY="bliao"
export WANDB_MODE="offline"
export WANDB_DIR="${save_dir}/logs"

torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node 1 \
    ./src/train.py \
    ./examples/train_full/qwen3_full.yaml 2>&1 | tee "${save_dir}/train.log"



    # --master_addr 127.0.0.1 \
    # --master_port 12345 \