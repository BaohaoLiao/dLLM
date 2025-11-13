#!/bin/bash

export WANDB_PROJECT="dLLM"

torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node 2 \
    --master_addr 127.0.0.1 \
    --master_port 12345 \
    ./src/llamafactory/v1/launcher.py \
    ./examples/train_full/qwen3_full.yaml