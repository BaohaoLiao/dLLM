#!/bin/bash

MODEL_NAME=JetLM/SDAR-1.7B-Chat
MODEL=/mnt/nushare2/data/baliao/PLLMs/${MODEL_NAME}
DATA=/mnt/nushare2/data/baliao/dlm/data/hendrydong/train.parquet
SAVE_DIR=/mnt/nushare2/data/baliao/dlm/00_start/decode_order/${MODEL_NAME}

mkdir -p ${SAVE_DIR}

GPUS=(0 1)

for block_len in 4; do
    echo "Evaluating block_length=${block_len}"

    for ((i=0; i<${#GPUS[@]}; i++)); do
        CUDA_VISIBLE_DEVICES=${GPUS[i]} python eval/sdar_ppl_parallel.py \
            --model_name_or_path ${MODEL} \
            --mask_token_id 151669 \
            --dataset_path ${DATA} \
            --max_samples 128 \
            --max_length 2048 \
            --block_length ${block_len} \
            --batch_size 64 \
            --save_decode_orders \
            --output_dir ${SAVE_DIR} \
            --world_size ${#GPUS[@]} \
            --local_idx ${i} &
    done
done