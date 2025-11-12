#!/bin/bash

MODEL_NAME=JetLM/SDAR-1.7B-Chat
MODEL=/mnt/nushare2/data/baliao/PLLMs/${MODEL_NAME}
DATA=/mnt/nushare2/data/baliao/dlm/data/hendrydong/test.json
SAVE_DIR=/mnt/nushare2/data/baliao/dlm/00_start/ppl_results/${MODEL_NAME}

mkdir -p ${SAVE_DIR}

for block_len in 4; do
    echo "Evaluating block_length=${block_len}"
    python eval/sdar_ppl.py \
        --model_name_or_path ${MODEL} \
        --mask_token_id 151669 \
        --dataset_path ${DATA} \
        --max_samples 128 \
        --max_length 2048 \
        --block_length ${block_len} \
        --batch_size 32 \
        --save_decode_orders \
        --output_dir ${SAVE_DIR}
done