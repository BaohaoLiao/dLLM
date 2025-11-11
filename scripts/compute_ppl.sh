#!/bin/bash

MODEL_NAME=JetLM/SDAR-1.7B-Chat
MODEL=/mnt/nushare2/data/baliao/PLLMs/${MODEL_NAME}
DATA=/mnt/nushare2/data/baliao/dlm/data/hendrydong/test.json
SAVE_DIR=/mnt/nushare2/data/baliao/dlm/00_start/ppl_results/${MODEL_NAME}

mkdir -p ${SAVE_DIR}

for block_len in 4; do
    echo "Evaluating block_length=${block_len}"
    python sample/ppl.py \
        --model_name_or_path ${MODEL} \
        --tensor_parallel_size 1 \
        --dataset_path ${DATA} \
        --block_length ${block_len} \
        --batch_size 8 \
        --max_concurrent 8 \
        --max_samples 128 \
        --max_length 2048 \
        --output_file ${SAVE_DIR}/blocklen_${block_len}.json
done


# --save_decode_orders