#!/bin/bash

save_dir=./outputs/TraDo-4B-Instruct
dataset_dir=./data
data_path=/mnt/nushare2/data/baliao/dynamic_filter/data

# Configuration
K=3
GPUS=(0)

models=("/mnt/nushare2/data/baliao/PLLMs/Gen-Verse/TraDo-4B-Instruct")
datasets=("AIME2024" "MATH500")

# Loop through models and datasets
for model_name in "${models[@]}"; do
    echo "Testing model: $model_name"
    for dataset in "${datasets[@]}"; do
        echo "Testing dataset: $dataset"
        
        # Create model/dataset specific output directory
        output_dir=${save_dir}/${dataset}
        echo "Output directory: ${output_dir}"
        
        # Generate data in parallel
        echo "Starting parallel data generation..."
        for ((i=0; i<${#GPUS[@]}; i++)); do
            CUDA_VISIBLE_DEVICES=$i python3 eval/sample.py \
                --dataset ${dataset} \
                --dataset_dir ${dataset_dir} \
                --model_name_or_path ${model_name} \
                --n ${K} \
                --temperature 1.0 \
                --top_p 1.0 \
                --top_k 0.0 \
                --max_new_tokens 2000 \
                --seed 0 \
                --dllm_unmasking_strategy "low_confidence_dynamic" \
                --dllm_block_length 4 \
                --dllm_denoising_steps 4 \
                --dllm_confidence_threshold 0.9 \
                --output_dir ${output_dir} \
                --num_gpus ${#GPUS[@]} \
                --gpu_idx ${i} &
        done
        
        # Wait for all parallel processes to complete
        wait
        echo "Data generation completed."
        
        # Merge the generated data
        echo "Merging data..."
        python3 eval/merge.py \
            --base_path ${output_dir}\
            --output_path ${output_dir}/merged_data.jsonl \
            --num_splits ${#GPUS[@]}
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to merge data for $model_name on $dataset"
            continue
        fi
        
        # Compute scores
        echo "Computing scores..."
        python3 eval/reward.py \
            --dataset_path ${output_dir}/merged_data.jsonl \
            --record_path ${output_dir}/record.txt
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to compute scores for ${model_name} on ${dataset}"
            continue
        fi
        
        echo "Completed evaluation for ${model_name} on ${dataset}"
        echo "Results saved to: ${output_dir}/record.txt"
        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"