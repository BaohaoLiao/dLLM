import os
import json
import argparse
import torch
import numpy as np
import datasets

from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.pytorch.tools.utils import Timer, visualize_pipe_out
from lmdeploy.model import MODELS, BaseChatTemplate


def parse_args():
    parser = argparse.ArgumentParser(description="Sampling from eval dataset.")

    # Eval dataset to sample from
    parser.add_argument(
        "--dataset",
        choices=[
            "PrimeIntellect",
            "MATH_train",
            "demon_openr1math",
            "MATH500",
            "GSM8K",
            "AIME2024",
        ],
        required=True,
        help="Which dataset to sample from",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="Where the eval dataset is stored",
    )

    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Which model to use for sampling",
    )

    # General sampling parameters
    parser.add_argument(
        "--n", type=int, default=1, help="Number of samples to generate per input"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--top_k",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    # dLLM sampling parameters
    parser.add_argument(
        "--dllm_unmasking_strategy",
        choices=["low_confidence_static", "low_confidence_dynamic", "sequential"],
        default="low_confidence_dynamic",
    )
    parser.add_argument(
        "--dllm_block_length",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dllm_denoising_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dllm_confidence_threshold",
        type=float,
        default=0.9,
        help="For low_confidence_dynamic",
    )

    # Save results
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save sampled traces",
    )

    # Parallel generation
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gpu_idx",
        type=int,
        default=0,
    )

    return parser.parse_args()


@MODELS.register_module(name='trado_thinking_model')
class TradoThinkingModel(BaseChatTemplate):
    def __init__(self,
                 system='',
                 meta_instruction='',
                 user='<|im_start|>user\n',
                 assistant='<|im_start|>assistant<think>\n',
                 eosys='<|im_end|>\n',
                 eoh='<|im_end|>\n',
                 eoa='<|im_end|>',
                 separator='\n',
                 stop_words=['<|im_end|>']):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         stop_words=stop_words)


def extract_boxed_answer(s: str):
    tag = r"\boxed{"
    start = s.rfind(tag)  # last \boxed{
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1  # we are already inside one '{'
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:  # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1

    return "".join(buf) if depth == 0 else "Can not extract the answer!"


def main(args):
    # Sanity check
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    data_path = os.path.join(args.dataset_dir, args.dataset + ".json")
    ds = datasets.load_dataset("json", data_files=data_path, split="train")

    def make_prompt(example):
        question = example["question"]
        question_suffix = (
            "\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
        message = [{"role": "user", "content": question + question_suffix}]
        return {"message": message}

    ds = ds.map(make_prompt)
    data_size = len(ds)

    ## Split the dataset equally among GPUs
    k, m = divmod(data_size, args.num_gpus)
    start = args.gpu_idx * k + min(args.gpu_idx, m)
    end = (args.gpu_idx + 1) * k + min(args.gpu_idx + 1, m)
    ds = ds.select(np.arange(start, end))

    print([start, end])
    print(ds, args.dataset)
    print(ds[0])

    # Init pipeline
    backend_config = PytorchEngineConfig(
        tp=1,
        dtype="bfloat16",
        max_prefill_token_num=4096,
        cache_max_entry_count=0.8,
        dllm_block_length=args.dllm_block_length,
        dllm_denoising_steps=args.dllm_denoising_steps,
        dllm_unmasking_strategy=args.dllm_unmasking_strategy,
        dllm_confidence_threshold=args.dllm_confidence_threshold,
    )

    if "TraDo-8B-Thinking" in args.model_name_or_path:
        pipe = pipeline(
            args.model_name_or_path, 
            backend_config=backend_config, 
            chat_template_config=ChatTemplateConfig('trado_thinking_model')
        )
    else:
        pipe = pipeline(
            args.model_name_or_path, 
            backend_config=backend_config, 
        )

    # Run sampling
    gen_config = GenerationConfig(
        n=1,  # only support n=1 for now
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        do_sample=True,
        max_new_tokens=args.max_new_tokens,
        random_seed=args.seed,
    )
    messages = [m for m in ds["message"] for _ in range(args.n)]  # Repeat n times
    outputs = pipe(messages, gen_config=gen_config, use_tqdm=True)

    responses = []
    response_lengths = []
    for i in range(len(ds)):
        responses.append([outputs[j].text for j in range(i * args.n, (i + 1) * args.n)])
        response_lengths.append(
            [outputs[j].generate_token_len for j in range(i * args.n, (i + 1) * args.n)]
        )

    # Format results
    saved_samples = []
    for i in range(len(ds)):
        sample = {
            "question": ds[i]["question"],
            "ground_truth_answer": ds[i]["ground_truth_answer"],
            "prediction": [extract_boxed_answer(r) for r in responses[i]],
            "response_length": response_lengths[i],
            "response": responses[i],
        }
        saved_samples.append(sample)

    # Save results
    output_path = os.path.join(output_dir, f"{args.gpu_idx}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(saved_samples, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
