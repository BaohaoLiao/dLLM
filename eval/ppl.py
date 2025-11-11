import json
import argparse
import numpy as np
from tqdm import tqdm
import datasets
from lmdeploy import pipeline, PytorchEngineConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity for block diffusion models"
    )

    # Model arguments
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Path to model directory"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="Tensor parallel size"
    )

    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )

    # Evaluation arguments
    parser.add_argument(
        "--block_length", type=int, required=True, help="Block length for evaluation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=8, help="Maximum concurrent sequences"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="ppl_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--save_decode_orders",
        action="store_true",
        help="Save decode orders for each sequence",
    )
    args = parser.parse_args()
    return args


def load_dataset(
    dataset_path: str, tokenizer, max_samples: int = None, max_length: int = 512
):
    """
    Load evaluation dataset and tokenize.

    Args:
        dataset_path: path to a json file
        tokenizer: Tokenizer instance
        max_samples: Maximum number of samples to evaluate (None = all)
        max_length: Maximum sequence length

    Returns:
        List of tokenized sequences
    """
    sequences = []

    dataset = datasets.load_dataset("json", data_files=dataset_path, split="train")

    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        text = example["text"].strip()
        if len(text) > 0:  # Skip empty lines
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > 0:
                # Truncate if too long
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                sequences.append(tokens)

    return sequences


def main(args):
    print("=" * 60)
    print("Block Diffusion Model Perplexity Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Block length: {args.block_length}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    backend_config = PytorchEngineConfig(
        tp=1,
        dtype="bfloat16",
        max_prefill_token_num=1024,
        cache_max_entry_count=0.8,
        dllm_block_length=args.block_length,
        dllm_denoising_steps=args.block_length,  # not used
        dllm_unmasking_strategy="low_confidence_dynamic",  # not used
        dllm_confidence_threshold=0.9,  # not used
    )
    pipe = pipeline(args.model_name_or_path, backend_config=backend_config)
    print("Model loaded successfully!")

    # Load dataset
    print("\nLoading dataset...")
    sentences = load_dataset(
        args.dataset_path,
        pipe.tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
    )

    if sentences is None or len(sentences) == 0:
        print("Failed to load dataset or dataset is empty")
        return

    print(f"Loaded {len(sentences)} sequences from {args.dataset_path}")
    print(f"Average sequence length: {np.mean([len(s) for s in sentences]):.1f}")
    print(f"Min sequence length: {min(len(s) for s in sentences)}")
    print(f"Max sequence length: {max(len(s) for s in sentences)}")

    # Evaluate perplexity in batches
    print("\nEvaluating perplexity...")
    all_results = []

    num_batches = (len(sentences) + args.batch_size - 1) // args.batch_size
    with tqdm(total=len(sentences), desc="Computing perplexity") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(sentences))
            batch = sentences[start_idx:end_idx]

            # Compute perplexity for this batch
            try:
                batch_results = pipe.get_dllm_ppl(
                    batch, max_concurrent=args.max_concurrent
                )

                for seq_idx, result in enumerate(batch_results):
                    global_idx = start_idx + seq_idx
                    decode_orders = result["decode_orders"]

                    # Prepare result entry
                    result_entry = {
                        "index": global_idx,
                        "num_tokens": result["num_tokens"],
                        "num_blocks": len(decode_orders),
                        "nll": result["nll"],
                        "perplexity": result["perplexity"],
                    }

                    if args.save_decode_orders:
                        result_entry["decode_orders"] = decode_orders

                    all_results.append(result_entry)

                pbar.update(len(batch))

            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                import traceback

                traceback.print_exc()
                continue

    # Compute statistics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    corpus_nll = 0
    corpus_num_tokens = 0
    for result in all_results:
        corpus_nll += result["nll"] * result["num_tokens"]
        corpus_num_tokens += result["num_tokens"]

    corpus_avg_nll = corpus_nll / corpus_num_tokens
    corpus_ppl = np.exp(corpus_avg_nll)
    avg_seq_len = corpus_num_tokens / len(all_results)

    print(f"Total sequences evaluated: {len(all_results)}")
    print(f"Perplexity: {corpus_ppl:.4f}")
    print(f"Avg sequence length: {avg_seq_len:.1f}")

    # Save results
    output_data = {
        "num_seqs": len(all_results),
        "avg_seq_len": avg_seq_len,
        "ppl": corpus_ppl,
    }

    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved successfully!")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
