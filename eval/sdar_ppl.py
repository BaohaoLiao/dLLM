import os
import json
import math
import argparse
import numpy as np
from tqdm import tqdm

import datasets
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity for block diffusion models"
    )

    # Model arguments
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Path to model directory"
    )
    parser.add_argument(
        "--mask_token_id", type=int, default=151669, help="Mask token ID"
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
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save_decode_orders",
        action="store_true",
        help="Save decode orders for each sequence",
    )
    args = parser.parse_args()
    return args


@torch.no_grad()
def compute_perplexity_block_diffusion_batch(
    model,
    token_lists,
    mask_id,
    block_length=4,
):
    """
    Compute perplexity using block diffusion with TRUE batching.
    Multiple token sequences are padded to same length and processed together in parallel.
    This is more efficient but requires all sequences to be padded to the same length.

    Args:
        model: The language model
        tokenizer: The tokenizer
        token_lists: List of token lists (each list is a sequence of token IDs)
                    e.g., [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        mask_id: Token ID for mask token
        block_length: Number of tokens per block

    Returns:
        List of (seq_len, nll, decoded_orders) tuples
    """
    # Handle single token list
    if isinstance(token_lists[0], int):
        # Single sequence passed as a flat list
        token_lists = [token_lists]

    batch_size = len(token_lists)

    # Get actual sequence lengths
    seq_lengths = [len(tokens) for tokens in token_lists]
    max_seq_length = max(seq_lengths)

    # Create padded tensor - use mask_id for padding
    input_ids = torch.full(
        (batch_size, max_seq_length), mask_id, dtype=torch.long, device=model.device
    )

    # Fill in the actual tokens
    for i, tokens in enumerate(token_lists):
        input_ids[i, : len(tokens)] = torch.tensor(
            tokens, dtype=torch.long, device=model.device
        )

    # Calculate number of blocks
    num_blocks = (max_seq_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Pad to total_length
    if total_length > max_seq_length:
        padding_length = total_length - max_seq_length
        input_ids = F.pad(input_ids, (0, padding_length), value=mask_id)

    # Store ground truth
    ground_truth = input_ids.clone()  # [batch_size, total_length]

    # Create block causal attention mask for batch
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(
        block_length, dim=0
    ).repeat_interleave(block_length, dim=1)
    # Expand for batch: [batch_size, 1, total_length, total_length]
    block_diffusion_attention_mask = (
        block_diffusion_attention_mask.unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, 1, -1, -1)
    )

    position_ids = (
        torch.arange(total_length, device=model.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    # Store results for all sequences
    all_token_log_probs = [[] for _ in range(batch_size)]
    all_decoded_orders = [[] for _ in range(batch_size)]

    past_key_values = DynamicCache()

    # Process blocks
    for block_idx in range(num_blocks):
        block_start = block_idx * block_length
        block_end = (block_idx + 1) * block_length

        # Initialize current block with masks for all sequences
        cur_x = torch.full(
            (batch_size, block_length), mask_id, dtype=torch.long, device=model.device
        )

        # Get ground truth for this block
        block_ground_truth = ground_truth[
            :, block_start:block_end
        ]  # [batch_size, block_length]

        # Track which positions are still masked (per sequence)
        mask_positions = torch.ones(
            (batch_size, block_length), dtype=torch.bool, device=model.device
        )

        # Track decoded order for each sequence in this block
        block_decoded_orders = [[] for _ in range(batch_size)]

        # Attention mask for current block
        cur_attn_mask = block_diffusion_attention_mask[
            :, :, block_start:block_end, :block_end
        ]
        cur_position_ids = position_ids[:, block_start:block_end]

        # Iteratively unmask tokens
        for step in range(block_length):
            if mask_positions.sum() == 0:
                break

            # Forward pass (batched!)
            outputs = model(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
            )
            logits = outputs.logits  # [batch_size, block_length, vocab_size]

            # Get probabilities for all sequences
            probs = F.softmax(logits, dim=-1)  # [batch_size, block_length, vocab_size]

            # Process each sequence in the batch
            for b in range(batch_size):
                if mask_positions[b].sum() == 0:
                    continue

                # Find peak probability at each masked position
                # Only consider positions within actual sequence length
                peak_probs = torch.full(
                    (block_length,), -float("inf"), device=model.device
                )
                for pos in range(block_length):
                    global_pos = block_start + pos
                    # Only consider if position is masked AND within actual sequence length
                    if mask_positions[b, pos] and global_pos < seq_lengths[b]:
                        peak_probs[pos] = probs[b, pos].max()

                # Check if there are any valid positions to unmask
                if peak_probs.max() == -float("inf"):
                    continue

                # Select position with highest peak probability
                selected_pos = peak_probs.argmax().item()

                # Gather the ground truth token's probability
                gt_token = block_ground_truth[b, selected_pos].item()
                gt_prob = probs[b, selected_pos, gt_token].item()

                # Record decoded order
                global_pos = block_start + selected_pos
                block_decoded_orders[b].append(selected_pos)

                # Record log prob
                all_token_log_probs[b].append(math.log(gt_prob + 1e-10))

                # Unmask this position with ground truth token
                cur_x[b, selected_pos] = gt_token
                mask_positions[b, selected_pos] = False

        # Store decoded orders for this block
        for b in range(batch_size):
            all_decoded_orders[b].append(block_decoded_orders[b])

        # Store KV cache for this block
        model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
        )

    # Compute perplexity for each sequence
    all_results = []
    for b in range(batch_size):
        avg_log_prob = sum(all_token_log_probs[b]) / len(all_token_log_probs[b])
        nll = -avg_log_prob

        all_results.append((seq_lengths[b], nll, all_decoded_orders[b]))

    return all_results


def load_dataset(
    dataset_path: str, tokenizer, max_samples: int = None, max_length: int = 512
):
    """
    Load evaluation dataset and tokenize, and order based on length

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
                sequences.append((text, tokens))

    # Sort sequences by length
    sequences.sort(key=lambda x: len(x[1]))

    return sequences


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Block Diffusion Model Perplexity Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"MASK id: {args.mask_token_id}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Block length: {args.block_length}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    print("Model loaded successfully!")

    # Load dataset
    print("\nLoading dataset...")
    sentences = load_dataset(
        args.dataset_path,
        tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
    )

    if sentences is None or len(sentences) == 0:
        print("Failed to load dataset or dataset is empty")
        return

    print(f"Loaded {len(sentences)} sequences from {args.dataset_path}")
    print(f"Average sequence length: {np.mean([len(s[1]) for s in sentences]):.1f}")
    print(f"Min sequence length: {min(len(s[1]) for s in sentences)}")
    print(f"Max sequence length: {max(len(s[1]) for s in sentences)}")

    # Evaluate perplexity in batches
    print("\nEvaluating perplexity...")
    all_results = []

    num_batches = (len(sentences) + args.batch_size - 1) // args.batch_size
    with tqdm(total=len(sentences), desc="Computing perplexity") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(sentences))
            batch = [s[1] for s in sentences[start_idx:end_idx]]

            # Compute perplexity for this batch
            batch_results = compute_perplexity_block_diffusion_batch(
                model=model,
                token_lists=batch,
                mask_id=args.mask_token_id,
                block_length=args.block_length,
            )

            for seq_idx, (seq_len, nll, decode_orders) in enumerate(batch_results):
                result_entry = {
                    "num_tokens": seq_len,
                    "num_blocks": len(decode_orders),
                    "nll": nll,
                }

                if args.save_decode_orders:
                    result_entry["text"] = sentences[start_idx:end_idx][seq_idx][0]
                    result_entry["decode_orders"] = decode_orders

                all_results.append(result_entry)

            pbar.update(len(batch))

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

    ppl_file = os.path.join(args.output_dir, "ppl_result.json")
    print(f"\nSaving ppl results to {ppl_file}...")
    with open(ppl_file, "w") as f:
        json.dump(output_data, f, indent=2)

    if args.save_decode_orders:
        decode_order_file = os.path.join(args.output_dir, "decode_orders.json")
        print(f"\nSaving decode orders to {decode_order_file}...")
        with open(decode_order_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    print(f"Results saved successfully!")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
