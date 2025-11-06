import os
import shutil
import argparse
from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download eval dataset.")
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
        help="Which dataset to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save downloaded dataset",
    )
    return parser.parse_args()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset in ["MATH_train", "PrimeIntellect", "demon_openr1math"]:
        split = "train"
    else:
        split = "test"

    cached_path = hf_hub_download(
        repo_id=f"Gen-Verse/{args.dataset}",
        repo_type="dataset",
        filename=f"{split}/{args.dataset}.json",
    )
    shutil.copy(cached_path, f"{args.output_dir}/{args.dataset}.json")


if __name__ == "__main__":
    args = parse_args()
    main(args)
