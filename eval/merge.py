import os
import json
import argparse
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Merge split")
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to splits",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=1,
    )

    return parser.parse_args()


def main(args):
    # Merge files
    all_dirs = []
    for i in range(args.num_splits):
        all_dirs.append(os.path.join(args.base_path, str(i) + ".json"))

    gathered_data = []
    for my_dir in all_dirs:
        ds = load_dataset("json", data_files=my_dir, split="train")
        print(len(ds))
        for sample in ds:
            gathered_data.append(sample)

    print("I collect ", len(gathered_data), "samples")

    # Save to file
    with open(args.output_path, "w", encoding="utf8") as f:
        for i in range(len(gathered_data)):
            json.dump(gathered_data[i], f, indent=2, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
