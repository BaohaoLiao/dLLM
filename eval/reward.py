import json
import argparse
import asyncio
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor

from eval.math_utils import is_equal


def parse_args():
    parser = argparse.ArgumentParser(description="Compute score.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--record_path",
        type=str,
        required=True,
    )

    return parser.parse_args()


def main(args):
    # Load dataset
    ds = []
    with open(args.dataset_path, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():                
                ds.append(json.loads(line))

    print("Loaded", len(ds), "samples")

    # Reformat
    index_list = []
    prediction_list = []
    gt_list = []
    response_length_list = []
    for i in range(len(ds)):
        index_list = index_list + [ds[i]["index"]] * len(ds[i]["prediction"])
        prediction_list = prediction_list + ds[i]["extracted_output"]
        response_length_list = response_length_list + ds[i]["response_length"]
        gt_list = gt_list + [ds[i]["ground_truth_answer"]] * len(ds[i]["prediction"])
        ds[i]["correctness"] = []

    # Compute scores
    nest_asyncio.apply()

    async def get_correctness():
        executor = ThreadPoolExecutor(max_workers=64)
        tasks = []
        for i in range(len(index_list)):
            tasks.append(is_equal(prediction_list[i], gt_list[i], executor))
        results = await asyncio.gather(*tasks)
        return results

    correctness_list = asyncio.run(get_correctness())
    for i in range(len(index_list)):
        index_i = index_list[i]
        ds[index_i]["correctness"].append(correctness_list[i])

    # Compute global acc
    acc = sum(correctness_list) / len(correctness_list)
    k = len(ds[0]["prediction"])
    pass_k = sum([any(ds[i]["correctness"]) for i in range(len(ds))]) / len(ds)
    scores = f"accuracy: {acc:.4f}\tpass@{k}: {pass_k:.4f}"

    # Save
    with open(args.record_path, "w") as f:
        f.write(args.dataset_path + " " + str(scores) + "\n")

    with open(
        args.dataset_path.split(".json")[0] + "_score.json", "w", encoding="utf8"
    ) as f:
        json.dump(ds, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
