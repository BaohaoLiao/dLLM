import json
import argparse
import asyncio
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor
import datasets
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
    ds = datasets.load_dataset("json", data_files=args.dataset_path, split="train")
    ds = list(ds)

    # Reformat
    index_list = []
    prediction_list = []
    gt_list = []
    response_length_list = []

    for i, sample in enumerate(ds):
        preds = sample["prediction"]
        n_preds = len(preds)
        
        index_list.extend([sample["index"]] * n_preds)
        prediction_list.extend(preds)
        gt_list.extend([sample["ground_truth_answer"]] * n_preds)
        response_length_list.extend(sample["response_length"])
        
        sample["correctness"] = []

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
    scores = f"#questions: {len(ds)}\taccuracy: {acc:.4f}\tpass@{k}: {pass_k:.4f}"
    print(scores)

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
