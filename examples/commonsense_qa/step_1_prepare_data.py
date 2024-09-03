#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path
import platform
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../"))

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="tau/commonsense_qa", type=str)
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--dataset_split", default=None, type=str)

    parser.add_argument(
        "--data_dir",
        default="data_dir/",
        type=str
    )

    parser.add_argument(
        "--dataset_cache_dir",
        default=(project_path / "hub_datasets").as_posix(),
        type=str
    )
    parser.add_argument("--dataset_streaming", default=False, type=bool)
    parser.add_argument(
        "--num_workers",
        default=None if platform.system() == "Windows" else os.cpu_count() // 2,
        type=str
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = load_dataset(
        path=args.dataset_path,
        split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
        num_proc=None if (args.dataset_streaming or platform.system() == "Windows") else args.num_workers,
        streaming=args.dataset_streaming,
        trust_remote_code=True,
    )

    for split, dataset in dataset_dict.items():
        print(f"split: {split}, samples count: {len(dataset)}")

        output_file = data_dir / f"{split}.jsonl"
        with open(output_file.as_posix(), "w", encoding="utf-8") as f:
            for sample in dataset:
                question = sample["question"]
                question_concept = sample["question_concept"]
                choices = sample["choices"]
                answer_key = sample["answerKey"]

                prompt = f"concept: {question_concept}\nquestion: {question}\nchoices: \n"
                for label, text in zip(choices["label"], choices["text"]):
                    row = f"{label}: {text}\n"
                    prompt += row
                prompt = prompt.strip()

                response = answer_key
                response = response.strip()

                if len(response) > 1:
                    raise AssertionError(f"prompt: {prompt}, response: {response}.")

                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                    {
                        "role": "assistant",
                        "content": response,
                    }
                ]
                row = {
                    "messages": messages,
                }
                row = json.dumps(row, ensure_ascii=False)
                f.write("{}\n".format(row))

    return


if __name__ == "__main__":
    main()
