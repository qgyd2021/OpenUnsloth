#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""

python3 step_1_prepare_data.py --data_file ../../data/nx_bot/judgment_list.jsonl --data_dir judgment-data_dir
python3 step_1_prepare_data.py --data_file ../../data/nx_bot/retrieval_list.jsonl --data_dir retrieval-data_dir
python3 step_1_prepare_data.py --data_file ../../data/nx_bot/talk_list.jsonl --data_dir talk-data_dir

"""
import argparse
import json
import os
from pathlib import Path
import platform
import random
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../"))

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        default=(project_path / "data/nx_bot/talk_list.jsonl").as_posix(),
        type=str
    )
    parser.add_argument("--train_file", default="train.jsonl", type=str)
    parser.add_argument("--valid_file", default="valid.jsonl", type=str)

    parser.add_argument(
        "--data_dir",
        default="data_dir/",
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # dataset
    data = list()
    with open(args.data_file, "r", encoding="utf-8") as f:
        for row in f:
            row = json.loads(row)
            data.append(row)

    print(f"dataset samples count: {len(data)}")

    train_file = data_dir / args.train_file
    valid_file = data_dir / args.valid_file
    with open(train_file.as_posix(), "w", encoding="utf-8") as ftrain, open(valid_file.as_posix(), "w", encoding="utf-8") as fvalid:
        random.shuffle(data)
        for row in data:
            row_ = json.dumps(row, ensure_ascii=False)
            flag = random.random()
            if flag < 0.8:
                ftrain.write(f"{row_}\n")
            else:
                fvalid.write(f"{row_}\n")
    return


if __name__ == '__main__':
    main()
