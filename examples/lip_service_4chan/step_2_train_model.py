#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path
import platform

from datasets import load_dataset, concatenate_datasets

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data_dir/train.jsonl", type=str)
    parser.add_argument("--valid_file", default="data_dir/valid.jsonl", type=str)

    parser.add_argument(
        "--data_dir",
        default="data_dir/",
        type=str
    )

    parser.add_argument(
        "--num_workers",
        default=None if platform.system() == "Windows" else os.cpu_count() // 2,
        type=str
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    train_dataset = load_dataset("json", data_files={"train": args.train_file,}, split="train")
    valid_dataset = load_dataset("json", data_files={"valid": args.valid_file,}, split="valid")
    print(train_dataset)
    print(valid_dataset)

    return


if __name__ == "__main__":
    main()
