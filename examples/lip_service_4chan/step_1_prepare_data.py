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

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="qgyd2021/lip_service_4chan", type=str)
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


def keywords_filter(example):
    # forbidden
    forbidden_words = [
        "助手", "脏话学习助手", "学习助手", "语言学习辅助助手", "狗屎助手",
        "我只是个", "我是", "我只是一個", "想问什么",
        "自己去学", "去学",
        "我才不会教你", "教你",
        "想学骂人"
    ]

    question = example["question"]
    answer = example["answer"]

    flag = True
    for keyword in forbidden_words:
        if answer.__contains__(keyword):
            flag = False
            break

    return flag


def main():
    args = get_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    name_list = [
        # "chatterbot_10",
        # "moss_003_sft_data_10",
        "weibo_1",
        # "xiaohuangji_10",
    ]

    dataset = list()
    for name in name_list:
        dataset_dict = load_dataset(
            path=args.dataset_path,
            name=name,
            split=args.dataset_split,
            cache_dir=args.dataset_cache_dir,
            num_proc= None if (args.dataset_streaming or platform.system() == "Windows") else args.num_workers,
            streaming=args.dataset_streaming,
            trust_remote_code=True,
        )
        # print(dataset_dict)
        dataset.append(dataset_dict["train"])
    dataset = concatenate_datasets(dataset)
    print(f"dataset samples count: {len(dataset)}")
    dataset = dataset.filter(function=keywords_filter)
    print(f"dataset after filer samples count: {len(dataset)}")

    if args.dataset_streaming:
        valid_dataset = dataset.take(args.valid_dataset_size)
        train_dataset = dataset.skip(args.valid_dataset_size)
        train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer_size, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=10000, seed=None)
        train_dataset = dataset["train"]
        valid_dataset = dataset["test"]

    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"

    for output_file, sub_dataset in zip([train_file, valid_file], [train_dataset, valid_dataset]):
        with open(output_file.as_posix(), "w", encoding="utf-8") as f:
            for example in sub_dataset:
                messages = [
                    {
                        "role": "user",
                        "content": example["question"],
                    },
                    {
                        "role": "assistant",
                        "content": example["answer"],
                    }
                ]
                row = {
                    "messages": messages,
                }
                row = json.dumps(row, ensure_ascii=False)
                f.write("{}\n".format(row))

    return


if __name__ == '__main__':
    main()
