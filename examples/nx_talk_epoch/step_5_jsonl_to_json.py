#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data_dir/evaluation.jsonl", type=str)
    parser.add_argument("--output_file", default="data_dir/evaluation.json", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    result = list()
    with open(args.input_file, "r", encoding="utf-8") as fin:
        for row in fin:
            row = json.loads(row)
            result.append(row)

    with open(args.output_file, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=4)

    return


if __name__ == "__main__":
    main()
