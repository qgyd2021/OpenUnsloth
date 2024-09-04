#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-1-8b-unsloth-notebook

python3 step_4_evaluation.py --model_name outputs/checkpoint-10000

"""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import platform
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../"))

import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_file", default="data_dir/valid.jsonl", type=str)
    parser.add_argument("--output_file", default="data_dir/evaluation.jsonl", type=str)

    parser.add_argument("--model_name", default="unsloth/Qwen2-1.5B-Instruct-bnb-4bit", type=str)
    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    print(f"loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    with open(args.valid_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"valid dataset samples count: {len(data)}")

    with open(args.evalation_file, 'w', encoding="utf-8") as f:
        for row in data:
            messages = row["messages"]
            text = tokenizer.apply_chat_template(
                conversation=messages[:-1],
                add_generation_prompt=True,
                tokenize=False,
            )
            print(f"prompt: {text}")

            inputs = tokenizer.__call__(
                text=[text],
                return_tensors="pt",
            ).to(args.device)

            response = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id
            )
            print(f"response: {response}")

            messages[-1]["gen_content"] = response
            row_ = {
                "messages": messages
            }
            row_ = json.dumps(row_, ensure_ascii=False)
            f.write(f"{row_}\n")

    return


if __name__ == "__main__":
    main()
