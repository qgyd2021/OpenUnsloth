#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-1-8b-unsloth-notebook
https://huggingface.co/docs/transformers/v4.36.1/zh/llm_tutorial

python3 step_4_evaluation.py --model_name output_dir/checkpoint-300

"""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import platform
import sys
from typing import List

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../"))

import torch
from transformers import TextStreamer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
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
    model: Qwen2Model = model

    with open(args.valid_file, "r", encoding="utf-8") as fin, open(args.output_file, 'w', encoding="utf-8") as fout:
        for row in fin:
            row = json.loads(row)
            messages: List[dict] = row["messages"]
            text = tokenizer.apply_chat_template(
                conversation=messages if messages[-1]["role"] == "user" else messages[:-1],
                add_generation_prompt=True,
                tokenize=False,
            )
            print(f"prompt: {text}")

            inputs = tokenizer.__call__(
                text=[text],
                return_tensors="pt",
            ).to(args.device)

            generate_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
            )
            input_length = inputs.input_ids.shape[1]
            response = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(f"response: {response}")

            if messages[-1]["role"] == "user":
                messages.append({
                    "role": "assistant",
                    "content": response,
                })
            else:
                messages[-1]["gen_content"] = response
            row_ = {
                "messages": messages
            }
            row_ = json.dumps(row_, ensure_ascii=False)
            fout.write(f"{row_}\n")

    return


if __name__ == "__main__":
    main()
