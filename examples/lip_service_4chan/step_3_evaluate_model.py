#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-1-8b-unsloth-notebook
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
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="你好。", type=str)

    parser.add_argument("--model_name", default="unsloth/Qwen2-1.5B-Instruct-bnb-4bit", type=str)
    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)

    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    text = tokenizer.apply_chat_template(
        conversation=[{
            "role": "user",
            "content": args.text,
        }],
        add_generation_prompt=True,
    )
    print(f"prompt: {text}")

    inputs = tokenizer.__call__(
        text=[text],
        return_tensors="pt",
    ).to(args.device)

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    _ = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        streamer=text_streamer,
        max_new_tokens=64,
        pad_token_id=tokenizer.eos_token_id
    )
    return


if __name__ == "__main__":
    pass
