#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path
import platform

from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data_dir/train.jsonl", type=str)
    parser.add_argument("--valid_file", default="data_dir/valid.jsonl", type=str)

    parser.add_argument("--model_name", default="unsloth/Qwen2-1.5B-Instruct-bnb-4bit", type=str)
    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)

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


def map_messages_to_text(sample: dict, tokenizer):
    messages = sample["messages"]

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    result = {
        "text": text,
    }
    return result


def main():
    args = get_args()

    # dataset
    train_dataset = load_dataset("json", data_files={"train": args.train_file,}, split="train")
    valid_dataset = load_dataset("json", data_files={"valid": args.valid_file,}, split="valid")
    print(f"train_dataset samples count: {len(train_dataset)}")
    print(f"train_dataset examples: ")
    for sample in train_dataset.take(3):
        messages = sample["messages"]
        print(f"\tprompt: {messages[0]['content']}, \tresponse: {messages[1]['content']}")
    print(f"valid_dataset samples count: {len(valid_dataset)}")
    print(f"valid_dataset examples: ")
    for sample in valid_dataset.take(3):
        messages = sample["messages"]
        print(f"\tprompt: {messages[0]['content']}, \tresponse: {messages[1]['content']}")

    # model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    print(model)
    print(tokenizer)

    # map
    train_dataset = train_dataset.map(map_messages_to_text)
    valid_dataset = valid_dataset.map(map_messages_to_text)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=args.max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    return


if __name__ == "__main__":
    main()
