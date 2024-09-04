#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
python3 step_2_train_model.py --model_name unsloth/Qwen2-1.5B-Instruct-bnb-4bit --output_dir Qwen2-1.5B-Instruct-talk
python3 step_2_train_model.py --model_name unsloth/Qwen2-0.5B-Instruct-bnb-4bit --output_dir Qwen2-0.5B-Instruct-talk
python3 step_2_train_model.py --model_name unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --output_dir Meta-Llama-3.1-8B-Instruct-talk


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

from datasets import load_dataset
from transformers.trainer_callback import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
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

    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--per_device_train_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)

    parser.add_argument("--data_dir", default="data_dir/", type=str)
    parser.add_argument("--cache_dir", default="cache_dir/", type=str)
    parser.add_argument("--output_dir", default="output_dir/", type=str)
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

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # dataset
    train_dataset = load_dataset("json", data_files={"train": args.train_file,}, split="train")
    valid_dataset = load_dataset("json", data_files={"valid": args.valid_file,}, split="valid")
    print(f"train_dataset samples count: {len(train_dataset)}\n")
    print(f"train_dataset examples: \n")
    for sample in train_dataset.take(3):
        messages = sample["messages"]
        for message in messages:
            print(f"\t{message['role']}: {message['content']}")
        print("\n")
    print(f"valid_dataset samples count: {len(valid_dataset)}")
    print(f"valid_dataset examples: \n")
    for sample in valid_dataset.take(3):
        messages = sample["messages"]
        for message in messages:
            print(f"\t{message['role']}: {message['content']}")
        print("\n")
    print("\n")

    # model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    print(f"model: \n{model}\n")
    print(f"tokenizer: \n{tokenizer}\n")

    # map
    map_messages_to_text_ = partial(map_messages_to_text, tokenizer=tokenizer)
    train_dataset = train_dataset.map(
        map_messages_to_text_,
        cache_file_name=(cache_dir / "train_dataset.cache").as_posix(),
    )
    valid_dataset = valid_dataset.map(
        map_messages_to_text_,
        cache_file_name=(cache_dir / "valid_dataset.cache").as_posix(),
    )
    print(f"mapped train_dataset examples: \n")
    for sample in train_dataset.take(3):
        text = sample["text"]
        print(f"\ttext: {text}\n")
    print(f"mapped valid_dataset examples: \n")
    for sample in valid_dataset.take(3):
        text = sample["text"]
        print(f"\ttext: {text}\n")
    print("\n")

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

    # train
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        compute_metrics=None,
        args=SFTConfig(
            output_dir=args.output_dir,

            eval_strategy="epoch",

            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=5e-5,
            num_train_epochs=args.num_train_epochs,
            warmup_steps=10,
            logging_strategy="epoch",

            save_strategy="epoch",
            save_total_limit=10,
            save_safetensors=True,

            seed=3407,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),

            optim="adamw_8bit",
        ),
    )
    trainer.evaluate()
    trainer.train()

    return


if __name__ == "__main__":
    main()
