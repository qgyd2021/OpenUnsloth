#!/usr/bin/python3
# -*- coding: utf-8 -*-
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
from transformers.trainer_utils import EvalPrediction
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data_dir/train.jsonl", type=str)
    parser.add_argument("--valid_file", default="data_dir/validation.jsonl", type=str)

    parser.add_argument("--model_name", default="unsloth/Meta-Llama-3.1-8B-bnb-4bit", type=str)
    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)

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
    print("\n")

    # model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    tokenizer.chat_template = chat_template
    print(f"model: \n{model}\n")
    print(f"tokenizer: \n{tokenizer}\n")

    # index of choices
    for token in ["A", "B", "C", "D", "E"]:
        idx = tokenizer.__call__(token, add_special_tokens=False,)
        print(f"token: {token}, index: {idx}")

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
    print(f"mapped train_dataset examples: ")
    for sample in train_dataset.take(3):
        text = sample["text"]
        print(f"\ttext: {text}")
    print(f"mapped valid_dataset examples: ")
    for sample in valid_dataset.take(3):
        text = sample["text"]
        print(f"\ttext: {text}")
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

    def compute_metrics(eval_prediction: EvalPrediction):
        predictions = eval_prediction.predictions
        label_ids = eval_prediction.label_ids
        print(f"predictions: {predictions}")
        print(f"label_ids: {label_ids}")
        return {}

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=5)
    ]

    # train
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        args=SFTConfig(
            output_dir=args.output_dir,

            eval_strategy="steps",

            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            eval_accumulation_steps=4,
            learning_rate=5e-5,
            max_steps=10000,
            warmup_steps=10,
            logging_steps=200,

            save_strategy="steps",
            save_steps=200,
            save_total_limit=10,
            save_safetensors=True,

            seed=3407,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),

            eval_steps=200,

            load_best_model_at_end=True,
            optim="adamw_8bit",
        ),
    )
    trainer.evaluate()
    trainer.train()

    return


if __name__ == "__main__":
    main()
