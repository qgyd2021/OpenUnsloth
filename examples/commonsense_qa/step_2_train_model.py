#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/docs/trl/sft_trainer
"""
import argparse
from functools import partial
import json
import logging
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
from trl import DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data_dir/train.jsonl", type=str)
    parser.add_argument("--valid_file", default="data_dir/validation.jsonl", type=str)

    parser.add_argument("--model_name", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", type=str)
    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)

    parser.add_argument(
        "--instruction_template",
        default="<|start_header_id|>user<|end_header_id|>\n\n",
        type=str
    )
    parser.add_argument(
        "--response_template",
        default="<|start_header_id|>assistant<|end_header_id|>\n\n",
        type=str
    )

    parser.add_argument("--data_dir", default="data_dir/", type=str)
    parser.add_argument("--cache_dir", default=None, type=str)
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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    cache_dir = None
    if args.cache_dir is not None:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # dataset
    train_dataset = load_dataset("json", data_files={"train": args.train_file,}, split="train")
    valid_dataset = load_dataset("json", data_files={"valid": args.valid_file,}, split="valid")

    msg = "\n"
    msg += f"train_dataset samples count: {len(train_dataset)}\n"
    msg += f"train_dataset examples: \n"
    for sample in train_dataset.take(3):
        messages = sample["messages"]
        msg += f"\tprompt: {messages[0]['content']}, \tresponse: {messages[1]['content']}\n"
    msg += f"valid_dataset samples count: {len(valid_dataset)}\n"
    msg += f"valid_dataset examples: \n"
    for sample in valid_dataset.take(3):
        messages = sample["messages"]
        msg += f"\tprompt: {messages[0]['content']}, \tresponse: {messages[1]['content']}\n"
    msg += "\n"
    logger.info(msg)

    # model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    msg = "\n"
    msg += f"model: \n{model}\n"
    msg += f"tokenizer: \n{tokenizer}\n"
    msg += "\n"
    logger.info(msg)

    # index of choices
    for token in ["A", "B", "C", "D", "E"]:
        idx = tokenizer.__call__(token, add_special_tokens=False,)
        logger.info(f"token: {token}, index: {idx}")

    # map
    map_messages_to_text_ = partial(map_messages_to_text, tokenizer=tokenizer)
    train_dataset = train_dataset.map(
        map_messages_to_text_,
        cache_file_name=None if cache_dir is None else (cache_dir / "train_dataset.cache").as_posix(),
    )
    valid_dataset = valid_dataset.map(
        map_messages_to_text_,
        cache_file_name=None if cache_dir is None else (cache_dir / "valid_dataset.cache").as_posix(),
    )
    msg = "\n"
    msg += "mapped train_dataset examples: \n"
    for sample in train_dataset.take(3):
        text = sample["text"]
        msg += f"\ttext: {text}\n"
    msg += "mapped valid_dataset examples: \n"
    for sample in valid_dataset.take(3):
        text = sample["text"]
        msg += f"\ttext: {text}\n"
    msg += "\n"
    logger.info(msg)

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

    # accuracy metric and cross entropy loss required.
    def compute_metrics(eval_prediction: EvalPrediction):
        predictions = eval_prediction.predictions
        label_ids = eval_prediction.label_ids
        logger.info(f"predictions: {predictions}")
        logger.info(f"label_ids: {label_ids}")
        return {"accuracy": 1.0}

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=5)
    ]

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=args.response_template,
        instruction_template=args.instruction_template,
        tokenizer=tokenizer
    )
    msg = "\n"
    msg += "data_collator: \n"
    msg += f"data_collator.instruction_template: {json.dumps(data_collator.instruction_template)}.\n"
    msg += f"data_collator.instruction_token_ids: {json.dumps(data_collator.instruction_token_ids)}.\n"
    msg += f"data_collator.response_template: {json.dumps(data_collator.response_template)}.\n"
    msg += f"data_collator.response_token_ids: {json.dumps(data_collator.response_token_ids)}.\n"
    msg += "\n"

    for sample in train_dataset.take(3):
        text = sample["text"]
        input_ids = tokenizer(text)
        sample_ = data_collator([input_ids])
        msg += f"text: {text}"
        msg += f"input sample: {sample_}"
    msg += "\n"
    logger.info(msg)

    # train
    trainer = SFTTrainer(
        model=model,
        data_collator=data_collator,
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
            max_steps=1000,
            warmup_steps=10,
            logging_steps=20,

            save_strategy="steps",
            save_steps=20,
            save_total_limit=10,
            save_safetensors=True,

            seed=3407,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),

            eval_steps=20,

            load_best_model_at_end=True,
            optim="adamw_8bit",
        ),
    )
    trainer.evaluate()
    trainer.train()

    return


if __name__ == "__main__":
    main()
