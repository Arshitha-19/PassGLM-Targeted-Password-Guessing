import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pii", "old", "combined"], required=True)
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--max-length", type=int, default=config.MAX_LENGTH)
    p.add_argument("--model-name", default=config.MODEL_NAME)
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    data_dir = config.PROCESSED_DIR / args.mode
    train_file = data_dir / "train.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(
            f"{train_file} not found. Run prepare_model_data.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    model_kwargs = {}

    if config.USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs.update(
            quantization_config=quant_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    if config.USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": str(train_file)})["train"]

    def tokenize(batch):
        texts = [
            p + " " + t + tokenizer.eos_token
            for p, t in zip(batch["prompt"], batch["target"])
        ]
        return tokenizer(texts, truncation=True, max_length=args.max_length)

    tokenized = ds.map(
        tokenize,
        batched=True,
        remove_columns=ds.column_names,
        desc=f"Tokenizing {args.mode}",
    )

    output_dir = config.CHECKPOINT_DIR / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print(f"Starting training: {args.mode}")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(final_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": args.mode,
                "model_name": args.model_name,
                "epochs": args.epochs,
                "max_length": args.max_length,
                "gpu": torch.cuda.get_device_name(0),
            },
            f,
            indent=2,
        )

    print(f"Saved model to: {final_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
