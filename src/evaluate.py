import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pii", "old", "combined"], required=True)
    p.add_argument("--limit", type=int, default=config.EVAL_LIMIT)
    p.add_argument("--beams", type=int, default=config.GENERATION_BEAMS)
    return p.parse_args()


def clean_prediction(text):
    text = text.strip().splitlines()[0].strip()
    text = re.split(r"[\s,;]+", text)[0]
    return text.strip("`'\"")


def main():
    args = parse_args()
    adapter_dir = config.CHECKPOINT_DIR / args.mode / "final"
    test_file = config.PROCESSED_DIR / args.mode / "test.jsonl"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_dir}")
    if not test_file.exists():
        raise FileNotFoundError(f"Evaluation data not found: {test_file}")

    df = pd.read_json(test_file, lines=True)
    if args.limit is not None:
        df = df.head(args.limit)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    device = next(model.parameters()).device

    rows = []
    print(f"Evaluating {args.mode}: {len(df)} examples, {args.beams} beam candidates")
    print("Evaluation only; no training.")

    for count, (_, row) in enumerate(df.iterrows(), start=1):
        inputs = tokenizer(
            row["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_LENGTH,
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                num_beams=args.beams,
                num_return_sequences=args.beams,
                do_sample=False,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        prompt_len = inputs["input_ids"].shape[1]
        candidates = []

        for seq in outputs:
            pred = clean_prediction(
                tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            )
            if pred and pred not in candidates:
                candidates.append(pred)

        target = str(row["target"])
        rank = next(
            (r for r, candidate in enumerate(candidates, start=1)
             if candidate == target),
            None,
        )

        rows.append({
            "user_id": row["user_id"],
            "target": target,
            "num_unique_candidates": len(candidates),
            "candidates": candidates,
            "rank": rank,
        })

        if count % 10 == 0 or count == len(df):
            print(f"Evaluated {count}/{len(df)}")

    metrics = {
        "mode": args.mode,
        "examples": len(rows),
        "requested_candidates": args.beams,
        "average_unique_candidates": (
            sum(r["num_unique_candidates"] for r in rows) / len(rows)
            if rows else 0.0
        ),
    }

    for k in config.TOP_K_VALUES:
        hits = sum(
            1 for r in rows
            if r["rank"] is not None and r["rank"] <= k
        )
        metrics[f"top_{k}"] = hits / len(rows) if rows else 0.0

    reciprocal = [1.0 / r["rank"] for r in rows if r["rank"] is not None]
    metrics["mrr"] = sum(reciprocal) / len(reciprocal) if reciprocal else 0.0

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    metrics_file = config.METRICS_DIR / f"{args.mode}_top100.json"
    predictions_file = config.PREDICTION_DIR / f"{args.mode}_top100.jsonl"

    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with predictions_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved: {metrics_file}")


if __name__ == "__main__":
    main()
