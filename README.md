# TargetedPasswordResearch

A safe, synthetic-data research implementation inspired by the experimental structure of the 2026 PassGLM paper:

> Wei Ou et al., "Improving targeted password guessing attacks by using personally identifiable information and old password", Cybersecurity, 2026.

## Safety boundary

This project generates and uses **entirely synthetic research records**. Do not replace the generated data with leaked credential dumps, real passwords, real PII, or real user records.

This is an **adaptation**, not an exact reproduction of the paper. The paper uses GLM-4-9B + LoRA and a real `hotel_mix` dataset; this project uses a small open model and synthetic data so it can run on a consumer GPU with 6 GB VRAM.

## Experimental conditions

1. `pii` — synthetic PII only -> synthetic target password
2. `old` — synthetic old password only -> synthetic target password
3. `combined` — synthetic PII + old password -> synthetic target password

Metrics:
- Top-1
- Top-5
- Top-10
- Top-50
- Top-100
- Mean Reciprocal Rank (MRR)

## Hardware target

Designed around:
- NVIDIA GTX 1660 Ti 6 GB
- Python 3.12
- Windows 11
- CUDA-capable PyTorch
- QLoRA / 4-bit loading where supported

The included defaults intentionally start small. After the end-to-end pipeline works, increase dataset size and epochs in `config.py`.

## First run

From the project root:

```bat
python src\generate_dataset.py
python src\prepare_dataset.py
python src\prepare_model_data.py
python tests\smoke_test.py
```

Then run:

```bat
python src\run_experiments.py
```

or:

```bat
run_pipeline.bat
```

The first model-training run downloads the configured Hugging Face model, so internet access is required.

## Recommended workflow for GTX 1660 Ti

Start with:
- 10,000 synthetic users
- 8,000 train / 2,000 test
- 1 epoch
- sequence length 192
- per-device batch size 1
- gradient accumulation 16
- LoRA rank 8
- 4-bit loading

Evaluation defaults to a 200-example test subset for a quick end-to-end run. Set `EVAL_LIMIT = None` in `config.py` for the complete test set.

## Research interpretation

Do not claim that this synthetic experiment proves the same absolute cracking rates as the paper. It demonstrates a controlled comparison of the information conditions and top-k evaluation methodology.

## Checkpoint format

Training saves LoRA adapters rather than a second full copy of the base model.
Evaluation reloads the base Qwen model in 4-bit mode and attaches the saved adapter.
This keeps disk and VRAM requirements manageable.
