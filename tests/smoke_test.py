from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import transformers
import peft
import datasets
import accelerate
import bitsandbytes
import config

print("=" * 60)
print("Environment and data smoke test")
print("=" * 60)
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
print("Datasets:", datasets.__version__)
print("Accelerate:", accelerate.__version__)
print("bitsandbytes: imported")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA runtime:", torch.version.cuda)
else:
    raise SystemExit("CUDA is not available.")

for path in [
    config.RAW_DATA,
    config.TRAIN_DATA,
    config.TEST_DATA,
    config.PROCESSED_DIR / "pii" / "train.jsonl",
    config.PROCESSED_DIR / "old" / "train.jsonl",
    config.PROCESSED_DIR / "combined" / "train.jsonl",
]:
    print("Exists:", path.exists(), path)

print("=" * 60)
print("SMOKE TEST PASSED")
print("=" * 60)
