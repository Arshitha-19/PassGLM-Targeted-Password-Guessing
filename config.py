from pathlib import Path

ROOT = Path(__file__).resolve().parent

SEED = 42
NUM_USERS = 10_000
TEST_SIZE = 0.20

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

NUM_EPOCHS = 1
MAX_LENGTH = 192
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

USE_4BIT = True

EVAL_LIMIT = 200
TOP_K_VALUES = [1, 5, 10, 50, 100]
GENERATION_BEAMS = 20
MAX_NEW_TOKENS = 20

RAW_DATA = ROOT / "data" / "raw" / "synthetic_users.csv"
TRAIN_DATA = ROOT / "data" / "splits" / "train.csv"
TEST_DATA = ROOT / "data" / "splits" / "test.csv"

PROCESSED_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "results" / "checkpoints"
METRICS_DIR = ROOT / "results" / "metrics"
PREDICTION_DIR = ROOT / "results" / "predictions"
PLOT_DIR = ROOT / "results" / "plots"
