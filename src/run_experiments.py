import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    cmd = [sys.executable, str(ROOT / "src" / script), *args]
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    run("generate_dataset.py")
    run("prepare_dataset.py")
    run("prepare_model_data.py")

    # Train and evaluate each controlled condition independently.
    for mode in ("pii", "old", "combined"):
        run("train.py", "--mode", mode)
        run("evaluate.py", "--mode", mode)

    print("\nAll three experiments completed.")
    print("Metrics: results/metrics/")
    print("Predictions: results/predictions/")


if __name__ == "__main__":
    main()
