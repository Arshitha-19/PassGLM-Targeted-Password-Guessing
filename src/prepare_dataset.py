from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def main():
    df = pd.read_csv(config.RAW_DATA)
    df = df.dropna().drop_duplicates(subset=["user_id"])

    test = df.sample(frac=config.TEST_SIZE, random_state=config.SEED)
    train = df.drop(test.index)

    train = train.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    test = test.sample(frac=1, random_state=config.SEED).reset_index(drop=True)

    config.TRAIN_DATA.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.TRAIN_DATA, index=False)
    test.to_csv(config.TEST_DATA, index=False)

    print("=" * 60)
    print("Train/test split created")
    print(f"Train: {len(train):,}")
    print(f"Test:  {len(test):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
