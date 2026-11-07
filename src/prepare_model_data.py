from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

MODES = ("pii", "old", "combined")


def build_prompt(row, mode):
    if mode == "pii":
        return (
            "Predict the synthetic target password from the synthetic user information.\n"
            f"Name: {row['name']}\n"
            f"Birthday: {row['birthday']}\n"
            f"Phone: {row['phone']}\n"
            f"ID: {row['id_number']}\n"
            "Target password:"
        )

    if mode == "old":
        return (
            "Predict the synthetic target password from the synthetic old password.\n"
            f"Old password: {row['old_password']}\n"
            "Target password:"
        )

    return (
        "Predict the synthetic target password using synthetic user information "
        "and the synthetic old password.\n"
        f"Name: {row['name']}\n"
        f"Birthday: {row['birthday']}\n"
        f"Phone: {row['phone']}\n"
        f"ID: {row['id_number']}\n"
        f"Old password: {row['old_password']}\n"
        "Target password:"
    )


def main():
    for mode in MODES:
        out_dir = config.PROCESSED_DIR / mode
        out_dir.mkdir(parents=True, exist_ok=True)

        for split_name, source in [
            ("train", config.TRAIN_DATA),
            ("test", config.TEST_DATA),
        ]:
            df = pd.read_csv(source)
            rows = [
                {
                    "prompt": build_prompt(row, mode),
                    "target": row["new_password"],
                    "user_id": row["user_id"],
                }
                for _, row in df.iterrows()
            ]
            pd.DataFrame(rows).to_json(
                out_dir / f"{split_name}.jsonl",
                orient="records",
                lines=True,
                force_ascii=False,
            )

        print(f"Prepared: {mode}")


if __name__ == "__main__":
    main()
