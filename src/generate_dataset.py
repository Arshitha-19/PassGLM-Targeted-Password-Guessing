import csv
import random
from datetime import date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

FIRST_NAMES = [
    "Aarav", "Arjun", "Aditya", "Akash", "Ananya",
    "Aisha", "Diya", "Ishaan", "Kabir", "Karan",
    "Meera", "Neha", "Nikhil", "Riya", "Rahul",
    "Rohan", "Sahil", "Simran", "Tanvi", "Varun",
    "Vikram", "Yash", "Zoya", "Aditi", "Maya",
]

LAST_NAMES = [
    "Sharma", "Verma", "Reddy", "Patel", "Singh",
    "Kumar", "Gupta", "Rao", "Mehta", "Nair",
    "Joshi", "Iyer", "Malhotra", "Kapoor", "Das",
    "Menon", "Bose", "Shah", "Mishra", "Pillai",
]

WORDS = [
    "Tiger", "Dragon", "Shadow", "Phoenix", "Falcon",
    "Storm", "Hunter", "Ranger", "Knight", "Wolf",
    "Eagle", "Thunder", "Matrix", "Galaxy", "Rocket",
    "Warrior",
]

SYMBOLS = ["!", "@", "#", "$", "%"]


def random_date(start_year=1995, end_year=2006):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))


def make_old_password(first_name, birth_year, word):
    patterns = [
        f"{first_name}{birth_year}",
        f"{first_name}{str(birth_year)[-2:]}",
        f"{word}{birth_year}",
        f"{first_name}_{birth_year}",
        f"{word}{str(birth_year)[-2:]}",
    ]
    return random.choice(patterns)


def make_new_password(first_name, birth_year, old_password, word):
    # Synthetic target intentionally combines PII-derived and
    # historical-password-derived signals.
    year2 = str(birth_year)[-2:]
    old_fragment = old_password[-3:]
    symbol = random.choice(SYMBOLS)

    templates = [
        f"{first_name[:3]}{old_fragment}{year2}{symbol}",
        f"{first_name}{year2}{old_fragment}",
        f"{word[:3]}{old_fragment}{year2}{symbol}",
        f"{first_name[:3]}_{old_fragment}_{year2}",
    ]
    return random.choice(templates)


def generate_user(i):
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    full_name = first + last

    birthday = random_date()
    year = birthday.year
    word = random.choice(WORDS)

    # Explicitly synthetic identifiers.
    phone = f"900{i:07d}"
    synthetic_id = f"SYN{i:09d}"

    old_password = make_old_password(first, year, word)
    new_password = make_new_password(first, year, old_password, word)

    return {
        "user_id": f"USER_{i:06d}",
        "name": full_name,
        "birthday": birthday.isoformat(),
        "phone": phone,
        "id_number": synthetic_id,
        "old_password": old_password,
        "new_password": new_password,
    }


def main():
    random.seed(config.SEED)
    config.RAW_DATA.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "user_id", "name", "birthday", "phone",
        "id_number", "old_password", "new_password",
    ]

    with config.RAW_DATA.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(1, config.NUM_USERS + 1):
            writer.writerow(generate_user(i))

    print("=" * 60)
    print("Synthetic dataset generated")
    print(f"Records: {config.NUM_USERS:,}")
    print(f"Output:  {config.RAW_DATA}")
    print("=" * 60)


if __name__ == "__main__":
    main()
