import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

print("\nModel comparison")
print("-" * 105)
print(f"{'Mode':<12}{'Top-1':>10}{'Top-5':>10}{'Top-10':>10}{'Top-50':>10}{'Top-100':>10}{'MRR':>12}{'Avg unique':>15}")
print("-" * 105)

for mode in ("pii", "old", "combined"):
    path = config.METRICS_DIR / f"{mode}_top100.json"
    if not path.exists():
        continue
    m = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"{mode:<12}"
        f"{m.get('top_1', 0):>9.1%}"
        f"{m.get('top_5', 0):>9.1%}"
        f"{m.get('top_10', 0):>9.1%}"
        f"{m.get('top_50', 0):>9.1%}"
        f"{m.get('top_100', 0):>9.1%}"
        f"{m.get('mrr', 0):>12.4f}"
        f"{m.get('average_unique_candidates', 0):>15.1f}"
    )

print("-" * 105)
