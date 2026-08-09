"""
Poison the IRIS dataset for the Week 8 MLSecOps assignment.

Design decisions (documented since the assignment brief left this ambiguous):

1. The train/test split is performed ONCE on the original clean dataset
   (stratified, test_size=0.4, random_state=42) and that exact split is
   reused for every poisoning level. Only the TRAINING portion is ever
   poisoned; the test set stays identical and clean across all four runs.
   This isolates "how well can the model still learn from corrupted
   training data" from "how corrupted is my evaluation" -- the latter
   would confound accuracy drops with bad ground truth rather than
   measuring what the model actually learned.

2. Poisoned feature values are drawn uniformly from the observed clean
   range of each feature (min-max over the clean training data), not from
   an obviously out-of-domain range. This simulates a realistic, harder
   -to-detect corruption (values look plausible) rather than an trivially
   detectable one (e.g. negative petal length).

3. The number of poisoned rows uses round-half-up integer arithmetic on
   the training set size so percentages are exact and reproducible.

The original data/raw/iris.csv (pulled via DVC) is never modified.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
POISON_LEVELS = [0, 5, 10, 50]

RAW_PATH = Path("data/raw/iris.csv")
DATA_DIR = Path("data")


def load_clean():
    df = pd.read_csv(RAW_PATH, encoding="utf-8-sig").reset_index(drop=True)
    df.insert(0, "iris_id", range(len(df)))
    return df


def fixed_split(df):
    train, test = train_test_split(
        df, test_size=0.4, stratify=df[TARGET], random_state=SEED
    )
    train = train.sort_values("iris_id").reset_index(drop=True)
    test = test.sort_values("iris_id").reset_index(drop=True)
    return train, test


def poison(train_df, pct, seed):
    df = train_df.copy()
    n = len(df)
    n_poison = (n * pct + 50) // 100  # round-half-up, integer-exact

    df["is_poisoned"] = False
    if n_poison > 0:
        rng = np.random.RandomState(seed)
        poison_idx = rng.choice(df.index, size=n_poison, replace=False)
        df.loc[poison_idx, "is_poisoned"] = True

        classes = sorted(train_df[TARGET].unique())
        for feat in FEATURES:
            lo, hi = train_df[feat].min(), train_df[feat].max()
            df.loc[poison_idx, feat] = rng.uniform(lo, hi, size=n_poison)
        df.loc[poison_idx, TARGET] = rng.choice(classes, size=n_poison)

    return df, n_poison


def train_variant_path(level):
    return DATA_DIR / "clean" / "train_0pct.csv" if level == 0 else DATA_DIR / "poisoned" / f"train_{level}pct.csv"


def main():
    df = load_clean()
    print(f"Loaded clean baseline: {len(df)} rows from {RAW_PATH}")
    print(df.head(3).to_string(index=False))

    train, test = fixed_split(df)
    print(f"\nFixed split (test_size=0.4, random_state={SEED}): "
          f"{len(train)} train / {len(test)} test (reused across all levels)")

    (DATA_DIR / "clean").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "poisoned").mkdir(parents=True, exist_ok=True)

    test_path = DATA_DIR / "clean" / "test.csv"
    test.to_csv(test_path, index=False)
    print(f"\nClean test set written -> {test_path} (never poisoned, identical across all runs)")

    summary = []
    for pct in POISON_LEVELS:
        seed = SEED + pct
        poisoned_train, n_poison = poison(train, pct, seed)
        expected = (len(train) * pct + 50) // 100
        assert n_poison == expected, f"poisoned-count mismatch at {pct}%"

        fname = train_variant_path(pct)
        poisoned_train.to_csv(fname, index=False)

        actual_pct = round(100 * n_poison / len(train), 2)
        summary.append({
            "poisoning_level": pct,
            "n_train_total": len(train),
            "n_poisoned": int(n_poison),
            "actual_pct": actual_pct,
            "file": str(fname),
        })
        print(f"  {pct:>3}% target -> {n_poison}/{len(train)} rows poisoned "
              f"({actual_pct}% actual) -> {fname}")

    summary_df = pd.DataFrame(summary)
    summary_path = DATA_DIR / "poisoning_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written -> {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
