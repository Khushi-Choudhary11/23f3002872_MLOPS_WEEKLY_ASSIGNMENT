"""
Task 2 -- Assess fairness with Fairlearn.

Loads the test-set predictions produced by fairness_data.py (which carry
the `location` sensitive attribute alongside y_true/y_pred, even though
location was never used as a training feature) and runs a Fairlearn
MetricFrame to disaggregate accuracy/precision/recall by location group.

`location` was assigned uniformly at random and independently of every
feature and the label (see fairness_data.py), so it carries no real
signal -- we *expect* near-equal metrics across groups 0 and 1. This
script demonstrates the auditing workflow itself: MetricFrame highlights
a performance gap if one exists, it does not fix it.
"""

from functools import partial
from pathlib import Path

import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn import metrics

DATA_DIR = Path("data/week9")

precision_macro = partial(metrics.precision_score, average="macro", zero_division=0)
recall_macro = partial(metrics.recall_score, average="macro", zero_division=0)


def main():
    df = pd.read_csv(DATA_DIR / "test_predictions.csv")

    mf = MetricFrame(
        metrics={
            "accuracy": metrics.accuracy_score,
            "precision": precision_macro,
            "recall": recall_macro,
        },
        y_true=df["y_true"],
        y_pred=df["y_pred"],
        sensitive_features=df["location"],
    )

    print("=== Overall metrics ===")
    print(mf.overall.to_string())

    print("\n=== Metrics by location group ===")
    print(mf.by_group.to_string())

    print("\n=== Disparity across groups ===")
    print("difference (max - min):")
    print(mf.difference().to_string())
    print("\nratio (min / max):")
    print(mf.ratio().to_string())

    n_by_group = df["location"].value_counts().sort_index()
    print("\n=== Group sizes (test set) ===")
    print(n_by_group.to_string())

    by_group_out = mf.by_group.copy()
    by_group_out.index.name = "location"
    by_group_out["n_samples"] = n_by_group
    out_path = DATA_DIR / "fairlearn_metrics.csv"
    by_group_out.to_csv(out_path)
    print(f"\nPer-group metrics saved -> {out_path}")

    disparity_path = DATA_DIR / "fairlearn_disparity.csv"
    disparity_df = pd.DataFrame({"difference": mf.difference(), "ratio": mf.ratio()})
    disparity_df.to_csv(disparity_path)
    print(f"Disparity summary saved -> {disparity_path}")


if __name__ == "__main__":
    main()
