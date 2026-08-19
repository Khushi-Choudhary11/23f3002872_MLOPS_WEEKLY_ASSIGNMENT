"""
Task 4 -- Detect data drift.

Simulates a "production" batch of IRIS data by deliberately shifting two
of the four feature distributions relative to the original training data,
while leaving the other two essentially untouched (small measurement-noise
only). This gives a clean pass/fail contrast for the statistical test:

  - petal_length : + constant offset (e.g. a sensor recalibration, or
                    measuring a later growth stage) -> distribution mean
                    shifts, shape stays the same.
  - petal_width   : multiplicative scale change (e.g. a unit/instrument
                    change) -> both mean and spread change.
  - sepal_length, sepal_width : small Gaussian measurement noise only,
                    no systematic shift -> should NOT be flagged as drift.

Each feature's reference (training) vs. production distribution is
compared with a two-sample Kolmogorov-Smirnov test (a standard,
model-free way to detect *data* drift -- it only looks at input feature
distributions, not predictions or labels, which is why it cannot detect
concept drift).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path("data/week9")
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SEED = 42
ALPHA = 0.05


def simulate_production(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    prod = df.copy()
    # Systematic shift: constant offset (location shift only).
    prod["petal_length"] = prod["petal_length"] + 1.5
    # Systematic shift: multiplicative scale change (mean + spread shift).
    prod["petal_width"] = prod["petal_width"] * 1.5
    # No systematic shift, just ordinary measurement noise.
    prod["sepal_length"] = prod["sepal_length"] + rng.normal(0, 0.05, size=len(df))
    prod["sepal_width"] = prod["sepal_width"] + rng.normal(0, 0.05, size=len(df))
    return prod


def main():
    reference = pd.read_csv("data/raw/iris.csv")
    production = simulate_production(reference)
    production.to_csv(DATA_DIR / "production_simulated.csv", index=False)
    print(f"Simulated production dataset saved -> {DATA_DIR / 'production_simulated.csv'}")

    results = []
    for feature in FEATURES:
        stat, p_value = stats.ks_2samp(reference[feature], production[feature])
        drifted = p_value < ALPHA
        results.append({
            "feature": feature,
            "ref_mean": reference[feature].mean(),
            "prod_mean": production[feature].mean(),
            "mean_shift": production[feature].mean() - reference[feature].mean(),
            "ks_statistic": stat,
            "p_value": p_value,
            "drifted": drifted,
        })

    results_df = pd.DataFrame(results)
    out_path = DATA_DIR / "drift_test_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nKS-test drift results (alpha={ALPHA}):")
    print(results_df.to_string(index=False))
    print(f"\nSaved -> {out_path}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, feature in zip(axes.flat, FEATURES):
        ax.hist(reference[feature], bins=20, alpha=0.5, density=True, label="reference (training)")
        ax.hist(production[feature], bins=20, alpha=0.5, density=True, label="production (simulated)")
        row = results_df[results_df.feature == feature].iloc[0]
        flag = "DRIFTED" if row.drifted else "no drift"
        ax.set_title(f"{feature}  (KS p={row.p_value:.2e}, {flag})")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plot_path = DATA_DIR / "drift_distributions.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Distribution comparison plot saved -> {plot_path}")

    drifted_features = results_df[results_df.drifted].feature.tolist()
    print(f"\nFeatures flagged as drifted: {drifted_features}")


if __name__ == "__main__":
    main()
