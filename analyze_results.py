"""
Pull the four Week 8 poisoning runs back out of MLflow and analyze them.

All numbers here come from mlflow.search_runs() against the actual
sqlite-backed experiment produced by train_mlflow.py -- nothing in this
script's output is hand-typed or assumed.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

EXPERIMENT = "IRIS_Poisoning_MLSecOps"
DATA_DIR = Path("data")
RANDOM_BASELINE = 1 / 3  # 3-class balanced problem


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise SystemExit(f"Experiment '{EXPERIMENT}' not found -- run train_mlflow.py first.")

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    runs["poisoning_level"] = runs["params.poisoning_level"].astype(int)
    runs = runs.sort_values("poisoning_level").reset_index(drop=True)

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    table = runs[["poisoning_level"] + [f"metrics.{m}" for m in metrics]].copy()
    table.columns = ["poisoning_level"] + metrics

    baseline = table.iloc[0]
    table_delta = table.copy()
    for m in metrics:
        table_delta[f"{m}_drop_from_baseline"] = (baseline[m] - table[m]).round(4)

    print("=== MLflow run comparison (actual results) ===")
    print(table.to_string(index=False))
    print("\n=== Drop from 0% baseline ===")
    print(table_delta[["poisoning_level"] + [f"{m}_drop_from_baseline" for m in metrics]].to_string(index=False))

    comparison_path = DATA_DIR / "mlflow_comparison.csv"
    table_delta.to_csv(comparison_path, index=False)
    print(f"\nComparison table written -> {comparison_path}")

    # ---- data-driven answers (computed, not assumed) ----
    print("\n=== Data-driven observations ===")
    for m in metrics:
        deltas = (baseline[m] - table[m]).values
        first_drop_idx = next((i for i, d in enumerate(deltas) if i > 0 and d > 0.01), None)
        if first_drop_idx is not None:
            lvl = table.iloc[first_drop_idx]["poisoning_level"]
            print(f"{m}: first noticeable drop (>1pt) at poisoning_level={int(lvl)}% "
                  f"(baseline={baseline[m]:.4f} -> {table.iloc[first_drop_idx][m]:.4f})")
        else:
            print(f"{m}: no drop >1pt observed across tested levels")

    worst_row = table.loc[table["poisoning_level"] == table["poisoning_level"].max()]
    worst_acc = float(worst_row["accuracy"].iloc[0])
    print(f"\nAt {int(worst_row['poisoning_level'].iloc[0])}% poisoning: accuracy={worst_acc:.4f} "
          f"vs random-guess baseline={RANDOM_BASELINE:.4f} for a balanced 3-class problem.")
    print("-> model is STILL far above random chance" if worst_acc > RANDOM_BASELINE + 0.15
          else "-> model performance is close to random chance")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in metrics:
        ax.plot(table["poisoning_level"], table[m], marker="o", label=m)
    ax.axhline(RANDOM_BASELINE, color="gray", linestyle="--", linewidth=1, label="random guess (1/3)")
    ax.set_xlabel("Poisoning level (%)")
    ax.set_ylabel("Score")
    ax.set_title("IRIS classifier metrics vs. training-set poisoning level")
    ax.set_xticks(table["poisoning_level"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plot_path = DATA_DIR / "poisoning_impact.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot written -> {plot_path}")


if __name__ == "__main__":
    main()
