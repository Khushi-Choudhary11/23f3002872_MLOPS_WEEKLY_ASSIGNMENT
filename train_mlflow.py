"""
Train the IRIS classifier separately for each poisoning level and log
every run to MLflow so the four runs can be compared directly.

Model architecture, train/test split, and evaluation methodology are held
IDENTICAL across all four runs -- poisoning_level is the only variable
that changes between them (per Task 3's requirement).

Metric averaging: "macro" is used for precision/recall/F1. IRIS is a
balanced 3-class problem (30 train / 20 test samples per class from the
fixed split). Macro-average weights every class equally, so if poisoning
degrades one class's predictions more than another, that shows up
directly in the metric instead of being smoothed out by class support
weighting (which is what "weighted" averaging would do). Since the split
is class-balanced, macro and weighted are numerically close on the clean
run, but macro is the more honest choice for a poisoning-impact study.

Features come from the isolated Feast BigQuery source
(iris_features_poison, set up by setup_feast_poison.py) via a real
point-in-time get_historical_features() call per poisoning level. Labels
(species) are merged in locally, matching the pattern already used by the
existing train.py.
"""

import sys
from pathlib import Path

import mlflow
import pandas as pd
from feast import FeatureStore
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
from setup_feast_poison import POISON_LEVELS, snapshot_ts  # noqa: E402

DATA_DIR = Path("data")
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"

MODEL_PARAMS = {"max_depth": 3, "criterion": "gini", "random_state": 42}
TEST_SIZE = 0.4
SEED = 42


def train_variant_path(level):
    return DATA_DIR / "clean" / "train_0pct.csv" if level == 0 else DATA_DIR / "poisoned" / f"train_{level}pct.csv"


def load_split_ids():
    train_ids = set(pd.read_csv(DATA_DIR / "clean" / "train_0pct.csv")["iris_id"])
    test_ids = set(pd.read_csv(DATA_DIR / "clean" / "test.csv")["iris_id"])
    return train_ids, test_ids


def run_level(store, level, train_ids, test_ids):
    entity_df = pd.DataFrame({
        "iris_id": list(range(150)),
        "event_timestamp": pd.to_datetime(snapshot_ts(level), utc=True),
    })
    historical = store.get_historical_features(
        entity_df=entity_df,
        features=[f"iris_features_poison:{f}" for f in FEATURES],
    ).to_df()

    train_variant = pd.read_csv(train_variant_path(level))
    test_variant = pd.read_csv(DATA_DIR / "clean" / "test.csv")
    labels = pd.concat([train_variant[["iris_id", TARGET]], test_variant[["iris_id", TARGET]]], ignore_index=True)

    full = historical.merge(labels, on="iris_id")
    train_df = full[full.iris_id.isin(train_ids)].sort_values("iris_id")
    test_df = full[full.iris_id.isin(test_ids)].sort_values("iris_id")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    n_poisoned = int((train_variant["is_poisoned"]).sum()) if "is_poisoned" in train_variant else 0

    with mlflow.start_run(run_name=f"poison_{level}pct"):
        mlflow.log_param("poisoning_level", level)
        mlflow.log_param("max_depth", MODEL_PARAMS["max_depth"])
        mlflow.log_param("criterion", MODEL_PARAMS["criterion"])
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_poisoned_train_rows", n_poisoned)

        model = DecisionTreeClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = metrics.accuracy_score(y_test, preds)
        prec = metrics.precision_score(y_test, preds, average="macro", zero_division=0)
        rec = metrics.recall_score(y_test, preds, average="macro", zero_division=0)
        f1 = metrics.f1_score(y_test, preds, average="macro", zero_division=0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        print(f"poisoning_level={level:>3}%  n_poisoned_train={n_poisoned:>2}/{len(X_train)}  "
              f"accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")

    return {"poisoning_level": level, "n_poisoned_train": n_poisoned, "n_train": len(X_train),
            "n_test": len(X_test), "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IRIS_Poisoning_MLSecOps")

    store = FeatureStore(repo_path="feast_repo/feature_repo")
    train_ids, test_ids = load_split_ids()
    print(f"Fixed split reused across all runs: {len(train_ids)} train / {len(test_ids)} test entities")

    results = [run_level(store, level, train_ids, test_ids) for level in POISON_LEVELS]

    results_df = pd.DataFrame(results)
    out_path = DATA_DIR / "training_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults summary written -> {out_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
