"""
Task 1 -- Introduce the `location` sensitive attribute and train the
baseline IRIS classifier.

`location` is assigned uniformly at random (0/1) per sample, independent of
every real feature and of the label. It is stored alongside the dataset so
it can be used later to *audit* the model (Task 2, Fairlearn), but it is
never passed to the model as a training feature: since it carries no real
signal, including it would only add noise, and the point of the exercise
is to demonstrate auditing a sensitive attribute that was deliberately
excluded from training (mirroring how real deployments often exclude a
protected attribute from the feature set but still must audit against it).

Model architecture matches the rest of the repo (DecisionTreeClassifier,
max_depth=3, criterion=gini, random_state=42, test_size=0.4) so results
are directly comparable to earlier weeks.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = Path("data/week9")
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
SENSITIVE = "location"

MODEL_PARAMS = {"max_depth": 3, "criterion": "gini", "random_state": 42}
TEST_SIZE = 0.4
SEED = 42


def add_location(df: pd.DataFrame) -> pd.DataFrame:
    """Randomly assign each sample to location group 0 or 1, independent
    of features/label. Fixed seed for reproducibility."""
    rng = np.random.RandomState(SEED)
    df = df.copy()
    df[SENSITIVE] = rng.randint(0, 2, size=len(df))
    return df


def main():
    raw = pd.read_csv("data/raw/iris.csv")
    df = add_location(raw)
    df.to_csv(DATA_DIR / "iris_with_location.csv", index=False)
    print(f"Added '{SENSITIVE}' attribute -> {DATA_DIR / 'iris_with_location.csv'}")
    print(df[SENSITIVE].value_counts().rename("count"))

    # Split the whole frame (not just X/y) so `location` stays attached to
    # each row through the split -- it is not used as a training feature,
    # only carried along for the Task 2 fairness audit.
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df[TARGET]
    )

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IRIS_Fairness_Explainability")

    with mlflow.start_run(run_name="baseline_with_location_audit"):
        mlflow.log_param("max_depth", MODEL_PARAMS["max_depth"])
        mlflow.log_param("criterion", MODEL_PARAMS["criterion"])
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", SEED)
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("sensitive_attribute", SENSITIVE)
        mlflow.log_param("sensitive_attribute_used_in_training", False)

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

        print(f"\nOverall test performance: accuracy={acc:.4f} precision={prec:.4f} "
              f"recall={rec:.4f} f1={f1:.4f}")

    joblib.dump(model, MODELS_DIR / "week9_model.joblib")
    print(f"Model saved -> {MODELS_DIR / 'week9_model.joblib'}")

    # Persist test-set predictions + location for the Task 2 fairness audit.
    out = test_df[FEATURES + [TARGET, SENSITIVE]].copy()
    out["y_true"] = y_test.values
    out["y_pred"] = preds
    out.to_csv(DATA_DIR / "test_predictions.csv", index=False)
    print(f"Test predictions (with location) saved -> {DATA_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
