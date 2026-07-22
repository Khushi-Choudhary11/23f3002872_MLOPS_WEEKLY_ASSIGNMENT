"""
IRIS training pipeline (two-bucket setup).

  data bucket      : gs://iris-data-23f3002872
  artifacts bucket : gs://iris-artifacts-23f3002872

Flow:
  1. Download chosen data version from the DATA bucket.
  2. Train multiple Decision Tree models using different hyperparameters.
  3. Log parameters, metrics, and models to MLflow.
  4. Register the trained models in the MLflow Model Registry.
  5. Upload metrics, evaluation set, and logs to the ARTIFACTS bucket.
Usage:
  python train.py --data-bucket iris-data-23f3002872 \
                  --artifact-bucket iris-artifacts-23f3002872 \
                  --data-version raw
"""

import argparse
import datetime
import io
import json
import logging
import os
import tempfile

import joblib
import pandas as pd
from feast import FeatureStore
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import mlflow
import mlflow.sklearn

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-bucket", required=True)
    p.add_argument("--artifact-bucket", required=True)
    p.add_argument("--data-version", default="raw", help="raw | v1 | v2")
    p.add_argument("--test-size", type=float, default=0.4)
    p.add_argument("--max-depth", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    client = storage.Client()
    data_bucket = client.bucket(args.data_bucket)
    artifact_bucket = client.bucket(args.artifact_bucket)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IRIS_Experiment")

    run_ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    artifact_prefix = f"artifacts/{run_ts}"

    log_buffer = io.StringIO()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("train")
    logger.addHandler(logging.StreamHandler(log_buffer))

    logger.info("Run timestamp: %s", run_ts)
    logger.info("Data version: %s", args.data_version)

    # ---- 1. Fetch data from the DATA bucket ----
    if args.data_version == "raw":
        data_blob_path = "data/raw/iris.csv"
    else:
        data_blob_path = f"data/{args.data_version}/data.csv"
    logger.info("Fetching gs://%s/%s", args.data_bucket, data_blob_path)

    local_csv = os.path.join(tempfile.gettempdir(), "data.csv")
    data_bucket.blob(data_blob_path).download_to_filename(local_csv)
    data = pd.read_csv(local_csv)
    logger.info("Loaded %d rows", len(data))
    data["iris_id"] = range(len(data))
    data["event_timestamp"] = pd.Timestamp.now()
    store = FeatureStore(repo_path="feast_repo/feature_repo")
    logger.info("Connected to Feast Feature Store")
    entity_df = data[["iris_id", "event_timestamp"]]
    

    # ---- 2. Split + train ----
    historical_features = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "iris_features:sepal_length",
            "iris_features:sepal_width",
            "iris_features:petal_length",
            "iris_features:petal_width",
        ],
    ).to_df()
    logger.info(
    "Retrieved %d historical feature rows from Feast",
    len(historical_features),
)
    
    # Combine Feast features with the target label
    training_data = historical_features.merge(
    data[["iris_id", "event_timestamp", TARGET]],
    on="iris_id",
)
    
    train, test = train_test_split(
        training_data,
        test_size=args.test_size,
        stratify=training_data[TARGET],
        random_state=42,
    )
    
    X_train = train[FEATURES]
    y_train = train[TARGET]
    
    X_test = test[FEATURES]
    y_test = test[TARGET]

    configs = [
    
        {
            "max_depth":2,
            "criterion":"gini"
        },
    
        {
            "max_depth":3,
            "criterion":"entropy"
        },
    
        {
            "max_depth":5,
            "criterion":"gini"
        }
    
    ]
    
    for config in configs:
    
        with mlflow.start_run():
    
            mlflow.log_param(
                "max_depth",
                config["max_depth"]
            )
    
            mlflow.log_param(
                "criterion",
                config["criterion"]
            )
    
            mlflow.log_param(
                "test_size",
                args.test_size
            )
    
            mlflow.log_param(
                "data_version",
                args.data_version
            )
    
            model = DecisionTreeClassifier(
    
                max_depth=config["max_depth"],
    
                criterion=config["criterion"],
    
                random_state=1
    
            )
    
            model.fit(X_train,y_train)
    
            predictions=model.predict(X_test)
    
            acc=metrics.accuracy_score(
                y_test,
                predictions
            )
    
            precision=metrics.precision_score(
                y_test,
                predictions,
                average="weighted"
            )
    
            recall=metrics.recall_score(
                y_test,
                predictions,
                average="weighted"
            )
    
            f1=metrics.f1_score(
                y_test,
                predictions,
                average="weighted"
            )
    
            mlflow.log_metric(
                "accuracy",
                acc
            )
    
            mlflow.log_metric(
                "precision",
                precision
            )
    
            mlflow.log_metric(
                "recall",
                recall
            )
    
            mlflow.log_metric(
                "f1_score",
                f1
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="iris_classifier"
            )
            
            logger.info(
                "Depth=%d Criterion=%s Accuracy=%.3f",
                config["max_depth"],
                config["criterion"],
                acc
            )

    # ---- 3. Save artifacts locally ----
    local_dir = tempfile.mkdtemp()

    metrics_path = os.path.join(local_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "run_timestamp": run_ts,
                "data_version": args.data_version,
                "n_rows": int(len(data)),
                "test_size": args.test_size,
                "max_depth": args.max_depth,
                "accuracy": round(float(acc), 4),
            },
            f,
            indent=2,
        )

    eval_path = os.path.join(local_dir, "eval.csv")
    test.to_csv(eval_path, index=False)

    # ---- 4. Upload to the ARTIFACTS bucket ----
    for fname, fpath in [
        ("metrics.json", metrics_path),
        ("eval.csv", eval_path),
    ]:
        dest = f"{artifact_prefix}/{fname}"
        artifact_bucket.blob(dest).upload_from_filename(fpath)
        logger.info("Uploaded gs://%s/%s", args.artifact_bucket, dest)

    artifact_bucket.blob(f"{artifact_prefix}/train.log").upload_from_string(
        log_buffer.getvalue()
    )

    print(f"\nDONE. Artifacts at: gs://{args.artifact_bucket}/{artifact_prefix}/")
    print(f"ARTIFACT_PREFIX={artifact_prefix}")


if __name__ == "__main__":
    main()
