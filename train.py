"""
IRIS training pipeline (two-bucket setup).

  data bucket      : gs://iris-data-23f3002872
  artifacts bucket : gs://iris-artifacts-23f3002872

Flow:
  1. Download chosen data version from the DATA bucket.
  2. Train a DecisionTree on a stratified train split.
  3. Save model.joblib + metrics + eval set + log locally.
  4. Upload all of them to the ARTIFACTS bucket under artifacts/<timestamp>/.

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
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

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

    # ---- 2. Split + train ----
    train, test = train_test_split(
        data, test_size=args.test_size, stratify=data[TARGET], random_state=42
    )
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model = DecisionTreeClassifier(max_depth=args.max_depth, random_state=1)
    model.fit(X_train, y_train)
    acc = metrics.accuracy_score(y_test, model.predict(X_test))
    logger.info("Holdout accuracy: %.3f", acc)

    # ---- 3. Save artifacts locally ----
    local_dir = tempfile.mkdtemp()
    model_path = os.path.join(local_dir, "model.joblib")
    joblib.dump(model, model_path)

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
        ("model.joblib", model_path),
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
