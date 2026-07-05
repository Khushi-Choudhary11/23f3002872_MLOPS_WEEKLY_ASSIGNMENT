"""
IRIS inference pipeline (two-bucket setup).

Flow:
  1. Download model.joblib + eval.csv from a chosen timestamp folder
     in the ARTIFACTS bucket.
  2. Run predictions on the eval set.
  3. Write predictions.csv back into the same folder.

Usage:
  python infer.py --artifact-bucket iris-artifacts-23f3002872 \
                  --artifact-prefix artifacts/2026-06-21T10-30-00
"""

import argparse
import os
import tempfile

import joblib
import pandas as pd
from feast import FeatureStore
from google.cloud import storage
from sklearn import metrics

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-bucket", required=True)
    p.add_argument(
        "--artifact-prefix",
        required=True,
        help="e.g. artifacts/2026-06-21T10-30-00 (timestamp folder from train.py)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    client = storage.Client()
    bucket = client.bucket(args.artifact_bucket)
    local_dir = tempfile.mkdtemp()

    # ---- 1. Fetch model + eval set from the ARTIFACTS bucket ----
    model_local = os.path.join(local_dir, "model.joblib")
    eval_local = os.path.join(local_dir, "eval.csv")
    bucket.blob(f"{args.artifact_prefix}/model.joblib").download_to_filename(model_local)
    bucket.blob(f"{args.artifact_prefix}/eval.csv").download_to_filename(eval_local)
    print(f"Fetched model from gs://{args.artifact_bucket}/{args.artifact_prefix}/")

    model = joblib.load(model_local)
    eval_df = pd.read_csv(eval_local)
    store = FeatureStore(repo_path="feast_repo/feature_repo")
    print("Connected to Feast Feature Store")

    # ---- 2. Inference ----
    predictions = []

    for _, row in eval_df.iterrows():
    
        online_features = store.get_online_features(
            features=[
                "iris_features:sepal_length",
                "iris_features:sepal_width",
                "iris_features:petal_length",
                "iris_features:petal_width",
            ],
            entity_rows=[
                {"iris_id": int(row["iris_id"])}
            ],
        ).to_dict()
    
        feature_vector = pd.DataFrame(
            {
                "sepal_length": [online_features["sepal_length"][0]],
                "sepal_width": [online_features["sepal_width"][0]],
                "petal_length": [online_features["petal_length"][0]],
                "petal_width": [online_features["petal_width"][0]],
            }
        )
    
        prediction = model.predict(feature_vector)[0]
        predictions.append(prediction)
    
    eval_df["predicted_species"] = predictions
    
    acc = metrics.accuracy_score(
        eval_df[TARGET],
        eval_df["predicted_species"],
    )
    
    print(f"Inference accuracy using Feast Online Store: {acc:.3f}")

    # ---- 3. Write predictions back to the ARTIFACTS bucket ----
    pred_local = os.path.join(local_dir, "predictions.csv")
    eval_df.to_csv(pred_local, index=False)
    dest = f"{args.artifact_prefix}/predictions.csv"
    bucket.blob(dest).upload_from_filename(pred_local)
    print(f"Wrote predictions to gs://{args.artifact_bucket}/{dest}")


if __name__ == "__main__":
    main()
