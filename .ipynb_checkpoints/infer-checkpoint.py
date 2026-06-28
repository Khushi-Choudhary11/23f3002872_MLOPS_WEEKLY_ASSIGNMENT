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

    # ---- 2. Inference ----
    eval_df["predicted_species"] = model.predict(eval_df[FEATURES])
    if TARGET in eval_df.columns:
        acc = metrics.accuracy_score(eval_df[TARGET], eval_df["predicted_species"])
        print(f"Inference accuracy on eval set: {acc:.3f}")

    # ---- 3. Write predictions back to the ARTIFACTS bucket ----
    pred_local = os.path.join(local_dir, "predictions.csv")
    eval_df.to_csv(pred_local, index=False)
    dest = f"{args.artifact_prefix}/predictions.csv"
    bucket.blob(dest).upload_from_filename(pred_local)
    print(f"Wrote predictions to gs://{args.artifact_bucket}/{dest}")


if __name__ == "__main__":
    main()
