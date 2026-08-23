"""Week 10: upload the v1/v2 JSONL fine-tuning files to GCS."""
import argparse
import pathlib

from google.cloud import storage

LOCAL_DIR = pathlib.Path("data/week10")
FILES = [
    "v1_train.jsonl", "v1_validation.jsonl", "v1_test.jsonl",
    "v1_train_gemini.jsonl", "v1_validation_gemini.jsonl",
    "v2_train.jsonl", "v2_validation.jsonl", "v2_test.jsonl",
    "v2_train_gemini.jsonl", "v2_validation_gemini.jsonl",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="iris-data-23f3002872")
    parser.add_argument("--prefix", default="week10")
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    for name in FILES:
        local_path = LOCAL_DIR / name
        blob_path = f"{args.prefix}/{name}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(local_path))
        print(f"uploaded gs://{args.bucket}/{blob_path}")


if __name__ == "__main__":
    main()
