"""Week 10 Task 3: submit two Vertex AI supervised fine-tuning jobs (v1, v2).

Same source model and same hyperparameters for both jobs -- the only
variable between them is the training data representation (raw feature
values vs. natural-language description).
"""
import argparse
import json
import pathlib
import time

import vertexai
from vertexai.tuning import sft

PROJECT = "mlops-project-500020"
LOCATION = "us-central1"
BUCKET = "iris-data-23f3002872"
PREFIX = "week10"
SOURCE_MODEL = "gemini-2.5-flash"
EPOCHS = 5
LEARNING_RATE_MULTIPLIER = 1.0
RESULTS_PATH = pathlib.Path("data/week10/tuning_jobs.json")


def submit(version: str) -> sft.SupervisedTuningJob:
    train_uri = f"gs://{BUCKET}/{PREFIX}/{version}_train_gemini.jsonl"
    val_uri = f"gs://{BUCKET}/{PREFIX}/{version}_validation_gemini.jsonl"
    job = sft.train(
        source_model=SOURCE_MODEL,
        train_dataset=train_uri,
        validation_dataset=val_uri,
        tuned_model_display_name=f"iris-{version}-{SOURCE_MODEL}",
        epochs=EPOCHS,
        learning_rate_multiplier=LEARNING_RATE_MULTIPLIER,
    )
    print(f"[{version}] submitted tuning job: {job.resource_name}")
    return job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true", help="Block until both jobs complete")
    args = parser.parse_args()

    vertexai.init(project=PROJECT, location=LOCATION)

    jobs = {"v1": submit("v1"), "v2": submit("v2")}

    results = {
        version: {
            "resource_name": job.resource_name,
            "source_model": SOURCE_MODEL,
            "epochs": EPOCHS,
            "learning_rate_multiplier": LEARNING_RATE_MULTIPLIER,
            "state": job.state.name,
        }
        for version, job in jobs.items()
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"wrote {RESULTS_PATH}")

    if args.wait:
        pending = set(jobs)
        while pending:
            time.sleep(60)
            for version in list(pending):
                job = jobs[version]
                job.refresh()
                print(f"[{version}] state={job.state.name}")
                if job.has_ended:
                    pending.discard(version)
                    results[version]["state"] = job.state.name
                    if job.has_succeeded:
                        results[version]["tuned_model_endpoint_name"] = job.tuned_model_endpoint_name
                        results[version]["tuned_model_name"] = job.tuned_model_name
                    RESULTS_PATH.write_text(json.dumps(results, indent=2))
        print("both tuning jobs finished")


if __name__ == "__main__":
    main()
