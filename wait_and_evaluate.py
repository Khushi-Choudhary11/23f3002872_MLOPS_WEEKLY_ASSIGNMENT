"""Poll both Vertex AI tuning jobs until they finish, then run evaluation."""
import json
import pathlib
import subprocess
import time

import vertexai
from vertexai.tuning import sft

PROJECT = "mlops-project-500020"
LOCATION = "us-central1"
JOBS_PATH = pathlib.Path("data/week10/tuning_jobs.json")


def main() -> None:
    vertexai.init(project=PROJECT, location=LOCATION)
    results = json.loads(JOBS_PATH.read_text())
    jobs = {v: sft.SupervisedTuningJob(r["resource_name"]) for v, r in results.items()}

    pending = set(jobs)
    while pending:
        for version in list(pending):
            job = jobs[version]
            job.refresh()
            print(f"[{version}] state={job.state.name}", flush=True)
            results[version]["state"] = job.state.name
            if job.has_ended:
                pending.discard(version)
                if job.has_succeeded:
                    results[version]["tuned_model_endpoint_name"] = job.tuned_model_endpoint_name
                    results[version]["tuned_model_name"] = job.tuned_model_name
                else:
                    results[version]["error"] = str(getattr(job, "error", "unknown"))
        JOBS_PATH.write_text(json.dumps(results, indent=2))
        if pending:
            time.sleep(60)

    print("both tuning jobs finished:", json.dumps(results, indent=2))

    if all(r["state"] == "JOB_STATE_SUCCEEDED" for r in results.values()):
        subprocess.run(["python3", "evaluate_llm.py"], check=True)
    else:
        print("skipping evaluation: at least one tuning job did not succeed")


if __name__ == "__main__":
    main()
