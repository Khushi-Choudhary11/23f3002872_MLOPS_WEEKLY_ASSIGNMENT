"""Week 10 Task 4: evaluate both fine-tuned Gemini models on the held-out test set.

Computes, per version (v1 raw / v2 description):
  - accuracy
  - per-class precision / recall
  - format compliance rate (did the model output a parseable, exactly-formatted
    valid species name, vs. a malformed response)

Malformed/non-compliant responses are scored as an incorrect prediction (a
distinct "invalid" label) rather than dropped, so they still count against
accuracy and against the recall of whatever the true class was.
"""
import argparse
import json
import pathlib
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import vertexai
from sklearn.metrics import classification_report
from vertexai.generative_models import GenerativeModel

PROJECT = "mlops-project-500020"
LOCATION = "us-central1"
DATA_DIR = pathlib.Path("data/week10")
JOBS_PATH = DATA_DIR / "tuning_jobs.json"
RESULTS_PATH = DATA_DIR / "evaluation_results.json"
PLOT_PATH = DATA_DIR / "llm_comparison.png"

VALID_CLASSES = ["setosa", "versicolor", "virginica"]

# v1 compliance: response is *exactly* one of the three class names, nothing else.
V1_COMPLIANT_RE = re.compile(r"^(setosa|versicolor|virginica)$")
# v2 compliance: response is *exactly* the expected sentence template.
V2_COMPLIANT_RE = re.compile(r"^This is Iris (setosa|versicolor|virginica)\.$")
# best-effort class extraction for accuracy scoring, independent of strict compliance
EXTRACT_RE = re.compile(r"(setosa|versicolor|virginica)", re.IGNORECASE)

GENERATION_CONFIG = {
    "temperature": 0.0,
    "max_output_tokens": 64,
    # gemini-2.5-flash defaults to a "thinking" pass before answering; a
    # classification task this small needs the visible-answer tokens, not the
    # thinking budget, so thinking is turned off at inference time.
    "thinking_config": {"thinking_budget": 0},
}

SYSTEM_INSTRUCTIONS = {
    "v1": "Respond with only the species name: setosa, versicolor, or virginica. No other words.",
    "v2": "Respond with exactly one sentence in the form: This is Iris <species>.",
}


def load_test_set(version: str) -> pd.DataFrame:
    records = [json.loads(line) for line in (DATA_DIR / f"{version}_test.jsonl").open()]
    return pd.DataFrame(records)


def predict(model: GenerativeModel, prompt: str) -> str:
    for attempt in range(3):
        try:
            response = model.generate_content(prompt, generation_config=GENERATION_CONFIG)
            return response.text.strip()
        except Exception as exc:  # malformed/blocked/empty responses count as non-compliant
            if attempt == 2:
                return f"__ERROR__: {exc}"
            time.sleep(2)
    return "__ERROR__"


def score(version: str, df: pd.DataFrame, raw_responses: list) -> dict:
    compliant_re = V1_COMPLIANT_RE if version == "v1" else V2_COMPLIANT_RE

    y_true, y_pred, compliant_flags = [], [], []
    for row, raw in zip(df.itertuples(), raw_responses):
        true_label = re.search(EXTRACT_RE, row.output_text).group(1).lower()
        is_compliant = bool(compliant_re.match(raw.strip()))
        match = EXTRACT_RE.search(raw)
        pred_label = match.group(1).lower() if match else "invalid"

        y_true.append(true_label)
        y_pred.append(pred_label)
        compliant_flags.append(is_compliant)

    report = classification_report(
        y_true, y_pred, labels=VALID_CLASSES, output_dict=True, zero_division=0
    )
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    compliance_rate = sum(compliant_flags) / len(compliant_flags)

    return {
        "version": version,
        "n_test": len(y_true),
        "accuracy": accuracy,
        "format_compliance_rate": compliance_rate,
        "per_class": {c: report[c] for c in VALID_CLASSES},
        "predictions": [
            {"input_text": row.input_text, "expected": t, "raw_response": r, "predicted": p, "compliant": c}
            for row, t, r, p, c in zip(df.itertuples(), y_true, raw_responses, y_pred, compliant_flags)
        ],
    }


def plot_comparison(results: dict) -> None:
    versions = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(versions, [results[v]["accuracy"] for v in versions], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Accuracy")
    axes[0].set_ylim(0, 1)

    axes[1].bar(versions, [results[v]["format_compliance_rate"] for v in versions], color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Format Compliance Rate")
    axes[1].set_ylim(0, 1)

    fig.suptitle("v1 (raw features) vs v2 (natural language description)")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"wrote {PLOT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-below", type=float, default=None, help="Exit non-zero if either version's accuracy is below this threshold (for CI gating)")
    args = parser.parse_args()

    vertexai.init(project=PROJECT, location=LOCATION)
    jobs = json.loads(JOBS_PATH.read_text())

    results = {}
    for version in ["v1", "v2"]:
        endpoint = jobs[version]["tuned_model_endpoint_name"]
        model = GenerativeModel(endpoint, system_instruction=[SYSTEM_INSTRUCTIONS[version]])
        df = load_test_set(version)
        raw_responses = [predict(model, row.input_text) for row in df.itertuples()]
        results[version] = score(version, df, raw_responses)
        print(f"[{version}] accuracy={results[version]['accuracy']:.3f} "
              f"compliance={results[version]['format_compliance_rate']:.3f}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"wrote {RESULTS_PATH}")
    plot_comparison(results)

    if args.fail_below is not None:
        for version, r in results.items():
            if r["accuracy"] < args.fail_below:
                raise SystemExit(
                    f"REGRESSION: {version} accuracy {r['accuracy']:.3f} < threshold {args.fail_below}"
                )


if __name__ == "__main__":
    main()
