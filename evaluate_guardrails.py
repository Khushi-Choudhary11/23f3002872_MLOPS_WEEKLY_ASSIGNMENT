"""Week 11 Task 5: measure guardrail effectiveness (before vs. after).

Re-runs the Task 1 (injection) and Task 2 (leakage) adversarial suites, plus
the Week 10 held-out test sets (data/week10/{v1,v2}_test.jsonl, 30 rows
each), through the guarded pipeline (InputGuardrail -> model ->
OutputGuardrail) and reports:

  - injection_block_rate      -- fraction of Task 1 attacks intercepted
  - leakage_block_rate        -- fraction of Task 2 probes intercepted
  - false_positive_rate       -- fraction of legitimate held-out inputs
                                  incorrectly blocked/filtered
  - accuracy_delta            -- Week 10 accuracy minus guarded-pipeline
                                  accuracy on the same held-out set, per version
"""
import argparse
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from guardrails import GuardedPredictor
from redteam_injection import ATTACKS as INJECTION_ATTACKS
from redteam_leakage import PROBES as LEAKAGE_PROBES

WEEK10_DIR = pathlib.Path("data/week10")
OUT_DIR = pathlib.Path("data/week11")
RESULTS_PATH = OUT_DIR / "guardrail_evaluation.json"
PLOT_PATH = OUT_DIR / "guardrail_metrics.png"

EXTRACT_RE = re.compile(r"(setosa|versicolor|virginica)", re.IGNORECASE)


def load_week10_accuracy() -> dict:
    results = json.loads((WEEK10_DIR / "evaluation_results.json").read_text())
    return {v: results[v]["accuracy"] for v in ("v1", "v2")}


def load_held_out(version: str) -> pd.DataFrame:
    records = [json.loads(line) for line in (WEEK10_DIR / f"{version}_test.jsonl").open()]
    return pd.DataFrame(records)


def run_adversarial_suite(predictor: GuardedPredictor, prompts: list, category: str) -> list[dict]:
    rows = []
    for prompt in prompts:
        for version in ("v1", "v2"):
            result = predictor.predict(version, prompt["input_prompt"])
            blocked = result["status"] != "ok"
            rows.append(
                {
                    "category": category,
                    "attack_pattern": prompt["attack_pattern"],
                    "model_version": version,
                    "status": result["status"],
                    "blocked": blocked,
                    "final_response": result["final_response"],
                }
            )
    return rows


def run_legitimate_suite(predictor: GuardedPredictor) -> tuple[list[dict], dict]:
    rows = []
    accuracy_guarded = {}
    for version in ("v1", "v2"):
        df = load_held_out(version)
        correct = 0
        for row in df.itertuples():
            expected_match = EXTRACT_RE.search(row.output_text)
            expected = expected_match.group(1).lower()
            result = predictor.predict(version, row.input_text)
            false_positive = result["status"] != "ok"

            predicted = None
            if result["final_response"]:
                match = EXTRACT_RE.search(result["final_response"])
                predicted = match.group(1).lower() if match else None
            if predicted == expected:
                correct += 1

            rows.append(
                {
                    "model_version": version,
                    "input_text": row.input_text,
                    "expected": expected,
                    "status": result["status"],
                    "false_positive": false_positive,
                    "final_response": result["final_response"],
                    "predicted": predicted,
                }
            )
        accuracy_guarded[version] = correct / len(df)
    return rows, accuracy_guarded


def plot_metrics(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    categories = ["Injection\nblock rate", "Leakage\nblock rate", "False positive\nrate"]
    values = [summary["injection_block_rate"], summary["leakage_block_rate"], summary["false_positive_rate"]]
    colors = ["#4C72B0", "#4C72B0", "#C44E52"]
    axes[0].bar(categories, values, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Guardrail Block Rate vs. False Positive Rate")

    versions = ["v1", "v2"]
    week10_acc = [summary["week10_accuracy"][v] for v in versions]
    guarded_acc = [summary["guarded_accuracy"][v] for v in versions]
    x = range(len(versions))
    width = 0.35
    axes[1].bar([i - width / 2 for i in x], week10_acc, width, label="Week 10 (ungated)", color="#4C72B0")
    axes[1].bar([i + width / 2 for i in x], guarded_acc, width, label="Guarded pipeline", color="#DD8452")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(versions)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy on Held-Out Set: Before vs. After Guardrails")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"wrote {PLOT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-below-block-rate", type=float, default=None,
                         help="Exit non-zero if injection or leakage block rate is below this threshold")
    parser.add_argument("--fail-above-fpr", type=float, default=None,
                         help="Exit non-zero if the false positive rate exceeds this threshold")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictor = GuardedPredictor()

    injection_rows = run_adversarial_suite(predictor, INJECTION_ATTACKS, "injection")
    leakage_rows = run_adversarial_suite(predictor, LEAKAGE_PROBES, "leakage")
    legit_rows, guarded_accuracy = run_legitimate_suite(predictor)

    injection_block_rate = sum(r["blocked"] for r in injection_rows) / len(injection_rows)
    leakage_block_rate = sum(r["blocked"] for r in leakage_rows) / len(leakage_rows)
    false_positive_rate = sum(r["false_positive"] for r in legit_rows) / len(legit_rows)
    week10_accuracy = load_week10_accuracy()
    accuracy_delta = {v: week10_accuracy[v] - guarded_accuracy[v] for v in ("v1", "v2")}

    summary = {
        "injection_block_rate": injection_block_rate,
        "leakage_block_rate": leakage_block_rate,
        "false_positive_rate": false_positive_rate,
        "week10_accuracy": week10_accuracy,
        "guarded_accuracy": guarded_accuracy,
        "accuracy_delta": accuracy_delta,
        "n_injection_attacks": len(injection_rows),
        "n_leakage_probes": len(leakage_rows),
        "n_legitimate_inputs": len(legit_rows),
    }

    RESULTS_PATH.write_text(json.dumps(
        {"summary": summary, "injection_results": injection_rows, "leakage_results": leakage_rows, "legitimate_results": legit_rows},
        indent=2,
    ))
    print(f"wrote {RESULTS_PATH}")

    print("\n=== Guardrail Effectiveness Summary ===")
    print(f"Injection block rate:  {injection_block_rate:.3f} ({sum(r['blocked'] for r in injection_rows)}/{len(injection_rows)})")
    print(f"Leakage block rate:    {leakage_block_rate:.3f} ({sum(r['blocked'] for r in leakage_rows)}/{len(leakage_rows)})")
    print(f"False positive rate:   {false_positive_rate:.3f} ({sum(r['false_positive'] for r in legit_rows)}/{len(legit_rows)})")
    for v in ("v1", "v2"):
        print(f"[{v}] accuracy week10={week10_accuracy[v]:.3f} guarded={guarded_accuracy[v]:.3f} delta={accuracy_delta[v]:+.3f}")

    plot_metrics(summary)

    if args.fail_below_block_rate is not None:
        if injection_block_rate < args.fail_below_block_rate:
            raise SystemExit(f"REGRESSION: injection block rate {injection_block_rate:.3f} < threshold {args.fail_below_block_rate}")
        if leakage_block_rate < args.fail_below_block_rate:
            raise SystemExit(f"REGRESSION: leakage block rate {leakage_block_rate:.3f} < threshold {args.fail_below_block_rate}")
    if args.fail_above_fpr is not None and false_positive_rate > args.fail_above_fpr:
        raise SystemExit(f"REGRESSION: false positive rate {false_positive_rate:.3f} > threshold {args.fail_above_fpr}")


if __name__ == "__main__":
    main()
