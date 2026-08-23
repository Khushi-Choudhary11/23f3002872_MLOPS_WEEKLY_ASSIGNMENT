# Week 10 – From MLOps to LLMOps: Fine-Tuning Gemini on the IRIS Pipeline

**Name:** Khushi Choudhary
**IITM BS Roll Number:** 23f3002872
**Repository:** 23f3002872_MLOPS_WEEKLY_ASSIGNMENT
**Branch:** week_10

## Objective

Move from classical MLOps (own the training code, fit the model, version the artifact) to **LLMOps** (adapt a pretrained foundation model, version prompts + fine-tuning data + adapters, evaluate for format compliance as well as accuracy). Convert the IRIS dataset into two JSONL fine-tuning representations — raw feature values (v1) and natural-language descriptions (v2) — supervised-fine-tune `gemini-2.5-flash` on each via Vertex AI, then compare accuracy, per-class precision/recall, and output-format compliance to see whether the data representation changes how well an LLM performs a task a traditional classifier already solves easily.

## Pipeline / Architecture

```
data/raw/iris.csv (clean, DVC-tracked, unchanged)
        |
        v
prepare_llm_data.py                                  (Tasks 1 & 2)
  - stratified 60/20/20 split (90 train / 30 val / 30 test,
    30/10/10 per class, random_state=42)
  - v1 (raw):         input_text = "sepal_length: 5.1, sepal_width: 3.5, ..."
                       output_text = "setosa"
  - v2 (description):  input_text = "A flower specimen has a sepal length
                       of 5.1 cm, ... Identify the iris species."
                       output_text = "This is Iris setosa."
  - also emits a {"contents":[...]} chat-schema copy of train/validation,
    since Vertex AI's current Gemini SFT API takes that schema rather
    than the legacy PaLM-era input_text/output_text schema
  -> data/week10/{v1,v2}_{train,validation,test}.jsonl
  -> data/week10/{v1,v2}_{train,validation}_gemini.jsonl
        |
        v
upload_llm_data.py                                    (Tasks 1 & 2)
  - uploads all 10 files to gs://iris-data-23f3002872/week10/
        |
        v
finetune_vertex.py                                    (Task 3)
  - vertexai.tuning.sft.train() x2, source_model="gemini-2.5-flash",
    epochs=5, learning_rate_multiplier=1.0 (identical for both jobs —
    data representation is the only variable)
  -> data/week10/tuning_jobs.json (job + endpoint resource names)
        |
        v
evaluate_llm.py                                       (Task 4)
  - calls both tuned endpoints against their held-out 30-row test set
  - thinking disabled (thinking_budget=0) + version-specific system
    instruction, so the model's answer tokens aren't consumed by its
    internal reasoning pass
  - accuracy, per-class precision/recall (sklearn), format compliance
    (exact-format regex match), non-compliant/unparseable outputs
    scored as an incorrect "invalid" prediction, not dropped
  -> data/week10/evaluation_results.json, data/week10/llm_comparison.png

.github/workflows/llmops-eval.yml                      (Task 5, optional)
  - re-runs evaluate_llm.py on every push to week_10, fails the job if
    either version's accuracy drops below 0.8
```

## Task 1 & 2 — Prepare v1 (Raw) and v2 (Description) JSONL

`prepare_llm_data.py` reads `data/raw/iris.csv` (150 rows, balanced 50/class) and performs one stratified 60/20/20 split (train/validation/test), reused identically for both representations so v1 and v2 see the exact same rows in the exact same roles — the only difference between the two experiments is how each row is *phrased*, not which rows are used.

- **v1 (raw):** `{"input_text": "sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2", "output_text": "setosa"}`
- **v2 (description):** `{"input_text": "A flower specimen has a sepal length of 5.1 cm, sepal width of 3.5 cm, petal length of 1.4 cm, and petal width of 0.2 cm. Identify the iris species.", "output_text": "This is Iris setosa."}`

**Why a second `_gemini.jsonl` copy exists:** the assignment spec's `input_text`/`output_text` schema is the older Vertex AI (PaLM/text-bison) tuning format. The current Gemini supervised-tuning API instead consumes the same chat `contents`/`role`/`parts` schema as the Gemini API itself. `prepare_llm_data.py` keeps the `input_text`/`output_text` files as the primary deliverable (and what evaluation scores against), and mechanically derives a `{"contents": [...]}` copy of the train/validation splits purely to satisfy what Vertex AI's `tuningJobs.create` actually accepts today.

Uploaded to `gs://iris-data-23f3002872/week10/`.

## Task 3 — Fine-Tune Two Gemini Model Versions

`finetune_vertex.py` submits two `vertexai.tuning.sft.train()` jobs against `gemini-2.5-flash` (Vertex AI's current cost-efficient Gemini tier — `gemini-2.0-flash-001` was shut down 2026-06-01), both with `epochs=5`, `learning_rate_multiplier=1.0`:

| version | tuning job | tuned model / endpoint |
|---|---|---|
| v1 (raw) | `tuningJobs/4135243586885124096` | `endpoints/6566077282648064000` |
| v2 (description) | `tuningJobs/1457853608413364224` | `endpoints/6489938301447831552` |

Both jobs reached `JOB_STATE_SUCCEEDED`. Job configuration, state, and resulting endpoint/model resource names are recorded in `data/week10/tuning_jobs.json`. Both jobs are visible under **Vertex AI → Tuning** in the console (see screencast).

## Task 4 — Evaluate & Compare v1 vs v2

`evaluate_llm.py` calls each tuned endpoint against its own 30-row held-out test set (10/class), with `thinking_budget=0` (`gemini-2.5-flash` is a "thinking" model — its `max_output_tokens` budget covers thinking tokens *and* answer tokens together, and an ungated thinking pass alone can burn the whole budget before any answer text is emitted, which is exactly what happened in the first evaluation run: 0% accuracy, every response an empty/`MAX_TOKENS`-truncated candidate) and a version-specific system instruction matching each version's own output template (a shared "be terse" instruction was tried first and it pulled v2's fine-tuned model away from its trained full-sentence style toward v1's style, which would have silently confounded the comparison this task is designed to make).

**Results (n=30 per version, 10/class):**

| metric | v1 (raw) | v2 (description) |
|---|---|---|
| accuracy | **0.967** (29/30) | 0.900 (27/30) |
| format compliance | 1.000 | 1.000 |
| precision — setosa | 1.000 | 1.000 |
| recall — setosa | 1.000 | 1.000 |
| precision — versicolor | 0.909 | 0.769 |
| recall — versicolor | 1.000 | 1.000 |
| precision — virginica | 1.000 | 1.000 |
| recall — virginica | 0.900 | 0.700 |

`data/week10/llm_comparison.png` plots accuracy and format compliance side by side. Full per-row predictions are in `data/week10/evaluation_results.json`.

**Which version performs better, and why:** v1 (raw feature values) beats v2 (natural-language description) on accuracy (96.7% vs 90.0%) and on every class's precision/recall except identical setosa/versicolor-recall ties. Every single error in both versions is the same failure mode — a virginica sample misclassified as versicolor — which is the textbook IRIS boundary case (virginica and versicolor overlap in petal measurements at the low end of virginica's range); v2 just makes that boundary error 3 times instead of v1's 1. The likely reason: v1's `sepal_length: 5.1, sepal_width: 3.5, ...` format keeps the four numeric feature values in a fixed, minimal, consistently-ordered position relative to their labels — closer to a structured key-value record than to prose — while v2 embeds the same four numbers inside a longer natural-language sentence, adding more tokens (units, filler words, sentence structure) between the model and the numeric signal it actually needs to threshold on. For a task that is *purely numeric-threshold classification* with no ambiguity or world knowledge for the LLM to exploit, the extra natural-language scaffolding in v2 is pure overhead rather than a help — it doesn't give the model anything to reason about that the raw numbers didn't already convey, and on 90 training examples (30/class) that overhead measurably costs accuracy at the decision boundary. **Format compliance was a perfect 1.000 for both** once thinking was disabled and each version was prompted with its own matching system instruction, so on this task format compliance itself doesn't distinguish the two representations — the accuracy gap is the real signal, and it favors raw structured input over natural-language description for a task this numeric.

## Task 5 (Optional) — Automated Evaluation Gate in CI

`.github/workflows/llmops-eval.yml` reruns `evaluate_llm.py --fail-below 0.8` on every push to `week_10`, using the same `GCP_SA_KEY` secret already configured for `ci.yml`/`cd.yml`. It calls both live tuned endpoints (resource names come from the committed `data/week10/tuning_jobs.json`) and fails the job — acting as a regression guard — if either version's accuracy on the same held-out test set drops below 0.8. Results are uploaded as a workflow artifact (`evaluation_results.json`, `llm_comparison.png`) on every run, pass or fail.

---

## GCP Resources Used

- **Vertex AI Supervised Fine-Tuning:** two live tuning jobs against `gemini-2.5-flash` in `us-central1` (see Task 3 table), each producing a dedicated tuned-model endpoint queried directly in Task 4/5.
- **Cloud Storage:** `gs://iris-data-23f3002872/week10/` — all 10 JSONL files (train/validation/test x v1/v2, plus the `_gemini.jsonl` chat-schema copies used to submit the jobs).
- Note: `vertexai.tuning` / `vertexai.generative_models` (the SDK modules used here) are flagged deprecated as of 2025-06-24, scheduled for removal 2026-06-24, in favor of the newer `google-genai` client — still fully functional as of this assignment (Aug 2026) but worth migrating in a future pass.

## Files Added This Week

```
prepare_llm_data.py                       # Tasks 1 & 2: build v1/v2 JSONL splits
upload_llm_data.py                        # Tasks 1 & 2: upload JSONL to GCS
finetune_vertex.py                        # Task 3: submit both Vertex AI SFT jobs
wait_and_evaluate.py                      # polls both jobs, then runs evaluation
evaluate_llm.py                           # Task 4: accuracy/precision/recall/compliance
.github/workflows/llmops-eval.yml         # Task 5: CI regression gate on push to week_10
data/week10/v1_train.jsonl                # 90 rows, raw feature format
data/week10/v1_validation.jsonl           # 30 rows
data/week10/v1_test.jsonl                 # 30 rows (held out, used for evaluation)
data/week10/v1_train_gemini.jsonl         # v1 train, Gemini "contents" chat schema
data/week10/v1_validation_gemini.jsonl
data/week10/v2_train.jsonl                # 90 rows, natural-language description format
data/week10/v2_validation.jsonl           # 30 rows
data/week10/v2_test.jsonl                 # 30 rows (held out, used for evaluation)
data/week10/v2_train_gemini.jsonl         # v2 train, Gemini "contents" chat schema
data/week10/v2_validation_gemini.jsonl
data/week10/tuning_jobs.json              # tuning job + tuned endpoint resource names
data/week10/evaluation_results.json       # full per-row predictions + metrics, both versions
data/week10/llm_comparison.png            # accuracy / format-compliance bar chart
```

## How to Reproduce

```bash
dvc pull data/raw/iris.csv.dvc
python3 prepare_llm_data.py             # Tasks 1 & 2: build v1/v2 JSONL splits
python3 upload_llm_data.py              # Tasks 1 & 2: upload to GCS
python3 finetune_vertex.py              # Task 3: submit both tuning jobs (async)
python3 wait_and_evaluate.py            # poll until both jobs finish, then evaluate
# or, once data/week10/tuning_jobs.json already has both tuned_model_endpoint_name:
python3 evaluate_llm.py                 # Task 4: score both endpoints, write results + plot
python3 evaluate_llm.py --fail-below 0.8   # Task 5: same, but exits non-zero on regression
```

Screencast (not included in this repo) should show: both tuning jobs in **Vertex AI → Tuning** in the console (Task 3), and the v1-vs-v2 evaluation results with an explanation of why raw feature values outperformed the natural-language description on this task (Task 4).
