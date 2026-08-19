# Week 9 – Explainability, Fairness, and Drift in the IRIS Pipeline

**Name:** Khushi Choudhary
**IITM BS Roll Number:** 23f3002872
**Repository:** 23f3002872_MLOPS_WEEKLY_ASSIGNMENT
**Branch:** week_9

## Objective

Move beyond securing the pipeline (Week 8) to making it *trustworthy*: introduce a synthetic sensitive attribute (`location`) into the IRIS dataset, audit the trained classifier for bias across that attribute with **Fairlearn**, explain individual predictions with **SHAP**, detect **data drift** between training and a simulated production batch, and document accountability with a **model card**.

## Pipeline / Architecture

```
data/raw/iris.csv (clean, DVC-tracked, unchanged)
        |
        v
fairness_data.py                                    (Task 1)
  - adds `location` in {0,1}, assigned uniformly at random,
    independent of every feature and of the label
  - stratified 60/40 train/test split (random_state=42),
    `location` carried through the split but NEVER used as
    a training feature
  - trains DecisionTreeClassifier(max_depth=3, criterion=gini)
    on the 4 original features only
  - logs params/metrics/model to MLflow experiment
    "IRIS_Fairness_Explainability" (sqlite:///mlflow.db)
  - saves model -> models/week9_model.joblib
  - saves test predictions + location -> data/week9/test_predictions.csv
        |
        +--> fairness_analysis.py                    (Task 2)
        |      - Fairlearn MetricFrame(accuracy/precision/recall,
        |        sensitive_features=location) on test predictions
        |      - data/week9/fairlearn_metrics.csv, fairlearn_disparity.csv
        |
        +--> shap_analysis.py                         (Task 3)
        |      - shap.TreeExplainer over the full 150-row dataset
        |        (4 real features only, `location` excluded)
        |      - beeswarm summary plot per class (setosa/versicolor/virginica)
        |      - data/week9/shap_summary_{class}.png
        |
        +--> drift_detection.py                       (Task 4)
               - simulates a production batch: petal_length +1.5 offset,
                 petal_width x1.5 scale, sepal_* small Gaussian noise only
               - two-sample Kolmogorov-Smirnov test per feature vs. training
               - data/week9/drift_test_results.csv, drift_distributions.png

data/week9/model_card.md                              (Task 5, optional)
  - purpose, training data, by-group performance, limitations, fairness
```

## Task 1 — Introduce the `location` Attribute

`fairness_data.py` adds `location` (0/1) to every row via `numpy.random.RandomState(42).randint(0, 2, ...)` — independent of `sepal_*`, `petal_*`, and `species`. The full frame (features + `species` + `location`) is split together (`stratify=species`) so `location` stays attached to each row through the split, but only the 4 original features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) are passed to `DecisionTreeClassifier.fit()`. `location` is carried forward purely as a group label for the Task 2 audit — never as a model input.

**Why exclude it from training:** it was assigned at random, so it has zero true relationship with the label; feeding it in would only add noise to the split criterion. This mirrors a real deployment pattern — a protected attribute is often withheld from the feature set, but its absence from training does **not** by itself guarantee the model behaves equitably across groups, which is exactly what Task 2 checks.

Output: `data/week9/iris_with_location.csv` (150 rows, location roughly balanced 0/1); model saved to `models/week9_model.joblib`; run logged under MLflow experiment `IRIS_Fairness_Explainability`.

## Task 2 — Assess Fairness with Fairlearn

`fairness_analysis.py` loads `data/week9/test_predictions.csv` (y_true, y_pred, and `location` for each of the 60 test rows) and runs a Fairlearn `MetricFrame` with `sensitive_features=location`.

**Overall (n=60):** accuracy 0.983, precision (macro) 0.984, recall (macro) 0.983.

**By `location` group** (`data/week9/fairlearn_metrics.csv`):

| location | n | accuracy | precision | recall |
|---|---|---|---|---|
| 0 | 29 | 1.000 | 1.000 | 1.000 |
| 1 | 31 | 0.968 | 0.963 | 0.970 |

**Disparity** (`data/week9/fairlearn_disparity.csv`): difference (max−min) ≈ 0.030–0.037; ratio (min/max) ≈ 0.963–0.970 across all three metrics.

**Interpretation:** the two groups perform near-identically, as expected — `location` was assigned independently of the features and label, so any gap is sampling noise from the small (60-row) test set (2 misclassified rows happened to fall in group 1), not a real disparity. This confirms the audit workflow is sound: `MetricFrame` surfaces a gap when one exists, it does not by itself distinguish "real bias" from "noise," which is why the near-zero, near-random-assignment baseline here is the right sanity check before trusting the metric on a real sensitive attribute.

## Task 3 — SHAP Summary Plots and Explaining Virginica

`shap_analysis.py` builds one `shap.TreeExplainer` over the full 150-sample dataset (exact SHAP values for a decision tree) and produces a beeswarm summary plot per class: `data/week9/shap_summary_setosa.png`, `shap_summary_versicolor.png`, `shap_summary_virginica.png`.

Mean |SHAP| per feature, class = virginica: `petal_length` 0.336, `petal_width` 0.109, `sepal_width`/`sepal_length` ≈ 0.

**Plain-language reading of the virginica plot:**
- **Pushes toward virginica:** high `petal_length` and high `petal_width` (red dots clustered on the positive/right side, SHAP ≈ +0.13 to +0.55). Large petals are the model's strongest evidence for virginica.
- **Pushes away from virginica:** low `petal_length` and low `petal_width` (blue dots clustered on the negative/left side, SHAP ≈ −0.25 to −0.30). Small petals strongly argue against virginica (these are the setosa/versicolor samples).
- **Right vs. left side:** the x-axis is the SHAP value for that sample-feature pair. Right of zero means that feature's actual value increased the model's confidence in virginica for that sample; left of zero means it decreased it.
- **Dot color:** red = the feature's value was high for that sample, blue = it was low, for that feature's own range. So "a red cluster on the right" (seen here for both petal features) means *high values of that feature are strong, consistent evidence for virginica* — the classic signature of a genuinely discriminative feature.
- `sepal_length`/`sepal_width` sit essentially on the zero line for every sample — the tree (max_depth=3) never actually splits on them for this class, so they carry no explanatory weight for virginica.

## Task 4 — Detect Data Drift

`drift_detection.py` simulates a production batch by shifting two of the four feature distributions and leaving the other two as near-identical (small-noise) controls:

- `petal_length`: **+1.5 constant offset** (e.g., a later growth stage or sensor recalibration) → mean shifts, shape unchanged.
- `petal_width`: **×1.5 multiplicative scale** (e.g., a unit/instrument change) → both mean and spread shift.
- `sepal_length`, `sepal_width`: small Gaussian measurement noise only (σ=0.05) → no systematic change.

Each reference-vs-production feature pair is compared with a two-sample **Kolmogorov–Smirnov** test (α=0.05) — a distribution-only, model-free way to detect *data* drift:

| feature | ks_statistic | p-value | drifted? |
|---|---|---|---|
| sepal_length | 0.053 | 0.984 | No |
| sepal_width | 0.100 | 0.443 | No |
| petal_length | 0.413 | 7.4e-12 | **Yes** |
| petal_width | 0.373 | 1.1e-09 | **Yes** |

Distribution overlays: `data/week9/drift_distributions.png`. Full results: `data/week9/drift_test_results.csv`.

**What this means for the deployed model:** `petal_length` and `petal_width` are exactly the two features the Task 3 SHAP analysis showed the model relies on almost entirely (combined mean |SHAP| ≫ the sepal features, for every class). A production stream whose petal measurements are shifted this far outside the training range would push samples into regions of feature space the `max_depth=3` decision boundaries were never fit against — accuracy on this stream cannot be assumed to match the reported test accuracy. This test only confirms **data drift** (the *input* distributions changed); it says nothing about whether the input→output relationship itself changed (**concept drift**), which would require scoring the model's predictions against ground-truth labels collected from production — statistical distribution tests alone cannot detect that.

## Task 5 (Optional) — Model Card

A structured model card covering intended use, training data, overall and by-group performance, SHAP-based explainability summary, drift findings, limitations, and fairness considerations is at [`data/week9/model_card.md`](data/week9/model_card.md).

---

## GCP Resources Used

No new GCP resources were required for this assignment — Tasks 1–5 run entirely on locally/DVC-tracked data (`data/raw/iris.csv`) and log to the existing local MLflow store (`mlflow.db`). No BigQuery, Feast, GKE, or Artifact Registry changes were made this week.

## Files Added This Week

```
fairness_data.py                     # Task 1: adds `location`, trains baseline model, logs to MLflow
fairness_analysis.py                 # Task 2: Fairlearn MetricFrame audit by location
shap_analysis.py                     # Task 3: SHAP beeswarm summaries, all 3 classes
drift_detection.py                   # Task 4: simulated production drift + KS tests
data/week9/iris_with_location.csv    # IRIS + location attribute
data/week9/test_predictions.csv      # test-set y_true/y_pred + location
data/week9/fairlearn_metrics.csv     # per-group accuracy/precision/recall
data/week9/fairlearn_disparity.csv   # difference/ratio across groups
data/week9/shap_summary_setosa.png
data/week9/shap_summary_versicolor.png
data/week9/shap_summary_virginica.png
data/week9/production_simulated.csv  # simulated drifted production batch
data/week9/drift_test_results.csv    # KS-test results per feature
data/week9/drift_distributions.png   # reference vs. production distribution overlays
data/week9/model_card.md             # Task 5: model card
models/week9_model.joblib            # trained DecisionTreeClassifier
```

MLflow tracking store: `mlflow.db` (sqlite, repo root, shared with prior weeks — experiment `IRIS_Fairness_Explainability` is separate from earlier experiments).

## How to Reproduce

```bash
dvc pull data/raw/iris.csv.dvc
python3 fairness_data.py        # Task 1: adds location, trains model, logs to MLflow
python3 fairness_analysis.py    # Task 2: Fairlearn fairness audit
python3 shap_analysis.py        # Task 3: SHAP summary plots
python3 drift_detection.py      # Task 4: data drift detection
mlflow ui --backend-store-uri sqlite:///mlflow.db   # to browse the run
```
