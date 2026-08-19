# Model Card — IRIS Species Classifier (Week 9)

## Model Details
- **Model type:** `DecisionTreeClassifier` (scikit-learn), `max_depth=3`, `criterion=gini`, `random_state=42`
- **Task:** Multi-class classification of iris species — `setosa`, `versicolor`, `virginica`
- **Training features (4):** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- **Excluded feature:** `location` (0/1) — carried in the dataset as a sensitive/group attribute for fairness auditing only; **never used as a model input**
- **Artifacts:** `models/week9_model.joblib`; MLflow run under experiment `IRIS_Fairness_Explainability` (tracking store: `mlflow.db`)
- **Code:** `fairness_data.py` (train), `fairness_analysis.py` (Fairlearn audit), `shap_analysis.py` (SHAP), `drift_detection.py` (drift)

## Intended Use
- **Primary use:** Educational/demonstration classifier for the IRIS dataset, used to practice explainability (SHAP), fairness auditing (Fairlearn), and drift monitoring workflows.
- **Not intended for:** Any production or decision-making use on real subjects. `location` here is a **synthetic, randomly-assigned** attribute (not a real demographic field) and is used purely to demonstrate the auditing methodology — it does not represent real-world location data or a validated protected attribute.

## Training Data
- **Source:** `data/raw/iris.csv` (classic 150-row Fisher IRIS dataset), versioned via DVC in earlier weeks.
- **Split:** 60/40 train/test, stratified by species, `random_state=42` (90 train / 60 test rows).
- **Sensitive attribute (`location`):** assigned uniformly at random (0/1) per row, independent of every feature and of the label (see `fairness_data.add_location`). Group sizes in the test set: location=0 → 29 rows, location=1 → 31 rows.

## Performance

**Overall (test set, n=60):**

| Metric | Value |
|---|---|
| Accuracy | 0.983 |
| Precision (macro) | 0.984 |
| Recall (macro) | 0.983 |

**By `location` group (Fairlearn `MetricFrame`, `data/week9/fairlearn_metrics.csv`):**

| location | n | Accuracy | Precision | Recall |
|---|---|---|---|---|
| 0 | 29 | 1.000 | 1.000 | 1.000 |
| 1 | 31 | 0.968 | 0.963 | 0.970 |

**Disparity (`data/week9/fairlearn_disparity.csv`):** max−min difference ≈ 0.030–0.037 across accuracy/precision/recall; min/max ratio ≈ 0.963–0.970. This gap is fully attributable to which two test-set errors happened to fall in group 1 (a 60-row test set), not to any real dependence on `location` — consistent with `location` being assigned uniformly at random and independently of the features/label.

## Explainability (SHAP)
Full-dataset `shap.TreeExplainer` beeswarm summaries for all three classes (`data/week9/shap_summary_*.png`). For **virginica**:
- `petal_length` and `petal_width` dominate (mean |SHAP| ≈ 0.336 and 0.109); `sepal_length`/`sepal_width` contribute ~0.
- High values (red dots) of petal_length/petal_width sit on the **positive** (right) side of the plot → push the prediction **toward** virginica. Low values (blue dots) sit on the **negative** (left) side → push the prediction **away from** virginica.
- Interpretation: the model has learned that virginica is essentially identified by having the largest petals of the three species; sepal measurements carry no discriminative weight for this class.

## Data & Concept Drift
Simulated a production batch (`drift_detection.py`) by adding a systematic offset to `petal_length` (+1.5) and rescaling `petal_width` (×1.5), while `sepal_length`/`sepal_width` received only small measurement noise. Two-sample Kolmogorov–Smirnov test vs. the training distribution (`data/week9/drift_test_results.csv`, α=0.05):

| feature | drifted? | p-value |
|---|---|---|
| sepal_length | No | 0.984 |
| sepal_width | No | 0.443 |
| petal_length | **Yes** | 7.4e-12 |
| petal_width | **Yes** | 1.1e-9 |

**Implication:** `petal_length` and `petal_width` are exactly the features the model relies on most (per SHAP). A model trained on the original distribution and served against this drifted production traffic would receive petal measurements systematically outside its training range, and its decision boundaries (fit at `max_depth=3` on the original scale) would no longer be reliable — this is *data* drift (input distributions shifted); confirming whether it becomes *concept* drift (whether accuracy actually degrades) requires ground-truth labels on the production batch, which this synthetic exercise does not have.

## Known Limitations
- Trained on only 150 total samples (90 train) — not representative of real-world data volume or diversity.
- `location` is synthetic and random; it demonstrates the auditing *methodology*, not a validated real-world fairness finding. A real sensitive attribute would need to be checked for correlation with the real features (proxy discrimination) before concluding "no bias."
- `max_depth=3` decision tree is intentionally shallow/interpretable; not tuned for maximum accuracy.
- No drift-triggered retraining pipeline exists yet — drift detection here is a one-off analysis, not a continuous monitor.

## Fairness Considerations
- `location` was deliberately excluded from training to avoid adding noise from a non-predictive random attribute, while still being retained for post-hoc auditing — mirroring how a real deployment might exclude a protected attribute from features but still audit predictions against it.
- The observed ~3 percentage-point gap between groups is within the noise expected from random group assignment on a 60-row test set and should **not** be interpreted as evidence of bias.
- If `location` were a real demographic attribute correlated with any of `sepal_length/sepal_width/petal_length/petal_width`, excluding it from training would **not** guarantee fairness — correlated features could still act as a proxy. This was not tested here since `location` is uncorrelated by construction.
