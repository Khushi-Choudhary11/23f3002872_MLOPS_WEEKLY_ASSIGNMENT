# IRIS ML Pipeline on Vertex AI — Week 1 MLOps Assignment

**Author:** Khushi Choudhary
**Roll Number:** 23f3002872
**Course:** MLOps (IIT Madras BS)

An end-to-end machine learning pipeline for the classic IRIS classification problem, built on Google Cloud Platform. Data and model artifacts are managed through Google Cloud Storage (GCS), and the pipeline runs on a Vertex AI Workbench instance.

The pipeline is intentionally minimal — fetch data from the cloud, train a model, and store artifacts back to the cloud — so it can serve as the foundation for adding MLOps tooling (DVC, Feast, experiment tracking, CI/CD) in later weeks.

---

## Architecture

```
Vertex AI Workbench  ──►  reads data from   ──►  gs://iris-data-23f3002872
        │
        │  train.py / infer.py
        ▼
Decision Tree model  ──►  writes artifacts  ──►  gs://iris-artifacts-23f3002872
                                                  └── artifacts/<timestamp>/
```

Two separate GCS buckets are used:

| Bucket | Purpose |
|--------|---------|
| `iris-data-23f3002872` | Stores the dataset (raw, v1, v2 versions) |
| `iris-artifacts-23f3002872` | Stores model + logs + predictions, organized by run timestamp |

---

## Dataset

R.A. Fisher's IRIS dataset — four numerical features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) and a `species` target (setosa, versicolor, virginica).

Three versions live in `data/` and are uploaded to the data bucket:

| Version | Path in bucket | Rows |
|---------|----------------|------|
| `raw` | `data/raw/iris.csv` | 150 |
| `v1`  | `data/v1/data.csv`  | 101 |
| `v2`  | `data/v2/data.csv`  | 49  |

---

## Repository structure

```
.
├── train.py          # fetch data from GCS, train, upload timestamped artifacts
├── infer.py          # fetch a trained model from GCS, predict on the eval set
├── requirements.txt  # Python dependencies
├── data/
│   ├── raw/iris.csv
│   ├── v1/data.csv
│   └── v2/data.csv
└── README.md
```

---

## How it works

### `train.py`
1. Downloads the chosen data version from the **data bucket**.
2. Splits the data into stratified train/eval sets (`test_size=0.4`, `random_state=42`).
3. Trains a `DecisionTreeClassifier` (`max_depth=3`).
4. Uploads the following to `gs://iris-artifacts-23f3002872/artifacts/<timestamp>/`:
   - `model.joblib` — the trained model
   - `metrics.json` — run metadata and accuracy
   - `eval.csv` — the held-out evaluation set (so inference reuses the exact split)
   - `train.log` — the training log

Each run gets its own UTC-timestamped folder, making runs traceable and comparable.

### `infer.py`
1. Downloads `model.joblib` and `eval.csv` from a chosen timestamp folder in the **artifacts bucket**.
2. Runs predictions on the evaluation set.
3. Writes `predictions.csv` back into the same folder.

---

## Setup

```bash
pip install -r requirements.txt
```

Authentication is handled automatically inside Vertex AI Workbench via the instance's service account — no key files needed.

---

## Usage

### Training

```bash
python train.py \
  --data-bucket iris-data-23f3002872 \
  --artifact-bucket iris-artifacts-23f3002872 \
  --data-version raw
```

The script prints an `ARTIFACT_PREFIX` line, e.g. `artifacts/2026-06-21T07-37-58`.

### Inference

Pass the prefix from the training run:

```bash
python infer.py \
  --artifact-bucket iris-artifacts-23f3002872 \
  --artifact-prefix artifacts/2026-06-21T07-37-58
```

---

## Results

The pipeline was executed multiple times across data versions. Accuracy on the held-out eval set:

| Data version | Rows | Holdout accuracy |
|--------------|------|------------------|
| `raw` (run 1) | 150 | 0.983 |
| `raw` (run 2) | 150 | 0.983 |
| `v1` | 101 | 0.951 |
| `v2` | 49  | 1.000 |

The `raw` runs produce two independent timestamped artifact folders (demonstrating reproducible, traceable runs). The `v1` / `v2` runs show how results vary across data versions — accuracy on the smaller `v2` subset reaches 1.0, while the larger `v1` sits lower at 0.951.

---

## Tasks completed

- **Task 1** — Activated GCP, set up Vertex AI Workbench, enabled required APIs.
- **Task 2** — Created GCS buckets and uploaded the IRIS dataset.
- **Task 3** — Executed the training pipeline; artifacts stored in timestamped GCS folders.
- **Task 4** — Separate inference script fetches the trained model from GCS and predicts on the eval set.
- **Task 5** — Ran training + inference twice → two separate timestamped artifact folders.
- **Task 6 (optional)** — Ran the pipeline across `v1` and `v2` data versions and compared results.