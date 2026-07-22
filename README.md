# Week 5 - MLflow Integration for IRIS ML Pipeline

## Overview

This week's assignment extends the IRIS machine learning pipeline by integrating **MLflow** for experiment tracking and model management.

The pipeline performs hyperparameter tuning, logs experiments to MLflow, registers trained models in the MLflow Model Registry, and modifies the inference pipeline to fetch the latest registered model instead of relying on locally stored model artifacts.

The implementation was completed on **Google Cloud Platform (Vertex AI Workbench)** using **Google Cloud Storage**, **Feast Feature Store**, **MLflow**, **DVC**, and **GitHub Actions**.

---

# Objectives

- Integrate MLflow into the training pipeline
- Perform hyperparameter tuning
- Log parameters, metrics, and trained models
- Register models using MLflow Model Registry
- Remove model dependency from DVC
- Load registered models during inference
- Update CI workflow for MLflow integration

---

# Project Structure

```
.
├── data/
├── feast_repo/
├── tests/
├── .github/workflows/
│   └── ci.yml
├── train.py
├── infer.py
├── requirements.txt
├── mlflow.db
├── mlruns/
└── README.md
```

---

# Technologies Used

- Python 3.12
- MLflow
- Feast Feature Store
- Google Cloud Storage
- Vertex AI Workbench
- DVC
- GitHub Actions
- Scikit-learn
- Pandas
- NumPy

---

# Google Cloud Storage Buckets

## Data Bucket

```
gs://iris-data-23f3002872
```

Stores:

- Raw dataset
- Versioned datasets

---

## Artifact Bucket

```
gs://iris-artifacts-23f3002872
```

Stores:

- metrics.json
- eval.csv
- predictions.csv
- train.log

---

# MLflow Integration

The training pipeline uses MLflow for:

- Experiment Tracking
- Hyperparameter Logging
- Metric Logging
- Model Logging
- Model Registry

Tracking URI:

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

Experiment Name

```
IRIS_Experiment
```

Registered Model

```
iris_classifier
```

---

# Hyperparameter Tuning

Three different configurations are evaluated.

| Run | Max Depth | Criterion |
|------|-----------|-----------|
| 1 | 2 | gini |
| 2 | 3 | entropy |
| 3 | 5 | gini |

Each configuration creates a separate MLflow run.

---

# Logged Parameters

- max_depth
- criterion
- test_size
- data_version

---

# Logged Metrics

- Accuracy
- Precision
- Recall
- F1 Score

---

# Model Registry

Every trained model is registered automatically in the MLflow Model Registry.

Example:

```
iris_classifier

Version 1
Version 2
Version 3
Version 4
Version 5
Version 6
```

---

# Training Pipeline

The training pipeline performs the following steps:

1. Download dataset from Google Cloud Storage.
2. Retrieve historical features from Feast.
3. Split the dataset into train and test sets.
4. Perform hyperparameter tuning.
5. Train Decision Tree models.
6. Log parameters and metrics to MLflow.
7. Register trained models.
8. Upload metrics, evaluation dataset, and logs to GCS.

Run:

```bash
python train.py \
  --data-bucket iris-data-23f3002872 \
  --artifact-bucket iris-artifacts-23f3002872 \
  --data-version raw
```

---

# Inference Pipeline

The inference pipeline performs:

1. Download evaluation dataset from GCS.
2. Load the latest model from MLflow Model Registry.
3. Retrieve online features from Feast.
4. Generate predictions.
5. Upload predictions.csv to GCS.

Run:

```bash
python infer.py \
  --artifact-bucket iris-artifacts-23f3002872 \
  --artifact-prefix artifacts/<timestamp>
```

---

# MLflow Tracking UI

Launch the MLflow UI:

```bash
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5001
```

The UI provides:

- Experiment comparison
- Parameter visualization
- Metric visualization
- Model Registry
- Registered model versions

---

# GitHub Actions

The CI workflow performs:

- Checkout repository
- Install dependencies
- Authenticate to Google Cloud
- Pull versioned data using DVC
- Configure MLflow
- Verify Model Registry
- Execute test suite
- Publish CML report

---

# Assignment Tasks Completed

| Task | Status |
|-------|--------|
| Hyperparameter Tuning | ✅ |
| MLflow Experiment Tracking | ✅ |
| Experiment Comparison | ✅ |
| Remove Model Dependency from DVC | ✅ |
| Fetch Models from MLflow Registry | ✅ |
| MLflow Integration in CI (Optional) | ✅ |

---

# Learning Outcomes

Through this assignment I learned how MLflow complements DVC by providing experiment tracking, model versioning, and centralized model management. I also gained practical experience integrating MLflow with Feast, Google Cloud Storage, Vertex AI Workbench, and GitHub Actions to build a complete MLOps workflow.

---

## Author

**Name:** Khushi Choudhary

**IIT Madras BS Roll Number:** 23f3002872