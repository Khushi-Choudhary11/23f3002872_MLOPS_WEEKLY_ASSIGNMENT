# IRIS ML Pipeline with Feast Feature Store on Vertex AI — Week 3 MLOps Assignment

**Author:** Khushi Choudhary  
**Roll Number:** 23f3002872  
**Course:** MLOps (IIT Madras BS Degree Programme)

---

# Overview

This repository extends the **Week 1 IRIS Machine Learning Pipeline** by integrating **Feast Feature Store** into the workflow.

The objective of this assignment is to eliminate **training-serving skew** by ensuring that the same engineered features are used during both **model training** and **online inference**.

The project combines:

- Google Cloud Storage (GCS)
- Vertex AI Workbench
- Feast Feature Store
- SQLite
- Git
- Scikit-learn

to build a production-style machine learning workflow where feature engineering is centralized and shared across training and inference.

---

# Learning Objectives

This implementation demonstrates:

- Understanding of Feature Stores
- Offline Feature Retrieval
- Online Feature Retrieval
- Feature Materialization
- Training-Serving Consistency
- End-to-End ML Pipeline on Google Cloud Platform

---

# Architecture

```text
                           Git Repository
                                  │
                                  ▼
                      Vertex AI Workbench
                                  │
            ┌─────────────────────┼──────────────────────┐
            │                     │                      │
            ▼                     ▼                      ▼
     Google Cloud Storage     Feast Offline Store   Feast Online Store
      (Raw IRIS Dataset)        (Parquet File)         (SQLite)
            │                     │                      │
            └───────────────┬─────┴──────────────────────┘
                            │
                            ▼
                  Training & Inference Pipeline
                            │
                            ▼
              Google Cloud Storage Artifact Bucket
```

---

# Google Cloud Storage Buckets

| Bucket | Purpose |
|----------|----------|
| **iris-data-23f3002872** | Stores the IRIS dataset used for training |
| **iris-artifacts-23f3002872** | Stores trained models, metrics, evaluation files, logs and predictions |

---

# Repository Structure

```text
.
├── data
│   ├── raw
│   │   └── iris.csv
│   ├── v1
│   └── v2
│
├── feast_repo
│   └── feature_repo
│       ├── data
│       │   ├── iris.parquet
│       │   ├── registry.db
│       │   └── online_store.db
│       ├── feature_definitions.py
│       ├── feature_store.yaml
│       └── test_workflow.py
│
├── train.py
├── infer.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset

The project uses the classic **IRIS dataset** containing 150 flower samples.

## Features

- sepal_length
- sepal_width
- petal_length
- petal_width

## Target

- species

---

# Feast Components

## Entity

```
iris_id
```

A unique identifier assigned to every IRIS sample.

---

## Data Source

The dataset is converted into a Parquet file and registered as a Feast FileSource.

```
iris.parquet
```

---

## Feature View

Feature View:

```
iris_features
```

contains

- sepal_length
- sepal_width
- petal_length
- petal_width

These features are shared by both training and inference.

---

# Week 3 Workflow

```text
Raw CSV Dataset
       │
       ▼
Download from GCS
       │
       ▼
Convert to Parquet
       │
       ▼
Feast Feature Store
       │
 ┌─────┴─────────┐
 │               │
 ▼               ▼
Offline Store   Online Store
 │               │
 ▼               ▼
Training      Inference
```

---

# Task 1 – Initialize Feast Repository

A Feast Feature Repository was created and configured with:

- SQLite Registry
- SQLite Online Store
- Local FileSource

Repository initialized using:

```bash
feast apply
```

---

# Task 2 – Define Entities, Data Sources & Feature Views

The following Feast objects were created:

### Entity

```
iris_id
```

### Data Source

```
iris.parquet
```

### Feature View

```
iris_features
```

containing

- sepal_length
- sepal_width
- petal_length
- petal_width

---

# Task 3 – Apply Definitions & Materialize Features

Feature definitions were registered using

```bash
feast apply
```

Features were materialized into the SQLite Online Store using

```bash
feast materialize 2020-01-01T00:00:00 <current_time>
```

Materialization populated the online feature store for real-time inference.

---

# Task 4 – Offline Feature Retrieval

Instead of reading feature columns directly from the dataset,

the training pipeline now retrieves historical features using Feast.

Workflow

```text
CSV Dataset
      │
      ▼
Entity DataFrame
      │
      ▼
get_historical_features()
      │
      ▼
Decision Tree Training
```

Training command

```bash
python train.py \
    --data-bucket iris-data-23f3002872 \
    --artifact-bucket iris-artifacts-23f3002872 \
    --data-version raw
```

Training artifacts generated

- model.joblib
- metrics.json
- eval.csv
- train.log

All artifacts are uploaded to the Google Cloud Storage artifact bucket.

---

# Task 5 – Online Feature Retrieval

Inference no longer uses feature columns directly from the evaluation dataset.

Instead,

features are retrieved from Feast Online Store using the entity

```
iris_id
```

Workflow

```text
iris_id
      │
      ▼
get_online_features()
      │
      ▼
Decision Tree Prediction
```

Inference command

```bash
python infer.py \
    --artifact-bucket iris-artifacts-23f3002872 \
    --artifact-prefix artifacts/<timestamp>
```

Generated output

```
predictions.csv
```

which is uploaded to the artifact bucket.

---

# Training-Serving Consistency

Traditional ML pipelines often duplicate feature engineering logic between training and inference, leading to **training-serving skew**.

Using Feast ensures that

- training retrieves historical features from the Offline Store
- inference retrieves live features from the Online Store

Both stages consume exactly the same feature definitions.

---

# Results

The pipeline successfully

- Initialized Feast Feature Repository
- Created Entity, Data Source and Feature View
- Materialized Features into SQLite Online Store
- Retrieved Historical Features using Feast Offline Store
- Trained Decision Tree Classifier using Feast Features
- Retrieved Online Features using Feast Online Store
- Generated Predictions
- Uploaded all artifacts to Google Cloud Storage

Final Model Accuracy

```
98.3%
```

---

# Assignment Tasks Completed

✅ Initialize Feast Feature Repository

✅ Configure SQLite Registry and Online Store

✅ Define Entity

✅ Define FileSource

✅ Define Feature View

✅ Apply Feature Definitions

✅ Materialize Features

✅ Retrieve Historical Features

✅ Train Model using Feast

✅ Retrieve Online Features

✅ Perform Online Inference

---

# Technologies Used

- Python
- Feast 0.64
- Google Cloud Platform
- Vertex AI Workbench
- Google Cloud Storage
- SQLite
- Scikit-learn
- Pandas
- Joblib
- Git

---

# References

- Feast Documentation: https://docs.feast.dev/
- Google Cloud Storage Documentation: https://cloud.google.com/storage/docs
- Vertex AI Documentation: https://cloud.google.com/vertex-ai/docs

---

# Author

**Khushi Choudhary**

**Roll Number:** 23f3002872

**IIT Madras BS Degree Programme**
