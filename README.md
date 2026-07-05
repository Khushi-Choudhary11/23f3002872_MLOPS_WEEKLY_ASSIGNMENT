# IRIS ML Pipeline with Feast Feature Store — Week 3 MLOps Assignment

# Overview

This repository extends the IRIS Machine Learning pipeline by integrating **Feast Feature Store** into the existing Google Cloud Platform workflow.

Week 1 focused on building a cloud-based ML pipeline.

Week 3 introduces a Feature Store layer so that both model training and real-time inference retrieve features from a single source of truth, eliminating training-serving skew.

The project was implemented using:

- Google Cloud Platform
- Vertex AI Workbench
- Google Cloud Storage
- Feast Feature Store
- BigQuery (Optional Task)
- SQLite Online Store
- Git
- Python
- scikit-learn

---

# Architecture

```
                     GitHub Repository
                            │
                            ▼
                  Vertex AI Workbench
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
        ▼                   ▼                    ▼
  Data Bucket         Feast Feature Store    Artifact Bucket
       │                 Offline Store             │
       │                 Online Store             │
       ▼                      │                   ▼
    Raw Data ─────────► Feature Retrieval ───► Model Training
                               │
                               ▼
                        Online Inference
```

---

# Google Cloud Resources

## Data Bucket

```
gs://iris-data-23f3002872
```

Stores IRIS datasets.

---

## Artifact Bucket

```
gs://iris-artifacts-23f3002872
```

Stores:

- trained model
- metrics
- evaluation data
- predictions
- logs

---

## Feast Repository

```
feast_repo/
```

Contains:

- feature_store.yaml
- feature_definitions.py
- registry
- online store
- feature metadata

---

# Repository Structure

```
.
├── data
│   ├── raw
│   ├── v1
│   └── v2
│
├── feast_repo
│   └── feature_repo
│       ├── feature_store.yaml
│       ├── feature_definitions.py
│       └── data
│
├── train.py
├── infer.py
├── prepare_feast_data.ipynb
├── requirements.txt
└── README.md
```

---

# Dataset

The IRIS dataset contains:

Features

- sepal_length
- sepal_width
- petal_length
- petal_width

Target

- species

Additional Feast columns

- iris_id
- event_timestamp

---

# Feast Concepts Used

## Entity

```
iris_id
```

Uniquely identifies every Iris sample.

---

## Data Source

Initially

```
Parquet File
```

Optional Task

```
BigQuery Table
```

---

## Feature View

```
iris_features
```

Contains

- sepal_length
- sepal_width
- petal_length
- petal_width

---

## Offline Store

Used during training through

```
get_historical_features()
```

---

## Online Store

SQLite Online Store

Used during inference through

```
get_online_features()
```

---

# Assignment Tasks

## Task 1

Initialized Feast Feature Repository.

Completed ✓

---

## Task 2

Created

- Entity
- Data Source
- Feature View

Completed ✓

---

## Task 3

Applied Feature Definitions

Materialized Features

Completed ✓

---

## Task 4

Training now retrieves historical features using Feast instead of directly reading the CSV.

Workflow

```
CSV
   │
   ▼
Feast Offline Store
   │
   ▼
Historical Features
   │
   ▼
Decision Tree Training
```

Completed ✓

---

## Task 5

Inference retrieves features from the Feast Online Store.

Workflow

```
Sample ID
    │
    ▼
SQLite Online Store
    │
    ▼
Feature Vector
    │
    ▼
Prediction
```

Completed ✓

---

## Task 6 (Optional)

Configured Feast to use BigQuery as the Offline Store.

Changes:

- Uploaded IRIS dataset to BigQuery
- Replaced FileSource with BigQuerySource
- Updated feature_store.yaml
- Applied Feature Definitions
- Materialized Features

Completed ✓

---

# Training

```
python train.py \
--data-bucket iris-data-23f3002872 \
--artifact-bucket iris-artifacts-23f3002872 \
--data-version raw
```

---

# Inference

```
python infer.py \
--artifact-bucket iris-artifacts-23f3002872 \
--artifact-prefix artifacts/<timestamp>
```

---

# Results

Training Accuracy

```
98.3%
```

Inference Accuracy

```
98.3%
```

Training and inference use identical Feast feature definitions, eliminating training-serving skew.

---

# Technologies

- Python
- Feast
- Google Cloud Platform
- Vertex AI Workbench
- Google Cloud Storage
- BigQuery
- SQLite
- scikit-learn
- pandas
- joblib
- Git

---

# Learning Outcomes

This assignment demonstrates

- Feature Store integration
- Historical feature retrieval
- Online feature serving
- Materialization
- Offline vs Online Stores
- Training-serving consistency
- BigQuery integration with Feast

---

# Author

Khushi Choudhary

Roll Number: 23f3002872

IIT Madras BS Degree Programme