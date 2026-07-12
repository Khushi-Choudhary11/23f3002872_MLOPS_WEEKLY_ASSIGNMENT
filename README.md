# IRIS ML Pipeline with Feast, DVC & GitHub Actions CI — Week 4 MLOps Assignment

## Demo Video

https://drive.google.com/drive/folders/1jaPKTEQNTdIMncwfoCD7LT8j0xLJR3nD?usp=sharing

---

# Overview

This repository implements a complete end-to-end MLOps pipeline for the IRIS classification problem on **Google Cloud Platform**.

The project evolved over multiple assignments:

- **Week 1:** Built a cloud-based ML training pipeline on Vertex AI and Google Cloud Storage.
- **Week 2:** Added **DVC** for version-controlled datasets and model artifacts.
- **Week 3:** Integrated **Feast Feature Store** to eliminate training-serving skew using offline and online feature stores.
- **Week 4:** Added **Continuous Integration (CI)** using **GitHub Actions**, automated testing with **pytest**, DVC data retrieval, and **CML** reporting for pull requests.

Every push and pull request automatically validates the ML pipeline before code is merged.

---

# Architecture

```text
                        GitHub Repository
                               │
                     Push / Pull Request
                               │
                               ▼
                     GitHub Actions CI
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
 Authenticate to GCP      DVC Pull              Install Dependencies
        │                      │                      │
        └──────────────┬───────┴──────────────────────┘
                       ▼
                  Run pytest Tests
                       │
        ┌──────────────┴───────────────┐
        │                              │
        ▼                              ▼
 Data Validation Tests         Model Evaluation Tests
        │                              │
        └──────────────┬───────────────┘
                       ▼
               Generate Test Report
                       │
                       ▼
               Publish using CML
```

---

# Google Cloud Resources

## Data Bucket

```
gs://iris-data-23f3002872
```

Stores versioned IRIS datasets.

---

## Artifact Bucket

```
gs://iris-artifacts-23f3002872
```

Stores:

- trained models
- evaluation metrics
- predictions
- evaluation dataset
- logs

---

## DVC Remote

```
gs://iris-dvc-23f3002872
```

Stores version-controlled datasets and models.

---

## Feast Repository

```
feast_repo/
```

Contains:

- feature_store.yaml
- feature_definitions.py
- registry
- SQLite online store
- offline feature metadata

---

# Repository Structure

```text
.
├── .github
│   └── workflows
│       └── ci.yml
│
├── data
│   ├── raw
│   ├── v1
│   └── v2
│
├── feast_repo
│   └── feature_repo
│
├── models
│
├── tests
│   ├── test_data.py
│   └── test_model.py
│
├── train.py
├── infer.py
├── requirements.txt
└── README.md
```

---

# Dataset

The IRIS dataset contains:

### Features

- sepal_length
- sepal_width
- petal_length
- petal_width

### Target

- species

### Feast Metadata

- iris_id
- event_timestamp

---

# Feast Integration

## Entity

```
iris_id
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

Used during training using

```
get_historical_features()
```

---

## Online Store

SQLite Online Store

Used during inference using

```
get_online_features()
```

---

# Continuous Integration (Week 4)

GitHub Actions automatically executes the CI pipeline whenever code is pushed or a Pull Request is opened.

Workflow includes:

- Repository checkout
- Python setup
- Dependency installation
- Google Cloud authentication
- DVC data retrieval
- Data validation tests
- Model evaluation tests
- Test report generation
- CML report publishing on Pull Requests

---

# Data Validation Tests

Implemented using **pytest**.

Tests verify:

- Dataset exists
- Expected schema
- No missing values
- Correct numeric feature types
- Reasonable feature value ranges

Example:

```bash
pytest tests/test_data.py
```

---

# Model Evaluation Tests

Tests verify:

- Model file exists
- Model loads successfully
- Accuracy exceeds threshold
- Precision exceeds threshold

Example:

```bash
pytest tests/test_model.py
```

---

# GitHub Actions Workflow

Location

```
.github/workflows/ci.yml
```

The workflow automatically:

- Checks out the repository
- Installs dependencies
- Authenticates with Google Cloud
- Executes

```
dvc pull
```

- Runs

```
pytest
```

- Generates a CI report
- Publishes results through CML

Workflow triggers:

- Push
- Pull Request

---

# DVC Integration

Version-controlled artifacts include

```
data/raw/iris.csv
models/model.joblib
```

Retrieve versioned data using

```bash
dvc pull
```

Upload new versions using

```bash
dvc push
```

---

# Training

```bash
python train.py \
--data-bucket iris-data-23f3002872 \
--artifact-bucket iris-artifacts-23f3002872 \
--data-version raw
```

---

# Inference

```bash
python infer.py \
--artifact-bucket iris-artifacts-23f3002872 \
--artifact-prefix artifacts/<timestamp>
```

---

# Assignment Tasks

## Task 1

Implemented automated data validation tests using pytest.

**Completed ✓**

---

## Task 2

Implemented automated model evaluation tests.

**Completed ✓**

---

## Task 3

Configured GitHub Actions to:

- install dependencies
- authenticate to GCP
- execute DVC pull
- run pytest

**Completed ✓**

---

## Task 4

Enabled CI execution on

- Push
- Pull Request

**Completed ✓**

---

## Task 5

Integrated CML to generate automated CI reports for Pull Requests.

**Completed ✓**

---

## Task 6

Merged Week 4 implementation into the main branch through Pull Request after successful CI validation.

**Completed ✓**

---

# Results

## Data Validation

✔ Dataset found

✔ Schema verified

✔ No missing values

✔ Valid feature ranges

---

## Model Validation

✔ Model loaded successfully

✔ Accuracy threshold satisfied

✔ Precision threshold satisfied

---

## CI Results

✔ GitHub Actions passed

✔ DVC Pull successful

✔ pytest passed

✔ Automated CI report generated

---

# Technologies Used

- Python
- scikit-learn
- pandas
- pytest
- DVC
- Feast
- GitHub Actions
- CML
- Google Cloud Platform
- Vertex AI Workbench
- Google Cloud Storage
- SQLite
- BigQuery
- Git

---

# Learning Outcomes

This assignment demonstrates:

- Continuous Integration for ML pipelines
- Automated testing using pytest
- GitHub Actions workflow creation
- DVC integration inside CI
- Secure authentication using GitHub Secrets
- Continuous Machine Learning (CML)
- Feature Store integration with Feast
- End-to-end reproducible MLOps pipeline

---

# Author

**Khushi Choudhary**

**Roll Number:** 23f3002872

**IIT Madras BS Degree Programme**