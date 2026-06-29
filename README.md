Video Link : https://drive.google.com/file/d/10Z5Ypc8RkvUKdrthdg5q7TX5di7y-NwH/view?usp=sharing

# IRIS ML Pipeline with DVC on Vertex AI — Week 2 MLOps Assignment

**Author:** Khushi Choudhary
**Roll Number:** 23f3002872
**Course:** MLOps (IIT Madras BS)

This repository extends the Week 1 IRIS Machine Learning pipeline by integrating **Data Version Control (DVC)** into the workflow. The project combines **Google Cloud Storage (GCS)**, **Vertex AI Workbench**, **Git**, and **DVC** to create a reproducible, version-controlled machine learning pipeline.

Week 1 focused on building and executing the training and inference pipeline on Google Cloud. Week 2 enhances that pipeline by introducing dataset and model versioning, enabling reproducibility across multiple training iterations.

---

# Architecture

```
                        Git Repository
                               │
                Code + .dvc Pointer Files
                               │
                               ▼
                    Vertex AI Workbench
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
 Data Bucket            DVC Remote Bucket      Artifact Bucket
(Training Data)      (Versioned Data/Models)   (Run Outputs)

gs://iris-data-23f3002872
gs://iris-dvc-23f3002872
gs://iris-artifacts-23f3002872
```

---

# Google Cloud Storage Buckets

| Bucket                        | Purpose                                                                    |
| ----------------------------- | -------------------------------------------------------------------------- |
| **iris-data-23f3002872**      | Stores IRIS dataset versions used for training                             |
| **iris-artifacts-23f3002872** | Stores timestamped model artifacts, logs, evaluation files and predictions |
| **iris-dvc-23f3002872**       | DVC remote used for versioning datasets and model artifacts                |

---

# Repository Structure

```
.
├── .dvc/
├── data/
│   ├── raw/
│   ├── v1/
│   └── v2/
├── models/
├── train.py
├── infer.py
├── requirements.txt
├── README.md
└── .dvcignore
```

---

# Dataset

The project uses the IRIS dataset consisting of four numerical features:

* sepal_length
* sepal_width
* petal_length
* petal_width

Target:

* species

Dataset versions:

| Version | Description                                         |
| ------- | --------------------------------------------------- |
| raw     | Original dataset                                    |
| v1      | Modified dataset used during experimentation        |
| v2      | Second modified dataset used during experimentation |

Each version is stored in Google Cloud Storage and tracked through DVC.

---

# Week 1 Pipeline

The original pipeline performs the following steps:

1. Downloads the selected dataset from the data bucket.
2. Splits the dataset into train and evaluation sets.
3. Trains a Decision Tree classifier.
4. Generates:

   * model.joblib
   * metrics.json
   * eval.csv
   * train.log
5. Uploads all artifacts into a timestamped folder inside the artifact bucket.

Inference downloads the model and evaluation dataset from the artifact bucket and generates predictions.

---

# Week 2 – DVC Integration

The Week 2 assignment integrates **Data Version Control (DVC)** into the existing pipeline.

The following tasks were completed:

* Initialized DVC inside the repository.
* Configured Google Cloud Storage as the DVC remote.
* Versioned IRIS dataset files using DVC.
* Versioned trained model artifacts using DVC.
* Stored DVC metadata inside Git while keeping large files inside GCS.
* Created multiple data/model iterations.
* Demonstrated switching between versions using Git and `dvc checkout`.

---

# DVC Workflow

Initialize DVC

```bash
dvc init
```

Configure remote storage

```bash
dvc remote add -d storage gs://iris-dvc-23f3002872
```

Track datasets

```bash
dvc add data/raw/iris.csv
dvc add data/v1/data.csv
dvc add data/v2/data.csv
```

Track model

```bash
dvc add models/model.joblib
```

Upload versioned files

```bash
dvc push
```

Restore previous versions

```bash
git checkout <commit_hash>
dvc checkout
```

Return to latest version

```bash
git checkout week_2
dvc checkout
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

# Version Control Workflow

```
Modify Dataset
      │
      ▼
Train Model
      │
      ▼
dvc add
      │
      ▼
dvc push
      │
      ▼
git commit
      │
      ▼
Switch Versions
git checkout + dvc checkout
```

---

# Results

The pipeline successfully:

* trained multiple IRIS models
* stored timestamped artifacts inside Google Cloud Storage
* versioned datasets using DVC
* versioned model artifacts using DVC
* restored previous dataset/model versions using Git + DVC
* maintained complete reproducibility across training iterations

---

# Assignment Tasks Completed

## Week 1

* Created Vertex AI Workbench instance.
* Created Google Cloud Storage buckets.
* Uploaded dataset.
* Implemented training pipeline.
* Implemented inference pipeline.
* Stored timestamped artifacts inside GCS.

## Week 2

* Initialized DVC.
* Configured Google Cloud Storage as DVC Remote.
* Versioned datasets using DVC.
* Versioned model artifacts using DVC.
* Created multiple training iterations.
* Demonstrated Git + DVC checkout for version restoration.
* Pushed all DVC metadata, pointer files and source code to GitHub.

---

# Technologies Used

* Python
* Google Cloud Platform
* Vertex AI Workbench
* Google Cloud Storage
* Git
* Data Version Control (DVC)
* scikit-learn
* pandas
* joblib

---

# Author

**Khushi Choudhary**
**Roll Number:** 23f3002872
**IIT Madras BS Degree Programme**
