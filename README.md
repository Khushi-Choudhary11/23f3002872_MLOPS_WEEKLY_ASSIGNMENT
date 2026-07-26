# Week 6 – Continuous Deployment (CD) for IRIS ML Pipeline

**Name:** Khushi Choudhary  
**IITM BS Roll Number:** 23f3002872  
**Repository:** 23f3002872_MLOPS_WEEKLY_ASSIGNMENT  
**Branch:** week_6

---

# Objective

The objective of this assignment is to automate the deployment of the IRIS inference API using Continuous Deployment (CD).

The API is containerized using Docker, pushed automatically to Google Artifact Registry through GitHub Actions, and deployed to a Google Kubernetes Engine (GKE) cluster. Every push to the `week_6` branch triggers the deployment pipeline automatically.

---

# Architecture

```
                Git Push
                    │
                    ▼
          GitHub Actions Workflow
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
 Docker Build           Authenticate to GCP
        │
        ▼
Artifact Registry
        │
        ▼
Google Kubernetes Engine
        │
        ▼
IRIS API Deployment
        │
        ▼
External LoadBalancer
        │
        ▼
Prediction API
```

---

# Technologies Used

- Python
- FastAPI
- Docker
- Kubernetes
- Google Kubernetes Engine (GKE)
- Google Artifact Registry
- GitHub Actions
- Google Cloud Platform
- MLflow
- Feast
- Scikit-learn

---

# Project Structure

```
.
├── api.py
├── infer.py
├── train.py
├── Dockerfile
├── requirements.txt
├── k8s
│   ├── deployment.yaml
│   └── service.yaml
├── .github
│   └── workflows
│       └── cd.yml
├── models
├── feast_repo
└── README.md
```

---

# Task 1 – Pod vs Container

## Docker Container

A Docker container is a lightweight package containing the application code, dependencies, libraries and runtime required to run an application consistently on any system.

For this assignment, the IRIS FastAPI application is packaged inside a Docker container.

---

## Kubernetes Pod

A Pod is the smallest deployable unit in Kubernetes.

A Pod can contain one or more Docker containers that share:

- Network
- Storage
- Lifecycle

In this project, one Pod runs one Docker container containing the IRIS API.

---

## Difference

| Docker Container | Kubernetes Pod |
|-----------------|----------------|
| Runs application | Runs one or more containers |
| Created using Docker | Managed by Kubernetes |
| No orchestration | Supports orchestration, scaling and self-healing |
| Independent | Smallest deployable Kubernetes object |

---

# Task 2 – Dockerizing the API

A Dockerfile was created to package the IRIS prediction API.

The Docker image contains:

- Python runtime
- Project source code
- Required Python packages
- FastAPI application

The application is served using Uvicorn on port **8000**.

Docker image was successfully built using:

```bash
docker build -t iris-api:v1 .
```

Local testing:

```bash
docker run -d -p 8000:8000 iris-api:v1
```

API endpoints tested successfully:

```
GET /
GET /health
POST /predict
```

---

# Task 3 – GCP Service Account

A dedicated GitHub Actions Service Account was created.

Roles assigned:

- Artifact Registry Writer
- Kubernetes Engine Developer
- Kubernetes Engine Cluster Viewer
- Storage Admin

A JSON key was generated and stored securely in GitHub Secrets.

---

# GitHub Repository Secrets

The following secrets were configured:

```
GCP_PROJECT
GAR_LOCATION
GAR_REPOSITORY
GCP_SA_KEY
GKE_CLUSTER
GKE_REGION
```

---

# Task 4 – GitHub Actions Workflow

A GitHub Actions workflow (`cd.yml`) automates deployment.

Workflow Steps:

1. Checkout repository
2. Authenticate to Google Cloud
3. Configure Docker
4. Build Docker image
5. Push image to Artifact Registry
6. Get GKE credentials
7. Update Kubernetes Deployment
8. Verify rollout

The workflow is triggered automatically on every push to the `week_6` branch.

---

# Task 5 – Deployment to Google Kubernetes Engine

A Kubernetes Deployment was created to manage the IRIS API container.

Deployment features:

- One replica
- Automatic rollout
- Image pulled from Artifact Registry

A Kubernetes Service of type **LoadBalancer** exposes the API publicly.

Deployment verification:

```bash
kubectl get deployments
```

```bash
kubectl get pods
```

```bash
kubectl get svc
```

The Pod reached the **Running** state successfully.

---

# API Testing

Health Endpoint

```bash
curl http://<EXTERNAL_IP>/health
```

Response

```json
{
  "status":"healthy"
}
```

---

Prediction Endpoint

```bash
curl -X POST http://<EXTERNAL_IP>/predict \
-H "Content-Type: application/json" \
-d '{
"sepal_length":5.1,
"sepal_width":3.5,
"petal_length":1.4,
"petal_width":0.2
}'
```

Response

```json
{
  "prediction":"setosa"
}
```

---

# Continuous Deployment Workflow

```
Developer Push
      │
      ▼
GitHub Actions
      │
      ▼
Docker Build
      │
      ▼
Artifact Registry
      │
      ▼
Google Kubernetes Engine
      │
      ▼
Rolling Deployment
      │
      ▼
Updated IRIS Prediction API
```

---

# Learning Outcomes

Through this assignment, I learned:

- Docker containerization
- Creating Docker images
- Using Google Artifact Registry
- Kubernetes Deployment and Services
- Google Kubernetes Engine
- GitHub Actions for Continuous Deployment
- Automating container deployment
- Managing cloud authentication using Service Accounts
- End-to-end CI/CD pipeline for ML deployment

---

# Conclusion

In this assignment, I successfully implemented a Continuous Deployment pipeline for the IRIS ML inference API.

The application was containerized using Docker, stored in Google Artifact Registry, and automatically deployed to Google Kubernetes Engine using GitHub Actions. The deployed API was successfully exposed through a Kubernetes LoadBalancer and verified using health and prediction endpoints.

This assignment completes the end-to-end MLOps workflow by extending the previous CI pipeline into a fully automated Continuous Deployment pipeline.