# Breast Cancer Triage Service

A machine-learning REST API that predicts a breast cancer classification from 30 numerical features. The project uses FastAPI for model serving, DVC for reproducible training, Docker for containerization, and Redis for prediction counting.

**Disclaimer:** This is an educational project, not a medical diagnostic tool.

## Features

- Breast cancer classification using scikit-learn
- Reproducible training pipeline with DVC
- REST API built with FastAPI
- Docker image and Docker Compose deployment
- Optional Redis prediction counter
- Persistent Redis data using a named volume
- Published Docker Hub images

## Tech Stack

Python · scikit-learn · FastAPI · DVC · Git · Docker · Docker Compose · Redis

## Project Structure

```text
app/                  API application
training/             Data preparation and model training
artifacts/            Trained model, metrics and model card
dvc.yaml              DVC training pipeline
dvc.lock              Pipeline lock file
params.yaml           Training parameters
Dockerfile            API image definition
docker-compose.yml    API and Redis services
docker-compose.hub.yml
                      Compose configuration using Docker Hub
.env.example          Environment configuration template
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/aRoora123/breast-cancer-triage-service.git
cd breast-cancer-triage-service
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
pip install -r requirements-serve.txt
```

### 3. Reproduce the DVC pipeline

Make sure the DVC remote is configured and the required data is available.

```bash
dvc pull
dvc repro
dvc metrics show
```

The training stage reads `data/raw/breast_cancer.csv` and generates the model, model card, and evaluation metrics.

## API

Run the API locally after the model artifacts are available:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/predict` | Return a predicted label and malignant probability |
| GET | `/model/info` | View model metadata and evaluation metrics |
| GET | `/stats` | View the Redis prediction count |

The `/predict` endpoint requires all 30 numerical feature fields. Feature names use underscores in place of spaces.

### Test the prediction endpoint

This command creates a complete example request from scikit-learn's breast cancer dataset:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "$(python -c 'import json; from sklearn.datasets import load_breast_cancer; d=load_breast_cancer(); print(json.dumps({k.replace(" ", "_"): float(v) for k,v in zip(d.feature_names,d.data[0])}))')"
```

## Docker Compose

Create the environment file:

```bash
cp .env.example .env
```

Set `DOCKERHUB_USERNAME=aroora123` in `.env`, then build and start the API and Redis:

```bash
docker compose up -d --build
docker compose ps
```

Check the services:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

Stop the services:

```bash
docker compose down
```

Redis uses the `redis-data` named volume, so its counter survives container recreation. To remove the volume and reset the counter:

```bash
docker compose down -v
```

Redis is optional for standalone API use. If `REDIS_HOST` is unset, predictions still work, but `/stats` reports that Redis is not configured.

## Docker Hub

**Image repository:** [aroora123/breast-cancer-triage-api](https://hub.docker.com/r/aroora123/breast-cancer-triage-api)

Published tags:

- `aroora123/breast-cancer-triage-api:1.0.0`
- `aroora123/breast-cancer-triage-api:latest`

Both tags point to the same published image digest.

### Run the published image

```bash
docker run --rm -p 8000:8000 \
  aroora123/breast-cancer-triage-api:1.0.0
```

### Run with the Hub Compose file

Set `DOCKERHUB_USERNAME=aroora123` in `.env`, then run:

```bash
docker compose -f docker-compose.hub.yml pull
docker compose -f docker-compose.hub.yml up -d
```

This configuration pulls the published API image instead of building it locally.

## Environment Variables

| Variable | Purpose |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub namespace |
| `IMAGE_TAG` | API image tag |
| `HOST_PORT` | Host port mapped to container port 8000 |
| `MODEL_VERSION` | Model version reported by the API |
| `DECISION_THRESHOLD` | Classification threshold |
| `REDIS_HOST` | Redis hostname; unset to disable Redis |
| `REDIS_PORT` | Redis port |
| `MODEL_PATH` | Path to the model artifact |
| `LOG_LEVEL` | Application logging level |
| `GIT_SHA` | Optional Git revision identifier |

Use `.env.example` as the configuration template. Do not commit `.env` or credentials.

## Workflow

1. Track code with Git.
2. Version the training pipeline and data with DVC.
3. Train and evaluate the model.
4. Serve predictions through FastAPI.
5. Package the service in Docker.
6. Run the API and Redis with Compose.
7. Publish versioned images to Docker Hub.
8. Verify deployment using the published image in a clean environment.

## Limitations

Predictions depend on the training data and model quality. This service is intended for educational use and must not replace medical testing or professional clinical judgement.