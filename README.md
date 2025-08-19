# Brake Failure Prediction — End-to-End MLOps Project

Production-ready template inspired by DVC + MLflow pipelines like **YT-MLOPS-Complete-ML-Pipeline**.
It uses **DVC for pipeline & data versioning (with AWS S3 remote)**, **MLflow for experiment tracking**, **FastAPI for serving**,
**Docker** for containerization, and **GitHub Actions** for CI/CD.

## Quickstart (Local)

```bash
# 1) Create & activate venv (Windows PowerShell)
py -3.11 -m venv venv
./venv/Scripts/Activate.ps1
# (Linux/Mac)
python3.11 -m venv venv && source venv/bin/activate

# 2) Install
pip install -r requirements.txt

# 3) Generate synthetic dataset (creates data/raw/brake_sensor_data.csv)
python scripts/generate_synthetic_data.py

# 4) Initialize DVC (first time only)
dvc init

# 5) Run pipeline (data -> features -> train -> evaluate -> register)
dvc repro

# 6) Start MLflow UI (optional, local)
mlflow ui --port 5000

# 7) Serve the model locally
uvicorn app.main:app --reload --port 8000
# Test:
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @sample_request.json
```

## Configure DVC Remote (AWS S3)

```bash
aws configure  # set AWS creds
dvc remote add -d s3remote s3://your-bucket-name/brake-failure-artifacts
dvc remote modify s3remote endpointurl https://s3.amazonaws.com  # if using AWS S3
# push data and artifacts
dvc push
```

## CI/CD (GitHub Actions -> AWS EC2)

- Lints & tests
- Reproduces pipeline with DVC (without pushing to S3 on PRs)
- Builds & pushes Docker image
- Deploys to EC2 via SSH (optional job)
Configure secrets:
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
- `EC2_HOST`, `EC2_USER`, `EC2_KEY` (base64 of private key), `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (or ECR creds)

## Project Structure

```
.
├── .github/workflows/cicd.yml
├── app/                # FastAPI service
├── data/               # raw/ and processed/ (DVC-tracked)
├── notebooks/          # EDA, experiments
├── scripts/            # utility scripts (data synth, eval, etc.)
├── src/                # pipeline code (stages)
├── dvc.yaml            # pipeline definition
├── params.yaml         # all hyperparams & paths
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml # optional MLflow + service
├── sample_request.json
└── Makefile
```

## References
- Upstream inspiration: vikashishere/YT-MLOPS-Complete-ML-Pipeline (DVC + S3 + params.yaml driven).