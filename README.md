# End-to-End MLOps Pipeline — IMDB Sentiment Analysis

![CI Pipeline](https://github.com/Vanshata17/MLOPS/actions/workflows/ci.yaml/badge.svg)

A production-grade MLOps project that classifies IMDB movie reviews as **positive** or **negative** using a fully automated pipeline — from raw data on S3 to a Dockerized Flask API deployed on AWS ECR.

---

## Architecture Overview

```
GitHub Push
    │
    ▼
GitHub Actions CI
    │
    ├─► DVC Pipeline (dvc repro)
    │       ├── Data Ingestion        (S3 → data/raw)
    │       ├── Data Preprocessing    (data/raw → data/interim)
    │       ├── Feature Engineering   (data/interim → data/processed)
    │       ├── Model Training        (data/processed → models/model.pkl)
    │       ├── Model Evaluation      (MLflow → DagsHub)
    │       └── Model Registration    (MLflow Model Registry → Staging)
    │
    ├─► Run Tests
    ├─► Promote Model (Staging → Production)
    └─► Build & Push Docker Image → AWS ECR
```

---

## Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.10 |
| ML | scikit-learn (Logistic Regression + Bag-of-Words) |
| Experiment Tracking | MLflow + DagsHub |
| Pipeline Versioning | DVC |
| Data Storage | AWS S3 |
| Model Registry | MLflow Model Registry |
| API | Flask + Gunicorn |
| Containerization | Docker |
| Container Registry | AWS ECR |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus |

---

## Pipeline Stages

### 1. Data Ingestion
Fetches `data.csv` from AWS S3, filters sentiment labels, splits 75/25 train/test, and saves to `data/raw/`.

### 2. Data Preprocessing
Applies a full NLP cleaning pipeline to each review:
- Remove URLs, numbers, punctuation
- Lowercase
- Remove English stop words (NLTK)
- Lemmatize (WordNetLemmatizer)

Output saved to `data/interim/`.

### 3. Feature Engineering
Converts cleaned text to numerical features using **Bag-of-Words** (`CountVectorizer`, top 50 features). Saves transformed CSVs to `data/processed/` and the fitted vectorizer to `models/vectorizer.pkl`.

### 4. Model Training
Trains a **Logistic Regression** model with L1 regularization and `liblinear` solver on the BoW features. Saves to `models/model.pkl`.

### 5. Model Evaluation
Evaluates on test data and logs to **MLflow on DagsHub**:
- Accuracy, Precision, Recall, AUC-ROC
- Model parameters and artifacts
- Saves `reports/experiment_info.json` for the registry step

### 6. Model Registration
Registers the evaluated model in the **MLflow Model Registry** and transitions it to **Staging**.

### 7. Model Promotion
After all tests pass in CI, `scripts/promote_model.py` promotes the Staging model to **Production** and archives the previous Production version.

---

## Project Structure

```
MLOPS/
├── .github/workflows/ci.yaml       # GitHub Actions CI/CD
├── src/
│   ├── connections/s3_connection.py # AWS S3 client
│   ├── data/
│   │   ├── data_ingestion.py
│   │   └── data_preprocessing.py
│   ├── features/feature_engineering.py
│   └── model/
│       ├── model_building.py
│       ├── model_evaluation.py
│       └── register_model.py
├── flask_app/
│   ├── app.py                       # Prediction API + Prometheus metrics
│   ├── requirements.txt
│   └── templates/index.html
├── scripts/promote_model.py
├── models/                          # DVC-tracked artifacts
├── data/                            # DVC-tracked data
├── reports/                         # Metrics and experiment info
├── dvc.yaml                         # Pipeline DAG
├── params.yaml                      # Hyperparameters
└── Dockerfile
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Docker
- AWS CLI configured
- DVC
- A [DagsHub](https://dagshub.com) account with an MLflow-enabled repo

### 1. Clone the repository
```bash
git clone https://github.com/Vanshata17/MLOPS.git
cd MLOPS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
```bash
export MLOPS_KEY=your_dagshub_token
export AWS_ACCESS_KEY_ID=your_aws_key
export AWS_SECRET_ACCESS_KEY=your_aws_secret
export AWS_BUCKET_NAME=your_s3_bucket
```

### 4. Run the pipeline
```bash
dvc repro
```

### 5. Register the model
```bash
python -m src.model.register_model
```

---

## Running the Flask App

### Locally
```bash
cd flask_app
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5000`

### With Docker
```bash
docker build -t mlops-app .
docker run -p 8888:5000 -e MLOPS_KEY=your_token mlops-app:latest
```
Open `http://localhost:8888`

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI for sentiment prediction |
| `/predict` | POST | Submit a review, returns positive/negative |
| `/metrics` | GET | Prometheus metrics (request count, latency, predictions) |

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yaml`) triggers on every push and:

1. Installs dependencies
2. Runs the full DVC pipeline (`dvc repro`)
3. Runs model unit tests
4. Promotes the model from Staging → Production
5. Runs Flask integration tests
6. Builds the Docker image
7. Pushes to AWS ECR

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `MLOPS_KEY` | DagsHub token for MLflow tracking |
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_BUCKET_NAME` | S3 bucket with the dataset |
| `AWS_ACCOUNT_ID` | AWS account ID for ECR |
| `AWS_REGION` | AWS region (e.g. `us-east-1`) |
| `ECR_REPOSITORY` | ECR repository name |

---

## Experiment Tracking

All experiments are tracked on DagsHub:
👉 [https://dagshub.com/vanshatajaiswal4/MLOPS.mlflow](https://dagshub.com/vanshatajaiswal4/MLOPS.mlflow)

---

## Hyperparameters

Configured in `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.25

feature_engineering:
  max_features: 50
```

---

## Model Performance

| Metric | Value |
|---|---|
| Accuracy | Tracked in MLflow |
| Precision | Tracked in MLflow |
| Recall | Tracked in MLflow |
| AUC-ROC | Tracked in MLflow |

---

## License

MIT
