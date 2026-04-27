# End-to-End MLOps Project Notes
## IMDB Sentiment Analysis Pipeline

---

## 1. Project Overview

This project implements a **production-grade MLOps pipeline** for sentiment analysis on IMDB movie reviews. It classifies reviews as **positive (1)** or **negative (0)** using a Logistic Regression model with Bag-of-Words features.

The project covers the full MLOps lifecycle:
- Data ingestion from AWS S3
- Text preprocessing and feature engineering
- Model training and evaluation
- Experiment tracking with MLflow + DagsHub
- Model registration and promotion
- Flask API for serving predictions
- Dockerization and deployment to AWS ECR
- CI/CD automation with GitHub Actions
- Pipeline versioning with DVC

---

## 2. Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.10 |
| ML Framework | scikit-learn |
| Experiment Tracking | MLflow + DagsHub |
| Pipeline Versioning | DVC |
| Data Storage | AWS S3 |
| Model Registry | MLflow Model Registry |
| API Framework | Flask + Gunicorn |
| Containerization | Docker |
| Container Registry | AWS ECR |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus |

---

## 3. Project Structure

```
MLOPS/
├── .github/workflows/ci.yaml      # CI/CD pipeline
├── src/
│   ├── connections/
│   │   └── s3_connection.py       # AWS S3 client
│   ├── data/
│   │   ├── data_ingestion.py      # Fetch data from S3, split, save
│   │   └── data_preprocessing.py  # Clean and normalize text
│   ├── features/
│   │   └── feature_engineering.py # Bag-of-Words vectorization
│   └── model/
│       ├── model_building.py      # Train Logistic Regression
│       ├── model_evaluation.py    # Evaluate + log to MLflow
│       └── register_model.py      # Register model in MLflow registry
├── flask_app/
│   ├── app.py                     # Flask prediction API
│   ├── requirements.txt           # App-specific dependencies
│   └── templates/index.html       # Frontend UI
├── scripts/
│   └── promote_model.py           # Promote model Staging → Production
├── models/                        # Saved model artifacts (DVC tracked)
├── data/                          # Data directories (DVC tracked)
├── reports/                       # Metrics and experiment info
├── dvc.yaml                       # DVC pipeline definition
├── params.yaml                    # Hyperparameters
├── Dockerfile                     # Container definition
└── requirements.txt               # Project dependencies
```

---

## 4. File-by-File Explanation

### 4.1 `params.yaml`
Central configuration file for all pipeline hyperparameters.

```yaml
data_ingestion:
  test_size: 0.25          # 75% train, 25% test split

feature_engineering:
  max_features: 50         # Top 50 words in the vocabulary
```

**Why this matters:** Keeping parameters here means changing a hyperparameter only requires editing one file. DVC detects the change and knows to re-run affected pipeline stages.

---

### 4.2 `src/connections/s3_connection.py`
Wraps the boto3 AWS SDK into a reusable class for fetching CSV files from S3.

**Key logic:**
- Initializes an S3 client with credentials passed via environment variables
- `fetch_file_from_s3(file_key)` downloads a file and returns it as a pandas DataFrame using `StringIO` (no disk write needed)

**Design decision:** Credentials are never hardcoded — they are passed in via `os.getenv()` to keep secrets out of code.

---

### 4.3 `src/data/data_ingestion.py`
**Purpose:** Fetch raw data, apply basic sentiment filtering, split into train/test, and save to `data/raw/`.

**Flow:**
1. Reads credentials from environment variables
2. Calls `s3_operations` to fetch `data.csv` from S3
3. `preprocess_data()` filters only `positive`/`negative` rows and encodes labels (positive=1, negative=0)
4. Splits with `train_test_split(test_size=0.25)`
5. Saves `train.csv` and `test.csv` to `data/raw/`

**DVC output:** `data/raw`

---

### 4.4 `src/data/data_preprocessing.py`
**Purpose:** Clean and normalize raw text for ML consumption. Saves to `data/interim/`.

**Text cleaning pipeline (applied to each review):**
1. Remove URLs (`https://...`, `www....`)
2. Remove numbers
3. Lowercase
4. Remove punctuation
5. Remove English stop words (NLTK)
6. Lemmatize (WordNetLemmatizer — reduces words to root form)

**Example:**
```
Input:  "This movie was absolutely AMAZING! Loved it in 2024."
Output: "movie absolutely amazing loved"
```

**DVC:** Input = `data/raw`, Output = `data/interim`

---

### 4.5 `src/features/feature_engineering.py`
**Purpose:** Convert cleaned text into numerical features using Bag-of-Words. Saves to `data/processed/`.

**Key logic:**
- `CountVectorizer(max_features=50)` builds a vocabulary of the top 50 most frequent words
- Fit on train data only, transform both train and test (prevents data leakage)
- Saves the fitted vectorizer as `models/vectorizer.pkl` (needed at inference time)
- Outputs `train_bow.csv` and `test_bow.csv` — each row is a 50-dimensional word count vector with a `label` column

**DVC:** Input = `data/interim`, Output = `data/processed` + `models/vectorizer.pkl`

---

### 4.6 `src/model/model_building.py`
**Purpose:** Train the Logistic Regression model and save it as a pickle file.

**Model config:**
```python
LogisticRegression(C=1, solver='liblinear', penalty='l1')
```
- L1 penalty: produces sparse coefficients, good for high-dimensional text features
- `liblinear` solver: efficient for small datasets

**MLflow setup:** Sets tracking URI to DagsHub using `MLOPS_KEY` token for authentication.

**DVC:** Input = `data/processed`, Output = `models/model.pkl`

---

### 4.7 `src/model/model_evaluation.py`
**Purpose:** Evaluate the model on test data and log everything to MLflow.

**Metrics logged:**
- Accuracy
- Precision
- Recall
- AUC-ROC

**MLflow logging:**
- Metrics → `mlflow.log_metric()`
- Model parameters → `mlflow.log_param()`
- Model artifact → `mlflow.sklearn.log_model(clf, "model")`
- Metrics file → `mlflow.log_artifact()`

**Critical output:** `reports/experiment_info.json` — stores the `run_id` and `model_path`, used by `register_model.py` to locate the artifact in the next stage.

**DVC:** Input = `models/model.pkl`, Output = `reports/experiment_info.json`

---

### 4.8 `src/model/register_model.py`
**Purpose:** Register the evaluated model into the MLflow Model Registry and transition it to **Staging**.

**Flow:**
1. Loads `run_id` and `model_path` from `reports/experiment_info.json`
2. Constructs the artifact URI: `{run.artifact_uri}/{model_path}`
3. Creates a registered model entry (if not already exists)
4. Creates a new model version pointing to the artifact
5. Transitions the version to **Staging** stage

**Why a registry?** Allows versioned, stage-based model lifecycle management (None → Staging → Production → Archived).

---

### 4.9 `scripts/promote_model.py`
**Purpose:** After tests pass in CI, promote the latest Staging model to **Production**.

**Flow:**
1. Finds the latest model version in Staging
2. Archives any existing Production versions
3. Transitions the Staging version to Production

This runs as the **"Promote model to production"** step in CI, only if all previous steps succeed.

---

### 4.10 `dvc.yaml`
**Purpose:** Defines the reproducible ML pipeline as a DAG (Directed Acyclic Graph).

```
data_ingestion → data_preprocessing → feature_engineering → model_building → model_evaluation → model_registration
```

Each stage declares:
- `cmd`: the Python command to run
- `deps`: files that trigger re-execution if changed
- `params`: hyperparameters from `params.yaml`
- `outs`: output artifacts DVC tracks and caches

**Key benefit:** `dvc repro` only re-runs stages whose dependencies have changed. Change `max_features` in `params.yaml` and DVC reruns from `feature_engineering` onwards — not from scratch.

---

### 4.11 `flask_app/app.py`
**Purpose:** Serve the trained model as a REST API with a web UI.

**Endpoints:**
| Route | Method | Description |
|---|---|---|
| `/` | GET | Renders the prediction form |
| `/predict` | POST | Preprocesses input text and returns prediction |
| `/metrics` | GET | Exposes Prometheus metrics |

**Inference pipeline:**
1. Receive raw text from form
2. Apply same preprocessing as training (lowercase, stop words, lemmatize, etc.)
3. Transform with saved `vectorizer.pkl`
4. Predict with `model.pkl`
5. Return "positive" or "negative"

**Prometheus metrics tracked:**
- `app_request_count` — total requests per endpoint
- `app_request_latency_seconds` — response time
- `model_prediction_count` — count of each prediction class

**Auth:** Uses `MLOPS_KEY` env var for DagsHub/MLflow token authentication (no browser OAuth in Docker).

---

### 4.12 `Dockerfile`
**Purpose:** Package the Flask app and models into a portable container.

```dockerfile
FROM python:3.11-slim          # Base image
WORKDIR /app
COPY flask_app/ /app/          # App code
COPY models/vectorizer.pkl ... # Pre-trained artifacts
COPY models/model.pkl ...
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet  # NLTK data
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```

**Why `python:3.11-slim`:** Minimal image size. Python 3.11 required because `pandas>=3.x` dropped support for Python 3.10.

---

### 4.13 `.github/workflows/ci.yaml`
**Purpose:** Automate the full pipeline on every `git push`.

**Pipeline stages:**
1. **Checkout** code
2. **Setup Python 3.10**
3. **Cache pip dependencies** (speeds up subsequent runs)
4. **Install dependencies** from `requirements.txt`
5. **Run DVC pipeline** (`dvc repro`) — trains model end-to-end
6. **Run model tests** — unit tests for model correctness
7. **Promote model** — moves Staging → Production if tests pass
8. **Run Flask tests** — integration tests for the API
9. **Login to AWS ECR** — authenticate with Docker registry
10. **Build Docker image**
11. **Tag and push** to ECR

**Secrets used:**
| Secret | Purpose |
|---|---|
| `MLOPS_KEY` | DagsHub token for MLflow |
| `AWS_ACCESS_KEY_ID` | AWS authentication |
| `AWS_SECRET_ACCESS_KEY` | AWS authentication |
| `AWS_BUCKET_NAME` | S3 bucket name |
| `AWS_ACCOUNT_ID` | ECR registry URL |
| `AWS_REGION` | AWS region |
| `ECR_REPOSITORY` | ECR repo name |

---

## 5. End-to-End Data Flow

```
GitHub Push
    │
    ▼
GitHub Actions CI
    │
    ├─► dvc repro
    │       │
    │       ├─► data_ingestion.py     → S3 → data/raw/
    │       ├─► data_preprocessing.py → data/interim/
    │       ├─► feature_engineering.py→ data/processed/ + vectorizer.pkl
    │       ├─► model_building.py     → model.pkl
    │       ├─► model_evaluation.py   → MLflow (DagsHub) + metrics.json
    │       └─► register_model.py     → MLflow Model Registry (Staging)
    │
    ├─► Run tests
    │
    ├─► promote_model.py              → Staging → Production
    │
    └─► Docker build → push to AWS ECR
                                │
                                ▼
                        AWS ECR Repository
                                │
                                ▼
                        Docker run (Flask app)
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              /predict                  /metrics
         (Prediction API)         (Prometheus metrics)
```

---

## 6. FAANG-Level Interview Questions & Answers

---

### Section A: MLOps & System Design

**Q1. Why use DVC instead of just committing data and model files to Git?**

**A:** Git is designed for code (text files) and performs poorly with large binary files — it stores the entire file on every change, bloating the repository. DVC solves this by storing only a small `.dvc` pointer file in Git while the actual data lives in a remote storage backend (S3 in this project). This gives you:
- Git-like version control semantics for data and models
- Reproducible pipelines via `dvc repro`
- Efficient storage — unchanged data is cached and not re-uploaded
- Stage-level caching — only re-run pipeline stages whose inputs changed

---

**Q2. What is the MLflow Model Registry and how does it differ from just saving a pickle file?**

**A:** A pickle file is just a binary blob with no metadata, versioning, or lifecycle management. The MLflow Model Registry provides:
- **Versioned model artifacts** with full lineage (which run, which params, which metrics produced it)
- **Stage-based lifecycle** (None → Staging → Production → Archived)
- **Centralized access** — any service can pull the model by name and stage rather than knowing a file path
- **Auditability** — you can compare versions and roll back

In this project, `register_model.py` creates versions in Staging, tests run against the Staging model, then `promote_model.py` moves it to Production only if all tests pass.

---

**Q3. Explain data leakage and how this project prevents it.**

**A:** Data leakage occurs when information from the test set influences the training process, causing unrealistically good test metrics that don't generalize to production.

This project prevents it in `feature_engineering.py`:
```python
vectorizer.fit_transform(X_train)   # Fit ONLY on training data
vectorizer.transform(X_test)        # Transform test using train's vocabulary
```
If we fit on all data, the vocabulary would include words from test reviews. The model would implicitly "know" the test set distribution, inflating metrics.

---

**Q4. Why use Bag-of-Words instead of TF-IDF or embeddings for this problem?**

**A:** This is a deliberate baseline choice for an MLOps-focused project. BoW:
- Is simple, fast, and interpretable
- Works well for binary sentiment classification on short texts
- Has low computational cost for a 50-feature vocabulary

**Limitations:** BoW ignores word order and semantics ("not good" ≠ "good not"). TF-IDF would downweight common words like "the", improving signal. Embeddings (Word2Vec, BERT) would capture semantic meaning but require significantly more infrastructure. The MLOps practices demonstrated here (DVC, MLflow, CI/CD) apply equally to all approaches.

---

**Q5. How would you handle model drift in production?**

**A:** Model drift occurs when the statistical properties of the input data or the relationship between features and labels change over time. Handling it requires:

1. **Data drift detection:** Monitor input feature distributions (e.g., KL divergence, Population Stability Index) against a training baseline. Tools: Evidently AI, WhyLabs.
2. **Concept drift detection:** Monitor prediction distribution and, when labels become available, actual model performance metrics.
3. **Alerting:** Set thresholds — if accuracy drops below X% or PSI exceeds 0.2, trigger a retraining pipeline.
4. **Retraining strategy:**
   - Scheduled: retrain weekly/monthly on fresh data
   - Triggered: retrain when drift is detected
5. **Canary deployment:** Route a small % of traffic to the new model before full rollout.

In this project, Prometheus metrics at `/metrics` expose prediction counts per class — a sudden shift in the positive/negative ratio is an early drift signal.

---

**Q6. Why is Gunicorn used in production instead of Flask's built-in server?**

**A:** Flask's built-in development server (`app.run()`) is single-threaded and not designed for concurrent requests. In production:
- Gunicorn spawns multiple **worker processes** that handle requests in parallel
- It manages worker lifecycle, restarts crashed workers
- It's WSGI-compliant and pairs with Nginx as a reverse proxy
- Flask's dev server has no access controls and exposes a debugger PIN that is a security risk

Gunicorn command in Dockerfile:
```
gunicorn --bind 0.0.0.0:5000 --timeout 120 app:app
```

---

**Q7. What are GitHub Actions secrets and why are they critical for this CI/CD pipeline?**

**A:** GitHub Actions secrets are encrypted key-value pairs stored at the repository level. They are injected as environment variables into workflow runs. They are critical because:
- AWS credentials, DagsHub tokens, and ECR repo names must never appear in code or logs
- GitHub automatically redacts secret values from logs (shown as `***`)
- They allow the CI runner to authenticate with AWS, DagsHub, and ECR without exposing credentials

This pipeline uses 7 secrets: `MLOPS_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_BUCKET_NAME`, `AWS_ACCOUNT_ID`, `AWS_REGION`, `ECR_REPOSITORY`.

---

### Section B: ML & Data Science

**Q8. Why use L1 (Lasso) regularization in Logistic Regression for text classification?**

**A:** Text features are typically high-dimensional and sparse. L1 regularization has a key property: it drives many coefficients exactly to zero, producing a **sparse model**. This is beneficial because:
- Most words are irrelevant to sentiment — L1 automatically performs feature selection
- Reduces model size and inference latency
- More interpretable — you can inspect which words have non-zero weights

L2 would shrink all coefficients toward zero but rarely eliminate any, keeping all 50 features active.

---

**Q9. Why use AUC-ROC as a metric alongside accuracy?**

**A:** Accuracy is misleading on imbalanced datasets. If 90% of reviews are positive, a model that always predicts positive achieves 90% accuracy while being useless. 

AUC-ROC measures the model's ability to **rank positive instances above negative ones** across all classification thresholds. An AUC of 0.5 is random; 1.0 is perfect. It is:
- Threshold-independent
- Robust to class imbalance
- Standard in binary classification evaluation

This project logs accuracy, precision, recall, and AUC together for a complete picture.

---

**Q10. What is lemmatization and how does it differ from stemming?**

**A:**
- **Stemming** (Porter, Snowball): Rule-based truncation. "running" → "run", "better" → "better" (imperfect). Fast but can produce non-words.
- **Lemmatization** (WordNet): Dictionary-based reduction to the canonical base form. "running" → "run", "better" → "good". Slower but linguistically correct.

This project uses `WordNetLemmatizer` because it produces valid English words, which preserves interpretability and avoids introducing noise from invalid stems.

---

### Section C: System Design & Production

**Q11. How would you scale this architecture to handle 10,000 requests per second?**

**A:**
1. **Horizontal scaling:** Run multiple Flask/Gunicorn containers behind a load balancer (AWS ALB)
2. **Kubernetes (EKS):** Auto-scale pods based on CPU/request metrics via HPA
3. **Async inference:** Use a message queue (SQS/Kafka) — clients post requests, workers process and return results asynchronously
4. **Feature caching:** Cache preprocessed features for repeated inputs (Redis)
5. **Model optimization:** Quantize or distill the model for faster inference
6. **CDN:** Cache static assets
7. **Monitoring:** Prometheus + Grafana for real-time latency and throughput dashboards

---

**Q12. The CI pipeline runs `dvc repro` on every push. How would you optimize this for a large dataset?**

**A:** DVC already caches stage outputs and only reruns changed stages. Further optimizations:
1. **Remote caching:** Push DVC cache to S3 so different CI runners share cached outputs — avoids retraining if nothing relevant changed
2. **Conditional triggers:** Only run the full pipeline on pushes to `main` or when `src/` files change. Use GitHub Actions path filters
3. **Parameterized experiments:** Use DVC experiments for hyperparameter tuning instead of rebuilding from scratch
4. **Lightweight CI:** Run fast unit tests on every push; run full `dvc repro` only on PRs to main
5. **Self-hosted runners:** Use a runner with GPU/large RAM if training is expensive

---

**Q13. How would you implement A/B testing for model versions?**

**A:**
1. Deploy both models (v1 and v2) as separate Flask services behind an API Gateway
2. Route X% of traffic to v2 using weighted routing (AWS ALB weighted target groups)
3. Log predictions with model version tag to a data store
4. Compare business metrics (click-through, conversion) and ML metrics (accuracy on labeled samples) between versions
5. Promote v2 to 100% if metrics improve beyond a statistical significance threshold
6. MLflow's Model Registry stages support this — Production can serve v1 while Staging serves v2

---

**Q14. What happens if the S3 bucket is unavailable during the CI pipeline run?**

**A:** Currently, `s3_connection.py` catches the exception, logs it, and returns `None`. Downstream code (`data_ingestion.py`) then fails with a `NoneType` error. Better production handling would include:

1. **Retry with exponential backoff:** Use `tenacity` (already a dependency) to retry the S3 call 3 times before failing
2. **Circuit breaker:** After N consecutive failures, skip and use cached data
3. **DVC cache fallback:** If the `data/raw` stage output is already cached from a previous successful run, DVC won't rerun it — protecting downstream stages
4. **Alerting:** Fail fast with a clear error message and notify via Slack/PagerDuty
5. **Data validation:** After ingestion, validate the DataFrame schema (row count, column names, null rates) before proceeding

---

## 7. How to Convert This to PDF

**Option 1 — VS Code:**
1. Install the "Markdown PDF" extension
2. Open this file in VS Code
3. Right-click → "Markdown PDF: Export (pdf)"

**Option 2 — Pandoc (command line):**
```bash
pandoc PROJECT_NOTES.md -o PROJECT_NOTES.pdf --pdf-engine=xelatex
```

**Option 3 — Online:**
Upload to [md2pdf.netlify.app](https://md2pdf.netlify.app) for instant conversion.
