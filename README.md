# Loan Default MLOps Pipeline

![CI](https://github.com/Yashwanth-R19/loan-default-mlops/actions/workflows/ci.yml/badge.svg)

Predicts whether a loan applicant will default, using the [Loan Default Prediction dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default) (Kaggle, 255,347 rows). Built as an end-to-end MLOps pipeline: DVC data/model versioning, MLflow experiment tracking + model registry (hosted on [DagsHub](https://dagshub.com/Yashwanth-R19/loan-default-mlops)), a FastAPI prediction service, Docker containerization, and GitHub Actions CI.

## Pipeline

```
Dataset -> DVC Versioning -> Training Script -> MLflow Tracking -> Best Model
        -> Model Registry -> FastAPI Prediction API -> Docker Container
        -> GitHub Repository -> GitHub Actions (Build -> Test -> Docker Build)
```

## Dataset

- **Source:** [kaggle.com/datasets/nikhil1e9/loan-default](https://www.kaggle.com/datasets/nikhil1e9/loan-default)
- **Size:** 255,347 rows, 18 columns, no missing values
- **Target:** `Default` (binary) — 11.6% positive class
- **Used for training:** a stratified 50,000-row subsample (80/20 stratified train/test split), for fast local iteration — see `params.yaml`

## Results

Three models were tuned with Optuna (25 trials, 3-fold stratified CV, optimizing ROC-AUC) and tracked in MLflow:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.672 | 0.213 | 0.680 | 0.325 | 0.739 | 0.313 |
| Random Forest | 0.851 | 0.350 | 0.331 | 0.340 | 0.739 | 0.309 |
| **XGBoost (best)** | 0.689 | 0.221 | 0.667 | 0.332 | **0.745** | 0.324 |

XGBoost was selected on ROC-AUC and registered in the MLflow Model Registry (`loan-default-classifier`, stage: Production). Precision is low across all three models because of the ~88/12 class imbalance and the choice to rank by ROC-AUC rather than accuracy — this is a deliberate tradeoff to keep recall (catching actual defaults) reasonably high. See `reports/shap_summary.png` for feature-level explainability of the winning model.

## Project structure

```
loan-default-mlops/
|-- data/
|   |-- raw/                # original dataset (DVC-tracked)
|   |-- processed/          # train/test splits (DVC-tracked)
|-- models/
|   |-- model.pkl            # best pipeline: preprocessing + XGBoost (DVC-tracked)
|-- reports/
|   |-- metrics.json         # test metrics for all 3 models (git-tracked)
|   |-- shap_summary.png     # SHAP explainability plot (DVC-tracked)
|-- src/
|   |-- data_prep.py         # stratified subsample + train/test split
|   |-- train.py             # Optuna tuning, MLflow tracking, model registry, SHAP
|   |-- predict.py            # inference logic (loads models/model.pkl)
|   |-- app.py                 # FastAPI app (/predict, /health)
|   |-- utils.py               # shared paths, config, preprocessing
|-- tests/                     # pytest suite
|-- .github/workflows/ci.yml   # CI: install -> test -> docker build
|-- Dockerfile
|-- .dockerignore
|-- requirements.txt
|-- params.yaml                # pipeline configuration
|-- dvc.yaml / dvc.lock        # DVC pipeline (prepare -> train)
|-- conftest.py
|-- README.md
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your DagsHub username + access token (used for MLflow tracking and the DVC remote). See `.env.example` for exactly which variables are needed.

## Reproducing the pipeline

```powershell
dvc pull          # fetch the dataset + trained model from the DagsHub remote
dvc repro         # re-run prepare + train stages if you want to retrain from scratch
```

Every run is logged to MLflow at `https://dagshub.com/Yashwanth-R19/loan-default-mlops.mlflow` (Experiments tab on the DagsHub repo page).

## Running the API

```powershell
uvicorn src.app:app --reload --port 8000
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

```powershell
curl -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"Age":35,"Income":65000,"LoanAmount":15000,"CreditScore":680,"MonthsEmployed":48,"NumCreditLines":3,"InterestRate":12.5,"LoanTerm":36,"DTIRatio":0.35,"Education":"Bachelor'"'"'s","EmploymentType":"Full-time","MaritalStatus":"Married","HasMortgage":"Yes","HasDependents":"No","LoanPurpose":"Auto","HasCoSigner":"No"}'
```

Response:
```json
{"prediction": 0, "default_probability": 0.4148, "risk_label": "Medium"}
```

## Running in Docker

```powershell
docker build --load -t loan-default-mlops:latest .
docker run -d --name loan-default-api -p 8000:8000 loan-default-mlops:latest
```

Note: `--load` is required on setups where `docker build` defaults to buildx's `docker-container` driver (build-cache only, not loaded into the local image list) — without it, `docker run` fails to find the image.

## Tests

```powershell
pytest -v
```

Covers config/preprocessing sanity checks, inference correctness and output bounds, and the API's success + validation-error paths.

## CI/CD

On every push/PR, `.github/workflows/ci.yml` checks out the repo, installs dependencies, pulls the trained model from the DagsHub DVC remote (requires the `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN` repository secrets), runs the test suite, and builds the Docker image.

## Tech stack

Python 3.12 - pandas / scikit-learn / XGBoost - Optuna - SHAP - MLflow (tracking + model registry, hosted on DagsHub) - DVC (data + model versioning, DagsHub remote) - FastAPI + Pydantic - Docker - GitHub Actions - pytest

## Submission checklist

- [ ] GitHub repository link
- [ ] Screenshot: MLflow experiments comparison (DagsHub Experiments tab, all 3 runs)
- [ ] Screenshot: registered model (DagsHub Models tab, `loan-default-classifier`, Production stage)
- [ ] Screenshot: successful DVC tracking (`dvc status` / DagsHub Data tab)
- [ ] Screenshot: successful GitHub Actions run (Actions tab, green check)
- [ ] Screenshot: FastAPI prediction endpoint (`/docs` Swagger UI or curl response)
