# Loan Default MLOps Pipeline

Predicts whether a loan applicant will default, using the [Loan Default Prediction dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default) (Kaggle). Built as an end-to-end MLOps capstone: DVC data versioning, MLflow experiment tracking + model registry (hosted on DagsHub), a FastAPI prediction service, Docker containerization, and a GitHub Actions CI pipeline.

> Status: work in progress, built phase by phase. This README will be filled in fully in the final phase.

## Pipeline

```
Dataset -> DVC Versioning -> Training Script -> MLflow Tracking -> Best Model
        -> Model Registry -> FastAPI Prediction API -> Docker Container
        -> GitHub Repository -> GitHub Actions (Build -> Test -> Docker Build)
```

## Project structure

```
loan-default-mlops/
|-- data/
|   |-- raw/            # original dataset (DVC-tracked)
|   |-- processed/      # train/test splits (DVC-tracked)
|-- models/              # trained model + preprocessor (DVC-tracked)
|-- src/
|   |-- train.py         # training pipeline: preprocessing, 3 models, Optuna, MLflow, SHAP
|   |-- app.py            # FastAPI app
|   |-- predict.py        # inference logic used by the API
|   |-- utils.py          # shared config, data loading, preprocessing helpers
|-- tests/
|-- .github/workflows/ci.yml
|-- Dockerfile
|-- requirements.txt
|-- dvc.yaml
|-- README.md
```
