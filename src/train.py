"""Tunes Logistic Regression, Random Forest, and XGBoost with Optuna, tracks
every run in MLflow (hosted on DagsHub), registers the best model in the
MLflow Model Registry, saves it locally for Docker to bake in, and writes a
SHAP summary plot explaining the best model. Run as: python -m src.train
"""
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import shap
from dotenv import load_dotenv
from huggingface_hub import HfApi
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.utils import (
    FEATURE_COLUMNS,
    METRICS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_PATH,
    MODEL_REGISTRY_NAME,
    PROJECT_ROOT,
    REPORTS_DIR,
    SHAP_PLOT_PATH,
    TARGET_COL,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    build_preprocessor,
    load_params,
)

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]


def setup_mlflow() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def build_model(model_name: str, params: dict, scale_pos_weight: float, random_state: int):
    if model_name == "logistic_regression":
        return LogisticRegression(
            **params,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    raise ValueError(f"Unknown model_name: {model_name}")


def tune_model(model_name, X_train, y_train, scale_pos_weight, n_trials, cv_folds, primary_metric, random_state):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        model = build_model(model_name, params, scale_pos_weight, random_state)
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", model)])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=primary_metric, n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def push_to_hub(model_path, all_metrics: dict, best_model_name: str) -> None:
    """Mirrors the winning pipeline to a Hugging Face Hub model repo, as a
    second, publicly browsable home for the model alongside the MLflow
    Model Registry. Best-effort: silently skipped if credentials aren't set,
    so training and CI never depend on a Hugging Face account existing."""
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    if not hf_token or not hf_repo_id:
        print("HF_TOKEN / HF_REPO_ID not set - skipping Hugging Face Hub push")
        return

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="model.pkl",
        repo_id=hf_repo_id,
        repo_type="model",
    )

    metrics_rows = "\n".join(
        f"| {name} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
        f"{m['f1']:.3f} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} |"
        for name, m in all_metrics.items()
        if name != "best_model"
    )
    model_card = f"""---
tags:
- tabular-classification
- xgboost
- scikit-learn
- loan-default
---

# Loan Default Classifier

Binary classifier predicting whether a loan applicant will default, mirrored here from the
MLflow Model Registry of the source project (`{MODEL_REGISTRY_NAME}`, best model: **{best_model_name}**).

## Test set performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
{metrics_rows}

## Usage

`model.pkl` is a scikit-learn `Pipeline` (preprocessing + classifier) saved with `joblib`. It expects
a single-row DataFrame with these raw feature columns:

`{", ".join(FEATURE_COLUMNS)}`

```python
import joblib
import pandas as pd

pipeline = joblib.load("model.pkl")
row = pd.DataFrame([{{...}}], columns={FEATURE_COLUMNS!r})
probability = pipeline.predict_proba(row)[0, 1]
```

Trained as part of an end-to-end MLOps pipeline (DVC + MLflow + FastAPI + Docker + GitHub Actions).
Source repository: https://github.com/Yashwanth-R19/loan-default-mlops
"""
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=hf_repo_id,
        repo_type="model",
    )
    print(f"Pushed model to https://huggingface.co/{hf_repo_id}")


def main() -> None:
    params = load_params()
    random_state = params["random_state"]
    n_trials = params["training"]["n_trials"]
    cv_folds = params["training"]["cv_folds"]
    primary_metric = params["training"]["primary_metric"]

    setup_mlflow()

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COL]
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    results = []
    for model_name in MODEL_NAMES:
        print(f"\n=== Tuning {model_name} ({n_trials} trials, {cv_folds}-fold CV) ===")
        best_params, best_cv_score = tune_model(
            model_name, X_train, y_train, scale_pos_weight, n_trials, cv_folds, primary_metric, random_state
        )
        print(f"Best CV {primary_metric}: {best_cv_score:.4f} | params: {best_params}")

        model = build_model(model_name, best_params, scale_pos_weight, random_state)
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_pred, y_proba)
        print(f"Test metrics: { {k: round(v, 4) for k, v in test_metrics.items()} }")

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_type", model_name)
            mlflow.log_params(best_params)
            mlflow.log_metric(f"cv_{primary_metric}", best_cv_score)
            mlflow.log_metrics(test_metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            run_id = mlflow.active_run().info.run_id

        results.append(
            {"model_name": model_name, "run_id": run_id, "pipeline": pipeline, "metrics": test_metrics}
        )

    best_result = max(results, key=lambda r: r["metrics"][primary_metric])
    print(
        f"\nBest model: {best_result['model_name']} "
        f"({primary_metric}={best_result['metrics'][primary_metric]:.4f})"
    )

    # Register the best model in the MLflow Model Registry
    client = MlflowClient()
    model_uri = f"runs:/{best_result['run_id']}/model"
    registered = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Registered '{MODEL_REGISTRY_NAME}' v{registered.version} -> Production")

    # Save the best pipeline locally so Docker can bake it into the image
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_result["pipeline"], MODEL_PATH)
    print(f"Saved best pipeline to {MODEL_PATH}")

    # Write metrics.json for all models (also DVC-tracked as a pipeline metric)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {r["model_name"]: r["metrics"] for r in results}
    all_metrics["best_model"] = best_result["model_name"]
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    push_to_hub(MODEL_PATH, all_metrics, best_result["model_name"])

    # SHAP summary plot for the best model
    best_pipeline = best_result["pipeline"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    classifier = best_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    background = preprocessor.transform(X_train)[:100]
    sample = preprocessor.transform(X_test)[:100]

    explainer = shap.Explainer(classifier, background, feature_names=feature_names)
    if best_result["model_name"] in ("random_forest", "xgboost"):
        # check_additivity=False: tree SHAP sums can miss exact additivity due
        # to floating-point summation order, which otherwise raises spuriously.
        # LinearExplainer (logistic_regression) doesn't accept this kwarg.
        shap_values = explainer(sample, check_additivity=False)
    else:
        shap_values = explainer(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved SHAP summary plot to {SHAP_PLOT_PATH}")

    client.log_artifact(best_result["run_id"], str(SHAP_PLOT_PATH))


if __name__ == "__main__":
    main()
