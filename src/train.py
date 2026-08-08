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
