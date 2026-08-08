"""Shared paths, config, and column definitions used across the pipeline."""
from pathlib import Path

import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Loan_default.csv"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
# The saved artifact is a full sklearn Pipeline (preprocessing + classifier),
# so it can be loaded and called directly on raw feature values.
MODEL_PATH = MODELS_DIR / "model.pkl"

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
SHAP_PLOT_PATH = REPORTS_DIR / "shap_summary.png"

PARAMS_PATH = PROJECT_ROOT / "params.yaml"

MLFLOW_EXPERIMENT_NAME = "loan-default-mlops"
MODEL_REGISTRY_NAME = "loan-default-classifier"

ID_COL = "LoanID"
TARGET_COL = "Default"

NUMERIC_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
]

CATEGORICAL_FEATURES = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def build_preprocessor() -> ColumnTransformer:
    """Numeric features are standardized; categorical features are one-hot
    encoded (binary Yes/No columns collapse to a single 0/1 column)."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
