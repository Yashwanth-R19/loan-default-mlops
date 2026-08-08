"""Inference logic used by the FastAPI app: loads the trained pipeline once
and turns raw feature values into a prediction + risk label."""
from functools import lru_cache

import joblib
import pandas as pd

from src.utils import FEATURE_COLUMNS, MODEL_PATH

LOW_RISK_THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.6


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def risk_label(probability: float) -> str:
    if probability < LOW_RISK_THRESHOLD:
        return "Low"
    if probability < HIGH_RISK_THRESHOLD:
        return "Medium"
    return "High"


def predict_one(features: dict) -> dict:
    model = load_model()
    row = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "default_probability": round(probability, 4),
        "risk_label": risk_label(probability),
    }
