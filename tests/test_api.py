from fastapi.testclient import TestClient

from src.app import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload(valid_application):
    with TestClient(app) as client:
        response = client.post("/predict", json=valid_application)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_label"] in ("Low", "Medium", "High")


def test_predict_rejects_unknown_category(valid_application):
    bad = dict(valid_application, Education="Not A Real Degree")
    with TestClient(app) as client:
        response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_rejects_missing_field(valid_application):
    bad = dict(valid_application)
    del bad["Income"]
    with TestClient(app) as client:
        response = client.post("/predict", json=bad)
    assert response.status_code == 422
