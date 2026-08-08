from fastapi.testclient import TestClient

from src.app import app


def test_root_serves_html():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Loan Default Predictor" in response.text
