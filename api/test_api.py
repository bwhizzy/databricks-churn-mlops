
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_rejects_invalid_input():
    response = client.post(
        "/predict",
        json={
            "age": 150,
            "tenure_months": 12,
            "monthly_charges": 75.5,
            "support_tickets": 3,
            "late_payments": 2,
            "contract_months": 12,
            "autopay_enabled": 1,
            "paperless_billing": 1,
        },
    )

    assert response.status_code == 422


def test_predict_rejects_missing_fields():
    response = client.post("/predict", json={"age": 35})

    assert response.status_code == 422
