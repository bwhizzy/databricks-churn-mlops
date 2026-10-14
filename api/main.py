
import os
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained ML model.",
    version="1.0.0",
)

FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "late_payments",
    "contract_months",
    "autopay_enabled",
    "paperless_billing",
]

MODEL_PATH = os.getenv("MODEL_PATH", "model")


class CustomerFeatures(BaseModel):
    age: int = Field(ge=18, le=100)
    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    support_tickets: int = Field(ge=0)
    late_payments: int = Field(ge=0)
    contract_months: int = Field(ge=0)
    autopay_enabled: int = Field(ge=0, le=1)
    paperless_billing: int = Field(ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "healthy"}


def load_model():
    path = Path(MODEL_PATH)
    if not path.exists():
        return None
    return mlflow.pyfunc.load_model(str(path))


@app.post("/predict")
def predict(customer: CustomerFeatures):
    try:
        model = load_model()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model artifact not found. Configure MODEL_PATH.",
            )

        features = pd.DataFrame(
            [customer.model_dump()],
            columns=FEATURES,
        )

        prediction = int(model.predict(features)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0][1])

        return {
            "churn_prediction": prediction,
            "churn_probability": probability,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc
