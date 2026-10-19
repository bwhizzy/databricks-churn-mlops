
import os
from functools import lru_cache

import mlflow
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a registered MLflow model.",
    version="1.0.0",
)

MODEL_NAME = os.getenv(
    "REGISTERED_MODEL_NAME",
    "main.default.customer_churn_model",
)
MODEL_VERSION = os.getenv("MODEL_VERSION", "2")

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


class CustomerFeatures(BaseModel):
    age: int = Field(ge=18, le=100)
    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    support_tickets: int = Field(ge=0)
    late_payments: int = Field(ge=0)
    contract_months: int = Field(ge=0)
    autopay_enabled: int = Field(ge=0, le=1)
    paperless_billing: int = Field(ge=0, le=1)


@lru_cache(maxsize=1)
def load_model():
    # Authentication is supplied through environment variables.
    if not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"):
        raise RuntimeError(
            "Set DATABRICKS_HOST and DATABRICKS_TOKEN before prediction."
        )

    mlflow.set_registry_uri("databricks-uc")
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    return mlflow.sklearn.load_model(model_uri)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    try:
        model = load_model()
        features = pd.DataFrame(
            [customer.model_dump()],
            columns=FEATURES,
        )

        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)
        churn_index = classes.index(1)

        return {
            "churn_prediction": prediction,
            "churn_probability": float(probabilities[churn_index]),
        }

    except Exception as exc:
        # Log the details on the server; don't expose credentials or
        # internal connection details in the HTTP response.
        print(f"Prediction error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Model unavailable. Check model access and server configuration.",
        ) from exc
