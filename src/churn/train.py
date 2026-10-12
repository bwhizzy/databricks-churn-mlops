
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "/Shared/churn-mlops",
)
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")


def create_dataset(n_samples=2000):
    """Create reproducible synthetic customer churn data."""
    rng = np.random.default_rng(RANDOM_STATE)

    data = pd.DataFrame({
        "age": rng.integers(18, 75, n_samples),
        "tenure_months": rng.integers(1, 73, n_samples),
        "monthly_charges": rng.uniform(20, 150, n_samples).round(2),
        "support_tickets": rng.poisson(2, n_samples),
        "late_payments": rng.poisson(1, n_samples),
        "contract_months": rng.choice(
            [1, 12, 24], n_samples, p=[0.55, 0.30, 0.15]
        ),
        "autopay_enabled": rng.integers(0, 2, n_samples),
        "paperless_billing": rng.integers(0, 2, n_samples),
    })

    # Synthetic relationship for demonstration, not a real business model.
    log_odds = (
        -1.0
        + 0.65 * (data["contract_months"] == 1)
        - 0.025 * data["tenure_months"]
        + 0.012 * (data["monthly_charges"] - 80)
        + 0.30 * data["support_tickets"]
        + 0.45 * data["late_payments"]
        - 0.55 * data["autopay_enabled"]
    )

    probabilities = 1 / (1 + np.exp(-log_odds))
    churn = rng.binomial(1, probabilities)

    return data, pd.Series(churn, name="churn")


def train():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = create_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": 150,
            "max_depth": 8,
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
        })
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_test.head(5),
        )

        print(f"Run ID: {run.info.run_id}")
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Register only when a Unity Catalog model name is configured.
        if REGISTERED_MODEL_NAME:
            mlflow.set_registry_uri("databricks-uc")

            model_uri = f"runs:/{run.info.run_id}/model"
            registered = mlflow.register_model(
                model_uri=model_uri,
                name=REGISTERED_MODEL_NAME,
            )
            print(
                "Registered model:",
                REGISTERED_MODEL_NAME,
                "version:",
                registered.version,
            )


if __name__ == "__main__":
    train()
