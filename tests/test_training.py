
import sys
from pathlib import Path

import pandas as pd

# Make the src directory importable when tests run from the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.train import create_dataset


def test_create_dataset_has_expected_columns():
    X, y = create_dataset(n_samples=100)

    expected_columns = {
        "age",
        "tenure_months",
        "monthly_charges",
        "support_tickets",
        "late_payments",
        "contract_months",
        "autopay_enabled",
        "paperless_billing",
    }

    assert set(X.columns) == expected_columns
    assert len(X) == 100
    assert len(y) == 100


def test_dataset_is_reproducible():
    X1, y1 = create_dataset(n_samples=100)
    X2, y2 = create_dataset(n_samples=100)

    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


def test_target_contains_both_classes():
    _, y = create_dataset(n_samples=2000)

    assert set(y.unique()) == {0, 1}
