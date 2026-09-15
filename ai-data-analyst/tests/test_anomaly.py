import numpy as np
import pandas as pd
import pytest

from app.ml.anomaly import detect_anomalies


@pytest.fixture
def df_with_outliers():
    rng = np.random.default_rng(42)
    normal = rng.normal(loc=50, scale=5, size=100)
    outliers = [200, 210, -100]
    values = np.concatenate([normal, outliers])
    return pd.DataFrame({"value": values, "other": rng.normal(0, 1, size=len(values))})


def test_detect_anomalies_flags_outliers(df_with_outliers):
    result = detect_anomalies(df_with_outliers, contamination=0.05)
    assert result["n_anomalies"] > 0
    assert "error" not in result


def test_detect_anomalies_no_numeric_columns():
    df = pd.DataFrame({"category": ["a", "b", "c"] * 10})
    result = detect_anomalies(df)
    assert "error" in result


def test_detect_anomalies_too_few_rows():
    df = pd.DataFrame({"value": [1, 2, 3]})
    result = detect_anomalies(df)
    assert "error" in result
