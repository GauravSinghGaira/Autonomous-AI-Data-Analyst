import numpy as np
import pandas as pd

from app.ml.prediction import predict_target


def test_predict_regression_target():
    rng = np.random.default_rng(0)
    x1 = rng.normal(0, 1, 200)
    x2 = rng.normal(0, 1, 200)
    y = 3 * x1 - 2 * x2 + rng.normal(0, 0.1, 200)
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})

    result = predict_target(df, target_column="target")
    assert result["task"] == "regression"
    assert result["r2_score"] > 0.5


def test_predict_classification_target():
    rng = np.random.default_rng(0)
    x1 = rng.normal(0, 1, 200)
    y = (x1 > 0).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": rng.normal(0, 1, 200), "target": y})

    result = predict_target(df, target_column="target")
    assert result["task"] == "classification"
    assert result["accuracy"] > 0.6


def test_predict_missing_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = predict_target(df, target_column="does_not_exist")
    assert "error" in result


def test_predict_not_enough_rows():
    df = pd.DataFrame({"a": range(5), "target": range(5)})
    result = predict_target(df, target_column="target")
    assert "error" in result
