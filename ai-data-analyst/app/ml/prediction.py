"""
Basic, automatic predictive modeling.

Not meant to replace real model development -- this gives the agent
a way to answer "can you predict X from the other columns?" with a
quick, honest baseline (train/test split + a standard model + real
metrics), rather than the LLM inventing plausible-sounding numbers.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.utils.logger import get_logger

logger = get_logger(__name__)

MIN_ROWS_FOR_MODELING = 30


def _is_classification_target(series: pd.Series) -> bool:
    if series.dtype == object or str(series.dtype) == "category":
        return True
    if pd.api.types.is_bool_dtype(series.dtype):
        return True
    # low-cardinality numeric columns are treated as classification targets
    return series.nunique(dropna=True) <= max(10, int(len(series) * 0.02))


def predict_target(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    if target_column not in df.columns:
        return {"error": f"Column '{target_column}' not found in dataset."}

    work_df = df.dropna(subset=[target_column]).copy()
    if len(work_df) < MIN_ROWS_FOR_MODELING:
        return {
            "error": f"Not enough rows ({len(work_df)}) with a non-null target to train a "
                     f"reliable model. Need at least {MIN_ROWS_FOR_MODELING}."
        }

    y_raw = work_df[target_column]
    feature_df = work_df.drop(columns=[target_column])

    # Keep it simple and robust: numeric features only, imputed.
    numeric_features = feature_df.select_dtypes(include=np.number)
    if numeric_features.shape[1] == 0:
        return {"error": "No numeric feature columns available to model with."}

    X = SimpleImputer(strategy="median").fit_transform(numeric_features)

    is_classification = _is_classification_target(y_raw)
    label_encoder = None
    if is_classification:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values

    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if is_classification:
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "task": "classification",
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        }
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "task": "regression",
            "r2_score": round(float(r2_score(y_test, y_pred)), 4),
            "mean_absolute_error": round(float(mean_absolute_error(y_test, y_pred)), 4),
        }

    importances = sorted(
        zip(numeric_features.columns, model.feature_importances_),
        key=lambda t: t[1], reverse=True
    )
    top_features = [{"feature": f, "importance": round(float(v), 4)} for f, v in importances[:10]]

    result = {
        "target_column": target_column,
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "features_used": numeric_features.columns.tolist(),
        "top_features": top_features,
        **metrics,
        "note": "Baseline RandomForest on numeric features only; a production model would "
                "include encoded categorical features and hyperparameter tuning.",
    }
    logger.info("Trained baseline model for target=%s task=%s", target_column, metrics["task"])
    return result
