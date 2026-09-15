"""
Anomaly detection using Isolation Forest.

Operates only on numeric columns (after imputing missing values with
the column median) so it works out-of-the-box on arbitrary uploaded
datasets without extra configuration.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def detect_anomalies(df: pd.DataFrame, contamination: float | None = None) -> dict[str, Any]:
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] == 0:
        return {"error": "No numeric columns available for anomaly detection."}
    if numeric_df.shape[0] < 10:
        return {"error": "Need at least 10 rows for meaningful anomaly detection."}

    contamination = contamination or settings.anomaly_contamination

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(numeric_df)
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    preds = model.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(X_scaled)  # higher = more normal

    anomaly_mask = preds == -1
    anomaly_indices = df.index[anomaly_mask].tolist()

    result = {
        "method": "IsolationForest",
        "contamination": contamination,
        "columns_used": numeric_df.columns.tolist(),
        "n_anomalies": int(anomaly_mask.sum()),
        "anomaly_pct": round(float(anomaly_mask.mean()) * 100, 2),
        "anomaly_row_indices": anomaly_indices[:200],  # cap payload size
        "anomaly_sample": df.loc[anomaly_indices].head(20).to_dict(orient="records"),
        "score_range": {"min": round(float(scores.min()), 4), "max": round(float(scores.max()), 4)},
    }
    logger.info("Anomaly detection: %d/%d rows flagged (contamination=%.3f)",
                result["n_anomalies"], len(df), contamination)
    return result
