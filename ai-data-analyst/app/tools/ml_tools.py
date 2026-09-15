"""
LangChain tools for ML: Isolation Forest anomaly detection and
baseline predictive modeling.
"""
from __future__ import annotations

import json

from langchain_core.tools import StructuredTool

from app.data_processing.session import registry
from app.ml.anomaly import detect_anomalies
from app.ml.prediction import predict_target
from app.tools.schemas import AnomalyDetectionInput, PredictTargetInput
from app.utils.logger import get_logger

logger = get_logger(__name__)


def make_ml_tools(session_id: str) -> list[StructuredTool]:

    def _detect_anomalies_fn(contamination: float | None = None) -> str:
        df = registry.get_df(session_id)
        if df is None:
            return json.dumps({"error": "No dataset loaded for this session."})
        result = detect_anomalies(df, contamination=contamination)
        # Trim the row-level sample before returning to the LLM to keep context small;
        # full detail is available via the API for the UI table.
        trimmed = {k: v for k, v in result.items() if k != "anomaly_sample"}
        return json.dumps(trimmed)

    def _predict_target_fn(target_column: str) -> str:
        df = registry.get_df(session_id)
        if df is None:
            return json.dumps({"error": "No dataset loaded for this session."})
        result = predict_target(df, target_column=target_column)
        return json.dumps(result)

    anomaly_tool = StructuredTool.from_function(
        func=_detect_anomalies_fn,
        name="detect_anomalies",
        description=(
            "Run Isolation Forest anomaly detection over all numeric columns of the current "
            "dataset and return how many/which rows look like outliers. Use for 'find anomalies', "
            "'any weird/suspicious rows', 'outlier detection' type questions."
        ),
        args_schema=AnomalyDetectionInput,
    )

    predict_tool = StructuredTool.from_function(
        func=_predict_target_fn,
        name="predict_target",
        description=(
            "Train a quick baseline model (classification or regression, auto-detected) to "
            "predict a target column from the other numeric columns, and return real accuracy/"
            "R2 metrics plus feature importances. Use when the user asks whether/how well a "
            "column can be predicted, or what drives a particular outcome."
        ),
        args_schema=PredictTargetInput,
    )

    return [anomaly_tool, predict_tool]
