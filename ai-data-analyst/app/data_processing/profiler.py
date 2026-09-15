"""
Automatic dataset profiling.

This produces the deterministic, numerically-verified summary of a
dataset that the LLM will later narrate in natural language. The LLM
never computes these numbers itself -- it only reads this output via
a tool call, which is the core "tool calling, not hallucinated math"
guarantee of the system.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)


def _dtype_group(dtype: np.dtype) -> str:
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    return "categorical"


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-serializable profile of the dataset."""
    n_rows, n_cols = df.shape

    columns_info = []
    for col in df.columns:
        series = df[col]
        group = _dtype_group(series.dtype)
        missing = int(series.isna().sum())
        col_info: dict[str, Any] = {
            "name": col,
            "dtype": str(series.dtype),
            "group": group,
            "missing_count": missing,
            "missing_pct": round(missing / n_rows * 100, 2) if n_rows else 0.0,
            "unique_count": int(series.nunique(dropna=True)),
        }

        if group == "numeric":
            desc = series.describe()
            col_info.update(
                {
                    "mean": _safe_round(desc.get("mean")),
                    "std": _safe_round(desc.get("std")),
                    "min": _safe_round(desc.get("min")),
                    "max": _safe_round(desc.get("max")),
                    "p25": _safe_round(desc.get("25%")),
                    "median": _safe_round(desc.get("50%")),
                    "p75": _safe_round(desc.get("75%")),
                    "skew": _safe_round(series.skew()),
                }
            )
        elif group in ("categorical", "boolean"):
            top = series.value_counts(dropna=True).head(5)
            col_info["top_values"] = {str(k): int(v) for k, v in top.items()}

        columns_info.append(col_info)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    correlations: dict[str, Any] = {}
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr(numeric_only=True).round(3)
        correlations["matrix"] = corr_matrix.to_dict()
        correlations["strong_pairs"] = _top_correlations(corr_matrix)

    duplicate_rows = int(df.duplicated().sum())

    profile = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns_info,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / n_rows * 100, 2) if n_rows else 0.0,
        "total_missing_cells": int(df.isna().sum().sum()),
        "numeric_columns": numeric_cols,
        "categorical_columns": [c["name"] for c in columns_info if c["group"] == "categorical"],
        "datetime_columns": [c["name"] for c in columns_info if c["group"] == "datetime"],
        "correlations": correlations,
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }
    logger.info("Profiled dataset: %d rows x %d cols, %d duplicates", n_rows, n_cols, duplicate_rows)
    return profile


def _safe_round(value: Any, ndigits: int = 4) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), ndigits)


def _top_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.5, top_n: int = 10) -> list[dict]:
    """Return the strongest pairwise correlations, excluding the diagonal."""
    pairs = []
    cols = corr_matrix.columns
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            val = corr_matrix.loc[c1, c2]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append({"col_a": c1, "col_b": c2, "correlation": round(float(val), 3)})
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return pairs[:top_n]


def query_dataframe(df: pd.DataFrame, column: str | None = None, agg: str | None = None,
                     groupby: str | None = None, filter_expr: str | None = None) -> dict[str, Any]:
    """
    Run a bounded, verified aggregation/query against the dataframe.

    This is intentionally a small, safe surface (no arbitrary eval of
    user text) so the LLM's tool-calls always produce real numbers
    instead of the LLM guessing at arithmetic.
    """
    work_df = df
    if filter_expr:
        try:
            work_df = df.query(filter_expr)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Invalid filter expression: {exc}"}

    if groupby and column and agg:
        if groupby not in work_df.columns or column not in work_df.columns:
            return {"error": "Unknown column name(s) supplied."}
        try:
            result = work_df.groupby(groupby)[column].agg(agg)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Aggregation failed: {exc}"}
        return {"result": {str(k): _safe_round(v) for k, v in result.items()}, "rows_considered": len(work_df)}

    if column and agg:
        if column not in work_df.columns:
            return {"error": f"Unknown column '{column}'."}
        try:
            result = getattr(work_df[column], agg)()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Aggregation '{agg}' failed: {exc}"}
        return {"result": _safe_round(result) if isinstance(result, (int, float, np.number)) else result,
                "rows_considered": len(work_df)}

    return {"result": {"row_count": len(work_df)}, "rows_considered": len(work_df)}
