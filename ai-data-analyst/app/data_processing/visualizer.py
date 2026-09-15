"""
Chart generation using Plotly.

Charts are returned as Plotly JSON (fig.to_dict()) so the same
generation code can serve both the FastAPI backend (JSON over HTTP)
and the Streamlit frontend (native Plotly rendering) without
duplicating logic.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.utils.logger import get_logger

logger = get_logger(__name__)


def auto_generate_charts(df: pd.DataFrame, max_charts: int = 6) -> list[dict[str, Any]]:
    """Heuristically pick a useful default set of charts for a new dataset."""
    charts: list[dict[str, Any]] = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in df.columns if df[c].dtype == object and df[c].nunique() <= 30]

    # Distributions for up to 3 numeric columns
    for col in numeric_cols[:3]:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        charts.append(_package(fig, "histogram", col))

    # Correlation heatmap if enough numeric columns
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", title="Correlation Heatmap", color_continuous_scale="RdBu_r")
        charts.append(_package(fig, "heatmap", "correlation"))

    # Bar chart for top categorical column
    if categorical_cols:
        col = categorical_cols[0]
        counts = df[col].value_counts().head(15).reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count", title=f"Top values of {col}")
        charts.append(_package(fig, "bar", col))

    # Scatter for the two most correlated numeric columns
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        np.fill_diagonal(corr.values, 0)
        if corr.values.max() > 0:
            idx = np.unravel_index(np.argmax(corr.values), corr.shape)
            col_x, col_y = numeric_cols[idx[0]], numeric_cols[idx[1]]
            fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_x} vs {col_y}", trendline=None)
            charts.append(_package(fig, "scatter", f"{col_x}_vs_{col_y}"))

    return charts[:max_charts]


def generate_chart(df: pd.DataFrame, chart_type: str, x: str | None = None,
                    y: str | None = None, color: str | None = None) -> dict[str, Any]:
    """On-demand chart generation driven by an LLM tool call."""
    try:
        if chart_type == "histogram":
            fig = px.histogram(df, x=x, color=color, title=f"Distribution of {x}")
        elif chart_type == "bar":
            fig = px.bar(df, x=x, y=y, color=color, title=f"{y or 'Count'} by {x}")
        elif chart_type == "line":
            fig = px.line(df, x=x, y=y, color=color, title=f"{y} over {x}")
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, title=f"{x} vs {y}")
        elif chart_type == "box":
            fig = px.box(df, x=x, y=y, color=color, title=f"Distribution of {y} by {x}")
        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include=np.number)
            fig = px.imshow(numeric_df.corr(), text_auto=".2f", title="Correlation Heatmap")
        else:
            return {"error": f"Unsupported chart_type '{chart_type}'."}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chart generation failed")
        return {"error": f"Could not generate chart: {exc}"}

    return _package(fig, chart_type, f"{x}_{y}" if y else str(x))


def _package(fig: go.Figure, chart_type: str, name: str) -> dict[str, Any]:
    return {
        "chart_type": chart_type,
        "name": name,
        "figure_json": fig.to_json(),
    }
