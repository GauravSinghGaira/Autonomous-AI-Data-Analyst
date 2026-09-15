"""
Pydantic models for structured tool inputs and API responses.

Using Pydantic here means every tool call has a validated schema
(so the LLM's tool-calling arguments are checked before they ever
touch pandas/sklearn) and every API response has a guaranteed shape.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProfileDatasetInput(BaseModel):
    pass  # operates on the currently loaded dataset; no arguments needed


class QueryDataInput(BaseModel):
    column: str | None = Field(default=None, description="Column to aggregate")
    agg: Literal["mean", "sum", "min", "max", "median", "std", "count", "nunique"] | None = Field(
        default=None, description="Aggregation function to apply"
    )
    groupby: str | None = Field(default=None, description="Optional column to group by")
    filter_expr: str | None = Field(
        default=None,
        description="Optional pandas query expression, e.g. \"age > 30 and country == 'US'\"",
    )


class GenerateChartInput(BaseModel):
    chart_type: Literal["histogram", "bar", "line", "scatter", "box", "heatmap"]
    x: str | None = Field(default=None, description="Column for the x-axis")
    y: str | None = Field(default=None, description="Column for the y-axis")
    color: str | None = Field(default=None, description="Optional column to color/group by")


class AnomalyDetectionInput(BaseModel):
    contamination: float | None = Field(
        default=None, ge=0.001, le=0.5,
        description="Expected proportion of outliers, e.g. 0.05 for 5%",
    )


class PredictTargetInput(BaseModel):
    target_column: str = Field(description="Column to predict from the other numeric columns")


class DocumentSearchInput(BaseModel):
    query: str = Field(description="Natural language question to search supporting documents for")
    top_k: int | None = Field(default=None, ge=1, le=10)


class AgentResponse(BaseModel):
    """Structured final response returned to the API/UI layer."""
    answer: str
    route_taken: str
    tool_results: dict[str, Any] = Field(default_factory=dict)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
