"""
Router node: classifies a user request into one of four branches so
the graph only loads/exposes the tools relevant to that branch. This
keeps each LLM call's tool list small and focused, which meaningfully
improves tool-calling reliability versus giving the model all tools
at once.

Classification itself is done by the LLM with a constrained, structured
output (Pydantic), falling back to keyword heuristics if the LLM call
fails -- so the router degrades gracefully rather than crashing the
whole graph.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.utils.logger import get_logger

logger = get_logger(__name__)

Route = Literal["data_analysis", "visualization", "ml_anomaly", "document_retrieval", "general"]

ROUTER_SYSTEM_PROMPT = """You are a routing classifier for a data analyst assistant.
Classify the user's latest message into exactly one category:

- data_analysis: questions about statistics, aggregations, summaries, profiling, comparisons,
  "how many", "average", "what's in this dataset", missing values, duplicates, correlations.
- visualization: explicit or implicit requests to see/plot/chart/graph/visualize something.
- ml_anomaly: requests about outliers, anomalies, predictions, "can we predict X", modeling,
  what drives/causes an outcome.
- document_retrieval: questions about the meaning of a metric/column/KPI, business definitions,
  or anything that sounds like it needs the data dictionary / documentation rather than the
  raw data itself.
- general: greetings, chit-chat, or anything that doesn't fit the above.

Respond with only the category label."""


class RouteDecision(BaseModel):
    route: Route = Field(description="The selected route category")


def keyword_fallback_route(text: str) -> Route:
    lowered = text.lower()
    if any(w in lowered for w in ["plot", "chart", "graph", "visuali", "show me a", "histogram", "heatmap"]):
        return "visualization"
    if any(w in lowered for w in ["anomal", "outlier", "predict", "forecast", "model", "drives", "causes"]):
        return "ml_anomaly"
    if any(w in lowered for w in ["mean by", "kpi", "definition", "what does", "means", "documentation", "data dictionary"]):
        return "document_retrieval"
    if any(w in lowered for w in ["average", "mean", "sum", "total", "count", "how many", "correlat",
                                   "missing", "duplicate", "distribution", "describe", "summary", "profile"]):
        return "data_analysis"
    return "general"


def classify_route(llm, user_text: str) -> Route:
    try:
        structured_llm = llm.with_structured_output(RouteDecision)
        decision = structured_llm.invoke(
            [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
        )
        return decision.route
    except Exception:  # noqa: BLE001
        logger.warning("Router LLM call failed, falling back to keyword heuristic.", exc_info=True)
        return keyword_fallback_route(user_text)
