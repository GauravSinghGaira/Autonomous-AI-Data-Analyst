"""
LangChain tools for on-demand chart generation.

Generated charts are stashed on the session (registry) rather than
returned inline in the tool string, since Plotly JSON is large; the
agent's final response assembles chart references from there.
"""
from __future__ import annotations

import json

from langchain_core.tools import StructuredTool

from app.data_processing.session import registry
from app.data_processing.visualizer import generate_chart
from app.tools.schemas import GenerateChartInput
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Session-scoped storage of charts produced during the current agent turn.
_pending_charts: dict[str, list[dict]] = {}


def pop_pending_charts(session_id: str) -> list[dict]:
    charts = _pending_charts.pop(session_id, [])
    return charts


def make_viz_tools(session_id: str) -> list[StructuredTool]:

    def _generate_chart_fn(chart_type: str, x: str | None = None, y: str | None = None,
                            color: str | None = None) -> str:
        df = registry.get_df(session_id)
        if df is None:
            return json.dumps({"error": "No dataset loaded for this session."})
        chart = generate_chart(df, chart_type=chart_type, x=x, y=y, color=color)
        if "error" in chart:
            return json.dumps(chart)
        _pending_charts.setdefault(session_id, []).append(chart)
        # Don't send the full figure JSON back to the LLM -- it's large and
        # not something the model needs to reason over; just confirm success.
        return json.dumps({
            "status": "chart_generated",
            "chart_type": chart["chart_type"],
            "name": chart["name"],
        })

    chart_tool = StructuredTool.from_function(
        func=_generate_chart_fn,
        name="generate_chart",
        description=(
            "Generate a Plotly chart from the current dataset (histogram, bar, line, scatter, "
            "box, or correlation heatmap) for a given column or pair of columns. Use this "
            "whenever the user asks to see/plot/visualize/chart something."
        ),
        args_schema=GenerateChartInput,
    )

    return [chart_tool]
