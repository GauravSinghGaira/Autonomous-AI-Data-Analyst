"""
LangChain tools for data profiling and querying.

Every tool here is a thin wrapper: it pulls the current session's
DataFrame, calls a pure function in app.data_processing, and returns
JSON. The LLM only ever sees these return values -- it never computes
statistics itself.
"""
from __future__ import annotations

import json

from langchain_core.tools import StructuredTool

from app.data_processing.profiler import profile_dataset, query_dataframe
from app.data_processing.session import registry
from app.tools.schemas import ProfileDatasetInput, QueryDataInput
from app.utils.logger import get_logger

logger = get_logger(__name__)


def make_data_tools(session_id: str) -> list[StructuredTool]:
    """Build session-bound tool instances (closures capture session_id)."""

    def _profile_dataset_fn() -> str:
        df = registry.get_df(session_id)
        if df is None:
            return json.dumps({"error": "No dataset loaded for this session."})
        session = registry.get(session_id)
        if session.profile_cache is None:
            session.profile_cache = profile_dataset(df)
        return json.dumps(session.profile_cache)

    def _query_dataframe_fn(column: str | None = None, agg: str | None = None,
                             groupby: str | None = None, filter_expr: str | None = None) -> str:
        df = registry.get_df(session_id)
        if df is None:
            return json.dumps({"error": "No dataset loaded for this session."})
        result = query_dataframe(df, column=column, agg=agg, groupby=groupby, filter_expr=filter_expr)
        return json.dumps(result)

    profile_tool = StructuredTool.from_function(
        func=_profile_dataset_fn,
        name="profile_dataset",
        description=(
            "Get a full automatic profile of the currently loaded dataset: row/column counts, "
            "dtypes, missing values, duplicates, distributions per column, and correlations "
            "between numeric columns. Call this first for any general 'tell me about this data' "
            "or 'what's in this dataset' question."
        ),
        args_schema=ProfileDatasetInput,
    )

    query_tool = StructuredTool.from_function(
        func=_query_dataframe_fn,
        name="query_dataframe",
        description=(
            "Run a verified numeric aggregation on the dataset, e.g. mean/sum/min/max/median of a "
            "column, optionally grouped by another column and/or filtered with a pandas query "
            "expression. Use this instead of doing arithmetic yourself whenever the user asks for "
            "a specific number (average, total, count, top group, etc.)."
        ),
        args_schema=QueryDataInput,
    )

    return [profile_tool, query_tool]
