"""
Shared state passed between LangGraph nodes.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    session_id: str
    messages: Annotated[list, add_messages]
    route: str | None
    tool_results: dict[str, Any]
    charts: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    final_answer: str | None
