"""
LangChain tool for RAG-based document retrieval (data dictionary,
KPI definitions, business documentation).
"""
from __future__ import annotations

import json

from langchain_core.tools import StructuredTool

from app.rag.vectorstore import query as vector_query
from app.tools.schemas import DocumentSearchInput
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Session-scoped storage so the final response can cite sources.
_pending_sources: dict[str, list[dict]] = {}


def pop_pending_sources(session_id: str) -> list[dict]:
    return _pending_sources.pop(session_id, [])


def make_rag_tools(session_id: str) -> list[StructuredTool]:

    def _search_documents_fn(query: str, top_k: int | None = None) -> str:
        hits = vector_query(query, top_k=top_k)
        if not hits:
            return json.dumps({"results": [], "note": "No supporting documents indexed or no relevant match found."})
        _pending_sources.setdefault(session_id, []).extend(hits)
        return json.dumps({"results": hits})

    search_tool = StructuredTool.from_function(
        func=_search_documents_fn,
        name="search_documents",
        description=(
            "Search the supporting business documentation (data dictionary, KPI definitions, "
            "business context docs) for information relevant to a question. Use this before "
            "answering any question that references a metric name, a column's business meaning, "
            "or company-specific terminology you're not certain about from the raw data alone."
        ),
        args_schema=DocumentSearchInput,
    )

    return [search_tool]
