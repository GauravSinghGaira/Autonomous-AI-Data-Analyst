"""
Facade class tying together dataset loading, automatic EDA,
and conversational turns through the LangGraph workflow.

This is the single entry point the API layer talks to, so FastAPI
routes stay thin and all orchestration logic lives in one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.data_processing.loader import load_dataset
from app.data_processing.profiler import profile_dataset
from app.data_processing.session import registry
from app.data_processing.visualizer import auto_generate_charts
from app.ml.anomaly import detect_anomalies
from app.utils.logger import get_logger
from app.workflow.graph import run_agent_turn

logger = get_logger(__name__)

# Conversation history per session (simple in-memory; see session.py note on scope).
_history: dict[str, list[dict]] = {}


class AnalystAgent:
    def load_dataset(self, session_id: str, file_path: str | Path) -> dict[str, Any]:
        """Load a dataset and run the automatic EDA pass (profile + default charts + anomalies)."""
        df = load_dataset(file_path)
        registry.set(session_id, df, filename=Path(file_path).name)
        _history[session_id] = []

        profile = profile_dataset(df)
        registry.get(session_id).profile_cache = profile

        charts = auto_generate_charts(df)
        anomalies = detect_anomalies(df) if df.select_dtypes(include="number").shape[1] > 0 else {
            "error": "No numeric columns for anomaly detection."
        }

        logger.info("Session %s: dataset loaded and auto-EDA complete.", session_id)
        return {
            "session_id": session_id,
            "filename": Path(file_path).name,
            "profile": profile,
            "charts": charts,
            "anomalies": anomalies,
        }

    def chat(self, session_id: str, message: str) -> dict[str, Any]:
        if not registry.exists(session_id):
            return {
                "answer": "No dataset is loaded for this session yet. Please upload a dataset first.",
                "route_taken": "none",
                "charts": [],
                "sources": [],
            }

        history = _history.setdefault(session_id, [])
        result = run_agent_turn(session_id, message, history=history)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": result["answer"]})
        # Keep history bounded so context doesn't grow unbounded across a long session.
        _history[session_id] = history[-20:]

        return result

    def get_profile(self, session_id: str) -> dict[str, Any] | None:
        session = registry.get(session_id)
        return session.profile_cache if session else None


analyst_agent = AnalystAgent()
