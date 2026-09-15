"""
In-memory registry mapping a session_id to its loaded DataFrame and
cached profile. Kept intentionally simple (a process-local dict) --
for a multi-worker production deployment this would move to Redis
or similar, but for this project's scope a single-process store is
the right amount of complexity.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DatasetSession:
    session_id: str
    df: pd.DataFrame
    filename: str
    profile_cache: dict[str, Any] | None = field(default=None)


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, DatasetSession] = {}
        self._lock = threading.Lock()

    def set(self, session_id: str, df: pd.DataFrame, filename: str) -> None:
        with self._lock:
            self._sessions[session_id] = DatasetSession(session_id=session_id, df=df, filename=filename)

    def get(self, session_id: str) -> DatasetSession | None:
        return self._sessions.get(session_id)

    def get_df(self, session_id: str) -> pd.DataFrame | None:
        session = self.get(session_id)
        return session.df if session else None

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


registry = SessionRegistry()
