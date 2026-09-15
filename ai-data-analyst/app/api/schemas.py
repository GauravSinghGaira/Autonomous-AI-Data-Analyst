from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    profile: dict[str, Any]
    charts: list[dict[str, Any]]
    anomalies: dict[str, Any]


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    route_taken: str
    charts: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []


class ReportRequest(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
