"""
FastAPI backend for the Autonomous AI Data Analyst.

Endpoints:
    POST /api/upload        - upload a CSV/Excel file, runs automatic EDA
    POST /api/chat          - ask a natural-language question about the loaded dataset
    POST /api/report        - generate & download a Markdown analysis report
    GET  /api/profile/{id}  - fetch the cached profile for a session
    GET  /api/health        - health check
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.agents.analyst_agent import analyst_agent
from app.api.schemas import ChatRequest, ChatResponse, HealthResponse, ReportRequest, UploadResponse
from app.data_processing.loader import DatasetLoadError, save_upload
from app.data_processing.report import save_report
from app.data_processing.session import registry
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Autonomous AI Data Analyst",
    description="Upload a dataset and interact with it through an AI analytical agent.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", llm_provider=settings.llm_provider)


@app.post("/api/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb}MB limit.")

    session_id = str(uuid.uuid4())
    try:
        path = save_upload(contents, file.filename, settings.upload_dir)
        result = analyst_agent.load_dataset(session_id, path)
    except DatasetLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during upload")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc

    return UploadResponse(**result)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not registry.exists(request.session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id. Upload a dataset first.")
    try:
        result = analyst_agent.chat(request.session_id, request.message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent turn failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return ChatResponse(
        answer=result["answer"],
        route_taken=result["route_taken"],
        charts=result.get("charts", []),
        sources=result.get("sources", []),
    )


@app.post("/api/report")
def generate_report(request: ReportRequest):
    session = registry.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id. Upload a dataset first.")

    from app.ml.anomaly import detect_anomalies

    profile = session.profile_cache
    anomalies = detect_anomalies(session.df)
    report_path = save_report(session.filename, profile, anomalies)
    return FileResponse(path=report_path, filename=report_path.name, media_type="text/markdown")


@app.get("/api/profile/{session_id}")
def get_profile(session_id: str):
    profile = analyst_agent.get_profile(session_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    return profile
