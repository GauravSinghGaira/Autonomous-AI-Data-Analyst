"""
Ingests supporting business documents (data dictionary, KPI
definitions, markdown/text docs) into the vector store.
"""
from __future__ import annotations

import os
from pathlib import Path

from app.rag.vectorstore import add_documents
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md"}
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def ingest_directory(docs_dir: str | None = None) -> int:
    """Chunk and index every supported document in docs_dir. Returns count indexed."""
    docs_dir = docs_dir or settings.docs_dir
    path = Path(docs_dir)
    if not path.exists():
        logger.warning("Docs directory %s does not exist; skipping RAG ingestion.", docs_dir)
        return 0

    all_chunks, all_meta, all_ids = [], [], []
    for file_path in path.rglob("*"):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_meta.append({"source": file_path.name, "chunk_index": i})
            all_ids.append(f"{file_path.stem}-{i}")

    add_documents(all_chunks, all_meta, all_ids)
    logger.info("Ingested %d chunks from %s", len(all_chunks), docs_dir)
    return len(all_chunks)


def ingest_single_file(file_path: str) -> int:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text)
    ids = [f"{path.stem}-{i}" for i in range(len(chunks))]
    metas = [{"source": path.name, "chunk_index": i} for i in range(len(chunks))]
    add_documents(chunks, metas, ids)
    return len(chunks)


if __name__ == "__main__":
    n = ingest_directory()
    print(f"Indexed {n} chunks from {settings.docs_dir}")
