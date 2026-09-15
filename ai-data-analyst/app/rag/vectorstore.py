"""
ChromaDB-backed vector store for the RAG pipeline.

Stores chunks of supporting documents (data dictionary, KPI
definitions, business docs) so the agent can ground its answers in
real documentation instead of guessing what a column or metric means.
"""
from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.rag.embeddings import embed_texts
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "business_docs"

_client = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    return get_client().get_or_create_collection(name=COLLECTION_NAME)


def add_documents(chunks: list[str], metadatas: list[dict[str, Any]], ids: list[str]) -> None:
    if not chunks:
        return
    collection = get_collection()
    embeddings = embed_texts(chunks)
    collection.upsert(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    logger.info("Upserted %d chunks into vector store", len(chunks))


def query(text: str, top_k: int | None = None) -> list[dict[str, Any]]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    top_k = top_k or settings.rag_top_k
    embedding = embed_texts([text])[0]
    results = collection.query(query_embeddings=[embedding], n_results=min(top_k, collection.count()))

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({"text": doc, "metadata": meta, "distance": round(float(dist), 4)})
    return hits


def reset_collection() -> None:
    get_client().delete_collection(COLLECTION_NAME)
    logger.info("Reset vector store collection '%s'", COLLECTION_NAME)
