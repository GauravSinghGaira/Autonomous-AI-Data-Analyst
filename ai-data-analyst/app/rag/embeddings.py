"""
Embedding model wrapper.

Wrapped in its own module (rather than instantiated inline wherever
needed) so the model is loaded once and reused, and so swapping
embedding models later only touches this file.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()
