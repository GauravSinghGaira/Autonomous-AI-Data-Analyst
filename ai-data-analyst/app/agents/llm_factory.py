"""
Builds the chat LLM instance based on configuration.

Kept as a single factory function so the rest of the codebase never
imports langchain_openai / langchain_ollama directly -- swapping
providers only means changing settings.llm_provider.
"""
from __future__ import annotations

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_llm(temperature: float | None = None):
    temp = temperature if temperature is not None else settings.llm_temperature

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise RuntimeError("llm_provider is 'openai' but OPENAI_API_KEY is not set.")
        logger.info("Using OpenAI model: %s", settings.openai_model)
        return ChatOpenAI(model=settings.openai_model, temperature=temp, api_key=settings.openai_api_key)

    if settings.llm_provider == "groq":
        from langchain_openai import ChatOpenAI

        if not settings.groq_api_key:
            raise RuntimeError("llm_provider is 'groq' but GROQ_API_KEY is not set.")
        logger.info("Using Groq model: %s", settings.groq_model)
        # Groq exposes an OpenAI-compatible /v1 endpoint, so we can reuse ChatOpenAI
        # with a custom base_url instead of needing a separate SDK/integration.
        return ChatOpenAI(
            model=settings.groq_model,
            temperature=temp,
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    # default: local Ollama model, no API key / cost required
    from langchain_ollama import ChatOllama

    logger.info("Using local Ollama model: %s", settings.ollama_model)
    return ChatOllama(model=settings.ollama_model, temperature=temp, base_url=settings.ollama_base_url)
