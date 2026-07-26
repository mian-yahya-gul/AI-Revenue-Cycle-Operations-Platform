"""
LLM client wrapper.

Provides a single ``get_llm()`` accessor used by every agent. When
``USE_MOCK_LLM`` is active (no OpenAI key configured), agents fall back
to deterministic rule-based logic implemented in each agent module —
this wrapper is only invoked when a live model call is actually needed.
"""

from __future__ import annotations

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_llm_singleton = None


def get_llm(temperature: float | None = None):
    """Return a cached ChatOpenAI instance. Raises if mock mode is active.

    Callers should check ``settings.use_mock_llm`` first and use their
    rule-based fallback path instead of calling this function.
    """
    global _llm_singleton
    if settings.use_mock_llm:
        raise RuntimeError(
            "get_llm() called while USE_MOCK_LLM is active. "
            "Agents must check settings.use_mock_llm and use their "
            "deterministic fallback instead of calling the live LLM."
        )

    from langchain_openai import ChatOpenAI

    if _llm_singleton is None:
        _llm_singleton = ChatOpenAI(
            model=settings.llm_model,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            api_key=settings.openai_api_key,
        )
        logger.info("Initialized ChatOpenAI model=%s", settings.llm_model)
    return _llm_singleton


def get_structured_llm(schema):
    """Return an LLM bound to a Pydantic schema for structured output."""
    return get_llm().with_structured_output(schema)
