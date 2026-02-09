"""LLM client factory for Anthropic and OpenRouter."""
from __future__ import annotations

from app.config import settings


def get_client():
    """Get AsyncAnthropic or OpenRouter client based on config.

    If openrouter_model is set, uses OpenRouter API.
    Otherwise, uses Anthropic API directly.
    """
    import anthropic

    if settings.openrouter_model and settings.openrouter_api_key:
        # Use OpenRouter API (compatible with Anthropic SDK)
        return anthropic.AsyncAnthropic(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.io/api/v1",
        )
    else:
        # Use Anthropic API directly
        return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


def get_model():
    """Get the active model based on config.

    Returns OpenRouter model if configured, otherwise default Anthropic model.
    """
    if settings.openrouter_model and settings.openrouter_api_key:
        return settings.openrouter_model
    return settings.default_model


# Singleton
_client = None


def client():
    """Get or create the LLM client."""
    global _client
    if _client is None:
        _client = get_client()
    return _client
