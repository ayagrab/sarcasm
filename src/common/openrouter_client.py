"""OpenRouter client factory."""
from __future__ import annotations
from openai import OpenAI
from config.settings import settings


def get_openrouter_client() -> OpenAI:
    """Create an OpenAI-compatible client configured for OpenRouter."""
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your .env file.")
    return OpenAI(api_key=settings.openrouter_api_key, base_url="https://openrouter.ai/api/v1")
