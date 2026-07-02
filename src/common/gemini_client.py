"""Gemini model factory."""
from __future__ import annotations
import google.generativeai as genai
from config.settings import settings


def get_gemini_model(model_name: str | None = None):
    """Return a configured Gemini GenerativeModel."""
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel(model_name or settings.default_gemini_model)
