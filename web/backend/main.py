"""Thin re-export so `uvicorn main:app --reload` works from `web/backend/`
without needing the `app.main:app` module path spelled out."""
from app.main import app  # noqa: F401
