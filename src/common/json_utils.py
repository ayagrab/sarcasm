"""JSON extraction helpers for LLM responses."""
from __future__ import annotations


def extract_json_array(text: str) -> str:
    """Extract the first JSON array from an LLM response string."""
    content = text.strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in response:\n{content}")
    return content[start : end + 1]
