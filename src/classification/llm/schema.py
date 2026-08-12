"""Structured-output parsing for LLM sarcasm-classification responses.

Every LLM approach (M2-M5) must return exactly one JSON object per example
with a `label` key restricted to the two canonical labels -- no free-form
label variants are accepted (task Section 7: "Do not allow arbitrary label
variants"). Kept local to `src/classification/` rather than added to
`src/common/json_utils.py` (which parses judge *arrays* for the existing
interpretation pipeline) to avoid touching shared, already-tested code for
an unrelated pipeline.
"""
from __future__ import annotations

import json

from config.classification_settings import classification_settings

VALID_LABELS = set(classification_settings.labels)


def extract_json_object(text: str) -> str:
    content = text.strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response:\n{content}")
    return content[start : end + 1]


def parse_label_response(text: str) -> dict:
    """Parses the model's raw response into a dict containing a validated
    `label`. Raises ValueError on malformed JSON, a missing `label` key, or
    any label outside `VALID_LABELS` -- callers are expected to retry
    rather than silently coerce an invalid label."""
    obj = json.loads(extract_json_object(text))
    if "label" not in obj:
        raise ValueError(f"Response JSON missing 'label' key: {obj}")
    normalized = str(obj["label"]).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in VALID_LABELS:
        raise ValueError(f"Invalid label '{obj['label']}' (expected one of {VALID_LABELS})")
    obj["label"] = normalized
    return obj
