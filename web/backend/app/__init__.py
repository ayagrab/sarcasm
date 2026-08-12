"""Web backend package. Adds the main repo root to `sys.path` on import so
adapters can import the existing classification code
(`src.classification.*`, `config.classification_settings`) directly,
without duplicating any model/prompt/training logic in the web layer."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
