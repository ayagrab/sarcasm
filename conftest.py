"""Ensures the project root is importable as `config`/`src` regardless of
how pytest is invoked (matches the `python -m src.<pkg>.<script>` convention
used to run the project's scripts from the repository root)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
