"""DSPy signature(s) for sarcasm classification.

Requires the `dspy` package (see `requirements-classification.txt`) --
**not installed in this environment** (see EXPERIMENT_LOG.md, Environment
Audit). This module is importable either way; only `build_signature()`
requires the real dependency, so the rest of `src/classification/` can be
imported/tested without `dspy` present.
"""
from __future__ import annotations

try:
    import dspy

    HAS_DSPY = True
except ImportError:  # pragma: no cover -- exercised once dspy is installed
    dspy = None
    HAS_DSPY = False


def build_signature():
    """Returns the `SarcasmClassification` dspy.Signature class. Defined
    lazily (not at import time) so this module stays importable without
    `dspy` installed."""
    if not HAS_DSPY:
        raise RuntimeError(
            "dspy is not installed. `pip install -r requirements-classification.txt`."
        )

    from typing import Literal

    class SarcasmClassification(dspy.Signature):
        """Classify whether an English sentence is sarcastic."""

        sentence: str = dspy.InputField(desc="The sentence to classify.")
        label: Literal["sarcastic", "not_sarcastic"] = dspy.OutputField(
            desc="Whether the sentence is sarcastic or not."
        )

    return SarcasmClassification
