"""DSPy signature(s) for sarcasm classification.

Requires the `dspy` package (see `requirements-classification.txt`).
This module is importable either way; only `build_signature()`
requires the real dependency, so the rest of `src/classification/` can be
imported/tested without `dspy` present (e.g. on a machine without the
Stage B GPU environment installed).

No `from __future__ import annotations` here deliberately: `SarcasmClassification`
is defined inside `build_signature()`, a local scope, so its `Literal[...]`
output-field annotation must evaluate eagerly to a real type object at class
-body execution time. With postponed evaluation, the annotation would be
stored as an unresolved string/ForwardRef (pydantic/dspy resolve annotations
against the enclosing *module*'s globals, which don't include a
function-local `Literal` import) -- this doesn't break `dspy.Predict`/
`BootstrapFewShot`, but breaks MIPROv2's `Signature.with_instructions()`,
which rebuilds the signature and re-validates field types
(`ValueError: Field types must be types, but received: ForwardRef(...)`).
"""

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
