"""Tests that SIGN labels follow the brief's contract: originals are
sarcastic, interpretations are not_sarcastic, and both use the exact
label strings Dataset A / the shared evaluation code already expect."""
from __future__ import annotations

from config.classification_settings import classification_settings
from config.sign_settings import sign_settings
from src.sign.data.load_sign import build_family_table


def _write_raw(tmp_path, rows):
    path = tmp_path / "raw.csv"
    path.write_text("\n".join(f"{a},{b}" for a, b in rows) + "\n", encoding="utf-8")
    return path


def test_sign_labels_are_a_subset_of_dataset_a_labels():
    assert set(sign_settings.labels) == set(classification_settings.labels)
    assert sign_settings.positive_label == classification_settings.positive_label


def test_every_original_row_is_labeled_sarcastic(tmp_path):
    rows = [("t", f"i{i}") for i in range(5)]
    path = _write_raw(tmp_path, rows)
    df = build_family_table("train", path=path)
    originals = df[df["role"] == "original"]
    assert (originals["label"] == "sarcastic").all()


def test_every_interpretation_row_is_labeled_not_sarcastic(tmp_path):
    rows = [("t", f"i{i}") for i in range(5)]
    path = _write_raw(tmp_path, rows)
    df = build_family_table("train", path=path)
    interps = df[df["role"] == "interpretation"]
    assert (interps["label"] == "not_sarcastic").all()


def test_sign_never_labels_every_row_sarcastic_regression_guard(tmp_path):
    """Guards against the brief's explicit warning: SIGN must never be
    treated as 15,000 independent sarcastic examples."""
    rows = [("t", f"i{i}") for i in range(5)]
    path = _write_raw(tmp_path, rows)
    df = build_family_table("train", path=path)
    assert not (df["label"] == "sarcastic").all()
    assert (df["label"] == "sarcastic").sum() == 1
    assert (df["label"] == "not_sarcastic").sum() == 5
