"""Tests for src/sign/data/load_sign.py -- raw-file parsing."""
from __future__ import annotations

import pytest

from src.sign.data.load_sign import load_raw_pairs


def _write_raw(tmp_path, rows):
    path = tmp_path / "raw.csv"
    path.write_text("\n".join(f"{a},{b}" for a, b in rows) + "\n", encoding="utf-8")
    return path


def test_load_raw_pairs_strips_whitespace(tmp_path):
    path = _write_raw(tmp_path, [(" best day of my life", " worst day of my life")])
    pairs = load_raw_pairs("train", path=path)
    assert pairs == [("best day of my life", "worst day of my life")]


def test_load_raw_pairs_preserves_row_order_and_count(tmp_path):
    rows = [(f"orig{i}", f"interp{i}") for i in range(10)]
    path = _write_raw(tmp_path, rows)
    pairs = load_raw_pairs("train", path=path)
    assert len(pairs) == 10
    assert pairs[0] == ("orig0", "interp0")
    assert pairs[-1] == ("orig9", "interp9")


def test_load_raw_pairs_rejects_rows_with_wrong_field_count(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("only_one_field\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_raw_pairs("train", path=path)


def test_load_raw_pairs_unknown_split_without_path_override_raises():
    with pytest.raises(ValueError):
        load_raw_pairs("not_a_real_split")
