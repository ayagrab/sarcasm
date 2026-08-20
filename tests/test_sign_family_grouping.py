"""Tests for src/sign/data/load_sign.py -- family grouping from raw pairs."""
from __future__ import annotations

from src.sign.data.load_sign import build_family_table, summarize


def _write_raw(tmp_path, rows):
    path = tmp_path / "raw.csv"
    path.write_text("\n".join(f"{a},{b}" for a, b in rows) + "\n", encoding="utf-8")
    return path


def test_build_family_table_groups_five_interpretations_into_one_clean_family(tmp_path):
    rows = [("tweet A", f"interp{i}") for i in range(5)]
    path = _write_raw(tmp_path, rows)
    df = build_family_table("train", path=path)

    assert df["family_id"].nunique() == 1
    fam = df["family_id"].iloc[0]
    assert (df["family_id"] == fam).all()
    assert (df["family_size"] == 5).all()
    assert df["is_clean_family"].all()
    assert (df[df["role"] == "original"]["text"] == "tweet A").all()
    assert len(df[df["role"] == "original"]) == 1
    assert len(df[df["role"] == "interpretation"]) == 5


def test_build_family_table_flags_anomalous_family_sizes(tmp_path):
    # 3 interpretations for one family, 5 for another -- only the second is "clean".
    rows = [("short", "i1"), ("short", "i2"), ("short", "i3")]
    rows += [(f"long", f"j{i}") for i in range(5)]
    path = _write_raw(tmp_path, rows)
    df = build_family_table("train", path=path)

    assert df["family_id"].nunique() == 2
    stats = summarize(df)
    assert stats["n_families"] == 2
    assert stats["n_clean_families"] == 1
    assert stats["n_anomalous_families"] == 1


def test_build_family_table_original_row_has_index_zero_interpretations_are_one_indexed(tmp_path):
    rows = [("tweet", f"interp{i}") for i in range(5)]
    path = _write_raw(tmp_path, rows)
    df = build_family_table("train", path=path)

    orig = df[df["role"] == "original"]
    assert (orig["interp_index"] == 0).all()
    interp_indices = sorted(df[df["role"] == "interpretation"]["interp_index"].tolist())
    assert interp_indices == [1, 2, 3, 4, 5]


def test_build_family_table_example_ids_are_unique(tmp_path):
    rows = [("a", "ia1"), ("a", "ia2"), ("b", "ib1")]
    path = _write_raw(tmp_path, rows)
    df = build_family_table("train", path=path)
    assert df["example_id"].is_unique


def test_summarize_matches_manual_counts_on_a_known_small_fixture(tmp_path):
    # 2 clean families (5 interps each) + 1 anomalous (2 interps).
    rows = []
    for fam in ("f1", "f2"):
        rows += [(fam, f"{fam}-i{i}") for i in range(5)]
    rows += [("f3", "f3-i0"), ("f3", "f3-i1")]
    path = _write_raw(tmp_path, rows)
    df = build_family_table("train", path=path)
    stats = summarize(df)
    assert stats == {
        "n_rows": 3 + 12,
        "n_originals": 3,
        "n_interpretations": 12,
        "n_families": 3,
        "n_clean_families": 2,
        "n_anomalous_families": 1,
    }
