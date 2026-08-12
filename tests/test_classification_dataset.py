"""Tests for src/classification/data/build_canonical_dataset.py and
audit_dataset.py. No real raw data is touched -- tiny fixture CSVs stand
in for GEN/HYP/RQ, matching the real column schema (class, id, text)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.classification.data.audit_dataset import audit_dataset
from src.classification.data.build_canonical_dataset import build_canonical_dataset


@pytest.fixture
def fixture_raw_dir(tmp_path):
    gen = pd.DataFrame(
        {
            "class": ["sarc", "notsarc", "sarc"],
            "id": [1, 2, 3],
            "text": ["oh great, another meeting", "I like pizza", "duplicate text here"],
        }
    )
    hyp = pd.DataFrame(
        {
            "class": ["sarc", "notsarc"],
            "id": [1, 2],
            "text": ["best day of my life ever", "the weather is cloudy today"],
        }
    )
    rq = pd.DataFrame(
        {
            "class": ["notsarc", "notsarc"],  # RQ-2 conflicts with GEN-3 ("sarc") on purpose
            "id": [1, 2],
            "text": ["is the sky blue", "Duplicate Text Here"],  # near-dup (case) of GEN-3
        }
    )
    gen.to_csv(tmp_path / "GEN-sarc-notsarc.csv", index=False)
    hyp.to_csv(tmp_path / "HYP-sarc-notsarc.csv", index=False)
    rq.to_csv(tmp_path / "RQ-sarc-notsarc.csv", index=False)
    return tmp_path


def test_build_canonical_dataset_combines_all_three_files(fixture_raw_dir):
    canonical = build_canonical_dataset(fixture_raw_dir)
    assert len(canonical) == 7
    assert set(canonical["category"]) == {"GEN", "HYP", "RQ"}


def test_build_canonical_dataset_assigns_unique_global_ids(fixture_raw_dir):
    canonical = build_canonical_dataset(fixture_raw_dir)
    assert canonical["example_id"].is_unique
    assert set(canonical["example_id"]) >= {"GEN-1", "GEN-2", "GEN-3", "HYP-1", "RQ-1"}


def test_build_canonical_dataset_maps_labels(fixture_raw_dir):
    canonical = build_canonical_dataset(fixture_raw_dir)
    assert set(canonical["label"]) == {"sarcastic", "not_sarcastic"}
    row = canonical[canonical["example_id"] == "GEN-1"].iloc[0]
    assert row["label"] == "sarcastic"


def test_build_canonical_dataset_flags_normalized_duplicates_across_files(fixture_raw_dir):
    canonical = build_canonical_dataset(fixture_raw_dir)
    gen3 = canonical[canonical["example_id"] == "GEN-3"].iloc[0]
    rq2 = canonical[canonical["example_id"] == "RQ-2"].iloc[0]
    assert gen3["dup_group_id"] == rq2["dup_group_id"]


def test_build_canonical_dataset_flags_label_conflicts(fixture_raw_dir):
    canonical = build_canonical_dataset(fixture_raw_dir)
    gen3 = canonical[canonical["example_id"] == "GEN-3"].iloc[0]
    rq2 = canonical[canonical["example_id"] == "RQ-2"].iloc[0]
    assert gen3["label_conflict"] is True or gen3["label_conflict"] == True  # noqa: E712
    assert rq2["label_conflict"] is True or rq2["label_conflict"] == True  # noqa: E712
    assert gen3["label"] != rq2["label"]


def test_build_canonical_dataset_never_drops_rows(fixture_raw_dir):
    canonical = build_canonical_dataset(fixture_raw_dir)
    assert canonical["text"].notna().all()
    assert len(canonical) == 3 + 2 + 2


def test_build_canonical_dataset_writes_output_file(fixture_raw_dir, tmp_path):
    output_path = tmp_path / "out" / "canonical.csv"
    build_canonical_dataset(fixture_raw_dir, output_path)
    assert output_path.exists()
    reloaded = pd.read_csv(output_path)
    assert len(reloaded) == 7


def test_build_canonical_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_canonical_dataset(tmp_path)


def test_build_canonical_dataset_unexpected_class_value_raises(tmp_path):
    bad = pd.DataFrame({"class": ["maybe"], "id": [1], "text": ["hmm"]})
    bad.to_csv(tmp_path / "GEN-sarc-notsarc.csv", index=False)
    pd.DataFrame({"class": [], "id": [], "text": []}).to_csv(tmp_path / "HYP-sarc-notsarc.csv", index=False)
    pd.DataFrame({"class": [], "id": [], "text": []}).to_csv(tmp_path / "RQ-sarc-notsarc.csv", index=False)
    with pytest.raises(ValueError):
        build_canonical_dataset(tmp_path)


def test_audit_dataset_reports_expected_fields(fixture_raw_dir):
    canonical = build_canonical_dataset(fixture_raw_dir)
    report = audit_dataset(canonical)
    assert report["n_rows"] == 7
    assert report["n_missing_text"] == 0
    assert report["n_rows_label_conflict"] == 2
    assert report["n_duplicate_text_groups"] == 1
    assert "warnings" in report
    assert any("label_conflict" in w or "conflicting labels" in w for w in report["warnings"])


def test_audit_dataset_flags_missing_text(fixture_raw_dir):
    canonical = build_canonical_dataset(fixture_raw_dir)
    canonical.loc[0, "text"] = None
    report = audit_dataset(canonical)
    assert report["n_missing_text"] == 1
    assert any("missing text" in w for w in report["warnings"])
