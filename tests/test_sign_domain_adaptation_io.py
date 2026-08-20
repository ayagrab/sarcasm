"""Tests for src/sign/domain_adaptation/io.py -- Phase 8's dual-target
(SIGN Test + Dataset A TEST) result saving."""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd

from src.sign.domain_adaptation.io import save_domain_adaptation_result


def _preds(labels):
    return pd.DataFrame(
        {
            "example_id": [f"e{i}" for i in range(len(labels))],
            "gold_label": [l[0] for l in labels],
            "predicted_label": [l[1] for l in labels],
        }
    )


def test_save_domain_adaptation_result_writes_both_targets(tmp_path, monkeypatch):
    from src.sign.domain_adaptation import io as io_module

    fake_settings = replace(io_module.sign_settings, results_dir=tmp_path)
    monkeypatch.setattr(io_module, "sign_settings", fake_settings)

    sign_preds = _preds([("sarcastic", "sarcastic"), ("not_sarcastic", "not_sarcastic")])
    dataset_a_preds = _preds([("sarcastic", "not_sarcastic"), ("not_sarcastic", "not_sarcastic")])

    out_dir = save_domain_adaptation_result(
        "EXP-SIGN-TEST", {"foo": "bar"}, sign_preds, dataset_a_preds
    )

    assert (out_dir / "config.json").exists()
    assert (out_dir / "sign_test" / "metrics.json").exists()
    assert (out_dir / "sign_test" / "predictions.csv").exists()
    assert (out_dir / "dataset_a_test" / "metrics.json").exists()
    assert (out_dir / "dataset_a_test" / "predictions.csv").exists()

    sign_metrics = json.load(open(out_dir / "sign_test" / "metrics.json"))
    assert sign_metrics["accuracy"] == 1.0

    dataset_a_metrics = json.load(open(out_dir / "dataset_a_test" / "metrics.json"))
    assert dataset_a_metrics["accuracy"] == 0.5
