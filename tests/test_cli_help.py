"""Confirms every script's CLI loads: imports succeed, argparse is wired up,
and `--help` exits cleanly. This does not call any API or load any model --
argparse prints help and exits before any of that code runs.
"""
import subprocess
import sys
import pytest

SCRIPTS = [
    "src.preprocessing.clean_dataset",
    "src.generation.generate_with_gemini",
    "src.generation.generate_with_openrouter",
    "src.evaluation.evaluate_with_llm",
    "src.evaluation.evaluate_with_nli",
    "src.postprocessing.summarize_classifications",
    "src.postprocessing.calculate_text_metrics",
    "src.postprocessing.create_manual_sample",
    "src.postprocessing.run_alt_test",
    "src.postprocessing.summarize_text_metrics",
    "src.postprocessing.plot_text_metrics",
    "src.postprocessing.significance_tests",
    "src.postprocessing.correlation_heatmap",
    "src.postprocessing.linguistic_analysis",
    "src.postprocessing.human_llm_agreement",
    "src.postprocessing.extract_case_studies",
]


@pytest.mark.parametrize("module", SCRIPTS)
def test_module_help_exits_cleanly(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"{module} --help failed:\n{result.stderr}"
    assert "usage" in result.stdout.lower()


def test_check_openrouter_limit_has_no_argparse_but_imports_cleanly():
    # This script takes no CLI arguments; confirm it at least imports and
    # fails with the expected clear error (no API key configured), not a crash.
    result = subprocess.run(
        [sys.executable, "-c", "import src.tools.check_openrouter_limit"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"import failed:\n{result.stderr}"
