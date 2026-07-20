"""Shared CSV and filesystem helpers."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a file if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv_flexible(path: Path, expected_columns: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV and optionally normalize unnamed columns.

    Many original files were created without headers. If expected_columns is
    provided and the file does not contain those columns, the file is read again
    with header=None and the provided column names.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    if expected_columns and not set(expected_columns).issubset(df.columns):
        df = pd.read_csv(path, header=None, encoding="utf-8-sig")
        df = df.iloc[:, : len(expected_columns)]
        df.columns = expected_columns
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as UTF-8-SIG CSV for Excel compatibility."""
    ensure_parent_dir(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_all_model_outputs(model_outputs_dir: Path) -> pd.DataFrame:
    """Load every classified CSV under data/model_outputs/ into one DataFrame.

    Adds a `prompt` column (the experiment number, e.g. `experiment_04` -> 4)
    and a `model` column (the first token of the filename, e.g.
    `nvidia_run_04.csv` -> "nvidia"). Rows without a numeric `classification`
    are dropped. Used by the significance/correlation/linguistic analyses,
    which need the full labeled dataset rather than a single file.
    """
    frames = []
    for csv_file in sorted(model_outputs_dir.glob("experiment_*/*.csv")):
        df = pd.read_csv(csv_file, encoding="utf-8-sig")
        df["prompt"] = int(csv_file.parent.name.split("_")[-1])
        df["model"] = csv_file.stem.split("_")[0]
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["classification"] = pd.to_numeric(combined["classification"], errors="coerce")
    return combined.dropna(subset=["classification"])
