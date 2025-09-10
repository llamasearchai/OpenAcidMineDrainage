from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .validation import validate_measurements_df
from .utils import ensure_dir, project_artifacts_dir


def load_csvs(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)
    if not frames:
        raise ValueError("No input CSV files provided")
    merged = pd.concat(frames, ignore_index=True)
    return merged


def ingest_csvs(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load, normalize, and validate measurements from the given CSV files."""
    raw = load_csvs(paths)
    clean = validate_measurements_df(raw)
    return clean


def write_clean_csv(df: pd.DataFrame, output: str | Path | None = None) -> Path:
    out_dir = project_artifacts_dir()
    if output is None:
        output = out_dir / "clean.csv"
    else:
        output = Path(output)
        ensure_dir(output.parent)
    df.to_csv(output, index=False)
    return Path(output)

