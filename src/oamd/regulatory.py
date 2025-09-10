from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import ensure_dir


def export_regulatory_csv(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Export a dataset with a fixed subset of columns and canonical names.

    This provides a simple, consistent format that can be mapped to various regulatory portals.
    """
    cols = [
        "timestamp",
        "site_id",
        "site_type",
        "pH",
        "conductivity_uScm",
        "Fe_mg_L",
        "Mn_mg_L",
        "Al_mg_L",
        "sulfate_mg_L",
    ]
    out = Path(out_path)
    ensure_dir(out.parent)
    df[cols].to_csv(out, index=False)
    return out

