from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from .data_models import Measurement, SiteType
from .utils import normalize_column_name


_CANONICAL_COLUMNS = [
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

# Common synonyms mapping -> canonical columns
_SYNONYMS = {
    "ph": "pH",
    "conductivity": "conductivity_uScm",
    "cond": "conductivity_uScm",
    "fe_mg_l": "Fe_mg_L",
    "mn_mg_l": "Mn_mg_L",
    "al_mg_l": "Al_mg_L",
    "so4_mg_l": "sulfate_mg_L",
    "sulfate": "sulfate_mg_L",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for c in df.columns:
        nc = normalize_column_name(c)
        # Preserve case for pH but match synonyms in lowercase
        lower = nc.lower()
        mapped = _SYNONYMS.get(lower, None)
        if mapped is not None:
            cols[c] = mapped
        elif lower == "ph":
            cols[c] = "pH"
        else:
            cols[c] = nc
    out = df.rename(columns=cols)
    return out


def validate_measurements_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce a DataFrame of measurements into canonical schema.

    The function returns a new DataFrame guaranteed to include exactly the canonical
    columns in the canonical order, with proper dtypes.
    """
    if df.empty:
        raise ValueError("Input dataset is empty")

    df = standardize_columns(df)

    missing = [c for c in _CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate each row via pydantic
    records: List[dict] = []
    for _, row in df[_CANONICAL_COLUMNS].iterrows():
        data = row.to_dict()
        # Normalize site_type
        st = str(data["site_type"]).strip().lower()
        if st not in {s.value for s in SiteType}:
            raise ValueError(f"Invalid site_type: {data['site_type']}")
        data["site_type"] = st
        m = Measurement(**data)
        records.append(m.to_record())

    out = pd.DataFrame.from_records(records, columns=_CANONICAL_COLUMNS)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["site_type"] = out["site_type"].astype("category")
    out["site_id"] = out["site_id"].astype("string")
    numeric_cols = [c for c in _CANONICAL_COLUMNS if c not in ("timestamp", "site_type", "site_id")]
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric)
    return out

