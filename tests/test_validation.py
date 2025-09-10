import numpy as np
import pandas as pd

from oamd.validation import validate_measurements_df


def test_validate_measurements_df_happy_path():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
            "site_id": ["U1", "D1"],
            "site_type": ["upstream", "downstream"],
            "pH": [7.2, 7.1],
            "conductivity_uScm": [350.0, 360.0],
            "Fe_mg_L": [0.05, 0.04],
            "Mn_mg_L": [0.01, 0.02],
            "Al_mg_L": [0.02, 0.02],
            "sulfate_mg_L": [20.0, 22.0],
        }
    )
    out = validate_measurements_df(df)
    assert list(out.columns) == [
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
    assert pd.api.types.is_datetime64tz_dtype(out["timestamp"])  # timezone-aware
    assert set(out["site_type"].cat.categories) >= {"upstream", "downstream"}

