import numpy as np
import pandas as pd

from oamd.ml_forecast import forecast_for_site


def _make_site_series(n: int = 200, site_id: str = "D1") -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    y = 7.0 + 0.2 * np.sin(np.linspace(0, 6 * np.pi, n))
    df = pd.DataFrame(
        {
            "timestamp": t,
            "site_id": site_id,
            "site_type": "downstream",
            "pH": y,
            "conductivity_uScm": 300 + 10 * np.cos(np.linspace(0, 6 * np.pi, n)),
            "Fe_mg_L": 0.05 + 0.005 * np.sin(np.linspace(0, 2 * np.pi, n)),
            "Mn_mg_L": 0.02 + 0.002 * np.cos(np.linspace(0, 2 * np.pi, n)),
            "Al_mg_L": 0.03 + 0.003 * np.sin(np.linspace(0, 4 * np.pi, n)),
            "sulfate_mg_L": 20 + 0.5 * np.cos(np.linspace(0, 2 * np.pi, n)),
        }
    )
    return df


def test_forecast_for_site_shapes():
    df = _make_site_series()
    res = forecast_for_site(df, "D1", "pH", horizon=14)
    assert len(res.predictions) == 14
    assert set(res.predictions.columns) == {"timestamp", "y_true", "y_pred"}

