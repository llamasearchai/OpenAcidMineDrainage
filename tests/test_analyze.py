import numpy as np
import pandas as pd

from oamd.analyze import anova_by_site, regression_by_parameter


def _make_equal_mean_dataset(n_per_group: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_per_group, freq="D", tz="UTC")
    base = 7.0 + 0.1 * np.sin(np.linspace(0, 8 * np.pi, n_per_group))

    def mk(site_id: str, site_type: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": dates,
                "site_id": site_id,
                "site_type": site_type,
                "pH": base,  # identical across groups to avoid flakiness
                "conductivity_uScm": 300 + 5 * np.cos(np.linspace(0, 6 * np.pi, n_per_group)),
                "Fe_mg_L": 0.05 + 0.005 * np.sin(np.linspace(0, 2 * np.pi, n_per_group)),
                "Mn_mg_L": 0.02 + 0.002 * np.cos(np.linspace(0, 2 * np.pi, n_per_group)),
                "Al_mg_L": 0.03 + 0.003 * np.sin(np.linspace(0, 4 * np.pi, n_per_group)),
                "sulfate_mg_L": 20 + 0.5 * np.cos(np.linspace(0, 2 * np.pi, n_per_group)),
            }
        )

    df = pd.concat(
        [
            mk("U1", "upstream"),
            mk("D1", "downstream"),
            mk("C1", "control"),
            mk("B1", "baseline"),
        ],
        ignore_index=True,
    )
    return df


def test_anova_by_site_pvalue_high_when_means_equal():
    df = _make_equal_mean_dataset()
    res = anova_by_site(df, "pH")
    assert res.p_value >= 0.99


def test_regression_by_parameter_has_valid_r2():
    df = _make_equal_mean_dataset()
    res = regression_by_parameter(df, "conductivity_uScm")
    assert 0.0 <= res.r2 <= 1.0

