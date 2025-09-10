from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


NUMERIC_PARAMETERS = [
    "pH",
    "conductivity_uScm",
    "Fe_mg_L",
    "Mn_mg_L",
    "Al_mg_L",
    "sulfate_mg_L",
]


@dataclass
class AnovaResult:
    parameter: str
    f_stat: float
    p_value: float
    group_sizes: Dict[str, int]


@dataclass
class RegressionResult:
    parameter: str
    r2: float
    coef: Dict[str, float]
    pvalues: Dict[str, float]


def anova_by_site(df: pd.DataFrame, parameter: str) -> AnovaResult:
    if parameter not in NUMERIC_PARAMETERS:
        raise ValueError(f"Unsupported parameter: {parameter}")
    groups = [g[parameter].values for _, g in df.groupby("site_type", observed=True)]
    labels = [str(k) for k in df.groupby("site_type", observed=True).groups.keys()]
    if len(groups) < 2:
        raise ValueError("Need at least two site groups for ANOVA")
    f_stat, p_val = stats.f_oneway(*groups)
    sizes = {label: int(len(vals)) for label, vals in zip(labels, groups)}
    return AnovaResult(parameter=parameter, f_stat=float(f_stat), p_value=float(p_val), group_sizes=sizes)


def _design_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Encode site_type; drop_first to avoid multicollinearity
    X_cat = pd.get_dummies(df["site_type"], prefix="site", drop_first=True)
    # Time index feature
    t0 = df["timestamp"].min()
    time_idx = (df["timestamp"] - t0).dt.total_seconds() / 3600.0  # hours since start
    X = pd.concat([X_cat, time_idx.rename("time_idx")], axis=1)
    X = sm.add_constant(X, has_constant="add")
    return X, time_idx


def regression_by_parameter(df: pd.DataFrame, parameter: str) -> RegressionResult:
    if parameter not in NUMERIC_PARAMETERS:
        raise ValueError(f"Unsupported parameter: {parameter}")
    X, _ = _design_matrix(df)
    y = df[parameter]
    model = sm.OLS(y, X, missing="drop")
    results = model.fit()
    coef = {k: float(v) for k, v in results.params.items()}
    pvals = {k: float(v) for k, v in results.pvalues.items()}
    return RegressionResult(parameter=parameter, r2=float(results.rsquared), coef=coef, pvalues=pvals)


def seasonal_decompose(df: pd.DataFrame, parameter: str, period: int | None = None) -> pd.DataFrame:
    series = df.set_index("timestamp")[parameter].sort_index()
    if period is None:
        # heuristic: daily series -> weekly seasonality default
        period = 7
    result = sm.tsa.seasonal_decompose(series, model="additive", period=period, two_sided=True)
    out = pd.DataFrame(
        {
            "observed": result.observed,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
        }
    )
    return out

