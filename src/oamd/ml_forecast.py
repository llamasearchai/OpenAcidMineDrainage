from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


@dataclass
class ForecastResult:
    parameter: str
    site_id: str
    mae: float
    horizon: int
    predictions: pd.DataFrame  # columns: timestamp, y_true, y_pred


def _make_features(df: pd.DataFrame, parameter: str, lags: int = 3) -> pd.DataFrame:
    work = df.copy().sort_values("timestamp")
    # time features
    work["month"] = work["timestamp"].dt.month
    work["dayofyear"] = work["timestamp"].dt.dayofyear
    work["dow"] = work["timestamp"].dt.dayofweek
    # lag features
    for i in range(1, lags + 1):
        work[f"lag_{i}"] = work[parameter].shift(i)
    # drop rows with NA from lags
    work = work.dropna().reset_index(drop=True)
    return work


def forecast_for_site(df: pd.DataFrame, site_id: str, parameter: str, horizon: int = 14) -> ForecastResult:
    site_df = df[df["site_id"] == site_id].copy()
    if site_df.empty:
        raise ValueError(f"No data for site_id={site_id}")
    work = _make_features(site_df, parameter, lags=3)

    feature_cols = [c for c in work.columns if c not in {parameter, "timestamp", "site_id", "site_type"}]
    X = work[feature_cols]
    y = work[parameter]

    # Train/test split (last horizon points as test)
    if len(work) <= horizon + 1:
        raise ValueError("Not enough data to create a forecast; provide a longer series")
    split = len(work) - horizon
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))

    pred_df = pd.DataFrame(
        {
            "timestamp": work["timestamp"].iloc[split:].values,
            "y_true": y_test.values,
            "y_pred": y_pred,
        }
    )
    return ForecastResult(parameter=parameter, site_id=site_id, mae=mae, horizon=horizon, predictions=pred_df)

