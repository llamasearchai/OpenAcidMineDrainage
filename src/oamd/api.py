from __future__ import annotations

from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from .analyze import NUMERIC_PARAMETERS, anova_by_site, regression_by_parameter
from .ml_forecast import forecast_for_site


app = FastAPI(title="OpenAcidMineDrainage API", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(payload["records"])  # expects timestamp parsed by client or ISO strings
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        param = payload.get("param", "pH")
        if param not in NUMERIC_PARAMETERS:
            raise HTTPException(status_code=400, detail=f"Unsupported parameter {param}")
        a = anova_by_site(df, param)
        r = regression_by_parameter(df, param)
        return {
            "anova": {"f_stat": a.f_stat, "p_value": a.p_value, "group_sizes": a.group_sizes},
            "regression": {"r2": r.r2, "coef": r.coef, "pvalues": r.pvalues},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/forecast")
def forecast(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(payload["records"])  # expects timestamp parsed by client or ISO strings
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        site_id = payload["site_id"]
        param = payload.get("param", "pH")
        horizon = int(payload.get("horizon", 14))
        res = forecast_for_site(df, site_id, param, horizon)
        return {
            "mae": res.mae,
            "predictions": res.predictions.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


