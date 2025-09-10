from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests


@dataclass
class Alert:
    timestamp: str
    site_id: str
    parameter: str
    value: float
    rule: str


DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "pH": {"min": 6.0, "max": 9.0},
    "conductivity_uScm": {"max": 2000.0},
    "Fe_mg_L": {"max": 1.5},
    "Mn_mg_L": {"max": 1.0},
    "Al_mg_L": {"max": 0.75},
    "sulfate_mg_L": {"max": 250.0},
}


def check_thresholds(df: pd.DataFrame, thresholds: Optional[Dict[str, Dict[str, float]]] = None) -> List[Alert]:
    th = thresholds or DEFAULT_THRESHOLDS
    alerts: List[Alert] = []
    for param, rules in th.items():
        if param not in df.columns:
            continue
        for _, row in df.iterrows():
            v = float(row[param])
            ts = str(row["timestamp"])  # already isoformat in cleaned CSV
            site_id = str(row["site_id"])
            if "min" in rules and v < rules["min"]:
                alerts.append(Alert(ts, site_id, param, v, f"{param} < {rules['min']}"))
            if "max" in rules and v > rules["max"]:
                alerts.append(Alert(ts, site_id, param, v, f"{param} > {rules['max']}"))
    return alerts


def send_alerts(alerts: List[Alert]) -> None:
    webhook = os.environ.get("OAMD_ALERT_WEBHOOK", "").strip()
    if not webhook or not alerts:
        return
    payload = {"alerts": [a.__dict__ for a in alerts]}
    try:
        requests.post(webhook, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=10)
    except Exception:
        # Do not raise in library code; logging could be added here if needed
        pass

