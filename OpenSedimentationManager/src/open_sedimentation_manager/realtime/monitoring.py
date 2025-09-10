from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


def _read_timeseries(path: str) -> List[Dict[str, float]]:
    p = Path(path)
    if p.suffix.lower() in {".json"}:
        data = json.loads(p.read_text())
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        if not isinstance(data, list):
            raise ValueError("JSON must be a list of records or have a 'records' list.")
        return data  # type: ignore
    elif p.suffix.lower() in {".csv"}:
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]  # type: ignore
    else:
        raise ValueError("Unsupported realtime input format. Use .json or .csv")


def process_realtime_flow(input_path: str, output_path: str) -> None:
    """Process real-time flow measurements and estimate immediate sedimentation impact risk.

    Expects JSON or CSV with fields like 'flow_cms' (cubic meters per second) and optionally 'turbidity_ntu'.
    Outputs a JSON summary with basic statistics and a qualitative risk score.
    """
    records = _read_timeseries(input_path)

    flows = []
    turbs = []
    for r in records:
        try:
            if "flow_cms" in r:
                flows.append(float(r["flow_cms"]))
            if "turbidity_ntu" in r:
                turbs.append(float(r["turbidity_ntu"]))
        except Exception:
            # Skip malformed rows
            continue

    summary: Dict[str, float | str] = {}
    if flows:
        summary.update(
            flow_mean_cms=mean(flows),
            flow_std_cms=pstdev(flows) if len(flows) > 1 else 0.0,
            flow_min_cms=min(flows),
            flow_max_cms=max(flows),
        )
    if turbs:
        summary.update(
            turb_mean_ntu=mean(turbs),
            turb_std_ntu=pstdev(turbs) if len(turbs) > 1 else 0.0,
            turb_min_ntu=min(turbs),
            turb_max_ntu=max(turbs),
        )

    # Qualitative risk assessment
    risk_score = 0.0
    if flows:
        # Higher flows can remobilize sediment; very low flows can promote deposition.
        f_mean = summary.get("flow_mean_cms", 0.0)  # type: ignore
        f_std = summary.get("flow_std_cms", 0.0)  # type: ignore
        risk_score += 0.5 * (f_std / (f_mean + 1e-6))
        if f_mean < 5.0:
            risk_score += 0.5  # deposition favored
        elif f_mean > 200.0:
            risk_score += 0.3  # scour/remobilization
    if turbs:
        t_mean = summary.get("turb_mean_ntu", 0.0)  # type: ignore
        if t_mean > 50.0:
            risk_score += 0.4

    level = "low"
    if risk_score > 1.0:
        level = "moderate"
    if risk_score > 1.8:
        level = "high"

    output = {
        "summary": summary,
        "risk_score": float(risk_score),
        "risk_level": level,
        "records_analyzed": int(len(flows)),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2))

