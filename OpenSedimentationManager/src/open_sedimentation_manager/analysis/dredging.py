from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from open_sedimentation_manager.utils.geoutils import Raster, align_two_rasters, write_raster


def compute_dredging_volume(current_path: str, design_path: str) -> Tuple[float, Dict[str, float]]:
    """Compute total dredging volume (m^3) to reach design depth.

    Volume = sum(max(design_depth - current_depth, 0) * pixel_area)
    """
    current, design_on_current = align_two_rasters(current_path, design_path)

    current_d = current.data.astype(float)
    design_d = design_on_current.data.astype(float)

    thickness = np.clip(design_d - current_d, 0.0, None)

    # Pixel area from transform
    transform = current.profile["transform"]
    pixel_width = abs(transform[0])
    pixel_height = abs(transform[4])
    pixel_area = pixel_width * pixel_height

    volume_m3 = float(np.nansum(thickness) * pixel_area)

    stats = {
        "pixel_area_m2": float(pixel_area),
        "mean_thickness_m": float(np.nanmean(thickness)) if np.isfinite(thickness).any() else 0.0,
        "area_m2": float(np.count_nonzero(thickness > 0) * pixel_area),
    }

    return volume_m3, stats


def cost_benefit_analysis(volume_m3: float, unit_cost: float, stats: Dict[str, float]) -> Dict[str, object]:
    """Return a simple cost-benefit breakdown for plausible strategies.

    Strategies are illustrative and use multipliers for mobilization, environmental mitigation,
    and production rates to estimate time and cost.
    """
    strategies = {
        "mechanical": {
            "mobilization_factor": 1.10,
            "mitigation_factor": 1.05,
            "production_m3_per_day": 5000.0,
        },
        "hydraulic": {
            "mobilization_factor": 1.20,
            "mitigation_factor": 1.02,
            "production_m3_per_day": 12000.0,
        },
        "phased": {
            "mobilization_factor": 1.05,
            "mitigation_factor": 1.10,
            "production_m3_per_day": 4000.0,
        },
    }

    breakdown = {}
    for name, p in strategies.items():
        base_cost = volume_m3 * unit_cost
        cost = base_cost * p["mobilization_factor"] * p["mitigation_factor"]
        days = volume_m3 / max(p["production_m3_per_day"], 1.0)
        breakdown[name] = {
            "cost_usd": float(cost),
            "duration_days": float(days),
            "unit_cost_usd_per_m3": float(cost / max(volume_m3, 1.0)),
        }

    # Simple recommendation: choose lowest cost strategy, adjusted by mean thickness
    mean_thickness = stats.get("mean_thickness_m", 0.0)
    recommended = min(breakdown.items(), key=lambda kv: kv[1]["cost_usd"])[0]
    if mean_thickness > 3.0:
        recommended = "hydraulic"  # deeper cuts benefit from hydraulic throughput
    elif mean_thickness < 1.0:
        recommended = "mechanical"  # thin lenses are efficient mechanically

    return {
        "input": {
            "volume_m3": float(volume_m3),
            "unit_cost_usd_per_m3": float(unit_cost),
            **stats,
        },
        "strategies": breakdown,
        "recommended": recommended,
    }

