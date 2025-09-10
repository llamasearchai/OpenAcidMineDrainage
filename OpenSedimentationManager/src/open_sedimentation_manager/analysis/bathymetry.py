from __future__ import annotations

import numpy as np
from open_sedimentation_manager.utils.geoutils import align_two_rasters, Raster, write_raster


def compute_sediment_accumulation(current_path: str, baseline_path: str, output_path: str) -> None:
    """Compute sediment accumulation thickness (m) = max(baseline_depth - current_depth, 0).

    Assumes depths are positive values in meters. If current is shallower than baseline, we infer
    positive accumulation thickness.
    """
    current, baseline_on_current = align_two_rasters(current_path, baseline_path)

    current_data = current.data.astype(float)
    baseline_data = baseline_on_current.data.astype(float)

    with np.errstate(invalid="ignore"):
        accum = np.clip(baseline_data - current_data, 0.0, None)

    out = Raster(data=accum, profile=current.profile)
    write_raster(output_path, out)

