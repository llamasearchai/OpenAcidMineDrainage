from __future__ import annotations

import numpy as np

from open_sedimentation_manager.utils.geoutils import Raster, align_two_rasters, write_raster


def detect_invasive_species(red_path: str, nir_path: str, output_path: str, ndvi_threshold: float = 0.3) -> None:
    """Detect vegetation via NDVI and output a binary mask GeoTIFF (1=vegetation, 0=non-vegetation).

    This function aligns NIR to the RED raster grid, computes NDVI = (NIR-RED)/(NIR+RED), and thresholds it.
    """
    red, nir_on_red = align_two_rasters(red_path, nir_path)
    red_band = red.data.astype(float)
    nir_band = nir_on_red.data.astype(float)

    eps = 1e-6
    ndvi = (nir_band - red_band) / (nir_band + red_band + eps)

    mask = (ndvi >= ndvi_threshold).astype(np.uint8)

    out = Raster(data=mask, profile={**red.profile, "dtype": "uint8", "count": 1, "nodata": 0})
    write_raster(output_path, out)

