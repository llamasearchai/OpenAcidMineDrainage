from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape, mapping

from open_sedimentation_manager.utils.geoutils import open_raster


def _vectorize_threshold(mask: np.ndarray, transform) -> List[Dict]:
    """Convert boolean mask to polygons (GeoJSON-like features without properties)."""
    geoms = []
    for geom, val in features.shapes(mask.astype(np.uint8), mask=mask > 0, transform=transform):
        if val == 1:
            geoms.append(geom)
    return geoms


def generate_work_orders(sediment_path: str, threshold_m: float, output_geojson_path: str) -> None:
    """Generate GeoJSON polygons for dredging areas where sediment accumulation >= threshold.

    Each feature includes area (m^2), estimated volume (m^3), and centroid coordinates.
    """
    with rasterio.open(sediment_path) as ds:
        accum = ds.read(1).astype(float)
        transform = ds.transform
        crs = ds.crs

    pixel_area = abs(transform[0]) * abs(transform[4])

    # Mask where accumulation meets threshold
    mask = np.where(np.isfinite(accum) & (accum >= threshold_m), 1, 0).astype(np.uint8)

    geoms = _vectorize_threshold(mask, transform)

    features_out = []
    for idx, geom in enumerate(geoms, start=1):
        shp = shape(geom)
        # Compute mean thickness within this polygon using a per-geometry mask
        poly_mask = features.geometry_mask([geom], out_shape=accum.shape, transform=transform, invert=True)
        vals = accum[poly_mask]
        mean_thickness = float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
        area = float(shp.area)  # in CRS units^2; assumes meters
        volume = mean_thickness * area
        centroid = shp.centroid
        features_out.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "id": idx,
                    "area_m2": area,
                    "mean_thickness_m": mean_thickness,
                    "volume_m3": volume,
                    "centroid_x": float(centroid.x),
                    "centroid_y": float(centroid.y),
                    "crs": str(crs),
                    "threshold_m": float(threshold_m),
                },
            }
        )

    fc = {"type": "FeatureCollection", "features": features_out}
    Path(output_geojson_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_geojson_path).write_text(json.dumps(fc, indent=2))

