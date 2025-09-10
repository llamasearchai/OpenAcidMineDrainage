from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.warp import reproject


@dataclass
class Raster:
    data: np.ndarray  # shape: (bands, rows, cols) or (rows, cols)
    profile: dict


def open_raster(path: str) -> Raster:
    """Open a raster and return data and profile.

    Returns single-band as 2D array and multi-band as (bands, rows, cols).
    """
    with rasterio.open(path) as ds:
        data = ds.read()
        if data.shape[0] == 1:
            data = data[0]
        profile = ds.profile
    return Raster(data=data, profile=profile)


def write_raster(path: str, raster: Raster) -> None:
    """Write a raster to disk. Converts 2D data to single-band."""
    profile = raster.profile.copy()
    data = raster.data
    if data.ndim == 2:
        profile.update(count=1)
        data_to_write = data[np.newaxis, :, :]
    elif data.ndim == 3:
        profile.update(count=data.shape[0])
        data_to_write = data
    else:
        raise ValueError("Unsupported raster data dimensions")

    profile.setdefault("compress", "deflate")
    profile.setdefault("tiled", True)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data_to_write)


def reproject_to_match(
    src: DatasetReader,
    match: DatasetReader,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, dict]:
    """Reproject src dataset to match dataset's grid.

    Returns data (bands, rows, cols) and profile dict.
    """
    dst_profile = match.profile.copy()
    dst_count = src.count
    dst_profile.update(count=dst_count)

    dst_data = np.zeros((dst_count, match.height, match.width), dtype=src.dtypes[0])

    for b in range(1, dst_count + 1):
        reproject(
            source=rasterio.band(src, b),
            destination=dst_data[b - 1],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=match.transform,
            dst_crs=match.crs,
            resampling=resampling,
        )
    return dst_data, dst_profile


def align_two_rasters(path_a: str, path_b: str) -> Tuple[Raster, Raster]:
    """Reproject B onto A's grid so both align. Returns (A, B_on_A)."""
    with rasterio.open(path_a) as ds_a, rasterio.open(path_b) as ds_b:
        a = ds_a.read()
        if a.shape[0] == 1:
            a = a[0]
        a_profile = ds_a.profile
        b_on_a, b_profile = reproject_to_match(ds_b, ds_a)
        if b_on_a.shape[0] == 1:
            b_on_a = b_on_a[0]
    return Raster(a, a_profile), Raster(b_on_a, b_profile)

