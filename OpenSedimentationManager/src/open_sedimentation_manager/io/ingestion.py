from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin

try:
    import xarray as xr  # type: ignore
    import rioxarray  # noqa: F401  # ensure .rio accessor is registered
except Exception:  # pragma: no cover - optional
    xr = None  # type: ignore

from open_sedimentation_manager.utils.geoutils import Raster, write_raster


def _as_dataset_handles(paths: Iterable[str]) -> Tuple[List[rasterio.io.DatasetReader], str]:
    """Open input paths as rasterio datasets. Supports GeoTIFF; NetCDF via rioxarray if available.

    Returns a list of open datasets and a destination CRS (from first dataset).
    """
    datasets: List[rasterio.io.DatasetReader] = []
    dst_crs = None
    for p in paths:
        ext = Path(p).suffix.lower()
        if ext in {".tif", ".tiff"}:
            ds = rasterio.open(p)
            datasets.append(ds)
            if dst_crs is None:
                dst_crs = ds.crs
        elif ext in {".nc", ".nc4"} and xr is not None:
            da = _netcdf_to_rasterio_da(p)
            # Write to an in-memory file that rasterio can read consistently
            memfile = rasterio.io.MemoryFile()
            profile = {
                "driver": "GTiff",
                "height": da.shape[0],
                "width": da.shape[1],
                "count": 1,
                "dtype": str(da.dtype),
                "crs": da.rio.crs,
                "transform": da.rio.transform(),
            }
            with memfile.open(**profile) as dst:
                dst.write(da.values, 1)
            datasets.append(memfile.open())
            if dst_crs is None:
                dst_crs = da.rio.crs
        else:
            raise ValueError(f"Unsupported input format for {p}. Expected GeoTIFF or NetCDF with georeferencing.")
    assert dst_crs is not None, "Failed to infer destination CRS from inputs"
    return datasets, str(dst_crs)


def _netcdf_to_rasterio_da(path: str):  # pragma: no cover - format-dependent
    ds = xr.open_dataset(path)
    # Heuristic: choose first 2D variable with georeferencing
    candidate = None
    for v in ds.data_vars:
        da = ds[v]
        if da.ndim == 2:
            candidate = da
            break
    if candidate is None:
        raise ValueError("No 2D variable found in NetCDF for rasterization.")
    da = candidate
    if not hasattr(da, "rio"):
        raise ValueError("xarray.rio accessor missing; install rioxarray.")
    if da.rio.crs is None:
        # Try to read CRS from dataset attributes
        crs = da.attrs.get("crs") or ds.attrs.get("crs")
        if crs is not None:
            da = da.rio.write_crs(crs, inplace=False)
        else:
            raise ValueError("NetCDF variable lacks CRS information.")
    return da


def ingest_rasters(inputs: List[str], output_path: str) -> None:
    """Ingest multiple rasters and mosaic into a single GeoTIFF aligned to the first input's CRS/resolution.

    - Supports GeoTIFF; NetCDF if rioxarray/xarray available and the file is georeferenced.
    - Resamples inputs to destination grid using bilinear resampling.
    """
    # Expand globs if provided
    expanded: List[str] = []
    for p in inputs:
        matches = glob.glob(p)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(p)
    if not expanded:
        raise ValueError("No input rasters found.")

    datasets, dst_crs = _as_dataset_handles(expanded)

    try:
        mosaic, out_trans = merge(datasets, resampling=Resampling.bilinear)
        # Ensure single-band output if all inputs were single-band
        if mosaic.shape[0] == 1:
            data = mosaic[0]
            count = 1
        else:
            data = mosaic
            count = mosaic.shape[0]

        meta = datasets[0].meta.copy()
        meta.update({
            "height": data.shape[-2],
            "width": data.shape[-1],
            "transform": out_trans,
            "count": count,
            "crs": datasets[0].crs,
        })

        # Write output
        write_raster(output_path, Raster(data=data, profile=meta))
    finally:
        for ds in datasets:
            ds.close()


def xyz_to_raster(
    xyz_path: str,
    output_path: str,
    x_col: int = 0,
    y_col: int = 1,
    z_col: int = 2,
    resolution: float = 1.0,
) -> None:
    """Convert XYZ point cloud (e.g., multibeam sonar) into a gridded GeoTIFF using nearest-neighbor binning.

    Assumes WGS84 UTM or projected coordinates in meters. Output uses a simple transform inferred
    from the data extents.
    """
    pts = np.loadtxt(xyz_path, usecols=[x_col, y_col, z_col], dtype=float)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    cols = int(np.ceil((xmax - xmin) / resolution))
    rows = int(np.ceil((ymax - ymin) / resolution))

    grid = np.full((rows, cols), np.nan, dtype=float)
    count = np.zeros_like(grid, dtype=int)

    col_idx = ((x - xmin) / resolution).astype(int)
    row_idx = ((ymax - y) / resolution).astype(int)

    # Accumulate means
    for r, c, val in zip(row_idx, col_idx, z):
        if 0 <= r < rows and 0 <= c < cols:
            if np.isnan(grid[r, c]):
                grid[r, c] = val
                count[r, c] = 1
            else:
                grid[r, c] += val
                count[r, c] += 1

    with np.errstate(invalid="ignore"):
        grid = np.where(count > 0, grid / np.maximum(count, 1), np.nan)

    transform = from_origin(xmin, ymax, resolution, resolution)

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": str(grid.dtype),
        "crs": "EPSG:3857",  # As a safe default; adjust to your projection as needed
        "transform": transform,
        "nodata": np.nan,
    }
    write_raster(output_path, Raster(data=grid, profile=profile))

