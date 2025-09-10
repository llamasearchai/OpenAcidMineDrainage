from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling

from open_sedimentation_manager.utils.geoutils import Raster, open_raster, write_raster


def _finite_diff_advection_diffusion(
    C: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    steps: int,
    diffusivity: float,
    sink_rate: float = 0.01,
    mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evolve scalar concentration C with advection-diffusion and linear sink.

    Returns (C_final, deposition), where deposition integrates sink term over time.
    """
    ny, nx = C.shape
    dep = np.zeros_like(C)

    for _ in range(steps):
        # 2nd order central differences for diffusion (5-point Laplacian)
        lap = (
            -4 * C
            + np.roll(C, 1, axis=0)
            + np.roll(C, -1, axis=0)
            + np.roll(C, 1, axis=1)
            + np.roll(C, -1, axis=1)
        ) / (dx * dy)

        # Upwind advection
        dCdx = np.where(u >= 0, C - np.roll(C, 1, axis=1), np.roll(C, -1, axis=1) - C) / dx
        dCdy = np.where(v >= 0, C - np.roll(C, 1, axis=0), np.roll(C, -1, axis=0) - C) / dy

        adv = -(u * dCdx + v * dCdy)
        diff = diffusivity * lap
        sink = -sink_rate * C

        dC = dt * (adv + diff + sink)
        C = C + dC

        # Reflective boundaries (Neumann ~0 gradient) via edge copies
        C[0, :] = C[1, :]
        C[-1, :] = C[-2, :]
        C[:, 0] = C[:, 1]
        C[:, -1] = C[:, -2]

        dep = dep + dt * (-sink)  # deposition accumulates the mass removed by sink

        if mask is not None:
            C = np.where(mask, C, 0.0)
            dep = np.where(mask, dep, 0.0)

    return C, dep


def run_transport_model(
    depth_path: str,
    u_path: str,
    v_path: str,
    output_path: str,
    steps: int = 200,
    dt: float = 0.5,
    diffusivity: float = 0.2,
    sink_rate: float = 0.01,
) -> None:
    """Run a simple advection-diffusion model to predict relative deposition potential.

    Inputs are rasters on the same grid (depth in meters, u and v in m/s). If grids differ, u and v are
    resampled to match depth.
    """
    depth = open_raster(depth_path)
    with rasterio.open(u_path) as ds_u, rasterio.open(v_path) as ds_v, rasterio.open(depth_path) as ds_ref:
        u_resampled = np.zeros((ds_ref.height, ds_ref.width), dtype=np.float32)
        v_resampled = np.zeros((ds_ref.height, ds_ref.width), dtype=np.float32)
        rasterio.warp.reproject(
            source=ds_u.read(1),
            destination=u_resampled,
            src_transform=ds_u.transform,
            src_crs=ds_u.crs,
            dst_transform=ds_ref.transform,
            dst_crs=ds_ref.crs,
            resampling=Resampling.bilinear,
        )
        rasterio.warp.reproject(
            source=ds_v.read(1),
            destination=v_resampled,
            src_transform=ds_v.transform,
            src_crs=ds_v.crs,
            dst_transform=ds_ref.transform,
            dst_crs=ds_ref.crs,
            resampling=Resampling.bilinear,
        )

    # Grid spacing from affine transform
    dx = abs(depth.profile["transform"][0])
    dy = abs(depth.profile["transform"][4])

    # Stability safety: CFL-like cap for dt
    max_u = float(np.nanmax(np.abs(u_resampled))) if np.isfinite(u_resampled).any() else 0.0
    max_v = float(np.nanmax(np.abs(v_resampled))) if np.isfinite(v_resampled).any() else 0.0
    cfl_dt = np.inf
    if max_u > 0:
        cfl_dt = min(cfl_dt, 0.4 * dx / max_u)
    if max_v > 0:
        cfl_dt = min(cfl_dt, 0.4 * dy / max_v)
    if np.isfinite(cfl_dt):
        dt = min(dt, cfl_dt)

    # Initialize concentration higher in regions of low velocity (proxy for suspended load residence)
    speed = np.sqrt(u_resampled ** 2 + v_resampled ** 2)
    eps = 1e-6
    C0 = 1.0 / (speed + eps)
    C0 = C0 / np.nanmax(C0)

    # Mask non-water where depth <= 0 (if depth is positive for water)
    mask = depth.data > 0

    _, dep = _finite_diff_advection_diffusion(
        C=C0.astype(np.float32),
        u=u_resampled.astype(np.float32),
        v=v_resampled.astype(np.float32),
        dx=float(dx),
        dy=float(dy),
        dt=float(dt),
        steps=int(steps),
        diffusivity=float(diffusivity),
        sink_rate=float(sink_rate),
        mask=mask,
    )

    # Normalize deposition and write
    dep = np.where(mask, dep, np.nan)
    if np.isfinite(dep).any():
        dep = dep / np.nanmax(dep)

    write_raster(output_path, Raster(data=dep.astype(np.float32), profile=depth.profile))

