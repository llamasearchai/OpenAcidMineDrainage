# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Repository overview
- Language/runtime: Python (3.10+). Package name: OpenSedimentationManager. Source lives under src/open_sedimentation_manager.
- Packaging: pyproject.toml with hatchling build backend. Console script entrypoint: osm (open_sedimentation_manager.cli:main).

Commands
- Environment setup (macOS/zsh)
  - python3 -m venv .venv
  - source .venv/bin/activate
  - python -m pip install --upgrade pip wheel
- Install
  - Editable install: pip install -e .
  - Optional extras: pip install -e .[hydro] (NetCDF), pip install -e .[pybathy], pip install -e .[gdal], pip install -e .[llm], pip install -e .[datasette]
  - Note: GDAL and some geospatial packages may require system libraries (e.g., gdal, geos, proj). Install them via your OS package manager before pip install if needed.
- uv (optional fast installer)
  - Create venv: uv venv
  - Install package: uv pip install -e .[hydro,pybathy,gdal]
  - Install LLM/Datasette extras as needed: uv pip install -e .[llm,datasette]
- CLI usage
  - List commands: osm --help
  - Typical flow examples:
    - Mosaic/align rasters: osm ingest-raster --inputs data/*.tif --output outputs/combined_bathy.tif
    - Accumulation vs baseline: osm bathy-analyze --current outputs/combined_bathy.tif --baseline data/baseline_bathy.tif --output outputs/sediment_accum.tif
    - Transport model: osm model-transport --depth outputs/combined_bathy.tif --u data/flow_u.tif --v data/flow_v.tif --output outputs/deposition_pred.tif
    - Vectorize dredging areas: osm work-orders --sediment outputs/sediment_accum.tif --threshold 0.5 --output outputs/work_orders.geojson
    - LLM narrative in report (set OPENAI_API_KEY; optional model via OSM_OPENAI_MODEL): osm compliance-report --inputs outputs/dredging_report.json --output outputs/compliance_report.md --use-llm
    - LLM-rewritten notifications: osm notify --source outputs/work_orders.geojson --output outputs/notifications.json --use-llm --tone "public advisory"
    - Export to SQLite for Datasette: osm export-datasette --source outputs/work_orders.geojson --sqlite outputs/osm.db --table work_orders
- Tests
  - Install pytest (not declared in pyproject): python -m pip install -U pytest
  - Run all tests: pytest -q
  - Run a single test: pytest tests/test_basic.py::test_sediment_accumulation -q
  - tox (isolated runner): tox -q
  - tox single test: tox -q -e py310 -- tests/test_basic.py::test_sediment_accumulation -q
- Build / task runners
  - Build sdist/wheel: python -m pip install build && python -m build
  - hatch build (if installed): hatch build
  - hatch test runner: hatch run test
- Explore with Datasette (optional)
  - Install CLI: pipx install datasette (or pip install datasette)
  - Explore DB: datasette outputs/osm.db
- Lint/format
  - No linter or formatter is configured in-repo as of this version.

High-level architecture (big picture)
- CLI orchestration (Typer): src/open_sedimentation_manager/cli.py wires subcommands to implementation functions. The osm entrypoint dispatches to these modules:
  - ingest-raster → io.ingestion.ingest_rasters: opens/mosaics GeoTIFF inputs (and NetCDF if xarray/rioxarray present), resamples to a common grid (bilinear), and writes a GeoTIFF.
  - bathy-analyze → analysis.bathymetry.compute_sediment_accumulation: aligns rasters and computes thickness = max(baseline − current, 0) assuming depths are positive meters.
  - model-transport → modeling.sediment.run_transport_model: 2D advection–diffusion with a sink term; resamples flow fields to depth grid, simulates, normalizes deposition potential, writes raster.
  - dredge-calc → analysis.dredging.compute_dredging_volume + cost_benefit_analysis: thickness above design depth, pixel-area weighted volume (m³), simple cost/duration strategies.
  - vegetation-detect → vegetation.management.detect_invasive_species: NDVI = (NIR−RED)/(NIR+RED), threshold to uint8 mask.
  - train-ml / predict-ml → ml.model.train_ml / predict_ml: RandomForest on numeric features; persists artifact with features/metrics; predicts to CSV.
  - compliance-report → compliance.report.generate_compliance_report: aggregates JSON artifacts into a Markdown report (includes brief methodology and considerations).
  - work-orders → work_orders.generator.generate_work_orders: threshold accumulation raster, polygonize to GeoJSON with area (m²), mean thickness (m), volume (m³), centroid, CRS.
  - realtime-process → realtime.monitoring.process_realtime_flow: summarize timeseries (flow_cms, turbidity_ntu) to basic stats and a qualitative risk score.
  - notify → notifications.notify.generate_notifications: produce audience-tagged notification messages from a GeoJSON FeatureCollection or generic JSON.
- Geospatial foundation (utils/geoutils.py)
  - Raster dataclass encapsulates data + profile. Helpers for open/write, reprojection to match grids, and pairwise alignment (align_two_rasters).
  - Pixel area is derived from GeoTIFF affine transform; resampling defaults to rasterio bilinear where applicable.
  - Many subsystems depend on these utilities for consistent CRS/grid alignment and I/O.
- Configuration
  - config/config.example.yaml provides a reference of commonly used paths and parameters (NDVI threshold, sediment threshold, model params). It is not auto-loaded by the CLI; pass explicit paths/options via commands.

README highlights to keep in mind here
- Requires Python 3.10+.
- Install with pip install -e . (use extras as needed).
- CLI entrypoint is osm; see README for extended examples beyond the minimal ones listed above.

Project-specific rule
- Never use emojis, stubs, or placeholders in this repository. Ensure all content is complete and concrete, including docs such as this file.

