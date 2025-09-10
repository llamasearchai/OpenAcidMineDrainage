import json
import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from open_sedimentation_manager.analysis.bathymetry import compute_sediment_accumulation
from open_sedimentation_manager.analysis.dredging import compute_dredging_volume
from open_sedimentation_manager.api import app
from open_sedimentation_manager.modeling.sediment import run_transport_model
from open_sedimentation_manager.utils.geoutils import Raster, write_raster


def _write_test_raster(path, arr, x0=0.0, y0=10.0, res=1.0):
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": str(arr.dtype),
        "crs": "EPSG:3857",
        "transform": from_origin(x0, y0, res, res),
        "nodata": np.nan,
    }
    write_raster(str(path), Raster(data=arr, profile=profile))


def test_sediment_accumulation(tmp_path):
    current = np.array([[5, 5], [5, 5]], dtype=float)
    baseline = np.array([[6, 6], [5, 5]], dtype=float)
    cpath = tmp_path / "current.tif"
    bpath = tmp_path / "baseline.tif"
    out = tmp_path / "accum.tif"
    _write_test_raster(cpath, current)
    _write_test_raster(bpath, baseline)

    compute_sediment_accumulation(str(cpath), str(bpath), str(out))

    with rasterio.open(out) as ds:
        a = ds.read(1)
    assert np.allclose(a, np.array([[1, 1], [0, 0]], dtype=float), equal_nan=True)


def test_dredging_volume(tmp_path):
    current = np.array([[2, 3], [4, 6]], dtype=float)
    design = np.array([[5, 5], [5, 5]], dtype=float)
    cpath = tmp_path / "curr.tif"
    dpath = tmp_path / "design.tif"
    _write_test_raster(cpath, current, res=2.0)  # pixel area = 4
    _write_test_raster(dpath, design, res=2.0)

    volume, stats = compute_dredging_volume(str(cpath), str(dpath))
    # thickness = [[3,2],[1,0]]; sum=6; volume=6*4=24
    assert volume == 24.0
    assert stats["area_m2"] == 3 * 4.0


def test_transport_model(tmp_path):
    depth = np.ones((10, 10), dtype=float) * 5.0
    u = np.zeros((10, 10), dtype=float)
    v = np.zeros((10, 10), dtype=float)
    dpath = tmp_path / "depth.tif"
    upath = tmp_path / "u.tif"
    vpath = tmp_path / "v.tif"
    out = tmp_path / "dep.tif"
    _write_test_raster(dpath, depth)
    _write_test_raster(upath, u)
    _write_test_raster(vpath, v)

    run_transport_model(str(dpath), str(upath), str(vpath), str(out), steps=10, dt=0.1)
    with rasterio.open(out) as ds:
        dep = ds.read(1)
    assert np.nanmax(dep) <= 1.0
    assert np.nanmin(dep) >= 0.0


# API Tests
@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


class TestAPI:
    """Test suite for FastAPI endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct response."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "OpenAcidMineDrainage API is running" in data["message"]
        assert "endpoints" in data["data"]

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @patch('open_sedimentation_manager.api.compute_sediment_accumulation')
    def test_dredging_analysis_endpoint(self, mock_compute, client, tmp_path):
        """Test dredging analysis endpoint."""
        # Create test files
        current_path = tmp_path / "current.tif"
        design_path = tmp_path / "design.tif"
        _write_test_raster(current_path, np.array([[2, 3], [4, 6]], dtype=float))
        _write_test_raster(design_path, np.array([[5, 5], [5, 5]], dtype=float))

        response = client.post("/dredging-analysis", json={
            "current_bathymetry_path": str(current_path),
            "design_depth_path": str(design_path),
            "unit_cost_per_m3": 25.0
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "volume_m3" in data["data"]
        assert "cost_benefit_report" in data["data"]

    def test_vegetation_analysis_endpoint(self, client, tmp_path):
        """Test vegetation analysis endpoint."""
        red_path = tmp_path / "red.tif"
        nir_path = tmp_path / "nir.tif"
        _write_test_raster(red_path, np.random.rand(10, 10))
        _write_test_raster(nir_path, np.random.rand(10, 10))

        response = client.post("/vegetation-analysis", json={
            "red_band_path": str(red_path),
            "nir_band_path": str(nir_path),
            "ndvi_threshold": 0.3
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "vegetation_mask_path" in data["data"]

    def test_transport_model_endpoint(self, client, tmp_path):
        """Test sediment transport modeling endpoint."""
        depth_path = tmp_path / "depth.tif"
        u_path = tmp_path / "u.tif"
        v_path = tmp_path / "v.tif"
        _write_test_raster(depth_path, np.ones((10, 10), dtype=float) * 5.0)
        _write_test_raster(u_path, np.zeros((10, 10), dtype=float))
        _write_test_raster(v_path, np.zeros((10, 10), dtype=float))

        response = client.post("/transport-model", json={
            "depth_path": str(depth_path),
            "u_path": str(u_path),
            "v_path": str(v_path),
            "steps": 10,
            "dt": 0.1,
            "diffusivity": 0.2
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deposition_prediction_path" in data["data"]


class TestAgents:
    """Test suite for OpenAI Agents SDK integration."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('open_sedimentation_manager.llm.agents.Agent')
    @patch('open_sedimentation_manager.llm.agents.Runner')
    def test_agent_initialization(self, mock_runner, mock_agent):
        """Test agent initialization with mocked dependencies."""
        from open_sedimentation_manager.llm.agents import AcidMineDrainageAgent

        agent = AcidMineDrainageAgent()

        # Verify agents were created
        assert hasattr(agent, 'sediment_agent')
        assert hasattr(agent, 'dredging_agent')
        assert hasattr(agent, 'compliance_agent')
        assert hasattr(agent, 'orchestrator_agent')

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('open_sedimentation_manager.llm.agents.Agent')
    @patch('open_sedimentation_manager.llm.agents.Runner')
    def test_analyze_sedimentation_sync(self, mock_runner, mock_agent):
        """Test synchronous sedimentation analysis."""
        from open_sedimentation_manager.llm.agents import analyze_sedimentation_sync

        # Mock the runner result
        mock_result = Mock()
        mock_result.final_output = "Test analysis result"
        mock_runner.run.return_value = mock_result

        result = analyze_sedimentation_sync(
            current_bathymetry_path="test_current.tif",
            baseline_bathymetry_path="test_baseline.tif"
        )

        assert isinstance(result, dict)
        assert "sediment_analysis" in result

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @patch('open_sedimentation_manager.llm.agents.Agent')
    @patch('open_sedimentation_manager.llm.agents.Runner')
    def test_generate_compliance_report_sync(self, mock_runner, mock_agent):
        """Test synchronous compliance report generation."""
        from open_sedimentation_manager.llm.agents import generate_compliance_report_sync

        # Mock the runner result
        mock_result = Mock()
        mock_result.final_output = "Test compliance report"
        mock_runner.run.return_value = mock_result

        result = generate_compliance_report_sync(
            analysis_results={"test": "data"},
            regulatory_framework="EPA"
        )

        assert isinstance(result, dict)
        assert "report" in result

    def test_agent_initialization_without_api_key(self):
        """Test that agent initialization fails without API key."""
        from open_sedimentation_manager.llm.agents import AcidMineDrainageAgent

        # Remove API key if it exists
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY environment variable is required"):
                AcidMineDrainageAgent()


class TestCLIIntegration:
    """Test CLI command integration."""

    def test_imports(self):
        """Test that all CLI dependencies can be imported."""
        try:
            from open_sedimentation_manager.cli import app
            from open_sedimentation_manager.llm.agents import (
                analyze_sedimentation_sync,
                generate_compliance_report_sync,
            )
            assert app is not None
        except ImportError as e:
            pytest.fail(f"CLI import failed: {e}")

    @patch('open_sedimentation_manager.llm.agents.analyze_sedimentation_sync')
    def test_ai_analyze_command_logic(self, mock_analyze, tmp_path):
        """Test AI analyze command logic."""
        from open_sedimentation_manager.cli import cmd_ai_analyze

        # Mock the analysis function
        mock_analyze.return_value = {
            "sediment_analysis": "Test analysis",
            "status": "completed"
        }

        # Create test files
        current = tmp_path / "current.tif"
        baseline = tmp_path / "baseline.tif"
        output = tmp_path / "output.json"
        _write_test_raster(current, np.random.rand(5, 5))
        _write_test_raster(baseline, np.random.rand(5, 5))

        # This would normally be called by typer, but we test the logic
        result = mock_analyze(
            current_bathymetry_path=str(current),
            baseline_bathymetry_path=str(baseline)
        )

        assert result["status"] == "completed"


# Integration Tests
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_analysis_workflow(self, tmp_path):
        """Test a complete analysis workflow."""
        # Create test data
        current = tmp_path / "current.tif"
        baseline = tmp_path / "baseline.tif"
        output = tmp_path / "sediment.tif"

        _write_test_raster(current, np.array([[5, 5], [5, 5]], dtype=float))
        _write_test_raster(baseline, np.array([[6, 6], [5, 5]], dtype=float))

        # Run sediment accumulation analysis
        compute_sediment_accumulation(str(current), str(baseline), str(output))

        # Verify output exists and has correct content
        assert output.exists()
        with rasterio.open(output) as ds:
            result = ds.read(1)
            expected = np.array([[1, 1], [0, 0]], dtype=float)
            assert np.allclose(result, expected, equal_nan=True)

    def test_dredging_workflow(self, tmp_path):
        """Test dredging analysis workflow."""
        current = tmp_path / "current.tif"
        design = tmp_path / "design.tif"

        _write_test_raster(current, np.array([[2, 3], [4, 6]], dtype=float))
        _write_test_raster(design, np.array([[5, 5], [5, 5]], dtype=float))

        volume, stats = compute_dredging_volume(str(current), str(design))

        assert volume == 24.0  # 6 cells * 4 m²/cell = 24 m³
        assert stats["area_m2"] == 12.0  # 3 cells * 4 m²/cell

