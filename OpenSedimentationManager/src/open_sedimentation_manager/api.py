"""
FastAPI endpoints for OpenAcidMineDrainage system.
Provides REST API for web-based interaction with the sedimentation analysis system.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from open_sedimentation_manager.analysis.bathymetry import (
    compute_sediment_accumulation,
)
from open_sedimentation_manager.analysis.dredging import (
    compute_dredging_volume,
    cost_benefit_analysis,
)
from open_sedimentation_manager.compliance.report import generate_compliance_report
from open_sedimentation_manager.io.ingestion import ingest_rasters
from open_sedimentation_manager.llm.agents import (
    analyze_sedimentation_sync,
    generate_compliance_report_sync,
    SedimentAnalysisRequest,
    ComplianceReportRequest,
)
from open_sedimentation_manager.ml.model import predict_ml, train_ml
from open_sedimentation_manager.modeling.sediment import run_transport_model
from open_sedimentation_manager.notifications.notify import generate_notifications
from open_sedimentation_manager.realtime.monitoring import process_realtime_flow
from open_sedimentation_manager.vegetation.management import detect_invasive_species
from open_sedimentation_manager.work_orders.generator import generate_work_orders
from open_sedimentation_manager.export.sqlite_export import export_geojson_to_sqlite


# Pydantic models for API requests and responses
class SedimentAnalysisAPIRequest(BaseModel):
    """API request model for sediment analysis."""
    current_bathymetry_path: str
    baseline_bathymetry_path: str
    flow_u_path: Optional[str] = None
    flow_v_path: Optional[str] = None
    analysis_type: str = Field(default="comprehensive", description="Type of analysis: comprehensive, dredging, or monitoring")
    use_ai_agents: bool = Field(default=True, description="Whether to use AI agents for enhanced analysis")


class ComplianceReportAPIRequest(BaseModel):
    """API request model for compliance reporting."""
    analysis_results: Dict[str, Any]
    regulatory_framework: str = Field(default="EPA", description="Regulatory framework: EPA, EU, or local")
    include_recommendations: bool = Field(default=True)
    use_ai_agents: bool = Field(default=True, description="Whether to use AI agents for enhanced reporting")


class DredgingAnalysisRequest(BaseModel):
    """API request model for dredging analysis."""
    current_bathymetry_path: str
    design_depth_path: str
    unit_cost_per_m3: float = Field(default=25.0, description="Cost per cubic meter of dredging")


class VegetationAnalysisRequest(BaseModel):
    """API request model for vegetation analysis."""
    red_band_path: str
    nir_band_path: str
    ndvi_threshold: float = Field(default=0.3, description="NDVI threshold for vegetation detection")


class TransportModelRequest(BaseModel):
    """API request model for sediment transport modeling."""
    depth_path: str
    u_path: str
    v_path: str
    steps: int = Field(default=200, description="Time steps for simulation")
    dt: float = Field(default=0.5, description="Time step size (s)")
    diffusivity: float = Field(default=0.2, description="Turbulent diffusivity (m^2/s)")


class MLTrainingRequest(BaseModel):
    """API request model for ML model training."""
    csv_path: str
    target_column: str
    model_output_path: str


class MLPredictionRequest(BaseModel):
    """API request model for ML predictions."""
    csv_path: str
    model_path: str
    predictions_output_path: str


class WorkOrderRequest(BaseModel):
    """API request model for work order generation."""
    sediment_path: str
    threshold: float = Field(default=0.5, description="Accumulation threshold (m) for dredging polygons")


class NotificationRequest(BaseModel):
    """API request model for notification generation."""
    source_path: str
    audience: Optional[str] = Field(default=None, description="Audience tag (e.g., 'navigators', 'residents', 'crews')")
    use_ai_agents: bool = Field(default=True, description="Whether to use AI agents for enhanced messaging")
    tone: str = Field(default="informational", description="Desired tone for messaging")


class APIResponse(BaseModel):
    """Generic API response model."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# FastAPI application
app = FastAPI(
    title="OpenAcidMineDrainage API",
    description="Multi-agent AI-powered system for acid mine drainage analysis and management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint providing API information."""
    return APIResponse(
        success=True,
        message="OpenAcidMineDrainage API is running",
        data={
            "version": "1.0.0",
            "description": "Multi-agent AI-powered system for acid mine drainage analysis",
            "endpoints": [
                "/analyze-sedimentation",
                "/dredging-analysis",
                "/compliance-report",
                "/vegetation-analysis",
                "/transport-model",
                "/ml-train",
                "/ml-predict",
                "/work-orders",
                "/notifications",
                "/realtime-process"
            ]
        }
    )


@app.post("/analyze-sedimentation", response_model=APIResponse)
async def analyze_sedimentation_endpoint(request: SedimentAnalysisAPIRequest):
    """Perform comprehensive sedimentation analysis using AI agents."""
    try:
        if request.use_ai_agents:
            # Use the multi-agent system
            result = analyze_sedimentation_sync(
                current_bathymetry_path=request.current_bathymetry_path,
                baseline_bathymetry_path=request.baseline_bathymetry_path,
                flow_u_path=request.flow_u_path,
                flow_v_path=request.flow_v_path,
                analysis_type=request.analysis_type
            )
        else:
            # Use traditional analysis
            output_path = f"outputs/sediment_accum_{request.analysis_type}.tif"
            compute_sediment_accumulation(
                request.current_bathymetry_path,
                request.baseline_bathymetry_path,
                output_path
            )
            result = {
                "sediment_accumulation_path": output_path,
                "status": "completed"
            }

        return APIResponse(
            success=True,
            message="Sedimentation analysis completed successfully",
            data=result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/dredging-analysis", response_model=APIResponse)
async def dredging_analysis_endpoint(request: DredgingAnalysisRequest):
    """Perform dredging volume and cost analysis."""
    try:
        volume_m3, stats = compute_dredging_volume(
            request.current_bathymetry_path,
            request.design_depth_path
        )

        report = cost_benefit_analysis(volume_m3, request.unit_cost_per_m3, stats)

        return APIResponse(
            success=True,
            message="Dredging analysis completed successfully",
            data={
                "volume_m3": volume_m3,
                "stats": stats,
                "cost_benefit_report": report
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dredging analysis failed: {str(e)}")


@app.post("/compliance-report", response_model=APIResponse)
async def compliance_report_endpoint(request: ComplianceReportAPIRequest):
    """Generate compliance report using AI agents."""
    try:
        if request.use_ai_agents:
            # Use AI agents for enhanced reporting
            result = generate_compliance_report_sync(
                analysis_results=request.analysis_results,
                regulatory_framework=request.regulatory_framework,
                include_recommendations=request.include_recommendations
            )
        else:
            # Use traditional reporting
            report_path = "outputs/compliance_report.md"
            # Convert dict to JSON string for traditional function
            analysis_json = f"[\n{json.dumps(request.analysis_results, indent=2)}\n]"
            generate_compliance_report(
                [analysis_json],
                report_path,
                use_llm=False
            )
            result = {
                "report_path": report_path,
                "status": "completed"
            }

        return APIResponse(
            success=True,
            message="Compliance report generated successfully",
            data=result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compliance report generation failed: {str(e)}")


@app.post("/vegetation-analysis", response_model=APIResponse)
async def vegetation_analysis_endpoint(request: VegetationAnalysisRequest):
    """Perform vegetation analysis for invasive species detection."""
    try:
        output_path = "outputs/vegetation_mask.tif"
        detect_invasive_species(
            request.red_band_path,
            request.nir_band_path,
            output_path,
            request.ndvi_threshold
        )

        return APIResponse(
            success=True,
            message="Vegetation analysis completed successfully",
            data={
                "vegetation_mask_path": output_path,
                "ndvi_threshold": request.ndvi_threshold
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vegetation analysis failed: {str(e)}")


@app.post("/transport-model", response_model=APIResponse)
async def transport_model_endpoint(request: TransportModelRequest):
    """Run sediment transport modeling simulation."""
    try:
        output_path = "outputs/deposition_pred.tif"
        run_transport_model(
            depth_path=request.depth_path,
            u_path=request.u_path,
            v_path=request.v_path,
            output_path=output_path,
            steps=request.steps,
            dt=request.dt,
            diffusivity=request.diffusivity,
        )

        return APIResponse(
            success=True,
            message="Sediment transport modeling completed successfully",
            data={
                "deposition_prediction_path": output_path,
                "simulation_parameters": {
                    "steps": request.steps,
                    "dt": request.dt,
                    "diffusivity": request.diffusivity
                }
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transport modeling failed: {str(e)}")


@app.post("/ml-train", response_model=APIResponse)
async def ml_train_endpoint(request: MLTrainingRequest):
    """Train machine learning model for sedimentation prediction."""
    try:
        train_ml(
            request.csv_path,
            request.target_column,
            request.model_output_path
        )

        return APIResponse(
            success=True,
            message="ML model training completed successfully",
            data={
                "model_path": request.model_output_path,
                "target_column": request.target_column
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML training failed: {str(e)}")


@app.post("/ml-predict", response_model=APIResponse)
async def ml_predict_endpoint(request: MLPredictionRequest):
    """Make predictions using trained ML model."""
    try:
        predict_ml(
            request.csv_path,
            request.model_path,
            request.predictions_output_path
        )

        return APIResponse(
            success=True,
            message="ML predictions completed successfully",
            data={
                "predictions_path": request.predictions_output_path,
                "model_path": request.model_path
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {str(e)}")


@app.post("/work-orders", response_model=APIResponse)
async def work_orders_endpoint(request: WorkOrderRequest):
    """Generate work orders for dredging operations."""
    try:
        output_path = "outputs/work_orders.geojson"
        generate_work_orders(
            request.sediment_path,
            request.threshold,
            output_path
        )

        return APIResponse(
            success=True,
            message="Work orders generated successfully",
            data={
                "work_orders_path": output_path,
                "threshold": request.threshold
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Work order generation failed: {str(e)}")


@app.post("/notifications", response_model=APIResponse)
async def notifications_endpoint(request: NotificationRequest):
    """Generate stakeholder notifications."""
    try:
        output_path = "outputs/notifications.json"
        generate_notifications(
            request.source_path,
            output_path,
            request.audience,
            use_llm=request.use_ai_agents,
            tone=request.tone
        )

        return APIResponse(
            success=True,
            message="Notifications generated successfully",
            data={
                "notifications_path": output_path,
                "audience": request.audience,
                "tone": request.tone
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Notification generation failed: {str(e)}")


@app.post("/realtime-process", response_model=APIResponse)
async def realtime_process_endpoint(input_path: str):
    """Process real-time flow monitoring data."""
    try:
        output_path = "outputs/rt_assessment.json"
        process_realtime_flow(input_path, output_path)

        return APIResponse(
            success=True,
            message="Real-time processing completed successfully",
            data={
                "assessment_path": output_path
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Real-time processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "2025-01-10T12:00:00Z"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
