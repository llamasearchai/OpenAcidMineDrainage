from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich import print
from rich.console import Console

from open_sedimentation_manager.analysis.bathymetry import (
    compute_sediment_accumulation,
)
from open_sedimentation_manager.analysis.dredging import (
    compute_dredging_volume,
    cost_benefit_analysis,
)
from open_sedimentation_manager.compliance.report import generate_compliance_report
from open_sedimentation_manager.io.ingestion import ingest_rasters
from open_sedimentation_manager.ml.model import predict_ml, train_ml
from open_sedimentation_manager.modeling.sediment import run_transport_model
from open_sedimentation_manager.notifications.notify import generate_notifications
from open_sedimentation_manager.realtime.monitoring import process_realtime_flow
from open_sedimentation_manager.vegetation.management import detect_invasive_species
from open_sedimentation_manager.work_orders.generator import generate_work_orders
from open_sedimentation_manager.export.sqlite_export import export_geojson_to_sqlite
from open_sedimentation_manager.llm.agents import (
    analyze_sedimentation_sync,
    generate_compliance_report_sync,
)

app = typer.Typer(help="OpenSedimentationManager CLI")
console = Console()


@app.command("ingest-raster")
def cmd_ingest_raster(
    inputs: List[Path] = typer.Option(..., exists=True, readable=True, help="Input raster paths (GeoTIFF/NetCDF)"),
    output: Path = typer.Option(..., help="Output combined raster GeoTIFF path"),
):
    """Ingest and mosaic rasters to a common grid and CRS."""
    ingest_rasters([str(p) for p in inputs], str(output))
    print(f"[green]Wrote[/green] {output}")


@app.command("bathy-analyze")
def cmd_bathy_analyze(
    current: Path = typer.Option(..., exists=True, readable=True, help="Current bathymetry raster"),
    baseline: Path = typer.Option(..., exists=True, readable=True, help="Baseline bathymetry raster"),
    output: Path = typer.Option(..., help="Output sediment accumulation raster"),
):
    compute_sediment_accumulation(str(current), str(baseline), str(output))
    print(f"[green]Wrote[/green] {output}")


@app.command("model-transport")
def cmd_model_transport(
    depth: Path = typer.Option(..., exists=True, readable=True, help="Depth raster (m)"),
    u: Path = typer.Option(..., exists=True, readable=True, help="Flow U component raster (m/s)"),
    v: Path = typer.Option(..., exists=True, readable=True, help="Flow V component raster (m/s)"),
    output: Path = typer.Option(..., help="Predicted deposition raster"),
    steps: int = typer.Option(200, help="Time steps for simulation"),
    dt: float = typer.Option(0.5, help="Time step size (s)"),
    diffusivity: float = typer.Option(0.2, help="Turbulent diffusivity (m^2/s)"),
):
    run_transport_model(
        depth_path=str(depth),
        u_path=str(u),
        v_path=str(v),
        output_path=str(output),
        steps=steps,
        dt=dt,
        diffusivity=diffusivity,
    )
    print(f"[green]Wrote[/green] {output}")


@app.command("dredge-calc")
def cmd_dredge_calc(
    current: Path = typer.Option(..., exists=True, readable=True, help="Current bathymetry"),
    design: Path = typer.Option(..., exists=True, readable=True, help="Design depth raster"),
    unit_cost: float = typer.Option(..., help="Unit cost per cubic meter"),
    output: Path = typer.Option(..., help="Output JSON report path"),
):
    volume_m3, stats = compute_dredging_volume(str(current), str(design))
    report = cost_benefit_analysis(volume_m3, unit_cost, stats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"[green]Wrote[/green] {output}")


@app.command("vegetation-detect")
def cmd_vegetation_detect(
    red: Path = typer.Option(..., exists=True, readable=True, help="Red band GeoTIFF"),
    nir: Path = typer.Option(..., exists=True, readable=True, help="NIR band GeoTIFF"),
    output: Path = typer.Option(..., help="Output vegetation mask GeoTIFF"),
    ndvi_threshold: float = typer.Option(0.3, help="NDVI threshold for vegetation"),
):
    detect_invasive_species(str(red), str(nir), str(output), ndvi_threshold)
    print(f"[green]Wrote[/green] {output}")


@app.command("train-ml")
def cmd_train_ml(
    csv: Path = typer.Option(..., exists=True, readable=True, help="Training CSV file"),
    target: str = typer.Option(..., help="Target column name"),
    model: Path = typer.Option(..., help="Output model path (.joblib)"),
):
    train_ml(str(csv), target, str(model))
    print(f"[green]Wrote[/green] {model}")


@app.command("predict-ml")
def cmd_predict_ml(
    csv: Path = typer.Option(..., exists=True, readable=True, help="Prediction CSV file"),
    model: Path = typer.Option(..., exists=True, readable=True, help="Trained model .joblib path"),
    output: Path = typer.Option(..., help="Output predictions CSV"),
):
    predict_ml(str(csv), str(model), str(output))
    print(f"[green]Wrote[/green] {output}")


@app.command("compliance-report")
def cmd_compliance_report(
    inputs: List[Path] = typer.Option(..., exists=True, readable=True, help="Input JSON(s) for report aggregation"),
    output: Path = typer.Option(..., help="Output Markdown report"),
    use_llm: bool = typer.Option(False, help="Include an LLM-generated narrative (requires OPENAI_API_KEY and llm extra)"),
    llm_model: Optional[str] = typer.Option(None, help="LLM model name (overrides OSM_OPENAI_MODEL)"),
):
    generate_compliance_report([str(p) for p in inputs], str(output), use_llm=use_llm, llm_model=llm_model)
    print(f"[green]Wrote[/green] {output}")


@app.command("work-orders")
def cmd_work_orders(
    sediment: Path = typer.Option(..., exists=True, readable=True, help="Sediment accumulation raster"),
    threshold: float = typer.Option(..., help="Accumulation threshold (m) for dredging polygons"),
    output: Path = typer.Option(..., help="Output GeoJSON path"),
):
    generate_work_orders(str(sediment), threshold, str(output))
    print(f"[green]Wrote[/green] {output}")


@app.command("realtime-process")
def cmd_realtime_process(
    input: Path = typer.Option(..., exists=True, readable=True, help="Real-time flow JSON/CSV input"),
    output: Path = typer.Option(..., help="Output assessment JSON"),
):
    process_realtime_flow(str(input), str(output))
    print(f"[green]Wrote[/green] {output}")


@app.command("notify")
def cmd_notify(
    source: Path = typer.Option(..., exists=True, readable=True, help="Source GeoJSON/JSON for notifications"),
    output: Path = typer.Option(..., help="Output JSON array of notification messages"),
    audience: Optional[str] = typer.Option(None, help="Audience tag (e.g., 'navigators', 'residents', 'crews')"),
    use_llm: bool = typer.Option(False, help="Rewrite message text via LLM (requires OPENAI_API_KEY and llm extra)"),
    llm_model: Optional[str] = typer.Option(None, help="LLM model name (overrides OSM_OPENAI_MODEL)"),
    tone: str = typer.Option("informational", help="Desired tone for LLM rewriting (e.g., 'informational', 'urgent', 'public advisory')"),
):
    generate_notifications(str(source), str(output), audience, use_llm=use_llm, llm_model=llm_model, tone=tone)
    print(f"[green]Wrote[/green] {output}")


@app.command("export-datasette")
def cmd_export_datasette(
    source: Path = typer.Option(..., exists=True, readable=True, help="Source GeoJSON FeatureCollection (e.g., work orders)"),
    sqlite: Path = typer.Option(..., help="Output SQLite database file"),
    table: str = typer.Option("work_orders", help="Destination table name"),
):
    inserted = export_geojson_to_sqlite(str(source), str(sqlite), table)
    print(f"[green]Wrote[/green] {sqlite} with {inserted} rows in table '{table}'")


@app.command("ai-analyze")
def cmd_ai_analyze(
    current: Path = typer.Option(..., exists=True, readable=True, help="Current bathymetry raster"),
    baseline: Path = typer.Option(..., exists=True, readable=True, help="Baseline bathymetry raster"),
    output: Path = typer.Option(..., help="Output JSON report path"),
    flow_u: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Flow U component raster (optional)"),
    flow_v: Optional[Path] = typer.Option(None, exists=True, readable=True, help="Flow V component raster (optional)"),
    analysis_type: str = typer.Option("comprehensive", help="Analysis type: comprehensive, dredging, or monitoring"),
):
    """Perform AI-powered sedimentation analysis using multi-agent system."""
    try:
        result = analyze_sedimentation_sync(
            current_bathymetry_path=str(current),
            baseline_bathymetry_path=str(baseline),
            flow_u_path=str(flow_u) if flow_u else None,
            flow_v_path=str(flow_v) if flow_v else None,
            analysis_type=analysis_type
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(f"[green]AI analysis completed[/green] - Results saved to {output}")

        # Print summary
        if "comprehensive_plan" in result:
            print(f"[blue]Summary:[/blue] {result['comprehensive_plan'][:200]}...")

    except Exception as e:
        print(f"[red]AI analysis failed:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("ai-compliance")
def cmd_ai_compliance(
    inputs: List[Path] = typer.Option(..., exists=True, readable=True, help="Input JSON files with analysis results"),
    output: Path = typer.Option(..., help="Output compliance report path"),
    framework: str = typer.Option("EPA", help="Regulatory framework: EPA, EU, or local"),
    include_recommendations: bool = typer.Option(True, help="Include AI-generated recommendations"),
):
    """Generate AI-powered compliance report using multi-agent system."""
    try:
        # Load and combine analysis results
        combined_results = {}
        for input_file in inputs:
            data = json.loads(input_file.read_text())
            combined_results.update(data)

        result = generate_compliance_report_sync(
            analysis_results=combined_results,
            regulatory_framework=framework,
            include_recommendations=include_recommendations
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(f"[green]AI compliance report generated[/green] - Results saved to {output}")

    except Exception as e:
        print(f"[red]Compliance report generation failed:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command("start-api")
def cmd_start_api(
    host: str = typer.Option("0.0.0.0", help="Host to bind the API server"),
    port: int = typer.Option(8000, help="Port to bind the API server"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Start the FastAPI server for web-based interaction."""
    try:
        import uvicorn
        print(f"[green]Starting OpenAcidMineDrainage API server on {host}:{port}[/green]")
        print(f"[blue]API documentation available at: http://{host}:{port}/docs[/blue]")

        uvicorn.run(
            "open_sedimentation_manager.api:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except ImportError:
        print("[red]FastAPI and uvicorn required. Install with: pip install fastapi uvicorn[/red]")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[red]Failed to start API server:[/red] {str(e)}")
        raise typer.Exit(1)


def main() -> None:
    app()

