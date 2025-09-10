# OpenAcidMineDrainage

OpenAcidMineDrainage is an advanced multi-agent AI system powered by OpenAI Agents SDK for comprehensive acid mine drainage analysis and management. The system combines bathymetric analysis, sediment transport modeling, vegetation management, dredging planning, and environmental compliance reporting to address acid mine drainage challenges.

**Key Features:**
- **Multi-Agent AI Architecture**: Specialized agents for sediment analysis, dredging planning, compliance reporting, and orchestration
- **OpenAI Agents SDK Integration**: Advanced AI-powered analysis and decision support
- **FastAPI Web API**: RESTful API for web-based interaction and integration
- **Docker & OrbStack Optimized**: Containerized deployment with macOS optimization
- **Comprehensive Analysis Suite**: From data ingestion to stakeholder notifications
- **Environmental Compliance**: Automated regulatory reporting and compliance tracking

## Key Capabilities

### Core Analysis Features
- **Data Ingestion**: Multibeam sonar (XYZ/GeoTIFF/NetCDF), LiDAR bathymetry, satellite imagery
- **Bathymetric Analysis**: High-resolution sediment accumulation mapping and change detection
- **Sediment Transport Modeling**: Finite-difference advection-diffusion approximation
- **Machine Learning**: Correlate sedimentation with land use, precipitation, seasonality (scikit-learn)
- **Dredging Analysis**: Volume estimation, cost-benefit analysis, equipment optimization
- **Vegetation Management**: NDVI-based invasive species detection and remediation planning

### AI-Powered Features
- **Multi-Agent Analysis**: Specialized AI agents for sediment analysis, dredging planning, and compliance
- **Intelligent Compliance Reporting**: Automated regulatory compliance assessment and reporting
- **Smart Notifications**: AI-generated stakeholder communications with appropriate tone and context
- **Predictive Analytics**: ML-driven forecasting of sedimentation patterns and remediation needs

### Operational Features
- **Work Order Automation**: GPS-enabled work order generation with volume estimates
- **Real-time Monitoring**: Flow monitoring integration for immediate impact detection
- **Environmental Compliance**: Automated permitting workflow documentation
- **Stakeholder Management**: Multi-channel notification system for navigation hazards and construction

### Technical Features
- **FastAPI Web API**: RESTful endpoints for programmatic access
- **Docker Containerization**: OrbStack-optimized for macOS deployment
- **CLI Interface**: Command-line tools with Rich UI for direct usage
- **Modular Architecture**: Extensible plugin system for custom analysis modules

## Installation

### Option 1: Docker (Recommended for OrbStack)
```bash
# Clone the repository
git clone https://github.com/llamasearchai/OpenAcidMineDrainage.git
cd OpenAcidMineDrainage

# Copy environment configuration
cp env.example .env
# Edit .env with your OpenAI API key

# Start with Docker Compose
docker-compose up -d

# Access the API at http://localhost:8000
# View API documentation at http://localhost:8000/docs
```

### Option 2: Local Python Installation
Requirements: Python 3.10+

1. Create and activate a virtual environment:
   ```bash
   # macOS/Linux
   python3 -m venv .venv && source .venv/bin/activate

   # Windows (PowerShell)
   py -3 -m venv .venv; .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   # Optional extras for enhanced functionality
   pip install -e .[hydro,gdal,dev]
   ```

Note: GDAL and geospatial packages may require system libraries. On macOS with Homebrew:
```bash
brew install gdal geos proj
```

## Quickstart

### CLI Usage
The CLI entrypoint is `osm`:

```bash
# Basic sediment analysis
osm bathy-analyze --current data/current.tif --baseline data/baseline.tif --output outputs/sediment.tif

# AI-powered comprehensive analysis
osm ai-analyze --current data/current.tif --baseline data/baseline.tif --output outputs/ai_analysis.json

# Start web API server
osm start-api --host 0.0.0.0 --port 8000
```

### API Usage
Start the FastAPI server and access interactive documentation:

```bash
# With Docker
docker-compose up

# Or locally
osm start-api

# API endpoints:
# - POST /analyze-sedimentation - AI-powered sediment analysis
# - POST /dredging-analysis - Dredging cost and volume analysis
# - POST /compliance-report - AI-generated compliance reports
# - POST /vegetation-analysis - Vegetation monitoring
# - GET /docs - Interactive API documentation
```

### Example Commands
```bash
# Data processing
osm ingest-raster --inputs data/*.tif --output outputs/combined_bathy.tif
osm model-transport --depth outputs/combined_bathy.tif --u data/flow_u.tif --v data/flow_v.tif --output outputs/deposition_pred.tif

# Analysis and reporting
osm dredge-calc --current outputs/combined_bathy.tif --design data/design_depth.tif --unit-cost 25.0 --output outputs/dredging_report.json
osm ai-compliance --inputs outputs/dredging_report.json --output outputs/compliance_report.json --framework EPA

# Operational
osm work-orders --sediment outputs/sediment_accum.tif --threshold 0.5 --output outputs/work_orders.geojson
osm notify --source outputs/work_orders.geojson --output outputs/notifications.json --use-ai
```

## Data
- Bathymetry: GeoTIFFs or gridded rasters (depth in meters, negative down or positive depth; this project treats positive depth values).
- Flow fields: GeoTIFF rasters for U and V components in m/s, aligned to the depth grid.
- Satellite imagery: Red and NIR bands (GeoTIFF) for NDVI.
- ML features: CSV with numeric columns; specify the target column via `--target`.

## Output Structure (suggested)
- `outputs/` directory for generated rasters, reports, and artifacts
- `logs/` directory for logs (if enabled)

## License
MIT License. See LICENSE for details.

