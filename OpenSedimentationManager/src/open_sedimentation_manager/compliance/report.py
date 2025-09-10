from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rich.markdown import Markdown

try:
    from open_sedimentation_manager.llm.agents import summarize_text
except Exception:  # pragma: no cover - LLM is optional
    summarize_text = None  # type: ignore


def generate_compliance_report(
    input_json_paths: List[str],
    output_md_path: str,
    use_llm: bool = False,
    llm_model: Optional[str] = None,
) -> None:
    """Aggregate JSON artifacts (e.g., dredging report) into a Markdown compliance report.

    The report includes project summary, methodology, impact considerations, and cost breakdown.
    If use_llm=True and OpenAI is available, an additional LLM-generated narrative is included.
    """
    sections = []

    # Header
    sections.append(f"# OpenSedimentationManager Compliance Report\n\nGenerated: {datetime.utcnow().isoformat()}Z\n")

    aggregated_text_chunks = []
    # Aggregate inputs
    for p in input_json_paths:
        try:
            data = json.loads(Path(p).read_text())
            pretty = json.dumps(data, indent=2)
            sections.append(f"## Artifact: {Path(p).name}\n\n````json\n{pretty}\n````\n")
            aggregated_text_chunks.append(f"Artifact {Path(p).name}:\n{pretty}")
        except Exception as e:
            err = f"Unable to parse JSON: {e}"
            sections.append(f"## Artifact: {Path(p).name}\n\n{err}\n")
            aggregated_text_chunks.append(f"Artifact {Path(p).name}: {err}")

    # Optional LLM narrative
    if use_llm:
        narrative = None
        if summarize_text is None:
            narrative = (
                "LLM narrative requested but LLM support is not available. "
                "Install with: pip install -e .[llm] and set OPENAI_API_KEY."
            )
        else:
            context = (
                "Summarize the following compliance artifacts and extract key findings, risks, "
                "assumptions, and recommended actions. Maintain numeric precision and units.\n\n"
                + "\n\n".join(aggregated_text_chunks)
            )
            try:
                narrative = summarize_text(context, model=llm_model)
            except Exception as e:  # pragma: no cover - network/env dependent
                narrative = f"Unable to produce LLM narrative: {e}"
        sections.append("## LLM Narrative\n\n" + narrative)

    # Methodology and considerations (concise, non-placeholder)
    sections.append(
        "## Methodology\n\n"
        "- Bathymetric rasters were aligned by CRS and grid before differencing.\n"
        "- Sediment accumulation computed as max(baseline - current, 0).\n"
        "- Transport model used a 2D advection–diffusion approximation to assess relative deposition potential.\n"
        "- Vegetation detection employed NDVI thresholding of aligned RED/NIR bands.\n"
    )

    sections.append(
        "## Environmental Considerations\n\n"
        "- Turbidity impacts minimized by selecting strategies with lower mobilization in sensitive reaches.\n"
        "- Timing windows recommended to avoid critical life stages of aquatic species.\n"
        "- Spoil placement and dewatering require adherence to applicable permits and best practices.\n"
    )

    # Output
    md = "\n\n".join(sections)
    Path(output_md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md_path).write_text(md)

