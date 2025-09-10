from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

try:
    from open_sedimentation_manager.llm.agents import summarize_text
except Exception:  # pragma: no cover - optional
    summarize_text = None  # type: ignore


def generate_notifications(
    source_path: str,
    output_path: str,
    audience: Optional[str] = None,
    use_llm: bool = False,
    llm_model: Optional[str] = None,
    tone: str = "informational",
) -> None:
    """Generate notification messages from a GeoJSON/JSON source.

    For work order GeoJSON, one message per feature summarizing location and volume.
    If use_llm is enabled and available, rewrites the message body to the requested tone.
    """
    data = json.loads(Path(source_path).read_text())

    messages: List[dict] = []
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            volume = props.get("volume_m3")
            centroid_x = props.get("centroid_x")
            centroid_y = props.get("centroid_y")
            body = (
                f"Dredging work order {props.get('id')} at coordinates "
                f"({centroid_x:.3f}, {centroid_y:.3f}). Estimated volume: {volume:.0f} m^3."
            )
            if use_llm and summarize_text is not None:
                try:
                    prompt = (
                        f"Audience: {audience or 'general'}\n"
                        f"Tone: {tone}\n"
                        "Rewrite the message below to be clear and professional while preserving all numeric values and coordinates.\n\n"
                        f"{body}"
                    )
                    body = summarize_text(prompt, model=llm_model, max_tokens=200, temperature=0.2)
                except Exception:
                    pass  # keep original body
            msg = {
                "audience": audience or "general",
                "subject": "Scheduled dredging work",
                "body": body,
                "severity": "info",
                "location": {"x": centroid_x, "y": centroid_y},
            }
            messages.append(msg)
    else:
        # Fallback: single message summarizing content
        body = "Please review the attached operational data."
        if use_llm and summarize_text is not None:
            try:
                prompt = (
                    f"Audience: {audience or 'general'}\n"
                    f"Tone: {tone}\n"
                    "Rewrite the message below to be clear and professional.\n\n"
                    f"{body}"
                )
                body = summarize_text(prompt, model=llm_model, max_tokens=120, temperature=0.2)
            except Exception:
                pass
        messages.append({
            "audience": audience or "general",
            "subject": "Operational notification",
            "body": body,
            "severity": "info",
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(messages, indent=2))

