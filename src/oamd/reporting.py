from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .analyze import NUMERIC_PARAMETERS, anova_by_site, regression_by_parameter
from .utils import ensure_dir


def _env(template_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(template_dir)), autoescape=select_autoescape(["html", "xml"])
    )


def _plot_timeseries_image(df: pd.DataFrame, parameter: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    for site, g in df.sort_values("timestamp").groupby("site_id"):
        ax.plot(g["timestamp"], g[parameter], label=str(site), linewidth=1.5)
    ax.set_title(f"{parameter} over time by site")
    ax.set_xlabel("Time")
    ax.set_ylabel(parameter)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def generate_report(input_csv: str | Path, output_html: str | Path) -> Path:
    input_csv = Path(input_csv)
    output_html = Path(output_html)
    ensure_dir(output_html.parent)

    df = pd.read_csv(input_csv, parse_dates=["timestamp"])

    # Compute analyses for each parameter
    anova_results = []
    regression_results = []
    for param in NUMERIC_PARAMETERS:
        try:
            a = anova_by_site(df, param)
            r = regression_by_parameter(df, param)
            anova_results.append(a)
            regression_results.append(r)
        except Exception as e:  # pragma: no cover - keep report robust
            # Skip parameters that cannot be analyzed due to missing data
            continue

    # Compose plots
    ts_image = _plot_timeseries_image(df, "pH")

    # Render
    template_dir = Path(__file__).parent / "templates"
    env = _env(template_dir)
    template = env.get_template("report.html.jinja")

    ctx = {
        "title": "OpenAcidMineDrainage Compliance and Impact Report",
        "anova_results": [a.__dict__ for a in anova_results],
        "regression_results": [
            {"parameter": r.parameter, "r2": r.r2, "coef": r.coef, "pvalues": r.pvalues}
            for r in regression_results
        ],
        "timeseries_image": ts_image,
    }

    html = template.render(**ctx)
    output_html.write_text(html, encoding="utf-8")
    return output_html

