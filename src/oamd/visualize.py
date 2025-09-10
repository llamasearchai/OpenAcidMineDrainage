from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def timeseries_plot(df: pd.DataFrame, parameter: str, out_png: str | Path) -> Path:
    out_path = Path(out_png)
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(9, 4))
    for site, g in df.sort_values("timestamp").groupby("site_id"):
        ax.plot(g["timestamp"], g[parameter], label=str(site), linewidth=1.5)
    ax.set_title(f"{parameter} over time by site")
    ax.set_xlabel("Time")
    ax.set_ylabel(parameter)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

