from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def ensure_dir(path: os.PathLike | str) -> Path:
    """Create a directory if it does not exist and return its Path.

    Args:
        path: Directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_artifacts_dir(root: os.PathLike | str | None = None) -> Path:
    """Return the artifacts directory path under project root (default CWD)."""
    base = Path(root) if root else Path.cwd()
    return ensure_dir(base / "artifacts")


def normalize_column_name(name: str) -> str:
    """Normalize a column name to a canonical snake_case-like form used internally."""
    return (
        name.strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "_")
        .replace(")", "_")
    )


def find_first_existing(paths: Iterable[os.PathLike | str]) -> Path | None:
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    return None

