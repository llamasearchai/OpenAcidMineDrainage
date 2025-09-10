from __future__ import annotations

from typing import Dict


def run_phreeqc_example() -> Dict[str, str]:
    """Run a minimal PHREEQC speciation calculation if phreeqpy is available.

    Returns a status dictionary and avoids raising if PHREEQC is not available.
    """
    try:
        from phreeqpy.iphreeqc.phreeqc import IPhreeqc  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        return {"status": "phreeqpy_not_installed", "detail": str(e)}

    iph = IPhreeqc()
    try:
        iph.load_database("phreeqc.dat")
    except Exception as e:
        return {"status": "phreeqc_database_not_found", "detail": str(e)}

    input_str = (
        "SOLUTION 1\n"
        "    temp      25.0\n"
        "    pH        7.0\n"
        "    units     mg/L\n"
        "    Na        10\n"
        "    Cl        10\n"
        "    Ca        40\n"
        "    Mg        12\n"
        "    SO4       50\n"
        "SELECTED_OUTPUT\n"
        "    -reset false\n"
        "    -high_precision true\n"
        "    -ionic_strength true\n"
        "    -saturation_indices Calcite Dolomite Gypsum\n"
    )

    try:
        iph.run_string(input_str)
        so = iph.get_selected_output_array()
        return {"status": "phreeqc_run_succeeded", "rows": str(len(so))}
    except Exception as e:  # pragma: no cover
        return {"status": "phreeqc_run_failed", "detail": str(e)}

