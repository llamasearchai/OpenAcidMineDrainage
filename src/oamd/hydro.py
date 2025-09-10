from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict


def run_mf6_minimal(workspace: str | Path) -> Dict[str, str]:
    """Build a minimal MF6 groundwater flow model.

    If the mf6 executable is available on PATH, attempt to run the simulation; otherwise
    write inputs only. Returns a status dictionary describing what was done.
    """
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    try:
        import flopy
        from flopy.mf6 import MFSimulation
        from flopy.mf6.modflow import (
            ModflowGwf,
            ModflowGwfchd,
            ModflowGwfdis,
            ModflowGwfic,
            ModflowGwfnpf,
            ModflowGwfoc,
            ModflowIms,
            ModflowTdis,
        )
    except Exception as e:  # pragma: no cover - optional dependency
        return {"status": "flopy_not_installed", "detail": str(e)}

    sim = MFSimulation(sim_name="oamd_mf6", version="mf6", exe_name="mf6", sim_ws=str(ws))
    # Temporal discretization: 1 stress period, 1 time step
    ModflowTdis(sim, time_units="DAYS", perioddata=[(1.0, 1, 1.0)])
    # Solver
    ModflowIms(sim, complexity="SIMPLE")

    # Groundwater flow (GWF) model
    gwf = ModflowGwf(sim, modelname="gwf", save_flows=True)

    nlay, nrow, ncol = 1, 10, 10
    delr = delc = 100.0
    top = 10.0
    botm = [0.0]

    ModflowGwfdis(gwf, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=top, botm=botm)
    ModflowGwfic(gwf, strt=5.0)
    ModflowGwfnpf(gwf, icelltype=1, k=10.0)

    # Constant head boundaries at left and right edges
    chd_spd = []
    for i in range(nrow):
        chd_spd.append(((0, i, 0), 6.0))  # left boundary
        chd_spd.append(((0, i, ncol - 1), 4.0))  # right boundary
    ModflowGwfchd(gwf, stress_period_data=chd_spd)

    # Output control
    ModflowGwfoc(
        gwf,
        head_filerecord="gwf.hds",
        budget_filerecord="gwf.cbc",
        saverecord={(0, 0): ["HEAD", "BUDGET"]},
        printrecord={(0, 0): ["HEAD", "BUDGET"]},
    )

    # Try run
    mf6 = shutil.which("mf6")
    if mf6:
        success, buff = sim.run_simulation()
        if not success:
            return {"status": "mf6_run_failed"}
        return {"status": "mf6_run_succeeded", "workspace": str(ws)}
    else:
        sim.write_simulation()
        return {"status": "mf6_not_found_inputs_written", "workspace": str(ws)}

