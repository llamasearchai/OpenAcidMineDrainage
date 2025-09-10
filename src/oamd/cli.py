from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import __version__
from .alerts import check_thresholds, send_alerts
from .analyze import NUMERIC_PARAMETERS, anova_by_site, regression_by_parameter, seasonal_decompose
from .geochem import run_phreeqc_example
from .hydro import run_mf6_minimal
from .ingest import ingest_csvs, write_clean_csv
from .ml_forecast import forecast_for_site
from .reporting import generate_report
from .regulatory import export_regulatory_csv
from .visualize import timeseries_plot


def cmd_ingest(args: argparse.Namespace) -> int:
    df = ingest_csvs(args.inputs)
    out = write_clean_csv(df, args.out)
    print(f"Wrote cleaned dataset: {out}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    if args.param not in NUMERIC_PARAMETERS:
        raise SystemExit(f"Unsupported parameter {args.param}. Choose from {NUMERIC_PARAMETERS}")
    a = anova_by_site(df, args.param)
    r = regression_by_parameter(df, args.param)
    print("ANOVA:", a)
    print("Regression:", r)
    if args.decompose:
        dec = seasonal_decompose(df, args.param)
        out = Path(args.decompose)
        out.parent.mkdir(parents=True, exist_ok=True)
        dec.to_csv(out)
        print(f"Decomposition saved to: {out}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    res = forecast_for_site(df, args.site_id, args.param, args.horizon)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.predictions.to_csv(out, index=False)
    print(f"Forecast MAE={res.mae:.4f}. Predictions written to {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    out = generate_report(args.input, args.out)
    print(f"Report written: {out}")
    return 0


def cmd_hydro(args: argparse.Namespace) -> int:
    status = run_mf6_minimal(args.workspace)
    print(status)
    return 0


def cmd_geochem(_: argparse.Namespace) -> int:
    status = run_phreeqc_example()
    print(status)
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    out = timeseries_plot(df, args.param, args.out)
    print(f"Plot saved: {out}")
    return 0


def cmd_alert(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    alerts = check_thresholds(df)
    for a in alerts:
        print(a)
    send_alerts(alerts)
    print(f"Alerts generated: {len(alerts)}")
    return 0


def cmd_regulatory(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    out = export_regulatory_csv(df, args.out)
    print(f"Regulatory CSV exported: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="oamd", description="OpenAcidMineDrainage CLI")
    p.add_argument("--version", action="version", version=f"oamd {__version__}")
    sp = p.add_subparsers(dest="cmd", required=True)

    ing = sp.add_parser("ingest", help="Ingest CSV files and write cleaned dataset")
    ing.add_argument("inputs", nargs="+", help="Input CSV file(s)")
    ing.add_argument("--out", default="artifacts/clean.csv", help="Output cleaned CSV path")
    ing.set_defaults(func=cmd_ingest)

    an = sp.add_parser("analyze", help="Run statistical analyses")
    an.add_argument("input", help="Cleaned CSV input")
    an.add_argument("--param", default="pH", help=f"Parameter to analyze; one of {NUMERIC_PARAMETERS}")
    an.add_argument("--decompose", default=None, help="Optional output CSV for time-series decomposition")
    an.set_defaults(func=cmd_analyze)

    fc = sp.add_parser("forecast", help="Forecast a parameter for a site")
    fc.add_argument("input", help="Cleaned CSV input")
    fc.add_argument("--site-id", required=True)
    fc.add_argument("--param", default="pH")
    fc.add_argument("--horizon", type=int, default=14)
    fc.add_argument("--out", default="artifacts/forecast.csv")
    fc.set_defaults(func=cmd_forecast)

    rp = sp.add_parser("report", help="Generate compliance/impact report")
    rp.add_argument("input", help="Cleaned CSV input")
    rp.add_argument("--out", default="artifacts/report.html")
    rp.set_defaults(func=cmd_report)

    hd = sp.add_parser("hydro", help="Run a minimal MF6 (FloPy) scenario")
    hd.add_argument("--workspace", default="artifacts/mf6")
    hd.set_defaults(func=cmd_hydro)

    gc = sp.add_parser("geochem", help="Run a minimal PHREEQC speciation example")
    gc.set_defaults(func=cmd_geochem)

    vz = sp.add_parser("visualize", help="Save a timeseries plot for a parameter")
    vz.add_argument("input")
    vz.add_argument("--param", default="pH")
    vz.add_argument("--out", default="artifacts/plot.png")
    vz.set_defaults(func=cmd_visualize)

    al = sp.add_parser("alert", help="Evaluate thresholds and optionally send alerts")
    al.add_argument("input")
    al.set_defaults(func=cmd_alert)

    rg = sp.add_parser("regulatory", help="Export a fixed-format regulatory CSV")
    rg.add_argument("input")
    rg.add_argument("--out", default="artifacts/regulatory.csv")
    rg.set_defaults(func=cmd_regulatory)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

