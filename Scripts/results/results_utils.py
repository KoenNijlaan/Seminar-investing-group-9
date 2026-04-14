"""
Results Utilities

Purpose:
  Provide shared helper functions for formatting and writing result tables.

Inputs:
  - Imported by results scripts.

Outputs:
  - Reusable utility functions; no primary standalone output file.

Main Steps:
  - Define formatting helpers.
  - Define significance/annotation helpers.
  - Provide reusable output-writing utilities.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FINAL = ROOT / "data_final"
RESULTS_DIR = DATA_FINAL / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

def ensure_output_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def stars_from_t(t: float) -> str:
    if pd.isna(t):
        return ""
    at = abs(float(t))
    if at > 2.576:
        return "***"
    if at > 1.96:
        return "**"
    if at > 1.645:
        return "*"
    return ""

def pretty_coef_bps(mean_bps: float, t_stat: float) -> str:
    if pd.isna(mean_bps):
        return ""
    return f"{mean_bps:.2f}{stars_from_t(t_stat)}"

def ci95_bps(mean_bps: float, se_bps: float) -> tuple[float, float]:
    if pd.isna(mean_bps) or pd.isna(se_bps):
        return (np.nan, np.nan)
    lo = mean_bps - 1.96 * se_bps
    hi = mean_bps + 1.96 * se_bps
    return (lo, hi)

def write_csv_and_tex(df: pd.DataFrame, stem: str, float_fmt: str = "%.6f") -> None:
    csv_path = TABLES_DIR / f"{stem}.csv"
    tex_path = TABLES_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=False, float_format=float_fmt)
    with open(tex_path, "w", encoding="utf-8") as f:

        f.write(df.to_latex(index=False, escape=True))
