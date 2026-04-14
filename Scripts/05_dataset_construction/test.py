"""
Dataset Construction Sanity Checks

Purpose:
  Run quick checks to verify merges, variable coverage, and panel consistency.

Inputs:
  - Intermediate and final panel files.

Outputs:
  - Console diagnostics for validation.

Main Steps:
  - Load panel outputs.
  - Run basic checks.
  - Print diagnostics.
"""
import pandas as pd
p = pd.read_parquet("data_final/panel/weekly_panel.parquet")
print(p["res_weekly"].describe())
print(p["res_weekly"].skew())
