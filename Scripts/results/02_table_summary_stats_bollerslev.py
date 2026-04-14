"""
Bollerslev-Style Summary Table

Purpose:
  Create the alternative summary-statistics table in the Bollerslev-style layout.

Inputs:
  - Final panel data and results utility helpers.

Outputs:
  - Bollerslev-style summary table in the results folder.

Main Steps:
  - Compute grouped statistics.
  - Apply table formatting.
  - Write Bollerslev-style output.
"""
from __future__ import annotations

import pandas as pd

from results_utils import DATA_FINAL, ensure_output_dirs, write_csv_and_tex

ROOT       = DATA_FINAL.parent
PANEL_FILE = DATA_FINAL / "panel" / "weekly_panel.parquet"
FF_FILE    = ROOT / "data_raw" / "wrds" / "ff_daily_factors_1992_2024.parquet"

MIN_STOCKS       = 50
BOLLERSLEV_END   = "2013-12-31"

SORT_VARIABLES = [
    "rsj_weekly",
    "rsj_sys_weekly",
    "rsj_idio_weekly",
    "rsj_sys",
    "rsj_idio",
    "res_weekly",
    "res_sys_p025",
    "res_idio_p025",
]

def load_ff3_weekly(ff_file) -> pd.DataFrame:
    ff = pd.read_parquet(ff_file)
    ff.columns = ff.columns.str.strip().str.lower()
    ff["date"] = pd.to_datetime(ff["date"])
    ff["week"] = ff["date"].dt.to_period("W-TUE").dt.end_time.dt.normalize()
    ff_w = (
        ff.groupby("week", as_index=False)
          .agg(mktrf=("mktrf", "sum"), smb=("smb", "sum"), hml=("hml", "sum"), rf=("rf", "sum"))
    )
    ff_w["week"] = pd.to_datetime(ff_w["week"]).dt.normalize()
    return ff_w

def compute_common_weeks(panel: pd.DataFrame, sort_cols: list[str], ff_idx: pd.DataFrame) -> list:
    base = panel[
        (panel["valid_R_i_w_plus_1"] == True) &
        panel["R_i_w_plus_1"].notna()
    ].copy()

    common = set(pd.to_datetime(ff_idx.index).normalize())

    for sort_col in sort_cols:
        eligible = base[base[sort_col].notna()]
        wc = eligible.groupby("week").size()
        weeks_ok = set(wc[wc >= MIN_STOCKS].index)
        common &= weeks_ok

    return sorted(common)

def main() -> None:
    ensure_output_dirs()

    panel = pd.read_parquet(PANEL_FILE)
    panel["week"] = pd.to_datetime(panel["week"]).dt.normalize()
    if "valid_R_i_w_plus_1" in panel.columns:
        panel["valid_R_i_w_plus_1"] = panel["valid_R_i_w_plus_1"].astype(bool)
    else:
        panel["valid_R_i_w_plus_1"] = panel["R_i_w_plus_1"].notna()

    panel = panel[panel["week"] <= BOLLERSLEV_END].copy()

    ff_w   = load_ff3_weekly(FF_FILE)
    ff_idx = ff_w.set_index("week")

    active_sort_cols = [c for c in SORT_VARIABLES if c in panel.columns]
    common_weeks = compute_common_weeks(panel, active_sort_cols, ff_idx)
    print(f"Bollerslev common sample: {len(common_weeks)} weeks")

    panel = panel[panel["week"].isin(common_weeks)].copy()

    numeric_cols = [
        "R_i_w",
        "R_i_w_plus_1",
        "rsj_weekly",
        "rsj_sys_weekly",
        "rsj_idio_weekly",
        "res_weekly",
        "res_sys_p025",
        "res_idio_p025",
        "me",
        "bm",
        "mom",
        "rev",
        "ivol",
        "illiq",
        "rvol",
        "rsk",
        "rkt",
        "n_days",
        "n_obs_total",
    ]
    use_cols = [c for c in numeric_cols if c in panel.columns]

    rows = []
    for c in use_cols:
        s = pd.to_numeric(panel[c], errors="coerce")
        rows.append(
            {
                "Variable": c,
                "N": int(s.notna().sum()),
                "Mean": s.mean(),
                "Std": s.std(),
                "P25": s.quantile(0.25),
                "Median": s.median(),
                "P75": s.quantile(0.75),
                "Min": s.min(),
                "Max": s.max(),
            }
        )

    out = pd.DataFrame(rows)
    write_csv_and_tex(out, "table_summary_stats_bollerslev", float_fmt="%.6f")
    print("Saved Bollerslev summary statistics table.")

if __name__ == "__main__":
    main()
