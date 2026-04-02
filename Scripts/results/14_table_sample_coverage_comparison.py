from __future__ import annotations

import pandas as pd

from results_utils import DATA_FINAL, ensure_output_dirs, write_csv_and_tex

# Variables intended to align with the core Bollerslev-style setup.
# Decomposition variables are intentionally excluded.
BOLLERSLEV_SUMMARY_VARS = [
    "R_i_w",
    "R_i_w_plus_1",
    "rsj_weekly",
    "alpha_rsj",
    "beta_rsj",
    "me",
    "me_raw",
    "bm",
    "mom",
    "rev",
    "ivol",
    "illiq",
    "n_days",
    "n_obs_total",
]

# Set the comparable sample window to align with the original Bollerslev period.
BOLLERSLEV_START = "1993-01-01"
BOLLERSLEV_END = "2013-12-31"


def summarize_series(name: str, s: pd.Series) -> dict[str, object]:
    s = pd.to_numeric(s, errors="coerce")
    return {
        "Variable": name,
        "N": int(s.notna().sum()),
        "Mean": s.mean(),
        "Std": s.std(),
        "P25": s.quantile(0.25),
        "Median": s.median(),
        "P75": s.quantile(0.75),
        "Min": s.min(),
        "Max": s.max(),
    }


def main() -> None:
    ensure_output_dirs()

    panel_path = DATA_FINAL / "panel" / "weekly_panel.parquet"
    panel = pd.read_parquet(panel_path).copy()

    if "week" in panel.columns:
        panel["week"] = pd.to_datetime(panel["week"], errors="coerce")
        panel = panel[panel["week"].notna()].copy()
        panel = panel[
            (panel["week"] >= pd.Timestamp(BOLLERSLEV_START))
            & (panel["week"] <= pd.Timestamp(BOLLERSLEV_END))
        ].copy()

    present_vars = [c for c in BOLLERSLEV_SUMMARY_VARS if c in panel.columns]
    if not present_vars:
        raise ValueError("None of the requested Bollerslev summary variables are present in weekly_panel.parquet.")

    # Comparable sample definition: valid lead return and non-missing overlap variables.
    if "valid_R_i_w_plus_1" in panel.columns and "R_i_w_plus_1" in panel.columns:
        common_mask = (panel["valid_R_i_w_plus_1"] == True) & panel["R_i_w_plus_1"].notna()
    elif "R_i_w_plus_1" in panel.columns:
        common_mask = panel["R_i_w_plus_1"].notna()
    else:
        common_mask = pd.Series(True, index=panel.index)

    common_mask = common_mask & panel[present_vars].notna().all(axis=1)
    comp = panel.loc[common_mask, present_vars].copy()

    rows = [summarize_series(v, comp[v]) for v in present_vars]
    out = pd.DataFrame(rows)

    write_csv_and_tex(out, "table_summary_stats_bollerslev_comparable", float_fmt="%.6f")
    print("Saved Bollerslev-comparable summary statistics table.")


if __name__ == "__main__":
    main()
