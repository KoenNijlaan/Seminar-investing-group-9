"""
Quintile portfolio sorts following Bollerslev, Li & Zhao (2020).

For each sorting variable:
  1. Each week t, sort eligible stocks into quintiles (NYSE breakpoints)
  2. Compute EW and VW portfolio returns for the HOLDING week t+1
  3. Report mean return, FF3 alpha, and t-statistics (Newey-West, 6 lags)
  4. Report spread portfolio: Q1-Q5 for RSJ (low = high risk), Q5-Q1 for RES

Timing convention:
  - Sort week t  : sort variable observed, quintiles assigned, VW weights set
  - Holding week t+1 : portfolio return R_i_w_plus_1 is earned
  - FF3 factors are matched to the HOLDING week, not the sort week

Eligibility per week: valid_R_i_w_plus_1 AND non-missing sort variable.
Minimum 50 eligible stocks per sort-week (thin weeks dropped).

VW weights: me_raw at sort week t (market cap known at formation).
Factor model: Fama-French 3-factor (Mkt-RF, SMB, HML).

Output: data_final/portfolio_sorts/
  - sort_results_{variable}.csv  — quintile table (EW + VW side by side)
  - sort_results_all.parquet     — all results combined
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ============================================================
# Settings
# ============================================================
ROOT = Path(__file__).resolve().parents[2]

PANEL_FILE = ROOT / "data_final" / "panel" / "weekly_panel.parquet"
FF_FILE    = ROOT / "data_raw" / "wrds" / "ff_daily_factors_1992_2024.parquet"
OUTPUT_DIR = ROOT / "data_final" / "portfolio_sorts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_QUINTILES          = 5
NW_LAGS              = 6    # Newey-West lags for weekly data
ANNUALIZE            = 52   # multiply weekly return by this to annualise
MIN_STOCKS_PER_WEEK  = 50   # drop sort-weeks with fewer eligible stocks

# Variables to sort on (column name → display label)
SORT_VARIABLES = {
    "rsj_weekly"     : "RSJ Total",
    "rsj_sys_weekly" : "RSJ Systematic (M1)",
    "rsj_idio_weekly": "RSJ Idiosyncratic (M1)",
    "rsj_sys"        : "RSJ Systematic (M2)",
    "rsj_idio"       : "RSJ Idiosyncratic (M2)",
    "res_weekly"     : "RES Total",
    "res_sys_p025"   : "RES Systematic",
    "res_idio_p025"  : "RES Idiosyncratic",
}

# Spread direction per variable.
# RSJ: lower RSJ = more downside risk = higher expected return → L-H (Q1 - Q5)
# RES: higher RES = more downside risk = higher expected return → H-L (Q5 - Q1)
SPREAD_DIRECTION = {
    "rsj_weekly"     : "L-H",
    "rsj_sys_weekly" : "L-H",
    "rsj_idio_weekly": "L-H",
    "rsj_sys"        : "L-H",
    "rsj_idio"       : "L-H",
    "res_weekly"     : "H-L",
    "res_sys_p025"   : "H-L",
    "res_idio_p025"  : "H-L",
}


# ============================================================
# Helpers
# ============================================================
def nw_tstat(y: pd.Series, X: pd.DataFrame, maxlags: int) -> dict:
    """OLS with Newey-West standard errors. Returns alpha, t_alpha, betas."""
    mask = y.notna() & X.notna().all(axis=1)
    y_c = y[mask]
    X_c = sm.add_constant(X[mask])

    if len(y_c) < 20:
        return {}

    model = sm.OLS(y_c, X_c).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags, "use_correction": True},
    )

    result = {
        "alpha_weekly": model.params["const"],
        "alpha_annual": model.params["const"] * ANNUALIZE,
        "t_alpha"     : model.tvalues["const"],
        "n_obs"       : int(len(y_c)),
    }
    for k in model.params.index:
        if k != "const":
            result[f"beta_{k}"] = model.params[k]
    return result


def assign_quintiles_nyse(sort_var: pd.Series, exchcd: pd.Series | None) -> pd.Series:
    """
    Assign quintile ranks 1-5 using NYSE breakpoints (20/40/60/80 pct).
    Uses explicit breakpoint assignment so ties never collapse bins.
    Falls back to full cross-section if NYSE stocks are too few.
    """
    if exchcd is not None and exchcd.notna().any():
        ref = sort_var[exchcd == 1].dropna()
    else:
        ref = sort_var.dropna()

    # Fall back to full cross-section if NYSE is too thin
    if len(ref) < 10:
        ref = sort_var.dropna()
    if len(ref) == 0:
        return pd.Series(np.nan, index=sort_var.index)

    q20, q40, q60, q80 = np.nanpercentile(ref, [20, 40, 60, 80])

    out = pd.Series(np.nan, index=sort_var.index)
    valid = sort_var.notna()
    v = sort_var[valid]

    out[valid & (sort_var <= q20)]                           = 1
    out[valid & (sort_var > q20) & (sort_var <= q40)]        = 2
    out[valid & (sort_var > q40) & (sort_var <= q60)]        = 3
    out[valid & (sort_var > q60) & (sort_var <= q80)]        = 4
    out[valid & (sort_var > q80)]                            = 5

    return out


# ============================================================
# Load FF3 factors aggregated to W-TUE weeks
# ============================================================
def load_ff3_weekly(ff_file: Path) -> pd.DataFrame:
    """
    Load daily FF3 factors, compound to weekly (W-TUE).
    Columns: week, mktrf_weekly, smb_weekly, hml_weekly, rf_weekly.
    Factors are in decimal units (verified: values ~ 0.0001 scale).
    """
    ff = pd.read_parquet(ff_file)
    ff.columns = [c.lower().strip() for c in ff.columns]
    ff["date"] = pd.to_datetime(ff["date"]).dt.normalize()

    for c in ["mktrf", "smb", "hml", "rf"]:
        ff[c] = pd.to_numeric(ff[c], errors="coerce")

    # W-TUE: week ending Tuesday, matching panel week definition
    ff["week"] = ff["date"].dt.to_period("W-TUE").dt.end_time.dt.normalize()

    def compound(x):
        return np.prod(1 + x.dropna()) - 1

    ff_weekly = ff.groupby("week", sort=True).agg(
        mktrf_weekly=("mktrf", compound),
        smb_weekly  =("smb",   compound),
        hml_weekly  =("hml",   compound),
        rf_weekly   =("rf",    compound),
    ).reset_index()

    return ff_weekly


# ============================================================
# Single sort
# ============================================================
def run_sort(panel: pd.DataFrame, sort_col: str, ff_weekly: pd.DataFrame,
             label: str) -> pd.DataFrame:
    """
    Quintile portfolio sort on sort_col.
    Returns summary DataFrame: one row per quintile (Q1–Q5) + spread, per weighting.
    """
    print(f"  Sorting on: {label} ({sort_col})")

    # ----------------------------------------------------------
    # Eligible rows
    # ----------------------------------------------------------
    eligible = panel[
        (panel["valid_R_i_w_plus_1"] == True) &
        panel[sort_col].notna() &
        panel["R_i_w_plus_1"].notna()
    ].copy()

    # Minimum cross-sectional size: drop thin sort-weeks
    week_counts = eligible.groupby("week").size()
    thick_weeks = week_counts[week_counts >= MIN_STOCKS_PER_WEEK].index
    eligible = eligible[eligible["week"].isin(thick_weeks)].copy()

    if len(eligible) == 0:
        print(f"    No eligible rows for {sort_col} after thin-week filter — skipping.")
        return pd.DataFrame()

    # ----------------------------------------------------------
    # Holding week: portfolio return is earned in week t+1
    # Sort week = t, holding week = t+1 (add 7 days for W-TUE weeks)
    # ----------------------------------------------------------
    eligible["holding_week"] = eligible["week"] + pd.Timedelta(weeks=1)

    # ----------------------------------------------------------
    # Assign quintiles at sort week t (NYSE breakpoints)
    # ----------------------------------------------------------
    exchcd_col = "exchcd" if "exchcd" in eligible.columns else None

    eligible["quintile"] = eligible.groupby("week", group_keys=False).apply(
        lambda g: assign_quintiles_nyse(
            g[sort_col],
            g[exchcd_col] if exchcd_col else None,
        ),
        include_groups=False,
    ).reindex(eligible.index)

    eligible = eligible.dropna(subset=["quintile"]).copy()
    eligible["quintile"] = eligible["quintile"].astype(int)

    # VW weight = market cap at sort week t
    eligible["vw_weight"] = eligible["me_raw"].where(eligible["me_raw"] > 0, np.nan)

    # ----------------------------------------------------------
    # Build portfolio return time series, indexed by HOLDING week
    # ----------------------------------------------------------
    rows = []
    for q in range(1, N_QUINTILES + 1):
        grp = eligible[eligible["quintile"] == q].copy()

        # EW: simple mean of R_i_w_plus_1, indexed by holding_week
        weekly_ew = (
            grp.groupby("holding_week")["R_i_w_plus_1"]
            .mean()
            .rename("ret_ew")
        )

        # Number of stocks per holding week
        weekly_n = grp.groupby("holding_week").size().rename("n_stocks")

        # VW: weight by me_raw from sort week t
        def vw_ret(g):
            w = g["vw_weight"]
            r = g["R_i_w_plus_1"]
            valid = w.notna() & r.notna()
            if valid.sum() == 0:
                return np.nan
            return float((r[valid] * w[valid]).sum() / w[valid].sum())

        weekly_vw = (
            grp.groupby("holding_week")
            .apply(vw_ret, include_groups=False)
            .rename("ret_vw")
        )

        rows.append({
            "sort_var"  : sort_col,
            "label"     : label,
            "quintile"  : q,
            "ew_series" : weekly_ew,
            "vw_series" : weekly_vw,
            "n_series"  : weekly_n,
        })

    # ----------------------------------------------------------
    # Spread portfolio (direction depends on variable)
    # ----------------------------------------------------------
    direction   = SPREAD_DIRECTION.get(sort_col, "H-L")
    q1_ew, q5_ew = rows[0]["ew_series"], rows[4]["ew_series"]
    q1_vw, q5_vw = rows[0]["vw_series"], rows[4]["vw_series"]

    if direction == "H-L":
        spread_ew = (q5_ew - q1_ew).dropna().rename("ret_ew")
        spread_vw = (q5_vw - q1_vw).dropna().rename("ret_vw")
    else:
        spread_ew = (q1_ew - q5_ew).dropna().rename("ret_ew")
        spread_vw = (q1_vw - q5_vw).dropna().rename("ret_vw")

    rows.append({
        "sort_var"  : sort_col,
        "label"     : label,
        "quintile"  : direction,
        "ew_series" : spread_ew,
        "vw_series" : spread_vw,
        "n_series"  : None,
    })

    # ----------------------------------------------------------
    # Summarize: mean return and FF3 alpha per portfolio
    # ----------------------------------------------------------
    ff_idx = ff_weekly.set_index("week")
    results = []

    def compute_stats(ts: pd.Series, weighting: str,
                      n_series: pd.Series | None) -> dict:
        """
        Align ts (indexed by holding_week) with FF3 factors (indexed by week).
        Both use the same W-TUE calendar so the index values match directly.
        """
        common = ts.index.intersection(ff_idx.index)
        if len(common) == 0:
            return {"weighting": weighting, "mean_ret_weekly": np.nan,
                    "mean_ret_annual": np.nan, "t_mean": np.nan}

        ts_a  = ts.loc[common]
        ff_a  = ff_idx.loc[common]

        mean_ret = ts_a.mean()
        nw_mean  = sm.OLS(
            ts_a.values,
            np.ones((len(ts_a), 1)),
        ).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS, "use_correction": True})

        excess_ts = ts_a.values - ff_a["rf_weekly"].values
        ff_X = pd.DataFrame({
            "mktrf": ff_a["mktrf_weekly"].values,
            "smb"  : ff_a["smb_weekly"].values,
            "hml"  : ff_a["hml_weekly"].values,
        }, index=common)

        ff_res = nw_tstat(pd.Series(excess_ts, index=common), ff_X, NW_LAGS)

        stats = {
            "weighting"      : weighting,
            "n_weeks"        : len(ts_a),
            "mean_ret_weekly": float(mean_ret),
            "mean_ret_annual": float(mean_ret * ANNUALIZE),
            "t_mean"         : float(nw_mean.tvalues[0]),
            **ff_res,
        }

        if n_series is not None:
            n_common = n_series.reindex(common).dropna()
            stats["avg_n_stocks"] = float(n_common.mean()) if len(n_common) > 0 else np.nan
            stats["med_n_stocks"] = float(n_common.median()) if len(n_common) > 0 else np.nan

        return stats

    for row in rows:
        q      = row["quintile"]
        ew_ts  = row["ew_series"].dropna()
        vw_ts  = row["vw_series"].dropna()
        n_ser  = row["n_series"]

        ew_stats = compute_stats(ew_ts, "EW", n_ser)
        vw_stats = compute_stats(vw_ts, "VW", n_ser)

        for stat in [ew_stats, vw_stats]:
            results.append({"sort_var": sort_col, "label": label,
                             "quintile": q, **stat})

    return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================
def main():
    print("=== Portfolio Sorts ===\n")

    if not PANEL_FILE.exists():
        raise FileNotFoundError(
            f"Panel file not found: {PANEL_FILE}\n"
            "Run 05_dataset_construction/01_build_data_panel.py first."
        )

    panel = pd.read_parquet(PANEL_FILE)
    panel["permno"] = pd.to_numeric(panel["permno"], errors="coerce").astype("Int64")
    panel["week"]   = pd.to_datetime(panel["week"]).dt.normalize()

    if "valid_R_i_w_plus_1" in panel.columns:
        panel["valid_R_i_w_plus_1"] = panel["valid_R_i_w_plus_1"].astype(bool)
    else:
        panel["valid_R_i_w_plus_1"] = panel["R_i_w_plus_1"].notna()

    print(f"Panel: {len(panel):,} stock-weeks | "
          f"{panel['permno'].nunique():,} stocks | "
          f"{panel['week'].nunique():,} weeks")
    print(f"Range: {panel['week'].min().date()} – {panel['week'].max().date()}\n")

    print("Loading FF3 factors (W-TUE weekly)...")
    ff_weekly = load_ff3_weekly(FF_FILE)
    print(f"  {len(ff_weekly):,} weekly factor observations\n")

    all_results = []
    for sort_col, label in SORT_VARIABLES.items():
        if sort_col not in panel.columns:
            print(f"  Skipping {sort_col} — not in panel.\n")
            continue

        result_df = run_sort(panel, sort_col, ff_weekly, label)
        if not result_df.empty:
            all_results.append(result_df)
            print()

    if not all_results:
        print("No results computed.")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Save combined parquet
    combined_path = OUTPUT_DIR / "sort_results_all.parquet"
    combined.to_parquet(combined_path, index=False)
    print(f"Saved: {combined_path}")

    # Save per-variable CSV
    for sort_col in combined["sort_var"].unique():
        df_var = combined[combined["sort_var"] == sort_col].copy()

        ew = df_var[df_var["weighting"] == "EW"].drop(
            columns=["sort_var", "label", "weighting"])
        vw = df_var[df_var["weighting"] == "VW"].drop(
            columns=["sort_var", "label", "weighting"])
        ew.columns = ["quintile"] + [f"EW_{c}" for c in ew.columns if c != "quintile"]
        vw.columns = ["quintile"] + [f"VW_{c}" for c in vw.columns if c != "quintile"]

        table = ew.merge(vw, on="quintile", how="outer")
        csv_path = OUTPUT_DIR / f"sort_results_{sort_col}.csv"
        table.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"  Saved: {csv_path.name}")

    # Console summary
    print("\n" + "=" * 80)
    print("SUMMARY — EW Mean Return (weekly) and FF3 Alpha by Quintile")
    print("=" * 80)
    for sort_col, label in SORT_VARIABLES.items():
        subset = combined[
            (combined["sort_var"] == sort_col) &
            (combined["weighting"] == "EW")
        ][["quintile", "mean_ret_weekly", "t_mean", "alpha_weekly", "t_alpha",
           "avg_n_stocks"]]
        if subset.empty:
            continue
        print(f"\n{label} ({sort_col})  |  spread: {SPREAD_DIRECTION[sort_col]}")
        print(subset.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
