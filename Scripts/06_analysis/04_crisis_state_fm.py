"""
Crisis-state Fama-MacBeth analysis using NBER recession dates.

Why this implementation:
- In weekly cross-sectional Fama-MacBeth, a week-level crisis dummy C_w is
  constant within each week and therefore cannot be estimated directly as a
  regressor in that week's cross-section.
- We instead estimate weekly cross-sectional slopes and test whether those
  slopes differ between crisis and non-crisis weeks.

Workflow:
1) Load weekly panel data.
2) Build NBER crisis indicator by week (W-TUE week end).
3) For each downside-risk variable Z:
   a) Estimate weekly cross-sectional regression:
        R_{i,w+1} = a_w + b_w * Z_{i,w} + g_w' X_{i,w} + e_{i,w+1}
   b) Save the time series of b_w.
   c) Test crisis dependence with HAC (Newey-West) inference:
        b_w = alpha + delta * C_w + u_w
      where delta measures the crisis vs non-crisis slope difference.

Outputs:
- data_final/crisis/fm_crisis_weekly_betas.parquet
- data_final/crisis/fm_crisis_summary.csv
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
OUTPUT_DIR = ROOT / "data_final" / "crisis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEEKLY_BETAS_OUT = OUTPUT_DIR / "fm_crisis_weekly_betas.parquet"
SUMMARY_OUT = OUTPUT_DIR / "fm_crisis_summary.csv"

MIN_STOCKS = 50
NW_LAGS = 6

# Same controls as the baseline Fama-MacBeth script
CONTROLS = ["me", "bm", "mom", "rev", "ivol", "illiq"]

# Variables to test in crisis-state analysis
RISK_VARS = [
    "rsj_weekly",
    "rsj_sys",
    "rsj_idio",
    "rsj_sys_weekly",
    "rsj_idio_weekly",
    "res_weekly",
    "res_sys_p025",
    "res_idio_p025",
]

# NBER recessions overlapping your sample (1993-2024)
# Dates are month-based NBER peaks/troughs; we map weekly endpoints into these ranges.
NBER_RECESSION_PERIODS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


# ============================================================
# Helpers
# ============================================================
def winsorize_cs(s: pd.Series, low: float = 0.01, high: float = 0.99) -> pd.Series:
    lo = s.quantile(low)
    hi = s.quantile(high)
    return s.clip(lo, hi)


def build_crisis_indicator(week_series: pd.Series) -> pd.Series:
    week = pd.to_datetime(week_series).dt.normalize()
    c = pd.Series(False, index=week.index)

    for start, end in NBER_RECESSION_PERIODS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        c = c | ((week >= s) & (week <= e))

    return c.astype(int)


def run_weekly_cs(df_week: pd.DataFrame, z_var: str) -> tuple[float, int] | tuple[None, int]:
    predictors = [z_var] + CONTROLS
    cols = ["R_i_w_plus_1"] + predictors

    sub = df_week[cols].dropna()
    n_obs = len(sub)
    if n_obs < MIN_STOCKS:
        return None, n_obs

    y = sub["R_i_w_plus_1"].to_numpy(dtype=float)
    X = sm.add_constant(sub[predictors].to_numpy(dtype=float), has_constant="add")

    try:
        fit = sm.OLS(y, X).fit()
    except Exception:
        return None, n_obs

    # Parameter order: const, z_var, controls...
    beta_z = float(fit.params[1])
    return beta_z, n_obs


def nw_mean_t(series: pd.Series, maxlags: int) -> tuple[float, float, int]:
    y = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(y)
    if n < 10:
        return np.nan, np.nan, n

    X = np.ones((n, 1), dtype=float)
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True})
    return float(fit.params[0]), float(fit.tvalues[0]), n


def crisis_diff_test(beta_series: pd.Series, crisis_series: pd.Series, maxlags: int) -> tuple[float, float, int]:
    df = pd.DataFrame({"beta": beta_series, "crisis": crisis_series}).dropna()
    n = len(df)
    if n < 10 or df["crisis"].nunique() < 2:
        return np.nan, np.nan, n

    X = sm.add_constant(df[["crisis"]].to_numpy(dtype=float), has_constant="add")
    y = df["beta"].to_numpy(dtype=float)

    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True})

    # crisis coefficient is difference: mean(beta | crisis=1) - mean(beta | crisis=0)
    diff = float(fit.params[1])
    t_diff = float(fit.tvalues[1])
    return diff, t_diff, n


# ============================================================
# Main
# ============================================================
def main():
    print("=== Crisis-State Fama-MacBeth (NBER) ===\n")

    if not PANEL_FILE.exists():
        raise FileNotFoundError(
            f"Panel file not found: {PANEL_FILE}\n"
            "Run Scripts/05_dataset_construction/01_build_data_panel.py first."
        )

    panel = pd.read_parquet(PANEL_FILE)
    panel["week"] = pd.to_datetime(panel["week"]).dt.normalize()

    # Keep rows with valid forward return
    if "valid_R_i_w_plus_1" in panel.columns:
        panel = panel[panel["valid_R_i_w_plus_1"].astype(bool)].copy()
    else:
        panel = panel[panel["R_i_w_plus_1"].notna()].copy()

    # Winsorize controls cross-sectionally each week
    for c in CONTROLS:
        if c in panel.columns:
            panel[c] = panel.groupby("week")[c].transform(winsorize_cs)

    # Crisis indicator by week
    panel["crisis_nber"] = build_crisis_indicator(panel["week"])

    print(f"Panel rows: {len(panel):,}")
    print(f"Stocks: {panel['permno'].nunique():,}")
    print(f"Weeks: {panel['week'].nunique():,}")
    print(
        "Crisis weeks: "
        f"{panel[['week', 'crisis_nber']].drop_duplicates()['crisis_nber'].sum():,}"
    )

    missing_vars = [z for z in RISK_VARS if z not in panel.columns]
    if missing_vars:
        print(f"\nSkipping missing risk variables: {missing_vars}")

    test_vars = [z for z in RISK_VARS if z in panel.columns]
    if not test_vars:
        raise ValueError("None of the configured RISK_VARS exist in the panel.")

    weekly_rows = []
    summary_rows = []

    unique_weeks = sorted(panel["week"].dropna().unique())

    for z_var in test_vars:
        print(f"\nRunning crisis-state FM for: {z_var}")

        # 1) First pass: weekly cross-sectional beta_z,w
        var_rows = []
        for w in unique_weeks:
            df_w = panel[panel["week"] == w]
            crisis_w = int(df_w["crisis_nber"].iloc[0]) if len(df_w) > 0 else 0

            beta_z, n_obs = run_weekly_cs(df_w, z_var)
            if beta_z is None:
                continue

            var_rows.append({
                "risk_var": z_var,
                "week": pd.Timestamp(w),
                "crisis_nber": crisis_w,
                "beta_weekly": beta_z,
                "n_obs_cs": n_obs,
            })

        if not var_rows:
            print("  No valid weekly regressions for this variable.")
            continue

        df_var = pd.DataFrame(var_rows).sort_values("week").reset_index(drop=True)
        weekly_rows.append(df_var)

        # 2) Means by state (HAC t-stat against zero)
        beta_all = df_var["beta_weekly"]
        beta_nc = df_var.loc[df_var["crisis_nber"] == 0, "beta_weekly"]
        beta_c = df_var.loc[df_var["crisis_nber"] == 1, "beta_weekly"]

        mean_all, t_all, n_all = nw_mean_t(beta_all, NW_LAGS)
        mean_nc, t_nc, n_nc = nw_mean_t(beta_nc, NW_LAGS)
        mean_c, t_c, n_c = nw_mean_t(beta_c, NW_LAGS)

        # 3) Crisis-minus-noncrisis difference test
        diff, t_diff, n_diff = crisis_diff_test(
            beta_series=df_var["beta_weekly"],
            crisis_series=df_var["crisis_nber"],
            maxlags=NW_LAGS,
        )

        summary_rows.append({
            "risk_var": z_var,
            "n_weeks_all": n_all,
            "n_weeks_noncrisis": n_nc,
            "n_weeks_crisis": n_c,
            "beta_mean_all": mean_all,
            "t_beta_all": t_all,
            "beta_mean_noncrisis": mean_nc,
            "t_beta_noncrisis": t_nc,
            "beta_mean_crisis": mean_c,
            "t_beta_crisis": t_c,
            "beta_diff_crisis_minus_noncrisis": diff,
            "t_beta_diff": t_diff,
            "n_weeks_diff_test": n_diff,
        })

        print(
            "  "
            f"mean(noncrisis)={mean_nc:.6f}, mean(crisis)={mean_c:.6f}, "
            f"diff={diff:.6f}, t_diff={t_diff:.3f}"
        )

    if not weekly_rows:
        raise ValueError("No weekly crisis-state betas were estimated.")

    weekly_out = pd.concat(weekly_rows, ignore_index=True)
    weekly_out.to_parquet(WEEKLY_BETAS_OUT, index=False)

    summary_out = pd.DataFrame(summary_rows).sort_values("risk_var").reset_index(drop=True)
    summary_out.to_csv(SUMMARY_OUT, index=False, float_format="%.8f")

    print("\nSaved files:")
    print(f"  - {WEEKLY_BETAS_OUT}")
    print(f"  - {SUMMARY_OUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
