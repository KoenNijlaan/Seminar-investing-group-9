"""
Crisis-state Fama-MacBeth analysis using NBER recession dates.

Econometric note:
- In weekly Fama-MacBeth cross-sections, a crisis dummy C_w is constant within
  week, so C_w and Z_{i,w}*C_w are not identified in that cross-section.
- We therefore estimate weekly cross-sectional slopes first, then test whether
  those slopes differ in NBER recession weeks.

Step 1 (weekly cross-section):
    R_{i,w+1} = a_w + b_w' Z_{i,w} + g_w' X_{i,w} + e_{i,w+1}

Step 2 (state dependence of each weekly slope):
    beta_{k,w} = alpha_k + delta_k * C_w + u_{k,w}

where delta_k is crisis-minus-noncrisis difference (HAC/Newey-West t-stats).

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

# Same controls as baseline Fama-MacBeth
CONTROLS = ["me", "bm", "mom", "rev", "ivol", "illiq"]

# Full specification set (independent + joint), aligned to your FM script
CRISIS_SPECS = {
    # Panel A: univariate
    "A_rsj": ["rsj_weekly"],
    "A_rsj_sys_m2": ["rsj_sys"],
    "A_rsj_idio_m2": ["rsj_idio"],
    "A_rsj_sys_m1": ["rsj_sys_weekly"],
    "A_rsj_idio_m1": ["rsj_idio_weekly"],
    "A_res": ["res_weekly"],
    "A_res_sys": ["res_sys_p025"],
    "A_res_idio": ["res_idio_p025"],
    # Panel B: with controls
    "B1_rsj_ctrl": ["rsj_weekly"] + CONTROLS,
    "B2_res_ctrl": ["res_weekly"] + CONTROLS,
    "B3_rsj_res_ctrl": ["rsj_weekly", "res_weekly"] + CONTROLS,
    "B4_decomp_m2": ["rsj_sys", "rsj_idio", "res_sys_p025", "res_idio_p025"] + CONTROLS,
    "B5_decomp_m1": ["rsj_sys_weekly", "rsj_idio_weekly", "res_sys_p025", "res_idio_p025"] + CONTROLS,
}

# If True, summary will also include controls; if False, only risk variables
REPORT_CONTROLS_IN_SUMMARY = False

# NBER recessions overlapping sample 1993-2024
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


def run_weekly_cs(df_week: pd.DataFrame, predictors: list[str]) -> tuple[dict | None, int]:
    cols = ["R_i_w_plus_1"] + predictors
    sub = df_week[cols].dropna()
    n_obs = len(sub)
    if n_obs < MIN_STOCKS:
        return None, n_obs

    # Force numeric float arrays (avoid pandas extension dtype issues)
    y = pd.to_numeric(sub["R_i_w_plus_1"], errors="coerce").to_numpy(dtype=float)
    X_raw = sub[predictors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = sm.add_constant(X_raw, has_constant="add")

    if not np.isfinite(y).all() or not np.isfinite(X).all():
        return None, n_obs

    try:
        fit = sm.OLS(y, X).fit()
    except Exception:
        return None, n_obs

    out = {"alpha": float(fit.params[0])}
    for i, p in enumerate(predictors):
        out[p] = float(fit.params[i + 1])
    return out, n_obs


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

    # crisis coefficient = mean(beta|crisis=1) - mean(beta|crisis=0)
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

    # Keep valid forward-return rows
    if "valid_R_i_w_plus_1" in panel.columns:
        panel = panel[panel["valid_R_i_w_plus_1"].astype(bool)].copy()
    else:
        panel = panel[panel["R_i_w_plus_1"].notna()].copy()

    # Winsorize controls cross-sectionally by week
    for c in CONTROLS:
        if c in panel.columns:
            panel[c] = panel.groupby("week")[c].transform(winsorize_cs)

    panel["crisis_nber"] = build_crisis_indicator(panel["week"])

    print(f"Panel rows: {len(panel):,}")
    print(f"Stocks: {panel['permno'].nunique():,}")
    print(f"Weeks: {panel['week'].nunique():,}")
    print(
        "Crisis weeks: "
        f"{panel[['week', 'crisis_nber']].drop_duplicates()['crisis_nber'].sum():,}"
    )

    unique_weeks = sorted(panel["week"].dropna().unique())

    weekly_rows = []
    summary_rows = []

    for spec_name, predictors in CRISIS_SPECS.items():
        print(f"\nSpec: {spec_name}  predictors: {predictors}")

        missing = [p for p in predictors if p not in panel.columns]
        if missing:
            print(f"  SKIP — missing columns: {missing}")
            continue

        spec_weekly_rows = []
        n_valid_weeks = 0

        for w in unique_weeks:
            df_w = panel[panel["week"] == w]
            crisis_w = int(df_w["crisis_nber"].iloc[0]) if len(df_w) > 0 else 0

            coefs, n_obs = run_weekly_cs(df_w, predictors)
            if coefs is None:
                continue

            n_valid_weeks += 1
            row = {
                "spec": spec_name,
                "week": pd.Timestamp(w),
                "crisis_nber": crisis_w,
                "n_obs_cs": n_obs,
            }
            row.update(coefs)
            spec_weekly_rows.append(row)

        print(f"  Valid weeks: {n_valid_weeks} / {len(unique_weeks)}")

        if not spec_weekly_rows:
            continue

        df_spec = pd.DataFrame(spec_weekly_rows).sort_values("week").reset_index(drop=True)
        weekly_rows.append(df_spec)

        # Crisis summary by predictor in this spec
        if REPORT_CONTROLS_IN_SUMMARY:
            targets = [v for v in predictors if v in df_spec.columns]
        else:
            targets = [v for v in predictors if v in df_spec.columns and v not in CONTROLS]

        for var in targets:
            beta_all = df_spec[var]
            beta_nc = df_spec.loc[df_spec["crisis_nber"] == 0, var]
            beta_c = df_spec.loc[df_spec["crisis_nber"] == 1, var]

            mean_all, t_all, n_all = nw_mean_t(beta_all, NW_LAGS)
            mean_nc, t_nc, n_nc = nw_mean_t(beta_nc, NW_LAGS)
            mean_c, t_c, n_c = nw_mean_t(beta_c, NW_LAGS)

            diff, t_diff, n_diff = crisis_diff_test(
                beta_series=df_spec[var],
                crisis_series=df_spec["crisis_nber"],
                maxlags=NW_LAGS,
            )

            summary_rows.append({
                "spec": spec_name,
                "risk_var": var,
                "n_weeks_all": n_all,
                "n_weeks_noncrisis": n_nc,
                "n_weeks_crisis": n_c,
                "beta_mean_all": mean_all,
                "beta_mean_all_bps": mean_all * 10000 if pd.notna(mean_all) else np.nan,
                "t_beta_all": t_all,
                "beta_mean_noncrisis": mean_nc,
                "beta_mean_noncrisis_bps": mean_nc * 10000 if pd.notna(mean_nc) else np.nan,
                "t_beta_noncrisis": t_nc,
                "beta_mean_crisis": mean_c,
                "beta_mean_crisis_bps": mean_c * 10000 if pd.notna(mean_c) else np.nan,
                "t_beta_crisis": t_c,
                "beta_diff_crisis_minus_noncrisis": diff,
                "beta_diff_crisis_minus_noncrisis_bps": diff * 10000 if pd.notna(diff) else np.nan,
                "t_beta_diff": t_diff,
                "n_weeks_diff_test": n_diff,
            })

            print(
                "   "
                f"{var}: noncrisis={mean_nc:.6f}, crisis={mean_c:.6f}, "
                f"diff={diff:.6f}, t_diff={t_diff:.3f}"
            )

    if not weekly_rows:
        raise ValueError("No weekly crisis-state regressions were estimated.")

    weekly_out = pd.concat(weekly_rows, ignore_index=True)
    weekly_out.to_parquet(WEEKLY_BETAS_OUT, index=False)

    summary_out = pd.DataFrame(summary_rows).sort_values(["spec", "risk_var"]).reset_index(drop=True)
    summary_out.to_csv(SUMMARY_OUT, index=False, float_format="%.8f")

    print("\nSaved files:")
    print(f"  - {WEEKLY_BETAS_OUT}")
    print(f"  - {SUMMARY_OUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()