"""
Fama-MacBeth cross-sectional regressions following Bollerslev, Li & Zhao (2020).

Each week t, estimate cross-sectional OLS:
    R_{i,t+1} = alpha_t + sum_j beta_{j,t} Z_{j,i,t} + epsilon_{i,t+1}

Then average coefficients over T weeks and compute Newey-West t-statistics.

Specifications:
  Panel A  — Univariate: each predictor alone
  Panel B1 — RSJ + controls  (M2 decomposition)
  Panel B2 — RES + controls
  Panel B3 — RSJ_sys + RSJ_idio + RES_sys + RES_idio + controls  (M2)
  Panel B4 — same but with Method-1 decomposition (rsj_sys_weekly / rsj_idio_weekly)
  Panel B5 — Full kitchen-sink (all 4 decomposed + controls, M2)

Wald tests (H1-H4) from the proposal are run on Panel B3 / B5.

Controls: log(ME), BM, MOM, REV, log(IVOL), log(ILLIQ)
All control variables are cross-sectionally winsorized at 1/99 pct each week.

Output:
  data_final/fama_macbeth/
    fm_results.parquet    — all weekly coefficients (long format)
    fm_summary.csv        — averaged estimates with NW t-stats (one row per spec/variable)
    fm_wald.csv           — Wald test results (H1-H4)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

# ============================================================
# Settings
# ============================================================
ROOT       = Path(__file__).resolve().parents[2]
PANEL_FILE = ROOT / "data_final" / "panel" / "weekly_panel.parquet"
OUTPUT_DIR = ROOT / "data_final" / "fama_macbeth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NW_LAGS   = 6    # Newey-West lags (weeks)
ANNUALIZE = 52   # multiply weekly return coefficient by this for annual bps
MIN_STOCKS = 50  # minimum cross-sectional obs per week

# Controls to include in multiple regressions
CONTROLS = ["me", "bm", "mom", "rev", "ivol", "illiq"]
CONTROL_LABELS = {
    "me":   "log(ME)",
    "bm":   "BM",
    "mom":  "MOM",
    "rev":  "REV",
    "ivol": "IVOL",
    "illiq":"ILLIQ",
}

# Winsorize controls at these percentiles each week
WINSOR_LOW  = 0.01
WINSOR_HIGH = 0.99

# ============================================================
# Regression specifications
# ============================================================
# Each spec: name → list of predictor column names
SPECS = {
    # Panel A: univariate
    "A_rsj"         : ["rsj_weekly"],
    "A_rsj_sys_m2"  : ["rsj_sys"],
    "A_rsj_idio_m2" : ["rsj_idio"],
    "A_rsj_sys_m1"  : ["rsj_sys_weekly"],
    "A_rsj_idio_m1" : ["rsj_idio_weekly"],
    "A_res"         : ["res_weekly"],
    "A_res_sys"     : ["res_sys_p025"],
    "A_res_idio"    : ["res_idio_p025"],
    # Panel B1: RSJ + controls (M2)
    "B1_rsj_ctrl"   : ["rsj_weekly"] + CONTROLS,
    # Panel B2: RES + controls
    "B2_res_ctrl"   : ["res_weekly"] + CONTROLS,
    # Panel B3: RSJ + RES together + controls (M2)
    "B3_rsj_res_ctrl": ["rsj_weekly", "res_weekly"] + CONTROLS,
    # Panel B4: Decomposed M2 + controls
    "B4_decomp_m2"  : ["rsj_sys", "rsj_idio", "res_sys_p025", "res_idio_p025"] + CONTROLS,
    # Panel B5: Decomposed M1 + controls
    "B5_decomp_m1"  : ["rsj_sys_weekly", "rsj_idio_weekly",
                        "res_sys_p025", "res_idio_p025"] + CONTROLS,
}

# Human-readable labels for predictors
PRED_LABELS = {
    "rsj_weekly"    : "RSJ",
    "rsj_sys"       : "RSJ_sys (M2)",
    "rsj_idio"      : "RSJ_idio (M2)",
    "rsj_sys_weekly": "RSJ_sys (M1)",
    "rsj_idio_weekly":"RSJ_idio (M1)",
    "res_weekly"    : "RES",
    "res_sys_p025"  : "RES_sys",
    "res_idio_p025" : "RES_idio",
    **CONTROL_LABELS,
}


# ============================================================
# Helpers
# ============================================================
def winsorize_cross_section(s: pd.Series) -> pd.Series:
    """Winsorize a cross-sectional series at 1/99 pct."""
    lo = s.quantile(WINSOR_LOW)
    hi = s.quantile(WINSOR_HIGH)
    return s.clip(lo, hi)


def nw_average(series: np.ndarray, maxlags: int):
    """
    OLS on a constant with Newey-West SE.
    Returns (mean, t_stat, std_err).
    """
    y = series[~np.isnan(series)]
    T = len(y)
    if T < 10:
        return np.nan, np.nan, np.nan
    ones = np.ones((T, 1))
    res = sm.OLS(y, ones).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags, "use_correction": True},
    )
    return float(res.params[0]), float(res.tvalues[0]), float(res.bse[0])


def run_weekly_cross_section(df_week: pd.DataFrame, predictors: list[str]) -> dict | None:
    """
    Run one week's cross-sectional OLS.
    Returns dict of {predictor: coefficient, 'n_obs': int}.
    """
    cols = ["R_i_w_plus_1"] + predictors
    sub = df_week[cols].dropna()
    if len(sub) < MIN_STOCKS:
        return None
    y = sub["R_i_w_plus_1"].values
    X = sm.add_constant(sub[predictors].values, has_constant="add")
    try:
        res = sm.OLS(y, X).fit()
    except Exception:
        return None
    out = {"n_obs": len(sub)}
    # params: [const, pred1, pred2, ...]
    for i, pred in enumerate(predictors):
        out[pred] = float(res.params[i + 1])
    return out


# ============================================================
# Wald tests
# ============================================================
def wald_test(mean_vec: np.ndarray, cov_mat: np.ndarray,
              R: np.ndarray, r: np.ndarray) -> dict:
    """
    Wald test: H0: R @ mean_vec = r
    W = (R b - r)' (R V R')^{-1} (R b - r) ~ chi2(q)
    """
    diff = R @ mean_vec - r
    var  = R @ cov_mat @ R.T
    try:
        W = float(diff @ np.linalg.solve(var, diff))
    except np.linalg.LinAlgError:
        return {"W": np.nan, "df": len(r), "p_value": np.nan}
    df = len(r)
    p  = float(1 - scipy_stats.chi2.cdf(W, df))
    return {"W": W, "df": df, "p_value": p}


# ============================================================
# Main
# ============================================================
def main():
    print("=== Fama-MacBeth Cross-Sectional Regressions ===\n")

    # ----------------------------------------------------------
    # Load panel
    # ----------------------------------------------------------
    panel = pd.read_parquet(PANEL_FILE)
    panel["week"] = pd.to_datetime(panel["week"]).dt.normalize()

    if "valid_R_i_w_plus_1" in panel.columns:
        panel["valid_R_i_w_plus_1"] = panel["valid_R_i_w_plus_1"].astype(bool)
    else:
        panel["valid_R_i_w_plus_1"] = panel["R_i_w_plus_1"].notna()

    # Keep only valid (R_{i,t+1} available)
    panel = panel[panel["valid_R_i_w_plus_1"]].copy()

    print(f"Panel: {len(panel):,} stock-weeks | "
          f"{panel['permno'].nunique():,} stocks | "
          f"{panel['week'].nunique():,} weeks\n")

    # ----------------------------------------------------------
    # Winsorize controls cross-sectionally each week
    # ----------------------------------------------------------
    print("Winsorizing controls cross-sectionally (1/99 pct per week)...")
    for ctrl in CONTROLS:
        if ctrl not in panel.columns:
            print(f"  WARNING: control '{ctrl}' not in panel — skipping.")
            continue
        panel[ctrl] = (
            panel.groupby("week")[ctrl]
            .transform(winsorize_cross_section)
        )
    print("Done.\n")

    # ----------------------------------------------------------
    # Run Fama-MacBeth week-by-week for each spec
    # ----------------------------------------------------------
    all_coef_rows = []  # for long-format parquet

    # summary: spec → predictor → list of weekly betas
    spec_betas: dict[str, dict[str, list]] = {}

    weeks = sorted(panel["week"].unique())
    total_specs = len(SPECS)

    for spec_idx, (spec_name, predictors) in enumerate(SPECS.items(), 1):
        print(f"[{spec_idx}/{total_specs}] Spec: {spec_name}  "
              f"predictors: {predictors}")

        # Check all predictors exist
        missing = [p for p in predictors if p not in panel.columns]
        if missing:
            print(f"  SKIP — missing columns: {missing}\n")
            continue

        spec_betas[spec_name] = {p: [] for p in predictors}

        n_valid = 0
        for week in weeks:
            df_week = panel[panel["week"] == week]
            result  = run_weekly_cross_section(df_week, predictors)
            if result is None:
                continue
            n_valid += 1
            for pred in predictors:
                spec_betas[spec_name][pred].append(result[pred])
            # store for long-format
            row = {"spec": spec_name, "week": week, "n_obs": result["n_obs"]}
            for pred in predictors:
                row[pred] = result[pred]
            all_coef_rows.append(row)

        print(f"  Valid weeks: {n_valid} / {len(weeks)}\n")

    # ----------------------------------------------------------
    # Save all weekly coefficients (long format)
    # ----------------------------------------------------------
    if all_coef_rows:
        df_coef = pd.DataFrame(all_coef_rows)
        df_coef.to_parquet(OUTPUT_DIR / "fm_results.parquet", index=False)
        print(f"Saved weekly coefficients: {OUTPUT_DIR / 'fm_results.parquet'}\n")

    # ----------------------------------------------------------
    # Summarize: average + NW t-statistics
    # ----------------------------------------------------------
    print("Computing Newey-West averages...\n")
    summary_rows = []

    for spec_name, betas_by_pred in spec_betas.items():
        for pred, beta_list in betas_by_pred.items():
            arr = np.array(beta_list, dtype=float)
            mean, tstat, se = nw_average(arr, NW_LAGS)
            # Convert to basis points for interpretability
            # (weekly return is in decimal; 1 bps = 0.0001)
            mean_bps = mean * 10_000
            se_bps   = se   * 10_000
            summary_rows.append({
                "spec"      : spec_name,
                "predictor" : pred,
                "label"     : PRED_LABELS.get(pred, pred),
                "n_weeks"   : len(beta_list),
                "mean"      : mean,
                "mean_bps"  : mean_bps,   # weekly, in bps
                "t_stat"    : tstat,
                "se"        : se,
                "se_bps"    : se_bps,
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "fm_summary.csv", index=False, float_format="%.6f")
    print(f"Saved summary: {OUTPUT_DIR / 'fm_summary.csv'}\n")

    # ----------------------------------------------------------
    # Console output — pretty print
    # ----------------------------------------------------------
    print("=" * 90)
    print("FAMA-MACBETH RESULTS  (weekly coefficients scaled to bps × 10^4)")
    print("=" * 90)
    for spec_name in SPECS:
        if spec_name not in spec_betas or not spec_betas[spec_name]:
            continue
        sub = df_summary[df_summary["spec"] == spec_name]
        if sub.empty:
            continue
        print(f"\n{spec_name}")
        print(f"  {'Predictor':<22} {'Mean (bps)':>10} {'t-stat':>8} {'N-weeks':>8}")
        print(f"  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*8}")
        for _, row in sub.iterrows():
            stars = ""
            if abs(row["t_stat"]) > 3.29:
                stars = "***"
            elif abs(row["t_stat"]) > 2.58:
                stars = "**"
            elif abs(row["t_stat"]) > 1.96:
                stars = "*"
            print(f"  {row['label']:<22} {row['mean_bps']:>10.2f} "
                  f"{row['t_stat']:>8.2f}{stars:3} {row['n_weeks']:>8}")

    # ----------------------------------------------------------
    # Wald tests (H1-H4) on spec B4 (decomposed M2)
    # ----------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("WALD TESTS on spec B4_decomp_m2  (Hypotheses H1-H4)")
    print("Variables order: [rsj_sys, rsj_idio, res_sys_p025, res_idio_p025, controls...]")
    print("=" * 90)

    spec_key = "B4_decomp_m2"
    if spec_key in spec_betas and spec_betas[spec_key]:
        preds_b4 = SPECS[spec_key]
        n_preds  = len(preds_b4)

        # Build T × n_preds matrix of weekly betas
        coef_arrays = []
        for pred in preds_b4:
            coef_arrays.append(np.array(spec_betas[spec_key][pred], dtype=float))

        T = len(coef_arrays[0])
        B = np.column_stack(coef_arrays)  # T × n_preds

        # Mean vector and NW covariance of the mean
        mean_b = B.mean(axis=0)

        # NW covariance matrix of sqrt(T) * mean_b
        # Use Newey-West on the multivariate series
        # Implement manually: Gamma_0 + sum_{l=1}^{L} w_l (Gamma_l + Gamma_l')
        def nw_cov_matrix(B: np.ndarray, lags: int) -> np.ndarray:
            """NW covariance matrix of the mean (Bartlett kernel)."""
            T, k  = B.shape
            demeaned = B - B.mean(axis=0)
            S = demeaned.T @ demeaned / T  # Gamma_0
            for l in range(1, lags + 1):
                w       = 1 - l / (lags + 1)
                gamma_l = demeaned[l:].T @ demeaned[:-l] / T
                S      += w * (gamma_l + gamma_l.T)
            return S / T  # covariance of the MEAN

        V_mean = nw_cov_matrix(B, NW_LAGS)  # covariance of mean_b

        # Indices of the 4 decomposition predictors
        idx = {p: i for i, p in enumerate(preds_b4)}

        # H1: beta_rsj_sys = beta_rsj_idio
        R1 = np.zeros((1, n_preds));
        R1[0, idx["rsj_sys"]]   =  1
        R1[0, idx["rsj_idio"]]  = -1
        r1 = np.zeros(1)

        # H2: beta_res_sys = beta_res_idio
        R2 = np.zeros((1, n_preds))
        R2[0, idx["res_sys_p025"]]  =  1
        R2[0, idx["res_idio_p025"]] = -1
        r2 = np.zeros(1)

        # H3: beta_res_sys = 0 AND beta_res_idio = 0
        R3 = np.zeros((2, n_preds))
        R3[0, idx["res_sys_p025"]]  = 1
        R3[1, idx["res_idio_p025"]] = 1
        r3 = np.zeros(2)

        # H4: all 4 decomp betas = 0
        R4 = np.zeros((4, n_preds))
        R4[0, idx["rsj_sys"]]       = 1
        R4[1, idx["rsj_idio"]]      = 1
        R4[2, idx["res_sys_p025"]]  = 1
        R4[3, idx["res_idio_p025"]] = 1
        r4 = np.zeros(4)

        wald_results = []
        for h_label, R, r in [("H1: β_rsj_sys = β_rsj_idio",       R1, r1),
                               ("H2: β_res_sys = β_res_idio",       R2, r2),
                               ("H3: β_res_sys = β_res_idio = 0",  R3, r3),
                               ("H4: all 4 betas = 0",              R4, r4)]:
            w = wald_test(mean_b, V_mean, R, r)
            wald_results.append({
                "hypothesis": h_label,
                "W"         : w["W"],
                "df"        : w["df"],
                "p_value"   : w["p_value"],
            })
            sig = "***" if w["p_value"] < 0.01 else ("**" if w["p_value"] < 0.05
                  else ("*" if w["p_value"] < 0.10 else ""))
            print(f"  {h_label:<38}  W={w['W']:7.2f}  df={w['df']}  "
                  f"p={w['p_value']:.4f} {sig}")

        df_wald = pd.DataFrame(wald_results)
        df_wald.to_csv(OUTPUT_DIR / "fm_wald.csv", index=False, float_format="%.6f")
        print(f"\nSaved Wald tests: {OUTPUT_DIR / 'fm_wald.csv'}")
    else:
        print(f"  Spec {spec_key} not computed — skipping Wald tests.")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
