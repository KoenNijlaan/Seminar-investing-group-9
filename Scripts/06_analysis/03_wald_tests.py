"""
Wald tests (H1-H4) on Fama-MacBeth weekly coefficients.

Loads fm_results.parquet (saved by 02_fama_macbeth.py) and tests four
hypotheses about the decomposed RSJ and RES betas, for both M1 and M2
decomposition methods.

Hypotheses (from proposal):
  H1: beta_rsj_sys = beta_rsj_idio        (1 restriction)
  H2: beta_res_sys = beta_res_idio        (1 restriction)
  H3: beta_res_sys = 0 AND beta_res_idio = 0  (2 restrictions, joint)
  H4: all 4 decomposition betas = 0      (4 restrictions, joint)

Test statistic: W = (Rb - r)' (R V R')^{-1} (Rb - r) ~ chi2(q) under H0
where V = Var(b_bar) is estimated with a Newey-West covariance matrix.

Output:
  data_final/fama_macbeth/fm_wald.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ============================================================
# Settings
# ============================================================
ROOT       = Path(__file__).resolve().parents[2]
FM_FILE    = ROOT / "data_final" / "fama_macbeth" / "fm_results.parquet"
OUTPUT_DIR = ROOT / "data_final" / "fama_macbeth"

NW_LAGS = 6   # must match 02_fama_macbeth.py

# Predictors for each decomposition spec (must match SPECS in 02_fama_macbeth.py)
CONTROLS = ["me", "bm", "mom", "rev", "ivol", "illiq"]

SPECS = {
    "B4_decomp_m2": ["rsj_sys", "rsj_idio", "res_sys_p025", "res_idio_p025"] + CONTROLS,
    "B5_decomp_m1": ["rsj_sys_weekly", "rsj_idio_weekly",
                     "res_sys_p025", "res_idio_p025"] + CONTROLS,
}

# Which columns are RSJ sys/idio and RES sys/idio for each spec
DECOMP_KEYS = {
    "B4_decomp_m2": {
        "rsj_sys":  "rsj_sys",
        "rsj_idio": "rsj_idio",
        "res_sys":  "res_sys_p025",
        "res_idio": "res_idio_p025",
    },
    "B5_decomp_m1": {
        "rsj_sys":  "rsj_sys_weekly",
        "rsj_idio": "rsj_idio_weekly",
        "res_sys":  "res_sys_p025",
        "res_idio": "res_idio_p025",
    },
}


# ============================================================
# Core functions
# ============================================================
def nw_cov_matrix(B: np.ndarray, lags: int) -> np.ndarray:
    """
    Newey-West covariance matrix of the sample mean of rows of B.

    Returns Var(b_bar) = S_NW / T, where S_NW is the long-run variance
    estimated with Bartlett kernel weights w_l = 1 - l/(lags+1).
    """
    T, k     = B.shape
    demeaned = B - B.mean(axis=0)
    S = demeaned.T @ demeaned / T           # Gamma_0
    for l in range(1, lags + 1):
        w       = 1 - l / (lags + 1)       # Bartlett weight
        gamma_l = demeaned[l:].T @ demeaned[:-l] / T
        S      += w * (gamma_l + gamma_l.T)
    return S / T                            # covariance of the MEAN


def wald_test(mean_vec: np.ndarray, cov_mat: np.ndarray,
              R: np.ndarray, r: np.ndarray) -> dict:
    """
    Wald test: H0: R @ mean_vec = r
    W = (Rb - r)' (R V R')^{-1} (Rb - r) ~ chi2(q)
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


def build_restriction_matrices(idx: dict, n_preds: int, keys: dict):
    """
    Build (label, R, r) triples for H1-H4.

    idx   : {predictor_name: column_index}
    keys  : {"rsj_sys": col_name, "rsj_idio": col_name, ...}
    """
    rsj_sys  = keys["rsj_sys"]
    rsj_idio = keys["rsj_idio"]
    res_sys  = keys["res_sys"]
    res_idio = keys["res_idio"]

    # H1: beta_rsj_sys = beta_rsj_idio
    R1 = np.zeros((1, n_preds))
    R1[0, idx[rsj_sys]]  =  1
    R1[0, idx[rsj_idio]] = -1
    r1 = np.zeros(1)

    # H2: beta_res_sys = beta_res_idio
    R2 = np.zeros((1, n_preds))
    R2[0, idx[res_sys]]  =  1
    R2[0, idx[res_idio]] = -1
    r2 = np.zeros(1)

    # H3: beta_res_sys = 0 AND beta_res_idio = 0 (joint)
    R3 = np.zeros((2, n_preds))
    R3[0, idx[res_sys]]  = 1
    R3[1, idx[res_idio]] = 1
    r3 = np.zeros(2)

    # H4: all 4 decomposition betas = 0 (joint)
    R4 = np.zeros((4, n_preds))
    R4[0, idx[rsj_sys]]  = 1
    R4[1, idx[rsj_idio]] = 1
    R4[2, idx[res_sys]]  = 1
    R4[3, idx[res_idio]] = 1
    r4 = np.zeros(4)

    return [
        ("H1: beta_rsj_sys = beta_rsj_idio",           R1, r1),
        ("H2: beta_res_sys = beta_res_idio",           R2, r2),
        ("H3: beta_res_sys = 0 AND beta_res_idio = 0", R3, r3),
        ("H4: all 4 decomp betas = 0",                 R4, r4),
    ]


def run_wald_tests_for_spec(spec_key: str, df_fm: pd.DataFrame) -> list[dict]:
    """
    Run H1-H4 for one decomposition spec. Returns a list of result dicts.
    """
    preds  = SPECS[spec_key]
    keys   = DECOMP_KEYS[spec_key]
    n_preds = len(preds)

    sub = df_fm[df_fm["spec"] == spec_key].copy()
    if sub.empty:
        print(f"  No data for spec {spec_key} — skipping.")
        return []

    # Build T × n_preds matrix: one row per week, one column per predictor
    weeks = sorted(sub["week"].unique())
    B_rows = []
    for week in weeks:
        row_data = sub[sub["week"] == week]
        # Each predictor's coefficient is stored in a column named after it
        coefs = []
        ok = True
        for p in preds:
            if p not in row_data.columns or row_data[p].isna().all():
                ok = False
                break
            coefs.append(float(row_data[p].iloc[0]))
        if ok:
            B_rows.append(coefs)

    if len(B_rows) < 10:
        print(f"  Too few weeks ({len(B_rows)}) for spec {spec_key} — skipping.")
        return []

    B      = np.array(B_rows, dtype=float)   # T × n_preds
    mean_b = B.mean(axis=0)
    V_mean = nw_cov_matrix(B, NW_LAGS)
    idx    = {p: i for i, p in enumerate(preds)}

    hypotheses = build_restriction_matrices(idx, n_preds, keys)

    rows = []
    for h_label, R, r in hypotheses:
        result = wald_test(mean_b, V_mean, R, r)
        rows.append({
            "spec"      : spec_key,
            "hypothesis": h_label,
            "W"         : result["W"],
            "df"        : result["df"],
            "p_value"   : result["p_value"],
            "n_weeks"   : len(B_rows),
        })
        sig = ("***" if result["p_value"] < 0.01 else
               "**"  if result["p_value"] < 0.05 else
               "*"   if result["p_value"] < 0.10 else "")
        print(f"  {h_label:<47}  W={result['W']:7.2f}  "
              f"df={result['df']}  p={result['p_value']:.4f} {sig}")
    return rows


# ============================================================
# Main
# ============================================================
def main():
    print("=== Wald Tests on Fama-MacBeth Coefficients ===\n")

    if not FM_FILE.exists():
        raise FileNotFoundError(
            f"FM results not found: {FM_FILE}\n"
            "Run 02_fama_macbeth.py first."
        )

    df_fm = pd.read_parquet(FM_FILE)
    df_fm["week"] = pd.to_datetime(df_fm["week"])
    print(f"Loaded: {len(df_fm):,} rows, {df_fm['spec'].nunique()} specs\n")

    all_rows = []

    for spec_key in SPECS:
        print("=" * 70)
        print(f"Spec: {spec_key}")
        print("=" * 70)
        rows = run_wald_tests_for_spec(spec_key, df_fm)
        all_rows.extend(rows)
        print()

    if all_rows:
        df_wald = pd.DataFrame(all_rows)
        out_path = OUTPUT_DIR / "fm_wald.csv"
        df_wald.to_csv(out_path, index=False, float_format="%.6f")
        print(f"Saved: {out_path}")
    else:
        print("No Wald test results to save.")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
