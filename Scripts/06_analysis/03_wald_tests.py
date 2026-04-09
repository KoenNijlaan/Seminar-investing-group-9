"""
Wald tests on decomposed Fama-MacBeth coefficients.

Loads the weekly slope series from:
    data_final/results/intermediate/fm_coefs_B7.parquet  (M1 full decomposition)
    data_final/results/intermediate/fm_coefs_B8.parquet  (M2 full decomposition)

Coefficient vector for each decomposition:
  β = [β_RSJ_sys, β_RSJ_idio, β_RES_sys, β_RES_idio]   (in this order)

Tests:
  H1: β_RSJ_sys = β_RSJ_idio              R=[1,-1, 0, 0],     r=0,    df=1
  H2: β_RES_sys = β_RES_idio              R=[0, 0, 1,-1],     r=0,    df=1
  H3: β_RSJ_idio=0 AND β_RES_idio=0       R=[[0,1,0,0],[0,0,0,1]], r=0, df=2
  H4: β_RSJ_sys=0  AND β_RES_sys=0        R=[[1,0,0,0],[0,0,1,0]], r=0, df=2

Wald statistic: W = (Rβ̄ - r)' [R Var(β̄) R']^{-1} (Rβ̄ - r) ~ chi²(df) under H0

Var(β̄) = S_NW / T  where S_NW is the NW(6) long-run covariance matrix.

Output:
  data_final/results/tables/tab_wald_tests.tex
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ============================================================
# Settings
# ============================================================
ROOT      = Path(__file__).resolve().parents[2]
INTER_DIR = ROOT / "data_final" / "results" / "intermediate"
TABLE_DIR = ROOT / "data_final" / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

NW_LAGS = 6

# Ordered coefficient columns for each decomposition spec
DECOMP_COLS = {
    "B7": ["rsj_sys_weekly", "rsj_idio_weekly", "res_sys_p025", "res_idio_p025"],
    "B8": ["rsj_sys",        "rsj_idio",        "res_sys_p025", "res_idio_p025"],
}

PRED_LABEL = {
    "B7": {"rsj_sys_weekly": "RSJ sys (M1)", "rsj_idio_weekly": "RSJ idio (M1)",
            "res_sys_p025": "RES sys", "res_idio_p025": "RES idio"},
    "B8": {"rsj_sys": "RSJ sys (M2)", "rsj_idio": "RSJ idio (M2)",
            "res_sys_p025": "RES sys", "res_idio_p025": "RES idio"},
}


# ============================================================
# NW covariance of mean
# ============================================================
def nw_cov_mean(B: np.ndarray, lags: int) -> np.ndarray:
    """
    NW long-run covariance matrix of the time-averaged coefficient vector.
    B : T × k  matrix of weekly coefficient vectors
    Returns Var(β̄) = S_NW / T
    """
    T, k     = B.shape
    demeaned = B - B.mean(axis=0)
    S = demeaned.T @ demeaned / T          # Gamma_0
    for l in range(1, lags + 1):
        w       = 1.0 - l / (lags + 1)    # Bartlett weight
        gamma_l = demeaned[l:].T @ demeaned[:-l] / T
        S      += w * (gamma_l + gamma_l.T)
    return S / T                           # Var(β̄)


# ============================================================
# Wald test
# ============================================================
def wald_test(mean_vec, cov_mat, R, r):
    diff = R @ mean_vec - r
    V    = R @ cov_mat @ R.T
    try:
        W = float(diff @ np.linalg.solve(V, diff))
    except np.linalg.LinAlgError:
        return dict(W=np.nan, df=len(r), p=np.nan)
    df = len(r)
    p  = float(1 - scipy_stats.chi2.cdf(W, df))
    return dict(W=W, df=df, p=p)


# ============================================================
# Build restriction matrices (fixed order: sys_RSJ, idio_RSJ, sys_RES, idio_RES)
# ============================================================
def make_restrictions():
    """
    Signed restrictions (H1–H4).
    β index: 0=RSJ_sys, 1=RSJ_idio, 2=RES_sys, 3=RES_idio
    """
    # H1: β_RSJ_sys = β_RSJ_idio  ↔  β[0] - β[1] = 0
    R1 = np.array([[1, -1, 0, 0]], dtype=float)
    r1 = np.zeros(1)

    # H2: β_RES_sys = β_RES_idio  ↔  β[2] - β[3] = 0
    R2 = np.array([[0, 0, 1, -1]], dtype=float)
    r2 = np.zeros(1)

    # H3: β_RSJ_idio = 0 AND β_RES_idio = 0
    R3 = np.array([[0, 1, 0, 0],
                   [0, 0, 0, 1]], dtype=float)
    r3 = np.zeros(2)

    # H4: β_RSJ_sys = 0 AND β_RES_sys = 0
    R4 = np.array([[1, 0, 0, 0],
                   [0, 0, 1, 0]], dtype=float)
    r4 = np.zeros(2)

    return [
        (r"$H_1$: $\beta^\mathrm{RSJ}_\mathrm{sys} = \beta^\mathrm{RSJ}_\mathrm{idio}$",  R1, r1),
        (r"$H_2$: $\beta^\mathrm{RES}_\mathrm{sys} = \beta^\mathrm{RES}_\mathrm{idio}$",  R2, r2),
        (r"$H_3$: $\beta^\mathrm{RSJ}_\mathrm{idio} = \beta^\mathrm{RES}_\mathrm{idio} = 0$", R3, r3),
        (r"$H_4$: $\beta^\mathrm{RSJ}_\mathrm{sys} = \beta^\mathrm{RES}_\mathrm{sys} = 0$",   R4, r4),
    ]


def make_abs_restrictions(mean_b: np.ndarray):
    """
    Absolute-value restrictions (H1_abs, H2_abs) via the delta method.

    For |β_a| = |β_b|, linearise around the estimated mean:
        |β_a| - |β_b| ≈ sign(β̄_a)·β_a - sign(β̄_b)·β_b

    so R_abs = [sign(β̄_a), -sign(β̄_b), ...]

    H3 and H4 test β = 0, which is the same as |β| = 0, so they are
    unchanged and not duplicated here.

    β index: 0=RSJ_sys, 1=RSJ_idio, 2=RES_sys, 3=RES_idio
    """
    s = np.sign(mean_b)   # signs of estimated means; shape (4,)
    # If a mean is exactly zero, sign=0; treat as +1 to avoid a zero row
    s = np.where(s == 0, 1.0, s)

    # H1_abs: |β_RSJ_sys| = |β_RSJ_idio|  ↔  s[0]*β[0] - s[1]*β[1] = 0
    R1a = np.array([[s[0], -s[1], 0, 0]], dtype=float)
    r1a = np.zeros(1)

    # H2_abs: |β_RES_sys| = |β_RES_idio|  ↔  s[2]*β[2] - s[3]*β[3] = 0
    R2a = np.array([[0, 0, s[2], -s[3]]], dtype=float)
    r2a = np.zeros(1)

    return [
        (r"$H_{1,\mathrm{abs}}$: $|\beta^\mathrm{RSJ}_\mathrm{sys}| = |\beta^\mathrm{RSJ}_\mathrm{idio}|$",
         R1a, r1a),
        (r"$H_{2,\mathrm{abs}}$: $|\beta^\mathrm{RES}_\mathrm{sys}| = |\beta^\mathrm{RES}_\mathrm{idio}|$",
         R2a, r2a),
    ]


# ============================================================
# Run Wald tests for one decomposition spec
# ============================================================
def run_wald(spec_key):
    fpath = INTER_DIR / f"fm_coefs_{spec_key}.parquet"
    if not fpath.exists():
        raise FileNotFoundError(
            f"{fpath}\nRun 02_fama_macbeth.py first."
        )

    df = pd.read_parquet(fpath)
    df["week"] = pd.to_datetime(df["week"])
    cols = DECOMP_COLS[spec_key]

    # Check all needed columns exist
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Spec {spec_key}: missing columns {missing} in {fpath.name}")

    # Build T × 4 matrix (drop weeks where any component is NaN)
    B = df[cols].dropna().to_numpy(dtype=float)
    T = len(B)
    print(f"  {spec_key}: {T} valid weeks")

    mean_b = B.mean(axis=0)
    V      = nw_cov_mean(B, NW_LAGS)

    all_restrictions = make_restrictions() + make_abs_restrictions(mean_b)
    rows = []
    for label, R, r in all_restrictions:
        res = wald_test(mean_b, V, R, r)
        rows.append({
            "hypothesis": label,
            "df": res["df"],
            "W_M1" if spec_key == "B7" else "W_M2": res["W"],
            "p_M1" if spec_key == "B7" else "p_M2": res["p"],
        })
        sig = "***" if res["p"] < 0.01 else "**" if res["p"] < 0.05 else "*" if res["p"] < 0.10 else ""
        print(f"    {label[:60]:<60}  W={res['W']:7.2f}  df={res['df']}  "
              f"p={res['p']:.4f}{sig}")

    return pd.DataFrame(rows), mean_b, V


# ============================================================
# LaTeX table
# ============================================================
def build_latex(df_m1, df_m2):
    def _p_stars(p):
        if np.isnan(p): return ""
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    def _fmt_w(val, p):
        if np.isnan(val): return ""
        return f"{val:.2f}{_p_stars(p)}"

    def _fmt_p(val):
        if np.isnan(val): return ""
        return f"{val:.4f}"

    # Merge on hypothesis
    m1 = df_m1.rename(columns={"W_M1": "W_M1", "p_M1": "p_M1"})
    m2 = df_m2.rename(columns={"W_M2": "W_M2", "p_M2": "p_M2"})
    merged = m1.merge(m2[["hypothesis","W_M2","p_M2"]], on="hypothesis", how="outer")

    lines = []
    lines += [r"\begin{table}[htbp]", r"\centering", r"\begin{threeparttable}"]
    lines += [r"\caption{Wald Tests on Decomposed Fama--MacBeth Coefficients}"]
    lines += [r"\label{tab:wald_tests}"]
    lines += [r"\begin{tabular}{lcrcrr}"]
    lines += [r"\toprule"]
    lines += [r"Hypothesis & df & $\chi^2$ (M1) & $p$-value & $\chi^2$ (M2) & $p$-value \\"]
    lines += [r"\midrule"]

    # Signed tests (H1–H4) then absolute-value tests (H1_abs, H2_abs)
    ABS_MARKER = r"$H_{1,\mathrm{abs}}$"
    first_abs_seen = False
    for _, row in merged.iterrows():
        hyp = row["hypothesis"]
        if not first_abs_seen and ABS_MARKER in hyp:
            lines += [r"\midrule",
                      r"\multicolumn{6}{l}{\textit{Absolute-value tests (delta method)}} \\"]
            first_abs_seen = True
        df  = int(row["df"]) if np.isfinite(row["df"]) else ""
        w1  = _fmt_w(row.get("W_M1", np.nan), row.get("p_M1", np.nan))
        p1  = _fmt_p(row.get("p_M1", np.nan))
        w2  = _fmt_w(row.get("W_M2", np.nan), row.get("p_M2", np.nan))
        p2  = _fmt_p(row.get("p_M2", np.nan))
        lines.append(f"{hyp} & {df} & {w1} & {p1} & {w2} & {p2} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [r"\begin{tablenotes}", r"\small",
              r"\item Notes: Coefficient vector $\boldsymbol{\beta} ="
              r" [\beta^\mathrm{RSJ}_\mathrm{sys},\, \beta^\mathrm{RSJ}_\mathrm{idio},"
              r"\, \beta^\mathrm{RES}_\mathrm{sys},\, \beta^\mathrm{RES}_\mathrm{idio}]$."
              r" M1 = intraday decomposition; M2 = rolling 52-week regression."
              r" Wald statistic $W = (\mathbf{R}\bar{\boldsymbol{\beta}} - \mathbf{r})'$"
              r" $[\mathbf{R}\widehat{\mathrm{Var}}(\bar{\boldsymbol{\beta}})\mathbf{R}']^{-1}$"
              r" $(\mathbf{R}\bar{\boldsymbol{\beta}} - \mathbf{r}) \sim \chi^2(df)$ under $H_0$."
              r" $\widehat{\mathrm{Var}}(\bar{\boldsymbol{\beta}})$ uses NW(6) Bartlett kernel."
              r" Absolute-value tests ($H_{1,\mathrm{abs}}$, $H_{2,\mathrm{abs}}$) apply the"
              r" delta-method linearisation $|\beta| \approx \mathrm{sign}(\bar{\beta})\cdot\beta$"
              r" and test $|\beta_\mathrm{sys}| = |\beta_\mathrm{idio}|$ regardless of sign."
              r" $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
              r"\end{tablenotes}",
              r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
def main():
    print("=== Wald Tests on FM Coefficients ===\n")

    print("--- M1 decomposition (B7) ---")
    df_m1, mean_m1, _ = run_wald("B7")

    print("\n--- M2 decomposition (B8) ---")
    df_m2, mean_m2, _ = run_wald("B8")

    # Build combined table
    tex = build_latex(df_m1, df_m2)
    out = TABLE_DIR / "tab_wald_tests.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"\nSaved: {out.name}")

    # Console summary
    print("\n--- M1 mean betas (bps) ---")
    for lbl, val in zip(DECOMP_COLS["B7"], mean_m1 * 10_000):
        print(f"  {lbl:<30} {val:>8.2f}")

    print("\n--- M2 mean betas (bps) ---")
    for lbl, val in zip(DECOMP_COLS["B8"], mean_m2 * 10_000):
        print(f"  {lbl:<30} {val:>8.2f}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
