"""
Wald Tests

Purpose:
  Test linear coefficient restrictions and cross-model hypotheses using Wald statistics.

Inputs:
  - Regression coefficients and covariance inputs from analysis outputs.

Outputs:
  - Wald test summary tables and supporting files.

Main Steps:
  - Define restriction matrices.
  - Compute Wald statistics and p-values.
  - Write formatted output tables.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT      = Path(__file__).resolve().parents[2]
INTER_DIR = ROOT / "data_final" / "results" / "intermediate"
TABLE_DIR = ROOT / "data_final" / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

NW_LAGS = 6

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

def nw_cov_mean(B: np.ndarray, lags: int) -> np.ndarray:
    T, k     = B.shape
    demeaned = B - B.mean(axis=0)
    S = demeaned.T @ demeaned / T
    for l in range(1, lags + 1):
        w       = 1.0 - l / (lags + 1)
        gamma_l = demeaned[l:].T @ demeaned[:-l] / T
        S      += w * (gamma_l + gamma_l.T)
    return S / T

def wald_test(g_val: np.ndarray, G: np.ndarray, cov_mat: np.ndarray) -> dict:
    V = G @ cov_mat @ G.T
    try:
        W = float(g_val @ np.linalg.solve(V, g_val))
    except np.linalg.LinAlgError:
        return dict(W=np.nan, df=len(g_val), p=np.nan)
    df = len(g_val)
    p  = float(1 - scipy_stats.chi2.cdf(W, df))
    return dict(W=W, df=df, p=p)

def make_restrictions(mean_b: np.ndarray):
    b = mean_b

    g1 = np.array([b[0]**2 - b[1]**2])
    G1 = np.array([[2*b[0], -2*b[1], 0.0, 0.0]])

    g2 = np.array([b[2]**2 - b[3]**2])
    G2 = np.array([[0.0, 0.0, 2*b[2], -2*b[3]]])

    R3 = np.array([[0, 1, 0, 0],
                   [0, 0, 0, 1]], dtype=float)
    g3 = R3 @ b
    G3 = R3

    R4 = np.array([[1, 0, 0, 0],
                   [0, 0, 1, 0]], dtype=float)
    g4 = R4 @ b
    G4 = R4

    return [
        (r"$H_1$: $|\beta^\mathrm{RSJ}_\mathrm{sys}| = |\beta^\mathrm{RSJ}_\mathrm{idio}|$", g1, G1),
        (r"$H_2$: $|\beta^\mathrm{RES}_\mathrm{sys}| = |\beta^\mathrm{RES}_\mathrm{idio}|$", g2, G2),
        (r"$H_3$: $\beta^\mathrm{RSJ}_\mathrm{idio} = \beta^\mathrm{RES}_\mathrm{idio} = 0$", g3, G3),
        (r"$H_4$: $\beta^\mathrm{RSJ}_\mathrm{sys} = \beta^\mathrm{RES}_\mathrm{sys} = 0$",   g4, G4),
    ]

def run_wald(spec_key):
    fpath = INTER_DIR / f"fm_coefs_{spec_key}.parquet"
    if not fpath.exists():
        raise FileNotFoundError(
            f"{fpath}\nRun 02_fama_macbeth.py first."
        )

    df = pd.read_parquet(fpath)
    df["week"] = pd.to_datetime(df["week"])
    cols = DECOMP_COLS[spec_key]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Spec {spec_key}: missing columns {missing} in {fpath.name}")

    B = df[cols].dropna().to_numpy(dtype=float)
    T = len(B)
    print(f"  {spec_key}: {T} valid weeks")

    mean_b = B.mean(axis=0)
    V      = nw_cov_mean(B, NW_LAGS)

    all_restrictions = make_restrictions(mean_b)
    rows = []
    for label, g_val, G in all_restrictions:
        res = wald_test(g_val, G, V)
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

    for _, row in merged.iterrows():
        hyp = row["hypothesis"]
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
              r" Nonlinear Wald statistic $W = g(\bar{\boldsymbol{\beta}})'$"
              r" $[G\,\widehat{\mathrm{Var}}(\bar{\boldsymbol{\beta}})\,G']^{-1}$"
              r" $g(\bar{\boldsymbol{\beta}}) \sim \chi^2(df)$ under $H_0$,"
              r" where $G = \partial g / \partial \bar{\boldsymbol{\beta}}'$ is the Jacobian."
              r" $\widehat{\mathrm{Var}}(\bar{\boldsymbol{\beta}})$ uses NW(6) Bartlett kernel."
              r" $H_1$ and $H_2$ use $g(\boldsymbol{\beta}) = \beta_\mathrm{sys}^2 - \beta_\mathrm{idio}^2$"
              r" with Jacobian $G = [2\beta_\mathrm{sys},\, {-2}\beta_\mathrm{idio},\, \ldots]$;"
              r" $H_3$ and $H_4$ are linear zero restrictions."
              r" $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
              r"\end{tablenotes}",
              r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(lines)

def main():
    print("=== Wald Tests on FM Coefficients ===\n")

    print("--- M1 decomposition (B7) ---")
    df_m1, mean_m1, _ = run_wald("B7")

    print("\n--- M2 decomposition (B8) ---")
    df_m2, mean_m2, _ = run_wald("B8")

    tex = build_latex(df_m1, df_m2)
    out = TABLE_DIR / "tab_wald_tests.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"\nSaved: {out.name}")

    print("\n--- M1 mean betas (bps) ---")
    for lbl, val in zip(DECOMP_COLS["B7"], mean_m1 * 10_000):
        print(f"  {lbl:<30} {val:>8.2f}")

    print("\n--- M2 mean betas (bps) ---")
    for lbl, val in zip(DECOMP_COLS["B8"], mean_m2 * 10_000):
        print(f"  {lbl:<30} {val:>8.2f}")

    print("\n=== Done ===")

if __name__ == "__main__":
    main()
