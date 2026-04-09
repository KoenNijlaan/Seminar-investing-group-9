"""
Illiquidity mechanism check following eq. (mech) in the paper.

Model:
    C_{i,w} = alpha_w + delta_w * ILLIQ_{i,w} + eta_w' * X_{i,w} + nu_{i,w}

where C_{i,w} in {RSJ^idio (M1), RES^idio} and
    X_{i,w} = (ME, BM, MOM, REV, IVOL)    [ILLIQ is the focal regressor, not a control]

Specifications per dependent variable:
  (1) Univariate:  C ~ ILLIQ
  (2) Controlled:  C ~ ILLIQ + X

Table layout (matches main FM tables):
  Rows = regressors (ILLIQ focal, then controls)
  Columns = (1) RSJ_idio Univ. | (2) RSJ_idio Ctrl. | (3) RES_idio Univ. | (4) RES_idio Ctrl.
  t-statistics in parentheses below each coefficient
  Footer: N weeks, Avg adj R²

Output:
    data_final/results/tables/tab_illiquidity.tex
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PANEL_FILE = ROOT / "data_final" / "panel" / "weekly_panel.parquet"
TABLE_DIR = ROOT / "data_final" / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

NW_LAGS     = 6
WINSOR_LOW  = 0.01
WINSOR_HIGH = 0.99
MIN_STOCKS  = 30

# Controls as defined in eq. (controls_mech): X = (ME, BM, MOM, REV, IVOL)
# ILLIQ is excluded because it is the focal regressor
CONTROLS = ["me", "bm", "mom", "rev", "ivol", "rvol", "rsk", "rkt", "n_obs_daily"]

PRED_LABEL = {
    "illiq":       r"ILLIQ",
    "me":          r"$\log(\mathrm{ME})$",
    "bm":          "BM",
    "mom":         "MOM",
    "rev":         "REV",
    "ivol":        "IVOL",
    "rvol":        "RVOL",
    "rsk":         "RSK",
    "rkt":         "RKT",
    "n_obs_daily": r"NTRANS",
}

# Only M1 idiosyncratic decomposition
IDIO_SIGNALS = [
    ("rsj_idio_weekly", r"RSJ$^{\mathrm{idio}}$ (M1)"),
    ("res_idio_p025",   r"RES$^{\mathrm{idio}}$"),
]

# Column definitions: (signal_col, spec_label, regressors)
SPECS = []
for sig_col, sig_lbl in IDIO_SIGNALS:
    SPECS.append((sig_col, f"{sig_lbl} (Univ.)", ["illiq"]))
    SPECS.append((sig_col, f"{sig_lbl} (Ctrl.)", ["illiq"] + CONTROLS))


def winsorize_cs(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile(WINSOR_LOW), s.quantile(WINSOR_HIGH)
    return s.clip(lo, hi)


def nw_mean(arr: np.ndarray, lags: int) -> tuple[float, float]:
    """NW HAC mean. Returns (mean, t-stat)."""
    y = np.asarray(arr, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return np.nan, np.nan
    res = sm.OLS(y, np.ones((len(y), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    return float(res.params[0]), float(res.tvalues[0])


def run_ols_week(df_week: pd.DataFrame, dep: str, indep: list[str],
                 min_n: int = MIN_STOCKS) -> dict | None:
    """Single-week OLS. Returns {pred: coef, ..., rsq_adj} or None."""
    sub = df_week[[dep] + indep].dropna()
    if len(sub) < min_n:
        return None
    y = pd.to_numeric(sub[dep], errors="coerce").to_numpy(dtype=float)
    X = sm.add_constant(
        sub[indep].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
        has_constant="add",
    )
    if not (np.isfinite(y).all() and np.isfinite(X).all()):
        return None
    try:
        res = sm.OLS(y, X).fit()
    except Exception:
        return None
    out = {"rsq_adj": float(res.rsquared_adj)}
    for i, p in enumerate(indep):
        out[p] = float(res.params[i + 1])
    return out


def _stars(t: float) -> str:
    if not np.isfinite(t):
        return ""
    a = abs(float(t))
    if a > 2.576: return "***"
    if a > 1.96:  return "**"
    if a > 1.645: return "*"
    return ""


def _fmt(val: float, t: float = None, dec: int = 4) -> str:
    if val is None or not np.isfinite(float(val)):
        return ""
    s = f"{float(val):.{dec}f}"
    if t is not None:
        s += _stars(t)
    return s


def _fmt_pct(val: float) -> str:
    if val is None or not np.isfinite(float(val)):
        return ""
    return f"{float(val) * 100:.2f}\\%"


def run_all_specs(panel: pd.DataFrame) -> dict:
    """
    For each spec (signal x univariate/controlled), collect weekly coefficient
    series and compute NW means + avg adj R².

    Returns dict keyed by spec index:
      {i: {"label": str, "regressors": [...],
           "coefs": {pred: (mean, t)}, "rsq": float, "n_weeks": int}}
    """
    weeks = sorted(panel["week"].unique())
    results = {}

    for i, (sig_col, spec_lbl, regressors) in enumerate(SPECS):
        if sig_col not in panel.columns:
            print(f"  SKIP {sig_col}: not in panel")
            continue

        coef_series = {p: [] for p in regressors}
        rsq_series  = []
        n_valid     = 0

        for week in weeks:
            wdf = panel[panel["week"] == week]
            r = run_ols_week(wdf, sig_col, regressors)
            if r is None:
                continue
            n_valid += 1
            rsq_series.append(r["rsq_adj"])
            for p in regressors:
                coef_series[p].append(r[p])

        coefs = {}
        for p in regressors:
            arr = np.array(coef_series[p], dtype=float)
            coefs[p] = nw_mean(arr, NW_LAGS)

        results[i] = {
            "label":      spec_lbl,
            "regressors": regressors,
            "coefs":      coefs,
            "rsq":        float(np.mean(rsq_series)) if rsq_series else np.nan,
            "n_weeks":    n_valid,
        }

        illiq_m, illiq_t = coefs.get("illiq", (np.nan, np.nan))
        print(f"  [{spec_lbl}]  ILLIQ: {illiq_m:.4f} (t={illiq_t:.2f})  "
              f"adj R²={results[i]['rsq']*100:.2f}%  N={n_valid}")

    return results


def build_latex(results: dict) -> str:
    """
    FM-style table:
      Rows  = regressors (ILLIQ, then controls)
      Cols  = one per spec
      Footer = N weeks, Avg adj R²
    """
    col_keys = sorted(results.keys())
    n_cols   = len(col_keys)
    col_spec = "l" + "c" * n_cols

    # All regressors that appear anywhere, in display order
    row_vars = ["illiq"] + CONTROLS

    def _cell(spec_idx, pred):
        spec = results.get(spec_idx, {})
        if pred not in spec.get("coefs", {}):
            return "", ""
        m, t = spec["coefs"][pred]
        return _fmt(m, t), (f"({t:.2f})" if np.isfinite(t) else "")

    # Column headers: split label at " (" for two-line header
    def _col_header(lbl):
        # e.g. "RSJ^idio (M1) (Univ.)" -> split before last " ("
        if " (Univ.)" in lbl:
            base = lbl.replace(" (Univ.)", "")
            return f"{base} \\\\ & (Univ.)"
        if " (Ctrl.)" in lbl:
            base = lbl.replace(" (Ctrl.)", "")
            return f"{base} \\\\ & (Ctrl.)"
        return lbl

    lines = []
    lines += [r"\begin{table}[htbp]", r"\centering", r"\begin{threeparttable}"]
    lines += [r"\caption{Illiquidity and Idiosyncratic Downside-Risk Characteristics}"]
    lines += [r"\label{tab:illiquidity}"]
    lines += [rf"\begin{{tabular}}{{{col_spec}}}"]
    lines += [r"\toprule"]

    # Header: group RSJ_idio and RES_idio with cmidrules
    rsj_cols = [k for k in col_keys if "RSJ" in results[k]["label"]]
    res_cols = [k for k in col_keys if "RES" in results[k]["label"]]

    def col_pos(k):
        return col_keys.index(k) + 2  # +2 because first col is row label

    if rsj_cols and res_cols:
        rsj_start, rsj_end = col_pos(rsj_cols[0]),  col_pos(rsj_cols[-1])
        res_start, res_end = col_pos(res_cols[0]),  col_pos(res_cols[-1])
        lines += [
            f"  & \\multicolumn{{{len(rsj_cols)}}}{{c}}"
            f"{{RSJ$^{{\\mathrm{{idio}}}}$ (M1)}}"
            f" & \\multicolumn{{{len(res_cols)}}}{{c}}"
            f"{{RES$^{{\\mathrm{{idio}}}}$}} \\\\",
            f"\\cmidrule(lr){{{rsj_start}-{rsj_end}}}"
            f"\\cmidrule(lr){{{res_start}-{res_end}}}",
        ]

    # Sub-headers: Univ. / Ctrl.
    sub_hdrs = []
    for k in col_keys:
        lbl = results[k]["label"]
        sub_hdrs.append("(Univ.)" if "Univ." in lbl else "(Ctrl.)")
    lines += ["  & " + " & ".join(sub_hdrs) + r" \\"]
    lines += [r"\midrule"]

    # Coefficient rows
    for pred in row_vars:
        # Check if this predictor appears in any spec
        if not any(pred in results[k]["regressors"] for k in col_keys):
            continue
        coef_cells = []
        t_cells    = []
        for k in col_keys:
            c, t = _cell(k, pred)
            coef_cells.append(c)
            t_cells.append(t)
        lbl = PRED_LABEL.get(pred, pred)
        lines.append(lbl + " & " + " & ".join(coef_cells) + r" \\")
        lines.append("  & " + " & ".join(t_cells) + r" \\")

    lines += [r"\midrule"]
    lines += [r"$N_{\text{weeks}}$" + " & " +
              " & ".join(str(results[k]["n_weeks"]) for k in col_keys) + r" \\"]
    lines += [r"Avg.\ Adj.\ $R^2$" + " & " +
              " & ".join(_fmt_pct(results[k]["rsq"]) for k in col_keys) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}"]

    lines += [
        r"\begin{tablenotes}", r"\small",
        r"\item Notes: Weekly cross-sectional regressions following"
        r" equation~(\ref{eq:mech}). The dependent variable $C_{i,w}$ is"
        r" either the M1 idiosyncratic realised skewness-jump measure"
        r" (RSJ$^{\mathrm{idio}}$, intraday decomposition) or the idiosyncratic"
        r" realised entropy measure (RES$^{\mathrm{idio}}$)."
        r" The focal regressor is log-Amihud illiquidity (ILLIQ)."
        r" The controlled specification adds $X_{i,w} = (\log\mathrm{ME},"
        r" \mathrm{BM}, \mathrm{MOM}, \mathrm{REV}, \mathrm{IVOL},"
        r" \mathrm{RVOL}, \mathrm{RSK}, \mathrm{RKT}, \mathrm{NTRANS})$"
        r" following equation~(\ref{eq:controls_mech}); ILLIQ is not"
        r" duplicated inside $X_{i,w}$."
        r" All coefficients are time-series averages of weekly OLS slopes;"
        r" $t$-statistics in parentheses use Newey--West HAC with 6 lags."
        r" Avg.\ Adj.\ $R^2$ is the time-series average of weekly adjusted $R^2$."
        r" All regressors winsorized cross-sectionally at the 1st/99th"
        r" percentile each week."
        r" $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
        r"\end{tablenotes}",
        r"\end{threeparttable}", r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    print("=== Illiquidity Mechanism Analysis ===\n")

    panel = pd.read_parquet(PANEL_FILE)
    panel["week"] = pd.to_datetime(panel["week"]).dt.normalize()

    # Average number of 5-minute observations per trading day
    panel["n_obs_daily"] = panel["n_obs_total"] / panel["n_days"]

    for c in ["illiq"] + CONTROLS:
        if c in panel.columns:
            panel[c] = panel.groupby("week")[c].transform(winsorize_cs)

    results = run_all_specs(panel)

    tex = build_latex(results)
    out = TABLE_DIR / "tab_illiquidity.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"\nSaved: {out.name}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
