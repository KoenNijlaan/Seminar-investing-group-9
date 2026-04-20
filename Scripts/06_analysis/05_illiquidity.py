"""
Illiquidity Analysis

Purpose:
  Estimate how signal effects vary with illiquidity using interaction-style models.

Inputs:
  - Final panel with signal and illiquidity variables.

Outputs:
  - Illiquidity analysis tables and intermediate summaries.

Main Steps:
  - Build interaction terms.
  - Run weekly specifications.
  - Summarize and export outputs.
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
    "n_obs_daily": r"$\log(\mathrm{NTRANS})$",
}

IDIO_SIGNALS = [
    ("rsj_idio_weekly", r"RSJ$^{\mathrm{idio}}$ (M1)"),
    ("res_idio_p025",   r"RES$^{\mathrm{idio}}$"),
]

SPECS = []
for sig_col, sig_lbl in IDIO_SIGNALS:
    SPECS.append((sig_col, f"{sig_lbl} (Univ.)", ["illiq"]))
    SPECS.append((sig_col, f"{sig_lbl} (Ctrl.)", ["illiq"] + CONTROLS))

def winsorize_cs(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile(WINSOR_LOW), s.quantile(WINSOR_HIGH)
    return s.clip(lo, hi)

def nw_mean(arr: np.ndarray, lags: int) -> tuple[float, float]:
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

def run_m2_illiq_specs(panel: pd.DataFrame) -> dict:
    """
    Run the illiquidity mechanism regression with rsj_idio (M2) as the
    dependent variable, for univariate and controlled specifications.
    Returns {("rsj_idio", "univ"): {...}, ("rsj_idio", "ctrl"): {...}}.
    """
    sig_col = "rsj_idio"
    weeks   = sorted(panel["week"].unique())
    out     = {}

    for spec_type, regressors in [("univ", ["illiq"]),
                                   ("ctrl", ["illiq"] + CONTROLS)]:
        key = (sig_col, spec_type)
        if sig_col not in panel.columns:
            print(f"  SKIP {key}: {sig_col} not in panel")
            out[key] = None
            continue

        coef_series = {p: [] for p in regressors}
        rsq_series  = []
        n_valid     = 0

        for week in weeks:
            r = run_ols_week(panel[panel["week"] == week], sig_col, regressors)
            if r is None:
                continue
            n_valid += 1
            rsq_series.append(r["rsq_adj"])
            for p in regressors:
                coef_series[p].append(r[p])

        coefs = {p: nw_mean(np.array(coef_series[p], dtype=float), NW_LAGS)
                 for p in regressors}
        out[key] = {
            "coefs":   coefs,
            "rsq":     float(np.mean(rsq_series)) if rsq_series else np.nan,
            "n_weeks": n_valid,
        }
        illiq_m, illiq_t = coefs.get("illiq", (np.nan, np.nan))
        print(f"  M2 illiq [{spec_type}]  ILLIQ: {illiq_m:.4f} (t={illiq_t:.2f})"
              f"  N={n_valid}")

    return out


def build_latex_robustness(existing: dict, m2: dict) -> str:
    """
    Two-panel LaTeX table comparing M1 vs M2 for the illiquidity mechanism.

    Columns : RSJ_idio M1 | RSJ_idio M2 | RES_idio
    Panels  : A – Univariate  |  B – Controlled
    Reports : time-averaged ILLIQ coefficient + NW(6) t-statistic only.

    existing : dict returned by run_all_specs()  (indices 0-3)
    m2       : dict returned by run_m2_illiq_specs()
    """
    # Map existing results to named slots.
    # SPECS order: 0=rsj_m1_univ, 1=rsj_m1_ctrl, 2=res_univ, 3=res_ctrl
    named = {
        ("rsj_idio_weekly", "univ"): existing.get(0),
        ("rsj_idio_weekly", "ctrl"): existing.get(1),
        ("res_idio_p025",   "univ"): existing.get(2),
        ("res_idio_p025",   "ctrl"): existing.get(3),
    }
    named.update({k: v for k, v in m2.items() if v is not None})

    COLS = [
        ("rsj_idio_weekly", r"(M1)"),
        ("rsj_idio",        r"(M2)"),
        ("res_idio_p025",   r"---"),
    ]
    n_data = len(COLS)
    col_spec = "l" + "c" * n_data

    def get_illiq(dep_col, spec_type):
        entry = named.get((dep_col, spec_type))
        if entry is None:
            return "", "", "", ""
        m, t = entry["coefs"].get("illiq", (np.nan, np.nan))
        return (
            _fmt(m, t) if np.isfinite(m) else "",
            f"({t:.2f})" if np.isfinite(t) else "",
            str(entry["n_weeks"]),
            _fmt_pct(entry["rsq"]),
        )

    def panel_rows(spec_type):
        coef_cells, t_cells, n_cells, rsq_cells = [], [], [], []
        for dep_col, _ in COLS:
            c, t, n, rsq = get_illiq(dep_col, spec_type)
            coef_cells.append(c); t_cells.append(t)
            n_cells.append(n);    rsq_cells.append(rsq)
        return (
            [r"ILLIQ & " + " & ".join(coef_cells) + r" \\",
             r"  & "    + " & ".join(t_cells)     + r" \\"],
            n_cells, rsq_cells,
        )

    n_total = 1 + n_data
    lines   = []
    lines += [r"\begin{table}[htbp]", r"\centering", r"\begin{threeparttable}"]
    lines += [r"\caption{Illiquidity Mechanism: Robustness to Decomposition Method"
              r" (M1 vs.\ M2)}"]
    lines += [r"\label{tab:robustness_illiquidity}"]
    lines += [rf"\begin{{tabular}}{{{col_spec}}}"]
    lines += [r"\toprule"]
    lines += [
        r"  & \multicolumn{2}{c}{RSJ$^{\mathrm{idio}}$}"
        r" & \multicolumn{1}{c}{RES$^{\mathrm{idio}}$} \\",
        r"\cmidrule(lr){2-3}",
    ]
    lines += [
        "  & " + " & ".join(lbl for _, lbl in COLS) + r" \\"
    ]
    lines += [r"\midrule"]

    # Panel A: Univariate
    lines += [
        rf"\multicolumn{{{n_total}}}{{l}}"
        rf"{{\textit{{Panel A: Univariate specification"
        rf" ($C_{{i,w}} = \alpha + \delta\,\mathrm{{ILLIQ}}_{{i,w}} + \nu_{{i,w}}$)"
        rf"}}}}\\"
    ]
    univ_rows, univ_n, univ_rsq = panel_rows("univ")
    lines += univ_rows
    lines += [r"\midrule"]

    # Panel B: Controlled
    lines += [
        rf"\multicolumn{{{n_total}}}{{l}}"
        rf"{{\textit{{Panel B: Controlled specification"
        rf" ($C_{{i,w}} = \alpha + \delta\,\mathrm{{ILLIQ}}_{{i,w}}"
        rf" + \eta^\top X_{{i,w}} + \nu_{{i,w}}$)"
        rf"}}}}\\"
    ]
    ctrl_rows, ctrl_n, ctrl_rsq = panel_rows("ctrl")
    lines += ctrl_rows
    lines += [r"\midrule"]

    lines += [r"$N_{\text{weeks}}$ (A) & " + " & ".join(univ_n)   + r" \\"]
    lines += [r"$N_{\text{weeks}}$ (B) & " + " & ".join(ctrl_n)   + r" \\"]
    lines += [r"Avg.\ Adj.\ $R^2$ (A) & " + " & ".join(univ_rsq) + r" \\"]
    lines += [r"Avg.\ Adj.\ $R^2$ (B) & " + " & ".join(ctrl_rsq) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [
        r"\begin{tablenotes}", r"\small",
        r"\item Notes: Weekly cross-sectional regressions following"
        r" equation~(\ref{eq:mech}). The dependent variable $C_{i,w}$ is"
        r" the idiosyncratic RSJ (M1: intraday decomposition; M2: rolling"
        r" 52-week market-RSJ regression) or the idiosyncratic RES"
        r" (RES$^{\mathrm{idio}}$; no M2 equivalent)."
        r" The focal regressor is log-Amihud illiquidity (ILLIQ)."
        r" Panel~A is the univariate specification; Panel~B adds controls"
        r" $X_{i,w} = (\log\mathrm{ME}, \mathrm{BM}, \mathrm{MOM},"
        r" \mathrm{REV}, \mathrm{IVOL}, \mathrm{RVOL}, \mathrm{RSK},"
        r" \mathrm{RKT}, \log\mathrm{NTRANS})$; ILLIQ is not duplicated"
        r" inside $X_{i,w}$."
        r" All coefficients are time-series averages of weekly OLS slopes;"
        r" $t$-statistics in parentheses use Newey--West HAC with 6 lags."
        r" $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
        r"\end{tablenotes}",
        r"\end{threeparttable}", r"\end{table}",
    ]
    return "\n".join(lines)


def build_latex(results: dict) -> str:
    col_keys = sorted(results.keys())
    n_cols   = len(col_keys)
    col_spec = "l" + "c" * n_cols

    row_vars = ["illiq"] + CONTROLS

    def _cell(spec_idx, pred):
        spec = results.get(spec_idx, {})
        if pred not in spec.get("coefs", {}):
            return "", ""
        m, t = spec["coefs"][pred]
        return _fmt(m, t), (f"({t:.2f})" if np.isfinite(t) else "")

    def _col_header(lbl):

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

    rsj_cols = [k for k in col_keys if "RSJ" in results[k]["label"]]
    res_cols = [k for k in col_keys if "RES" in results[k]["label"]]

    def col_pos(k):
        return col_keys.index(k) + 2

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

    sub_hdrs = []
    for k in col_keys:
        lbl = results[k]["label"]
        sub_hdrs.append("(Univ.)" if "Univ." in lbl else "(Ctrl.)")
    lines += ["  & " + " & ".join(sub_hdrs) + r" \\"]
    lines += [r"\midrule"]

    for pred in row_vars:

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
        r" \mathrm{RVOL}, \mathrm{RSK}, \mathrm{RKT}, \log\mathrm{NTRANS})$"
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

    panel["n_obs_daily"] = np.nan
    valid_ntrans = panel["n_days"].gt(0) & panel["n_obs_total"].gt(0)
    panel.loc[valid_ntrans, "n_obs_daily"] = np.log(
        panel.loc[valid_ntrans, "n_obs_total"] / panel.loc[valid_ntrans, "n_days"]
    )

    for c in ["illiq"] + CONTROLS:
        if c in panel.columns:
            panel[c] = panel.groupby("week")[c].transform(winsorize_cs)

    results = run_all_specs(panel)

    tex = build_latex(results)
    out = TABLE_DIR / "tab_illiquidity.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"\nSaved: {out.name}")

    print("\n--- M2 illiquidity mechanism specs ---")
    m2_results = run_m2_illiq_specs(panel)

    tex_rob = build_latex_robustness(results, m2_results)
    out_rob = TABLE_DIR / "tab_illiquidity_robustness.tex"
    out_rob.write_text(tex_rob, encoding="utf-8")
    print(f"Saved: {out_rob.name}")

    print("\n=== Done ===")

if __name__ == "__main__":
    main()
