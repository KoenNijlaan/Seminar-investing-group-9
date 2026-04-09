"""
Subsample analysis: Pre-2008 vs. Post-2008 (from 2008-01-01 onwards).

Runs the same quintile portfolio sorts and Fama-MacBeth regressions
as 01_portfolio_sorts.py and 02_fama_macbeth.py, but separately on each
subsample. Produces comparison LaTeX tables.

Outputs:
  data_final/results/tables/tab_subsample_portfolio_spreads.tex
  data_final/results/tables/tab_subsample_fm_aggregate.tex
  data_final/results/tables/tab_subsample_fm_decomposition.tex
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ============================================================
# Settings
# ============================================================
ROOT       = Path(__file__).resolve().parents[2]
PANEL_FILE = ROOT / "data_final" / "panel" / "weekly_panel.parquet"
FF_FILE    = ROOT / "data_raw" / "wrds" / "ff_daily_factors_1992_2024.parquet"

TABLE_DIR  = ROOT / "data_final" / "results" / "tables"
INTER_DIR  = ROOT / "data_final" / "results" / "intermediate"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
INTER_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DATE = pd.Timestamp("2008-01-01")   # Post-2008: week >= this date

NW_LAGS    = 6
MIN_STOCKS = 50
WINSOR_LOW  = 0.01
WINSOR_HIGH = 0.99
CONTROLS    = ["me", "bm", "mom", "rev", "ivol", "illiq", "rvol", "rsk", "rkt"]

SORT_VARIABLES = {
    "rsj_weekly"     : "RSJ",
    "rsj_sys_weekly" : r"RSJ$_{\mathrm{sys}}$ (M1)",
    "rsj_idio_weekly": r"RSJ$_{\mathrm{idio}}$ (M1)",
    "rsj_sys"        : r"RSJ$_{\mathrm{sys}}$ (M2)",
    "rsj_idio"       : r"RSJ$_{\mathrm{idio}}$ (M2)",
    "res_weekly"     : "RES",
    "res_sys_p025"   : r"RES$_{\mathrm{sys}}$",
    "res_idio_p025"  : r"RES$_{\mathrm{idio}}$",
}

SPREAD_DIR = {
    "rsj_weekly"     : "L-H",
    "rsj_sys_weekly" : "L-H",
    "rsj_idio_weekly": "L-H",
    "rsj_sys"        : "L-H",
    "rsj_idio"       : "L-H",
    "res_weekly"     : "H-L",
    "res_sys_p025"   : "H-L",
    "res_idio_p025"  : "H-L",
}

PANEL_A = ["rsj_weekly", "rsj_sys_weekly", "rsj_idio_weekly", "rsj_sys", "rsj_idio"]
PANEL_B = ["res_weekly", "res_sys_p025", "res_idio_p025"]

SPECS = {
    "B1":  ["rsj_weekly"] + CONTROLS,
    "B2":  ["res_weekly"] + CONTROLS,
    "B3":  ["rsj_weekly", "res_weekly"] + CONTROLS,
    "B4":  ["rsj_sys_weekly", "rsj_idio_weekly"] + CONTROLS,
    "B5":  ["rsj_sys", "rsj_idio"] + CONTROLS,
    "B6":  ["res_sys_p025", "res_idio_p025"] + CONTROLS,
    "B7":  ["rsj_sys_weekly", "rsj_idio_weekly", "res_sys_p025", "res_idio_p025"] + CONTROLS,
    "B8":  ["rsj_sys", "rsj_idio", "res_sys_p025", "res_idio_p025"] + CONTROLS,
    "B9":  ["rsj_idio_weekly", "res_idio_p025"] + CONTROLS,
    "B10": ["rsj_sys_weekly", "res_idio_p025"] + CONTROLS,
}

PRED_LABEL = {
    "rsj_weekly":      "RSJ",
    "res_weekly":      "RES",
    "rsj_sys_weekly":  r"RSJ$_{\mathrm{sys}}$ (M1)",
    "rsj_idio_weekly": r"RSJ$_{\mathrm{idio}}$ (M1)",
    "rsj_sys":         r"RSJ$_{\mathrm{sys}}$ (M2)",
    "rsj_idio":        r"RSJ$_{\mathrm{idio}}$ (M2)",
    "res_sys_p025":    r"RES$_{\mathrm{sys}}$",
    "res_idio_p025":   r"RES$_{\mathrm{idio}}$",
    "me":    r"$\log(\mathrm{ME})$",
    "bm":    "BM",
    "mom":   "MOM",
    "rev":   "REV",
    "ivol":  "IVOL",
    "illiq": "ILLIQ",
    "rvol":  "RVOL",
    "rsk":   "RSK",
    "rkt":   "RKT",
}

CTRL_LABEL = {k: PRED_LABEL[k] for k in CONTROLS}


# ============================================================
# Shared helpers
# ============================================================
def _stars(t):
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    t = abs(float(t))
    if t > 2.576: return "***"
    if t > 1.96:  return "**"
    if t > 1.645: return "*"
    return ""


def _fmt(val, t=None, dec=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = f"{float(val):.{dec}f}"
    if t is not None:
        s += _stars(t)
    return s


def _fmt_pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return f"{float(val) * 100:.2f}\\%"


def nw_mean(series, lags):
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return np.nan, np.nan, np.nan
    res = sm.OLS(y, np.ones((len(y), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    return float(res.params[0]), float(res.tvalues[0]), float(res.bse[0])


def nw_alpha(excess_ret, factors_df, lags):
    df = pd.DataFrame({"y": excess_ret}).join(factors_df).dropna()
    if len(df) < 20:
        return np.nan, np.nan
    y = df["y"].to_numpy(dtype=float)
    X = sm.add_constant(df.drop(columns=["y"]).to_numpy(dtype=float))
    res = sm.OLS(y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    return float(res.params[0]), float(res.tvalues[0])


def assign_quintiles_nyse(sort_var: pd.Series, exchcd: pd.Series) -> pd.Series:
    ref = sort_var[exchcd == 1].dropna() if exchcd is not None else sort_var.dropna()
    if len(ref) < 10:
        ref = sort_var.dropna()
    if len(ref) == 0:
        return pd.Series(np.nan, index=sort_var.index)
    q20, q40, q60, q80 = np.nanpercentile(ref, [20, 40, 60, 80])
    out = pd.Series(np.nan, index=sort_var.index)
    v = sort_var.notna()
    out[v & (sort_var <= q20)]                     = 1
    out[v & (sort_var > q20) & (sort_var <= q40)]  = 2
    out[v & (sort_var > q40) & (sort_var <= q60)]  = 3
    out[v & (sort_var > q60) & (sort_var <= q80)]  = 4
    out[v & (sort_var > q80)]                      = 5
    return out


def winsorize_cs(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile(WINSOR_LOW), s.quantile(WINSOR_HIGH)
    return s.clip(lo, hi)


def load_ff3_weekly(ff_file):
    ff = pd.read_parquet(ff_file)
    ff.columns = [c.lower().strip() for c in ff.columns]
    ff["date"] = pd.to_datetime(ff["date"]).dt.normalize()
    for c in ["mktrf", "smb", "hml", "rf"]:
        ff[c] = pd.to_numeric(ff[c], errors="coerce")
    ff["week"] = ff["date"].dt.to_period("W-TUE").dt.end_time.dt.normalize()
    def compound(x):
        return np.prod(1 + x.dropna()) - 1
    ff_w = ff.groupby("week", sort=True).agg(
        mktrf=("mktrf", compound),
        smb  =("smb",   compound),
        hml  =("hml",   compound),
        rf   =("rf",    compound),
    ).reset_index()
    return ff_w


# ============================================================
# Portfolio sort for one subsample
# ============================================================
def run_sort_subsample(panel, sort_col, ff_idx):
    """Returns (mean_ew_bps, t_ew, alpha_ew_bps, t_alpha_ew,
                mean_vw_bps, t_vw, alpha_vw_bps, t_alpha_vw, n_weeks)."""
    eligible = panel[
        (panel["valid_R_i_w_plus_1"] == True) &
        panel[sort_col].notna() &
        panel["R_i_w_plus_1"].notna()
    ].copy()

    wc = eligible.groupby("week").size()
    eligible = eligible[eligible["week"].isin(wc[wc >= MIN_STOCKS].index)].copy()
    if len(eligible) == 0:
        return (np.nan,) * 9

    eligible["quintile"] = eligible.groupby("week", group_keys=False).apply(
        lambda g: assign_quintiles_nyse(
            g[sort_col],
            g["exchcd"] if "exchcd" in g.columns else None
        ),
        include_groups=False
    ).reindex(eligible.index)
    eligible = eligible.dropna(subset=["quintile"])
    eligible["quintile"] = eligible["quintile"].astype(int)
    eligible["vw_weight"] = eligible["me_raw"].where(eligible["me_raw"] > 0, np.nan)

    qt_ew, qt_vw = {}, {}
    for q in range(1, 6):
        g = eligible[eligible["quintile"] == q]
        qt_ew[q] = g.groupby("week")["R_i_w_plus_1"].mean()
        def vw_ret(grp):
            w = grp["vw_weight"]; r = grp["R_i_w_plus_1"]
            ok = w.notna() & r.notna()
            return (r[ok] * w[ok]).sum() / w[ok].sum() if ok.sum() > 0 else np.nan
        qt_vw[q] = g.groupby("week").apply(vw_ret, include_groups=False)

    direction = SPREAD_DIR[sort_col]
    if direction == "H-L":
        sp_ew = (qt_ew[5] - qt_ew[1]).dropna()
        sp_vw = (qt_vw[5] - qt_vw[1]).dropna()
    else:
        sp_ew = (qt_ew[1] - qt_ew[5]).dropna()
        sp_vw = (qt_vw[1] - qt_vw[5]).dropna()

    common = sp_ew.index.intersection(ff_idx.index)
    if len(common) < 20:
        return (np.nan,) * 9

    sp_ew_c = sp_ew.reindex(common)
    sp_vw_c = sp_vw.reindex(common)
    ff_c    = ff_idx.loc[common]

    factors = pd.DataFrame({
        "mktrf": ff_c["mktrf"].values,
        "smb":   ff_c["smb"].values,
        "hml":   ff_c["hml"].values,
    })

    mean_ew, t_ew, _ = nw_mean(sp_ew_c.values, NW_LAGS)
    mean_vw, t_vw, _ = nw_mean(sp_vw_c.values, NW_LAGS)

    excess_ew = sp_ew_c.values - ff_c["rf"].values
    excess_vw = sp_vw_c.values - ff_c["rf"].values
    alpha_ew, t_alpha_ew = nw_alpha(excess_ew, factors, NW_LAGS)
    alpha_vw, t_alpha_vw = nw_alpha(excess_vw, factors, NW_LAGS)

    return (
        mean_ew * 10_000, t_ew,
        alpha_ew * 10_000, t_alpha_ew,
        mean_vw * 10_000, t_vw,
        alpha_vw * 10_000, t_alpha_vw,
        len(common),
    )


def run_all_sorts(panel, ff_idx, label=""):
    results = {}
    for sort_col in SORT_VARIABLES:
        if sort_col not in panel.columns:
            continue
        r = run_sort_subsample(panel, sort_col, ff_idx)
        results[sort_col] = r
        m, t = r[0], r[1]
        print(f"    {SORT_VARIABLES[sort_col]:<35}  {m:>8.2f}{_stars(t):3}  (t={t:.2f})")
    return results


# ============================================================
# Fama-MacBeth for one subsample
# ============================================================
def run_weekly_cs(df_week, predictors):
    sub = df_week[["R_i_w_plus_1"] + predictors].dropna()
    if len(sub) < MIN_STOCKS:
        return None
    y     = pd.to_numeric(sub["R_i_w_plus_1"], errors="coerce").to_numpy(dtype=float)
    X_raw = sub[predictors].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X     = sm.add_constant(X_raw, has_constant="add")
    if not (np.isfinite(y).all() and np.isfinite(X).all()):
        return None
    try:
        res = sm.OLS(y, X).fit()
    except Exception:
        return None
    out = {"n": len(sub), "rsq_adj": float(res.rsquared_adj)}
    for i, p in enumerate(predictors):
        out[p] = float(res.params[i + 1])
    return out


def nw_average(arr, lags):
    y = arr[np.isfinite(arr)]
    if len(y) < 10:
        return np.nan, np.nan, np.nan
    res = sm.OLS(y, np.ones((len(y), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    return float(res.params[0]) * 10_000, float(res.tvalues[0]), float(res.bse[0]) * 10_000


def run_fm_subsample(panel, specs):
    summary  = {}
    avg_rsq  = {}
    n_weeks  = {}
    weeks    = sorted(panel["week"].unique())

    for spec, predictors in specs.items():
        missing = [p for p in predictors if p not in panel.columns]
        if missing:
            print(f"    SKIP {spec}: missing {missing}")
            continue

        betas    = {p: [] for p in predictors}
        rsq_list = []
        valid    = 0

        for week in weeks:
            df_w   = panel[panel["week"] == week]
            result = run_weekly_cs(df_w, predictors)
            if result is None:
                continue
            rsq_list.append(result["rsq_adj"])
            valid += 1
            for p in predictors:
                betas[p].append(result[p])

        if valid == 0:
            continue

        avg_rsq[spec] = float(np.mean(rsq_list))
        n_weeks[spec] = valid

        spec_summ = {}
        for p in predictors:
            arr = np.array(betas[p], dtype=float)
            m_bps, t, se_bps = nw_average(arr, NW_LAGS)
            spec_summ[p] = (m_bps, t, se_bps)
        summary[spec] = spec_summ

        focal = [p for p in predictors if p not in CONTROLS]
        parts = [f"{p}={summary[spec][p][0]:.1f}(t={summary[spec][p][1]:.2f})"
                 for p in focal if p in summary[spec]]
        print(f"    {spec} ({valid}w): {', '.join(parts)}")

    return summary, avg_rsq, n_weeks


# ============================================================
# LaTeX: subsample portfolio spreads
# ============================================================
def build_tex_sort_subsample(pre_res, post_res):
    """
    Two-panel table (Panel A: RSJ, Panel B: RES).
    Columns: Signal | Pre-2008 EW | VW | Post-2008 EW | VW  (bps, t-stats below)
    """
    def signal_rows(sort_col):
        pre  = pre_res.get(sort_col, (np.nan,) * 9)
        post = post_res.get(sort_col, (np.nan,) * 9)
        lbl  = SORT_VARIABLES[sort_col].replace("_", r"\_")
        spr  = SPREAD_DIR[sort_col]

        def pair(res, w_idx, t_idx):
            return _fmt(res[w_idx], res[t_idx]), \
                   f"({res[t_idx]:.2f})" if np.isfinite(res[t_idx]) else ""

        # EW: index 0/1, VW: index 4/5
        pre_ew_m,  pre_ew_t  = pair(pre,  0, 1)
        pre_vw_m,  pre_vw_t  = pair(pre,  4, 5)
        post_ew_m, post_ew_t = pair(post, 0, 1)
        post_vw_m, post_vw_t = pair(post, 4, 5)
        pre_nw  = int(pre[8])  if np.isfinite(pre[8])  else "--"
        post_nw = int(post[8]) if np.isfinite(post[8]) else "--"

        coef = (f"{lbl} ({spr}) & "
                f"{pre_ew_m} & {pre_vw_m} & {post_ew_m} & {post_vw_m} & "
                f"{pre_nw} & {post_nw} \\\\")
        tstat = (f" & "
                 f"({pre_ew_t}) & ({pre_vw_t}) & ({post_ew_t}) & ({post_vw_t}) & & \\\\")
        return coef + "\n" + tstat + "\n"

    panel_a = "".join(signal_rows(s) for s in PANEL_A if s in pre_res or s in post_res)
    panel_b = "".join(signal_rows(s) for s in PANEL_B if s in pre_res or s in post_res)

    tex = r"""\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{Quintile Portfolio Spreads: Pre-2008 vs.\ Post-2008 Subsamples}
\label{tab:subsample_portfolio_spreads}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{Pre-2008} & \multicolumn{2}{c}{Post-2008} & \multicolumn{2}{c}{$N$ weeks} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Signal (spread) & EW & VW & EW & VW & Pre & Post \\
 & (bps) & (bps) & (bps) & (bps) & & \\
\midrule
\multicolumn{7}{l}{\textit{Panel A: RSJ measures}} \\
""" + panel_a + r"""\midrule
\multicolumn{7}{l}{\textit{Panel B: RES measures}} \\
""" + panel_b + r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Each week stocks are sorted into quintiles based on the signal using
NYSE breakpoints. The spread return is Q1$-$Q5 for RSJ signals (L$-$H) and
Q5$-$Q1 for RES signals (H$-$L). EW and VW denote equal- and value-weighted
portfolios. Returns are in basis points per week.
The sample is split at January 1, 2008.
$t$-statistics in parentheses use Newey--West standard errors (NW(6)).
$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    return tex


# ============================================================
# LaTeX: subsample FM tables
# ============================================================
def _fm_tex_block(pre_summ, post_summ, pre_rsq, post_rsq,
                  pre_nw, post_nw, specs, focal_vars, title, label):
    """Generic subsample FM table builder."""
    spec_list = [s for s in specs if s in pre_summ or s in post_summ]
    if not spec_list:
        return ""

    # Column header: Pre | Post for each spec
    pre_labels  = [f"\\multicolumn{{1}}{{c}}{{Pre}}" for s in spec_list]
    post_labels = [f"\\multicolumn{{1}}{{c}}{{Post}}" for s in spec_list]
    spec_labels = [f"\\multicolumn{{2}}{{c}}{{{s}}}" for s in spec_list]

    n_data_cols = 2 * len(spec_list)
    col_spec = "l" + "cc" * len(spec_list)

    cmidrule = " ".join(
        rf"\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}"
        for i in range(len(spec_list))
    )

    def cell(summ, spec, pred):
        if spec not in summ or pred not in summ[spec]:
            return "", ""
        m, t, _ = summ[spec][pred]
        return _fmt(m, t), f"({t:.2f})" if np.isfinite(t) else ""

    rows_vars = focal_vars + CONTROLS

    lines = []
    lines += [r"\begin{table}[htbp]", r"\centering", r"\begin{threeparttable}"]
    lines += [rf"\caption{{{title}}}"]
    lines += [rf"\label{{{label}}}"]
    lines += [rf"\begin{{tabular}}{{{col_spec}}}"]
    lines += [r"\toprule"]
    lines += ["  & " + " & ".join(spec_labels) + r" \\"]
    lines += [f"  {cmidrule}"]
    lines += ["  & " + " & ".join(
        f"Pre-2008 & Post-2008" for _ in spec_list
    ) + r" \\"]
    lines += [r"\midrule"]

    for v in rows_vars:
        if all(v not in pre_summ.get(s, {}) and v not in post_summ.get(s, {})
               for s in spec_list):
            continue
        lbl = PRED_LABEL.get(v, v)
        coef_cells = []
        t_cells    = []
        for s in spec_list:
            pre_c,  pre_t  = cell(pre_summ,  s, v)
            post_c, post_t = cell(post_summ, s, v)
            coef_cells += [pre_c, post_c]
            t_cells    += [pre_t, post_t]
        lines.append(lbl + " & " + " & ".join(coef_cells) + r" \\")
        lines.append("  & " + " & ".join(t_cells) + r" \\")

    lines += [r"\midrule"]
    nw_row = r"$N_{\text{weeks}}$" + " & " + " & ".join(
        f"{pre_nw.get(s, '--')} & {post_nw.get(s, '--')}" for s in spec_list
    ) + r" \\"
    lines.append(nw_row)

    rsq_row = r"Avg.\ Adj.\ $R^2$" + " & " + " & ".join(
        f"{_fmt_pct(pre_rsq.get(s, np.nan))} & {_fmt_pct(post_rsq.get(s, np.nan))}"
        for s in spec_list
    ) + r" \\"
    lines.append(rsq_row)

    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [
        r"\begin{tablenotes}", r"\small",
        r"\item Notes: Fama--MacBeth cross-sectional regressions. Dependent variable"
        r" is $R_{i,w+1}$ (next-week return). All slope coefficients in bps/week"
        r" ($\times 10^4$). Controls: $\log(\mathrm{ME})$, BM, MOM, REV, IVOL, ILLIQ"
        r" (winsorized 1/99\% per week). The sample is split at January 1, 2008."
        r" $t$-statistics (in parentheses) use Newey--West HAC with 6 lags."
        r" $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
        r"\end{tablenotes}",
        r"\end{threeparttable}", r"\end{table}",
    ]
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
def main():
    print("=== Subsample Analysis: Pre-2008 vs. Post-2008 ===\n")

    panel = pd.read_parquet(PANEL_FILE)
    panel["week"] = pd.to_datetime(panel["week"]).dt.normalize()
    if "valid_R_i_w_plus_1" in panel.columns:
        panel["valid_R_i_w_plus_1"] = panel["valid_R_i_w_plus_1"].astype(bool)
    else:
        panel["valid_R_i_w_plus_1"] = panel["R_i_w_plus_1"].notna()

    ff_w   = load_ff3_weekly(FF_FILE)
    ff_idx = ff_w.set_index("week")

    pre_panel  = panel[panel["week"] <  SPLIT_DATE].copy()
    post_panel = panel[panel["week"] >= SPLIT_DATE].copy()

    print(f"Full panel : {panel['week'].nunique():>5} weeks")
    print(f"Pre-2008   : {pre_panel['week'].nunique():>5} weeks  "
          f"({pre_panel['week'].min().date()} – {pre_panel['week'].max().date()})")
    print(f"Post-2008  : {post_panel['week'].nunique():>5} weeks  "
          f"({post_panel['week'].min().date()} – {post_panel['week'].max().date()})")

    # ----------------------------------------------------------
    # 1) Portfolio sorts
    # ----------------------------------------------------------
    print("\n--- Portfolio Sorts (Pre-2008) ---")
    pre_sort = run_all_sorts(pre_panel, ff_idx)

    print("\n--- Portfolio Sorts (Post-2008) ---")
    post_sort = run_all_sorts(post_panel, ff_idx)

    tex_sort = build_tex_sort_subsample(pre_sort, post_sort)
    p = TABLE_DIR / "tab_subsample_portfolio_spreads.tex"
    p.write_text(tex_sort, encoding="utf-8")
    print(f"\nSaved: {p.name}")

    # ----------------------------------------------------------
    # 2) Fama-MacBeth
    # ----------------------------------------------------------
    # Winsorize controls within each subsample
    for sub in (pre_panel, post_panel):
        for c in CONTROLS:
            if c in sub.columns:
                sub[c] = sub.groupby("week")[c].transform(winsorize_cs)

    # Filter to stock-weeks with valid forward return
    pre_fm  = pre_panel[pre_panel["valid_R_i_w_plus_1"]].copy()
    post_fm = post_panel[post_panel["valid_R_i_w_plus_1"]].copy()

    print("\n--- Fama-MacBeth (Pre-2008) ---")
    pre_summ, pre_rsq, pre_nw = run_fm_subsample(pre_fm, SPECS)

    print("\n--- Fama-MacBeth (Post-2008) ---")
    post_summ, post_rsq, post_nw = run_fm_subsample(post_fm, SPECS)

    # Aggregate table: B1–B3
    tex_agg = _fm_tex_block(
        pre_summ, post_summ, pre_rsq, post_rsq, pre_nw, post_nw,
        specs      = ["B1", "B2", "B3"],
        focal_vars = ["rsj_weekly", "res_weekly"],
        title      = r"Fama--MacBeth Regressions: Aggregate Measures --- Pre-2008 vs.\ Post-2008",
        label      = "tab:subsample_fm_aggregate",
    )
    p = TABLE_DIR / "tab_subsample_fm_aggregate.tex"
    p.write_text(tex_agg, encoding="utf-8")
    print(f"\nSaved: {p.name}")

    # Decomposition table: B4–B8
    tex_decomp = _fm_tex_block(
        pre_summ, post_summ, pre_rsq, post_rsq, pre_nw, post_nw,
        specs      = ["B4", "B5", "B6", "B7", "B8"],
        focal_vars = ["rsj_sys_weekly", "rsj_idio_weekly",
                      "rsj_sys", "rsj_idio",
                      "res_sys_p025", "res_idio_p025"],
        title      = r"Fama--MacBeth Regressions: Decomposed Measures --- Pre-2008 vs.\ Post-2008",
        label      = "tab:subsample_fm_decomposition",
    )
    p = TABLE_DIR / "tab_subsample_fm_decomposition.tex"
    p.write_text(tex_decomp, encoding="utf-8")
    print(f"\nSaved: {p.name}")

    # Supplementary table: B9, B10
    tex_supp = _fm_tex_block(
        pre_summ, post_summ, pre_rsq, post_rsq, pre_nw, post_nw,
        specs      = ["B9", "B10"],
        focal_vars = ["rsj_idio_weekly", "rsj_sys_weekly", "res_idio_p025"],
        title      = r"Fama--MacBeth Regressions: Supplementary Specifications --- Pre-2008 vs.\ Post-2008",
        label      = "tab:subsample_fm_supplementary",
    )
    p = TABLE_DIR / "tab_subsample_fm_supplementary.tex"
    p.write_text(tex_supp, encoding="utf-8")
    print(f"\nSaved: {p.name}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
