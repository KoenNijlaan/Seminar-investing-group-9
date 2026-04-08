"""
Quintile portfolio sorts following Bollerslev, Li & Zhao (2020).

For each of 8 sorting variables, each week t:
  1. Sort eligible stocks into quintiles Q1–Q5 (NYSE breakpoints)
  2. Compute EW and VW portfolio returns for the HOLDING week t+1
  3. Compute spread: L-H for RSJ signals, H-L for RES signals
  4. Time-series mean (NW, 6 lags) and FF3 alpha (NW, 4 lags)

FF3 alpha is reported. The Carhart UMD factor is deliberately omitted because
momentum is already controlled for via the MOM variable in the cross-sections.

Outputs:
  data_final/portfolio_sorts/sort_results_all.parquet   ← intermediate
  data_final/results/intermediate/spread_returns.parquet ← for cumulative figure
  data_final/results/tables/tab_portfolio_spreads.tex
  data_final/results/figures/fig_portfolio_spreads_ew_bps.{pdf,png}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Settings
# ============================================================
ROOT       = Path(__file__).resolve().parents[2]
PANEL_FILE = ROOT / "data_final" / "panel" / "weekly_panel.parquet"
FF_FILE    = ROOT / "data_raw" / "wrds" / "ff_daily_factors_1992_2024.parquet"

SORT_OUT   = ROOT / "data_final" / "portfolio_sorts"
INTER_DIR  = ROOT / "data_final" / "results" / "intermediate"
TABLE_DIR  = ROOT / "data_final" / "results" / "tables"
FIG_DIR    = ROOT / "data_final" / "results" / "figures"
for d in (SORT_OUT, INTER_DIR, TABLE_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

NW_LAGS        = 6   # Newey-West lags — applied everywhere uniformly
MIN_STOCKS     = 50

SORT_VARIABLES = {
    "rsj_weekly"     : "RSJ",
    "rsj_sys_weekly" : "RSJ Systematic (M1)",
    "rsj_idio_weekly": "RSJ Idiosyncratic (M1)",
    "rsj_sys"        : "RSJ Systematic (M2)",
    "rsj_idio"       : "RSJ Idiosyncratic (M2)",
    "res_weekly"     : "RES",
    "res_sys_p025"   : "RES Systematic",
    "res_idio_p025"  : "RES Idiosyncratic",
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

# Panel A = RSJ signals, Panel B = RES signals
PANEL_A = ["rsj_weekly","rsj_sys_weekly","rsj_idio_weekly","rsj_sys","rsj_idio"]
PANEL_B = ["res_weekly","res_sys_p025","res_idio_p025"]


# ============================================================
# Helpers
# ============================================================
def _stars(t):
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    t = abs(float(t))
    if t > 2.576: return "***"
    if t > 1.96: return "**"
    if t > 1.645: return "*"
    return ""


def _fmt(val, t=None, dec=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = f"{float(val):.{dec}f}"
    if t is not None:
        s += _stars(t)
    return s


def nw_mean(series, lags):
    """NW-HAC mean of a series. Returns (mean, t_stat, se)."""
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return np.nan, np.nan, np.nan
    res = sm.OLS(y, np.ones((len(y), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    return float(res.params[0]), float(res.tvalues[0]), float(res.bse[0])


def nw_alpha(excess_ret, factors_df, lags):
    """OLS of excess_ret on factors with HAC-NW. Returns (alpha, t_alpha)."""
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
    out[v & (sort_var <= q20)]                          = 1
    out[v & (sort_var > q20) & (sort_var <= q40)]       = 2
    out[v & (sort_var > q40) & (sort_var <= q60)]       = 3
    out[v & (sort_var > q60) & (sort_var <= q80)]       = 4
    out[v & (sort_var > q80)]                           = 5
    return out


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


def compute_common_sort_weeks(panel: pd.DataFrame, sort_cols: list[str], ff_idx: pd.DataFrame) -> list[pd.Timestamp]:
    """
    Compute the intersection of weeks that satisfy eligibility for all sort variables.
    This enforces a common time sample across all portfolio sorts.
    """
    base = panel[
        (panel["valid_R_i_w_plus_1"] == True) &
        panel["R_i_w_plus_1"].notna()
    ].copy()

    # Start from weeks present in the FF factor index.
    common = set(pd.to_datetime(ff_idx.index).normalize())

    for sort_col in sort_cols:
        eligible = base[base[sort_col].notna()]
        wc = eligible.groupby("week").size()
        weeks_ok = set(wc[wc >= MIN_STOCKS].index)
        common &= weeks_ok

    return sorted(common)


# ============================================================
# Single sort
# ============================================================
def run_sort(panel, sort_col, ff_idx, label, common_weeks=None):
    eligible = panel[
        (panel["valid_R_i_w_plus_1"] == True) &
        panel[sort_col].notna() &
        panel["R_i_w_plus_1"].notna()
    ].copy()

    if common_weeks is not None:
        eligible = eligible[eligible["week"].isin(common_weeks)].copy()

    wc = eligible.groupby("week").size()
    eligible = eligible[eligible["week"].isin(wc[wc >= MIN_STOCKS].index)].copy()
    if len(eligible) == 0:
        return pd.DataFrame(), None

    exchcd = eligible["exchcd"] if "exchcd" in eligible.columns else None
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

    # Build EW/VW series per quintile
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

    # Summarise each quintile + spread
    rows = []
    common_week_index = pd.Index(sorted(common_weeks)).intersection(ff_idx.index) if common_weeks is not None else None

    for label_q, ts_ew, ts_vw in (
        [(str(q), qt_ew[q].dropna(), qt_vw[q].dropna()) for q in range(1, 6)] +
        [(direction, sp_ew, sp_vw)]
    ):
        for w, ts in [("EW", ts_ew), ("VW", ts_vw)]:
            common = common_week_index if common_week_index is not None else ts.index.intersection(ff_idx.index)
            if len(common) < 20:
                rows.append({"sort_var": sort_col, "label": SORT_VARIABLES[sort_col],
                             "quintile": label_q, "weighting": w,
                             "n_weeks": len(common),
                             "mean_ret_weekly": np.nan, "t_mean": np.nan,
                             "se_mean": np.nan,
                             "alpha_weekly": np.nan, "t_alpha": np.nan})
                continue
            ts_c  = ts.reindex(common)
            ff_c  = ff_idx.loc[common]
            mean_r, t_mean, se_mean = nw_mean(ts_c.values, NW_LAGS)
            excess = ts_c.values - ff_c["rf"].values
            factors = pd.DataFrame({
                "mktrf": ff_c["mktrf"].values,
                "smb":   ff_c["smb"].values,
                "hml":   ff_c["hml"].values,
            })
            alpha, t_alpha = nw_alpha(excess, factors, NW_LAGS)
            rows.append({
                "sort_var":       sort_col,
                "label":          SORT_VARIABLES[sort_col],
                "quintile":       label_q,
                "weighting":      w,
                "n_weeks":        len(common),
                "mean_ret_weekly": mean_r,
                "t_mean":          t_mean,
                "se_mean":         se_mean,
                "alpha_weekly":    alpha,
                "t_alpha":         t_alpha,
            })

    spread_ew_ts = sp_ew.rename(sort_col)
    return pd.DataFrame(rows), spread_ew_ts


# ============================================================
# LaTeX table
# ============================================================
def build_latex(results_df):
    """Build tab_portfolio_spreads.tex (booktabs + threeparttable)."""
    spread = results_df[results_df["quintile"].isin(["L-H", "H-L"])].copy()

    def row_pair(sort_col, weighting, label):
        r = spread[
            (spread["sort_var"] == sort_col) &
            (spread["weighting"] == weighting)
        ]
        if r.empty:
            return None, None
        r = r.iloc[0]
        ew_mean  = _fmt(r["mean_ret_weekly"] * 10_000, r["t_mean"])
        ew_t     = f"({r['t_mean']:.2f})" if np.isfinite(r["t_mean"]) else ""
        ew_alpha = _fmt(r["alpha_weekly"] * 10_000, r["t_alpha"])
        ew_ta    = f"({r['t_alpha']:.2f})" if np.isfinite(r["t_alpha"]) else ""
        nw = int(r["n_weeks"]) if np.isfinite(r["n_weeks"]) else ""
        return (ew_mean, ew_t, ew_alpha, ew_ta, nw)

    col_header = (
        r"\toprule" "\n"
        r"Signal & Spread & EW Mean & VW Mean & EW $\alpha$ & VW $\alpha$ & $N$ \\" "\n"
        r" & & (bps) & (bps) & (bps) & (bps) & weeks \\" "\n"
        r"\midrule"
    )

    def signal_rows(sort_col):
        ew = spread[(spread["sort_var"] == sort_col) & (spread["weighting"] == "EW")]
        vw = spread[(spread["sort_var"] == sort_col) & (spread["weighting"] == "VW")]
        if ew.empty or vw.empty:
            return ""
        ew = ew.iloc[0]; vw = vw.iloc[0]
        spr = ew["quintile"]
        ew_m  = _fmt(ew["mean_ret_weekly"]  * 10_000, ew["t_mean"])
        vw_m  = _fmt(vw["mean_ret_weekly"]  * 10_000, vw["t_mean"])
        ew_a  = _fmt(ew["alpha_weekly"]     * 10_000, ew["t_alpha"])
        vw_a  = _fmt(vw["alpha_weekly"]     * 10_000, vw["t_alpha"])
        nw    = int(ew["n_weeks"]) if np.isfinite(ew["n_weeks"]) else ""
        ew_tm = f"({ew['t_mean']:.2f})" if np.isfinite(ew["t_mean"]) else ""
        vw_tm = f"({vw['t_mean']:.2f})" if np.isfinite(vw["t_mean"]) else ""
        ew_ta = f"({ew['t_alpha']:.2f})" if np.isfinite(ew["t_alpha"]) else ""
        vw_ta = f"({vw['t_alpha']:.2f})" if np.isfinite(vw["t_alpha"]) else ""
        lbl   = SORT_VARIABLES[sort_col].replace("_", r"\_")
        coef_row = f"{lbl} & {spr} & {ew_m} & {vw_m} & {ew_a} & {vw_a} & {nw} \\\\"
        tstat_row = f" & & {ew_tm} & {vw_tm} & {ew_ta} & {vw_ta} & \\\\"
        return coef_row + "\n" + tstat_row + "\n"

    panel_a_rows = "".join(signal_rows(s) for s in PANEL_A if s in spread["sort_var"].values)
    panel_b_rows = "".join(signal_rows(s) for s in PANEL_B if s in spread["sort_var"].values)

    tex = r"""\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{Quintile Portfolio Sort Spreads}
\label{tab:portfolio_spreads}
\begin{tabular}{llccccr}
""" + col_header + r"""
\multicolumn{7}{l}{\textit{Panel A: RSJ measures}} \\
""" + panel_a_rows + r"""\midrule
\multicolumn{7}{l}{\textit{Panel B: RES measures}} \\
""" + panel_b_rows + r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Each week stocks are sorted into quintiles based on the signal.
Spread is Q1$-$Q5 for RSJ signals (L$-$H) and Q5$-$Q1 for RES signals (H$-$L).
EW and VW denote equal- and value-weighted portfolios.
$\alpha$ is the intercept from a time-series regression on the FF3 factors
(Fama--French three-factor model; the Carhart momentum factor is omitted
because momentum is already controlled for via the MOM control variable
in the cross-sectional regressions).
$t$-statistics in parentheses use Newey--West standard errors
(NW(6) throughout).
$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    return tex


# ============================================================
# Figure
# ============================================================
def build_figure(results_df):
    spread = results_df[
        results_df["quintile"].isin(["L-H", "H-L"]) &
        (results_df["weighting"] == "EW")
    ].copy()
    spread["mean_bps"] = spread["mean_ret_weekly"] * 10_000
    spread["se_bps"]   = spread["se_mean"] * 10_000

    # Order: RSJ signals then RES signals
    order = PANEL_A + PANEL_B
    spread["_ord"] = spread["sort_var"].map({s: i for i, s in enumerate(order)})
    spread = spread.sort_values("_ord", ascending=False).reset_index(drop=True)

    labels = [SORT_VARIABLES.get(s, s) for s in spread["sort_var"]]
    means  = spread["mean_bps"].values
    errs   = (1.96 * spread["se_bps"]).values
    colors = ["#2a9d8f" if m >= 0 else "#e76f51" for m in means]

    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, means, color=colors, alpha=0.85, height=0.6)
    ax.errorbar(means, y_pos, xerr=errs, fmt="none", color="black", capsize=3, lw=1)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("EW Mean Spread Return (bps/week)", fontsize=10)
    ax.set_title("Quintile Spread Returns by Sorting Signal", fontsize=11, pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
    ax.grid(False)
    fig.tight_layout()
    return fig


# ============================================================
# Main
# ============================================================
def main():
    print("=== Portfolio Sorts ===\n")

    panel = pd.read_parquet(PANEL_FILE)
    panel["week"] = pd.to_datetime(panel["week"]).dt.normalize()
    if "valid_R_i_w_plus_1" in panel.columns:
        panel["valid_R_i_w_plus_1"] = panel["valid_R_i_w_plus_1"].astype(bool)
    else:
        panel["valid_R_i_w_plus_1"] = panel["R_i_w_plus_1"].notna()

    ff_w   = load_ff3_weekly(FF_FILE)
    ff_idx = ff_w.set_index("week")

    active_sort_cols = [c for c in SORT_VARIABLES if c in panel.columns]
    common_weeks = compute_common_sort_weeks(panel, active_sort_cols, ff_idx)
    print(f"Common sort sample: {len(common_weeks)} weeks shared across all signals")

    all_results = []
    spread_returns = {}  # sort_col → weekly EW spread series

    for sort_col in SORT_VARIABLES:
        if sort_col not in panel.columns:
            print(f"  Skipping {sort_col} — not in panel")
            continue
        print(f"  Sorting on: {SORT_VARIABLES[sort_col]}")
        df_res, sp_ts = run_sort(
            panel,
            sort_col,
            ff_idx,
            SORT_VARIABLES[sort_col],
            common_weeks=common_weeks,
        )
        if not df_res.empty:
            all_results.append(df_res)
        if sp_ts is not None and len(sp_ts) > 0:
            spread_returns[sort_col] = sp_ts

    if not all_results:
        print("No results.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined["quintile"] = combined["quintile"].astype(str)

    # Save intermediate
    combined.to_parquet(SORT_OUT / "sort_results_all.parquet", index=False)
    print(f"\nSaved: sort_results_all.parquet")

    # Save spread return time series for cumulative figure
    if spread_returns:
        sp_df = pd.DataFrame(spread_returns).reset_index().rename(columns={"index": "week"})
        sp_df.to_parquet(INTER_DIR / "spread_returns.parquet", index=False)
        print(f"Saved: spread_returns.parquet")

    # LaTeX table
    tex = build_latex(combined)
    tex_path = TABLE_DIR / "tab_portfolio_spreads.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"Saved: {tex_path.name}")

    # Figure
    fig = build_figure(combined)
    for ext in ("pdf", "png"):
        fpath = FIG_DIR / f"fig_portfolio_spreads_ew_bps.{ext}"
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: fig_portfolio_spreads_ew_bps.pdf/.png")

    # Console summary
    print("\n" + "=" * 70)
    print("EW SPREAD RETURNS (bps/week)")
    print("=" * 70)
    sp = combined[combined["quintile"].isin(["L-H","H-L"]) & (combined["weighting"] == "EW")]
    for _, r in sp.iterrows():
        m = r["mean_ret_weekly"] * 10_000
        t = r["t_mean"]
        stars = _stars(t)
        print(f"  {SORT_VARIABLES.get(r['sort_var'], r['sort_var']):<30} "
              f"{m:>8.2f}{stars}  (t={t:.2f})")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
