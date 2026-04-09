"""
Crisis-state Fama-MacBeth following equation (crisis) in the paper.

Model (eq. crisis):
    R_{i,w+1} = alpha_w
                + beta_1  * RSJ^sys_{i,w}
                + beta_2  * RES^sys_{i,w}
                + beta_3  * C_w                   <- absorbed by alpha_w
                + beta_4  * (RSJ^sys_{i,w} x C_w)
                + beta_5  * (RES^sys_{i,w} x C_w)
                + gamma_w' X_{i,w}
                + eps_{i,w+1}

C_w = 1 during NBER recession weeks, 0 otherwise.

Because C_w is constant across all stocks in a given week, it is
collinear with the FM intercept alpha_w and cannot be separately
identified in a cross-sectional regression. beta_3 is therefore
absorbed into alpha_w and not reported.

FM implementation:
  Expansion weeks (C_w = 0):
      cross-section OLS on [RSJ_sys, RES_sys, controls]
      -> weekly slopes b1_w, b2_w
  Recession weeks (C_w = 1):
      cross-section OLS on [RSJ_sys, RES_sys, controls]
      -> weekly slopes (b1+b4)_w, (b2+b5)_w

  beta_1          = NW mean of {b1_w}              expansion baseline
  beta_2          = NW mean of {b2_w}              expansion baseline
  beta_4          = crisis increment for RSJ_sys   (tested via pooled OLS)
  beta_5          = crisis increment for RES_sys   (tested via pooled OLS)
  beta_1 + beta_4 = NW mean of recession slopes    implied recession effect
  beta_2 + beta_5 = NW mean of recession slopes    implied recession effect

The crisis increment test: pool expansion and recession weekly slopes and
run OLS of slope_t ~ 1 + C_w with NW(6) SEs. The coefficient on C_w is
the interaction increment (beta_4 or beta_5).

NBER recession periods in sample:
    2001-03-01 to 2001-11-30   (Dot-com recession)
    2007-12-01 to 2009-06-30   (Global Financial Crisis)
    2020-02-01 to 2020-04-30   (Covid-19 recession)
Source: National Bureau of Economic Research (nber.org)

Outputs:
    data_final/results/tables/tab_crisis_baseline.tex
    data_final/results/tables/tab_crisis_state.tex
    data_final/results/figures/fig_rolling_fm_crisis.{pdf,png}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# Settings
# ============================================================
ROOT       = Path(__file__).resolve().parents[2]
PANEL_FILE = ROOT / "data_final" / "panel" / "weekly_panel.parquet"
INTER_DIR  = ROOT / "data_final" / "results" / "intermediate"
TABLE_DIR  = ROOT / "data_final" / "results" / "tables"
FIG_DIR    = ROOT / "data_final" / "results" / "figures"
for d in (INTER_DIR, TABLE_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

NW_LAGS     = 6
MIN_STOCKS  = 50
WINSOR_LOW  = 0.01
WINSOR_HIGH = 0.99

CONTROLS = ["me", "bm", "mom", "rev", "ivol", "illiq", "rvol", "rsk", "rkt"]

# Systematic predictors: M1 (intraday decomp) and M2 (rolling regression)
SYS_SPECS = {
    "M1": ["rsj_sys_weekly", "res_sys_p025"] + CONTROLS,
}
SYS_FOCAL = {
    "M1": ["rsj_sys_weekly", "res_sys_p025"],
}

PRED_LABEL = {
    "rsj_sys_weekly": r"RSJ$_{\mathrm{sys}}$ (M1)",
    "rsj_sys":        r"RSJ$_{\mathrm{sys}}$ (M2)",
    "res_sys_p025":   r"RES$_{\mathrm{sys}}$",
    "me":    r"$\log(\mathrm{ME})$",
    "bm":    "BM",   "mom":   "MOM",   "rev":   "REV",
    "ivol":  "IVOL", "illiq": "ILLIQ", "rvol":  "RVOL",
    "rsk":   "RSK",  "rkt":   "RKT",
}

NBER_PERIODS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


# ============================================================
# Helpers
# ============================================================
def winsorize_cs(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile(WINSOR_LOW), s.quantile(WINSOR_HIGH)
    return s.clip(lo, hi)


def build_nber_indicator(week_series: pd.Series) -> pd.Series:
    """Returns 1 for NBER recession weeks, 0 for expansion weeks."""
    week = pd.to_datetime(week_series).dt.normalize()
    out  = pd.Series(False, index=week.index)
    for start, end in NBER_PERIODS:
        out |= (week >= pd.Timestamp(start)) & (week <= pd.Timestamp(end))
    return out.astype(int)


def nw_mean(arr: np.ndarray, lags: int) -> tuple[float, float, float]:
    """NW HAC mean. Returns (mean_bps, t, se_bps)."""
    y = arr[np.isfinite(arr)]
    if len(y) < 10:
        return np.nan, np.nan, np.nan
    res = sm.OLS(y, np.ones((len(y), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    return float(res.params[0]) * 10_000, float(res.tvalues[0]), float(res.bse[0]) * 10_000


def crisis_increment(betas_exp: list, betas_cri: list, lags: int) -> tuple[float, float]:
    """
    Estimate beta_4 (or beta_5): the crisis interaction increment.

    Pools expansion and recession weekly slopes and runs:
        slope_t = a + b * C_t + u_t,  C_t in {0, 1}
    with NW(6) HAC standard errors. Returns (b * 10000, t_b).
    """
    slopes = np.concatenate([
        np.asarray(betas_exp, dtype=float),
        np.asarray(betas_cri, dtype=float),
    ])
    nber = np.concatenate([
        np.zeros(len(betas_exp)),
        np.ones(len(betas_cri)),
    ])
    ok = np.isfinite(slopes)
    slopes, nber = slopes[ok], nber[ok]
    if len(slopes) < 10 or nber.sum() < 3:
        return np.nan, np.nan
    X = sm.add_constant(nber.reshape(-1, 1), has_constant="add")
    res = sm.OLS(slopes, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    return float(res.params[1]) * 10_000, float(res.tvalues[1])


def run_weekly_cs(df_week: pd.DataFrame, predictors: list[str]) -> dict | None:
    """Single-week OLS. Returns {pred: coef, rsq_adj, n} or None."""
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


def run_fm_on_panel(panel: pd.DataFrame, specs: dict) -> tuple[dict, dict, dict]:
    """
    Run FM for each spec on the given panel.
    Returns (summary, avg_rsq, n_weeks).
      summary[spec][pred] = (mean_bps, t, se_bps)
    """
    summary = {}
    avg_rsq = {}
    n_weeks = {}
    weeks   = sorted(panel["week"].unique())

    for spec, predictors in specs.items():
        missing = [p for p in predictors if p not in panel.columns]
        if missing:
            print(f"    SKIP {spec}: missing columns {missing}")
            continue

        betas    = {p: [] for p in predictors}
        rsq_list = []
        valid    = 0

        for week in weeks:
            result = run_weekly_cs(panel[panel["week"] == week], predictors)
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
            m, t, se = nw_mean(arr, NW_LAGS)
            spec_summ[p] = (m, t, se)
        summary[spec] = spec_summ

    return summary, avg_rsq, n_weeks


# ============================================================
# Crisis interaction: collect weekly slopes per state
# ============================================================
def collect_weekly_slopes(panel: pd.DataFrame, nber_map: dict,
                          specs: dict) -> dict:
    """
    For each spec and predictor, collect lists of weekly slopes
    split by expansion (C=0) and recession (C=1) weeks.

    Returns dict[spec][pred] = {"exp": [...], "cri": [...],
                                 "rsq_exp": [...], "rsq_cri": [...]}
    """
    weeks = sorted(panel["week"].unique())
    result = {
        spec: {p: {"exp": [], "cri": [], "rsq_exp": [], "rsq_cri": []}
               for p in predictors}
        for spec, predictors in specs.items()
    }

    for week in weeks:
        df_w   = panel[panel["week"] == week]
        nber_w = int(nber_map.get(week, 0))
        for spec, predictors in specs.items():
            missing = [p for p in predictors if p not in panel.columns]
            if missing:
                continue
            out = run_weekly_cs(df_w, predictors)
            if out is None:
                continue
            state = "cri" if nber_w == 1 else "exp"
            rsq_key = "rsq_cri" if nber_w == 1 else "rsq_exp"
            for p in predictors:
                result[spec][p][state].append(out[p])
            result[spec][predictors[0]][rsq_key].append(out["rsq_adj"])

    return result


# ============================================================
# LaTeX helpers
# ============================================================
def _stars(t: float) -> str:
    if t is None or not np.isfinite(t): return ""
    a = abs(float(t))
    if a > 2.576: return "***"
    if a > 1.96:  return "**"
    if a > 1.645: return "*"
    return ""

def _fmt(val, t=None, dec=2) -> str:
    if val is None or not np.isfinite(float(val)): return ""
    s = f"{float(val):.{dec}f}"
    if t is not None: s += _stars(t)
    return s

def _fmt_pct(val) -> str:
    if val is None or not np.isfinite(float(val)): return ""
    return f"{float(val) * 100:.2f}\\%"

def _cell(summ: dict, spec: str, pred: str) -> tuple[str, str]:
    if spec not in summ or pred not in summ[spec]: return "", ""
    m, t, _ = summ[spec][pred]
    return _fmt(m, t), f"({t:.2f})" if np.isfinite(t) else ""


# ============================================================
# Table 1: Full-sample baseline
# ============================================================
def build_baseline_table(summary: dict, avg_rsq: dict, n_weeks: dict) -> str:
    """
    Full-sample FM for systematic spec (M1 and M2).
    Identical layout to the main FM tables: focal vars + controls,
    t-stats below each coefficient, N weeks, avg adj R².
    Serves as the reference before the crisis interaction table.
    """
    specs  = [s for s in ["M1", "M2"] if s in summary]
    labels = {"M1": r"(M1) RSJ$_\mathrm{sys}$ + RES$_\mathrm{sys}$"}
    col_spec = "l" + "c" * len(specs)

    all_focal = []
    for s in specs:
        for v in SYS_FOCAL.get(s, []):
            if v not in all_focal:
                all_focal.append(v)
    rows_vars = all_focal + CONTROLS

    lines = []
    lines += [r"\begin{table}[htbp]", r"\centering", r"\begin{threeparttable}"]
    lines += [r"\caption{Crisis-State Analysis: Full-Sample Baseline (Systematic Components)}"]
    lines += [r"\label{tab:crisis_baseline}"]
    lines += [rf"\begin{{tabular}}{{{col_spec}}}"]
    lines += [r"\toprule"]
    lines += ["  & " + " & ".join(labels[s] for s in specs) + r" \\"]
    lines += [r"\midrule"]

    for v in rows_vars:
        if all(v not in summary.get(s, {}) for s in specs):
            continue
        coef_row = [PRED_LABEL.get(v, v)] + [_cell(summary, s, v)[0] for s in specs]
        t_row    = [""]                    + [_cell(summary, s, v)[1] for s in specs]
        lines.append(" & ".join(coef_row) + r" \\")
        lines.append(" & ".join(t_row)    + r" \\")

    lines += [r"\midrule"]
    lines += [r"$N_{\text{weeks}}$" + " & " +
              " & ".join(str(n_weeks.get(s, "--")) for s in specs) + r" \\"]
    lines += [r"Avg.\ Adj.\ $R^2$" + " & " +
              " & ".join(_fmt_pct(avg_rsq.get(s)) for s in specs) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [r"\begin{tablenotes}", r"\small",
              r"\item Notes: Full-sample Fama--MacBeth regressions. Dependent variable"
              r" is $R_{i,w+1}$. M1 uses the intraday decomposition."
              r" All slope coefficients in bps/week ($\times 10^4$)."
              r" Controls $X$: $\log(\mathrm{ME})$, BM, MOM, REV, IVOL, ILLIQ,"
              r" RVOL, RSK, RKT (winsorized 1/99\% per week)."
              r" $t$-statistics in parentheses use Newey--West HAC with 6 lags."
              r" $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
              r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(lines)


# ============================================================
# Table 2: Crisis interaction
# ============================================================
def build_crisis_table(slopes: dict,
                       n_exp_weeks: int, n_cri_weeks: int) -> str:
    """
    Reports the full crisis-interaction specification for M1 and M2.

    For each systematic predictor Z in {RSJ_sys, RES_sys}:

      beta_1   = expansion baseline slope       (NW mean of expansion-week slopes)
      beta_4   = crisis increment               (pooled OLS on crisis dummy, NW HAC)
      beta_1+4 = implied recession slope        (NW mean of recession-week slopes)

    Layout (two columns: M1, M2):
      Panel A — Expansion baseline (beta_1, beta_2)
      Panel B — Crisis interaction increment (beta_4, beta_5)
      Panel C — Implied recession slopes (beta_1+beta_4, beta_2+beta_5)
      Footer  — N expansion weeks, N recession weeks, Avg adj R² per state
    """
    specs = [s for s in ["M1", "M2"] if s in slopes]
    col_spec = "l" + "c" * len(specs)
    labels = {"M1": r"M1 (intraday decomp.)"}

    # Precompute all needed statistics
    stat = {}   # stat[spec][pred] = dict with expansion/crisis/increment entries
    for spec in specs:
        stat[spec] = {}
        for pred in SYS_FOCAL.get(spec, []):
            d    = slopes[spec][pred]
            exp  = np.array(d["exp"], dtype=float)
            cri  = np.array(d["cri"], dtype=float)

            m_e, t_e, _  = nw_mean(exp, NW_LAGS)   # beta_1
            m_c, t_c, _  = nw_mean(cri, NW_LAGS)   # beta_1 + beta_4
            inc, t_inc   = crisis_increment(d["exp"], d["cri"], NW_LAGS)  # beta_4

            rsq_exp = float(np.mean([x for x in d["rsq_exp"] if np.isfinite(x)])) \
                      if d["rsq_exp"] else np.nan
            rsq_cri = float(np.mean([x for x in d["rsq_cri"] if np.isfinite(x)])) \
                      if d["rsq_cri"] else np.nan

            stat[spec][pred] = {
                "exp": (m_e, t_e),     # expansion baseline
                "inc": (inc, t_inc),   # crisis increment (interaction)
                "cri": (m_c, t_c),     # implied recession slope
                "rsq_exp": rsq_exp,
                "rsq_cri": rsq_cri,
                "n_exp": len(exp[np.isfinite(exp)]),
                "n_cri": len(cri[np.isfinite(cri)]),
            }

    def coef_t_rows(pred, state_key, panel_label_first=False):
        """Returns (coef_row_str, t_row_str) for a predictor across all specs."""
        lbl = PRED_LABEL.get(pred, pred)
        coef_cells = []
        t_cells    = []
        for spec in specs:
            if pred not in stat.get(spec, {}):
                coef_cells += [""]
                t_cells    += [""]
                continue
            val, t = stat[spec][pred][state_key]
            coef_cells.append(_fmt(val, t))
            t_cells.append(f"({t:.2f})" if np.isfinite(t) else "")
        return (lbl + " & " + " & ".join(coef_cells) + r" \\",
                "  & " + " & ".join(t_cells) + r" \\")

    lines = []
    lines += [r"\begin{table}[htbp]", r"\centering", r"\begin{threeparttable}"]
    lines += [r"\caption{Crisis-State Fama--MacBeth: Systematic Components"
              r" with NBER Recession Interactions}"]
    lines += [r"\label{tab:crisis_state}"]
    lines += [rf"\begin{{tabular}}{{{col_spec}}}"]
    lines += [r"\toprule"]
    lines += ["  & " + " & ".join(labels[s] for s in specs) + r" \\"]
    lines += [r"\midrule"]

    # ---- Panel A: Expansion baseline ----
    lines += [r"\multicolumn{" + str(1 + len(specs)) + r"}{l}"
              r"{\textit{Panel A: Expansion baseline ($\hat{\beta}_1$,"
              r" $\hat{\beta}_2$) --- non-recession weeks}} \\"]

    all_focal_preds = []
    for spec in specs:
        for p in SYS_FOCAL.get(spec, []):
            if p not in all_focal_preds:
                all_focal_preds.append(p)

    for pred in all_focal_preds:
        lbl = PRED_LABEL.get(pred, pred)
        coef_cells, t_cells = [], []
        for spec in specs:
            # find matching pred in this spec (RSJ_sys label same for M1/M2)
            matching = [p for p in SYS_FOCAL.get(spec, [])
                        if PRED_LABEL.get(p) == PRED_LABEL.get(pred)
                        or p == pred]
            if not matching or matching[0] not in stat.get(spec, {}):
                coef_cells += [""]
                t_cells    += [""]
            else:
                val, t = stat[spec][matching[0]]["exp"]
                coef_cells.append(_fmt(val, t))
                t_cells.append(f"({t:.2f})" if np.isfinite(t) else "")
        lines.append(lbl + " & " + " & ".join(coef_cells) + r" \\")
        lines.append("  & " + " & ".join(t_cells) + r" \\")

    # ---- Panel B: Crisis interaction increment ----
    lines += [r"\midrule"]
    lines += [r"\multicolumn{" + str(1 + len(specs)) + r"}{l}"
              r"{\textit{Panel B: Crisis interaction increment ($\hat{\beta}_4$,"
              r" $\hat{\beta}_5$) --- change during NBER recessions}} \\"]

    for pred in all_focal_preds:
        lbl = PRED_LABEL.get(pred, pred) + r" $\times$ Crisis"
        coef_cells, t_cells = [], []
        for spec in specs:
            matching = [p for p in SYS_FOCAL.get(spec, [])
                        if PRED_LABEL.get(p) == PRED_LABEL.get(pred) or p == pred]
            if not matching or matching[0] not in stat.get(spec, {}):
                coef_cells += [""]
                t_cells    += [""]
            else:
                val, t = stat[spec][matching[0]]["inc"]
                coef_cells.append(_fmt(val, t))
                t_cells.append(f"({t:.2f})" if np.isfinite(t) else "")
        lines.append(lbl + " & " + " & ".join(coef_cells) + r" \\")
        lines.append("  & " + " & ".join(t_cells) + r" \\")

    # ---- Panel C: Implied recession slopes ----
    lines += [r"\midrule"]
    lines += [r"\multicolumn{" + str(1 + len(specs)) + r"}{l}"
              r"{\textit{Panel C: Implied recession slopes"
              r" ($\hat{\beta}_1{+}\hat{\beta}_4$, $\hat{\beta}_2{+}\hat{\beta}_5$)}} \\"]

    for pred in all_focal_preds:
        lbl = PRED_LABEL.get(pred, pred) + " (recession)"
        coef_cells, t_cells = [], []
        for spec in specs:
            matching = [p for p in SYS_FOCAL.get(spec, [])
                        if PRED_LABEL.get(p) == PRED_LABEL.get(pred) or p == pred]
            if not matching or matching[0] not in stat.get(spec, {}):
                coef_cells += [""]
                t_cells    += [""]
            else:
                val, t = stat[spec][matching[0]]["cri"]
                coef_cells.append(_fmt(val, t))
                t_cells.append(f"({t:.2f})" if np.isfinite(t) else "")
        lines.append(lbl + " & " + " & ".join(coef_cells) + r" \\")
        lines.append("  & " + " & ".join(t_cells) + r" \\")

    # ---- Footer ----
    lines += [r"\midrule"]
    # N expansion weeks (use first spec's first pred)
    n_exp_cells, n_cri_cells = [], []
    rsq_exp_cells, rsq_cri_cells = [], []
    for spec in specs:
        first_pred = SYS_FOCAL.get(spec, [None])[0]
        if first_pred and first_pred in stat.get(spec, {}):
            s = stat[spec][first_pred]
            n_exp_cells.append(str(s["n_exp"]))
            n_cri_cells.append(str(s["n_cri"]))
            rsq_exp_cells.append(_fmt_pct(s["rsq_exp"]))
            rsq_cri_cells.append(_fmt_pct(s["rsq_cri"]))
        else:
            n_exp_cells  += ["--"]
            n_cri_cells  += ["--"]
            rsq_exp_cells += [""]
            rsq_cri_cells += [""]

    lines += [r"$N_{\text{expansion}}$" + " & " + " & ".join(n_exp_cells) + r" \\"]
    lines += [r"$N_{\text{recession}}$"  + " & " + " & ".join(n_cri_cells) + r" \\"]
    lines += [r"Avg.\ Adj.\ $R^2$ (exp.)" + " & " +
              " & ".join(rsq_exp_cells) + r" \\"]
    lines += [r"Avg.\ Adj.\ $R^2$ (rec.)" + " & " +
              " & ".join(rsq_cri_cells) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}"]

    nber_str = "; ".join(f"{s} to {e}" for s, e in NBER_PERIODS)
    lines += [r"\begin{tablenotes}", r"\small",
              r"\item Notes: Fama--MacBeth interaction regressions following"
              r" equation~(\ref{eq:crisis}). The crisis-state dummy $C_w = 1$"
              r" during NBER recession weeks (0 otherwise)."
              r" Because $C_w$ is constant across all stocks within a week, it is"
              r" absorbed by the FM intercept $\alpha_w$ and not separately reported."
              r" Panel~A reports expansion-week FM slopes ($\hat\beta_1$,"
              r" $\hat\beta_2$); Panel~B the crisis increment"
              r" ($\hat\beta_4$, $\hat\beta_5$) estimated via pooled OLS of"
              r" weekly slopes on $C_w$ with NW(6) standard errors;"
              r" Panel~C the implied recession slopes"
              r" ($\hat\beta_1{+}\hat\beta_4$, $\hat\beta_2{+}\hat\beta_5$)"
              r" as the NW mean of recession-week FM slopes."
              rf" NBER recession periods: {nber_str}."
              r" Source: National Bureau of Economic Research."
              r" All coefficients in bps/week ($\times 10^4$)."
              r" Controls $X$ included but not reported."
              r" $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
              r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(lines)


# ============================================================
# Figure: rolling 52-week FM slopes
# ============================================================
def build_rolling_figure() -> plt.Figure | None:
    b7_path = INTER_DIR / "fm_coefs_B7.parquet"
    if not b7_path.exists():
        print(f"  {b7_path.name} not found — run 02_fama_macbeth.py first.")
        return None
    df = pd.read_parquet(b7_path)
    df["week"] = pd.to_datetime(df["week"])
    needed = ["week", "rsj_sys_weekly", "res_sys_p025"]
    if not all(c in df.columns for c in needed):
        print("  B7 parquet missing sys columns — skipping figure.")
        return None
    df = df[needed].dropna().sort_values("week").set_index("week")
    if len(df) < 52:
        return None

    roll_rsj = df["rsj_sys_weekly"].rolling(52, min_periods=26).mean() * 10_000
    roll_res = df["res_sys_p025"].rolling(52, min_periods=26).mean() * 10_000

    fig, ax = plt.subplots(figsize=(9, 4))
    for start, end in NBER_PERIODS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color="gray", lw=0)
    ax.plot(roll_rsj.index, roll_rsj.values, color="#1f77b4", lw=1.5,
            label=r"RSJ$_\mathrm{sys}$ (M1)")
    ax.plot(roll_res.index, roll_res.values, color="#ff7f0e", lw=1.5,
            label=r"RES$_\mathrm{sys}$")
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    handles, lbls = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="gray", alpha=0.3, label="NBER recession"))
    ax.legend(handles=handles, fontsize=9)
    ax.set_ylabel("FM slope (bps/week, 52-week rolling mean)", fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_title("Rolling FM Slopes: Systematic Components with NBER Recessions",
                 fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
    ax.grid(False)
    fig.tight_layout()
    return fig


# ============================================================
# Main
# ============================================================
def main():
    print("=== Crisis-State Fama-MacBeth ===\n")

    panel = pd.read_parquet(PANEL_FILE)
    panel["week"] = pd.to_datetime(panel["week"]).dt.normalize()
    if "valid_R_i_w_plus_1" in panel.columns:
        panel = panel[panel["valid_R_i_w_plus_1"].astype(bool)].copy()
    else:
        panel = panel[panel["R_i_w_plus_1"].notna()].copy()

    for c in CONTROLS:
        if c in panel.columns:
            panel[c] = panel.groupby("week")[c].transform(winsorize_cs)

    # NBER indicator
    panel["nber"] = build_nber_indicator(panel["week"])
    nber_map = (panel[["week", "nber"]].drop_duplicates()
                .set_index("week")["nber"].to_dict())

    n_exp = sum(v == 0 for v in nber_map.values())
    n_cri = sum(v == 1 for v in nber_map.values())
    print(f"Panel: {len(panel):,} stock-weeks | {panel['week'].nunique():,} weeks total")
    print(f"  Expansion weeks : {n_exp}")
    print(f"  Recession weeks : {n_cri}")
    print(f"  NBER periods    : {'; '.join(f'{s} to {e}' for s,e in NBER_PERIODS)}\n")

    # ----------------------------------------------------------
    # Table 1: Full-sample baseline
    # ----------------------------------------------------------
    print("--- Full-sample baseline (M1) ---")
    full_summ, full_rsq, full_nw = run_fm_on_panel(panel, SYS_SPECS)
    for spec in ["M1"]:
        if spec not in full_summ:
            continue
        for pred in SYS_FOCAL.get(spec, []):
            if pred not in full_summ[spec]:
                continue
            m, t, _ = full_summ[spec][pred]
            st = "***" if abs(t) > 2.576 else "**" if abs(t) > 1.96 else \
                 "*" if abs(t) > 1.645 else ""
            print(f"  {spec} {PRED_LABEL.get(pred, pred):<35} "
                  f"{m:>8.2f}{st:<3} (t={t:.2f})")

    tex_base = build_baseline_table(full_summ, full_rsq, full_nw)
    p = TABLE_DIR / "tab_crisis_baseline.tex"
    p.write_text(tex_base, encoding="utf-8")
    print(f"\nSaved: {p.name}")

    # ----------------------------------------------------------
    # Table 2: Crisis interaction
    # ----------------------------------------------------------
    print("\n--- Collecting weekly slopes by state (M1) ---")
    slopes = collect_weekly_slopes(panel, nber_map, SYS_SPECS)

    # Console summary
    print(f"\n{'Predictor':<38} {'Expansion':>12}  {'Increment':>12}  {'Recession':>12}")
    print("-" * 80)
    for spec in ["M1"]:
        if spec not in slopes:
            continue
        print(f"  [{spec}]")
        for pred in SYS_FOCAL.get(spec, []):
            if pred not in slopes.get(spec, {}):
                continue
            d   = slopes[spec][pred]
            exp = np.array(d["exp"], dtype=float)
            cri = np.array(d["cri"], dtype=float)
            m_e, t_e, _ = nw_mean(exp, NW_LAGS)
            m_c, t_c, _ = nw_mean(cri, NW_LAGS)
            inc, t_inc  = crisis_increment(d["exp"], d["cri"], NW_LAGS)
            st = lambda t: "***" if abs(t)>2.576 else "**" if abs(t)>1.96 else \
                           "*" if abs(t)>1.645 else "" if np.isfinite(t) else ""
            print(f"  {PRED_LABEL.get(pred, pred):<36} "
                  f"{m_e:>8.2f}{st(t_e):<3}  "
                  f"{inc:>8.2f}{st(t_inc):<3}  "
                  f"{m_c:>8.2f}{st(t_c):<3}")

    tex_crisis = build_crisis_table(slopes, n_exp, n_cri)
    p = TABLE_DIR / "tab_crisis_state.tex"
    p.write_text(tex_crisis, encoding="utf-8")
    print(f"\nSaved: {p.name}")

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------
    fig = build_rolling_figure()
    if fig is not None:
        for ext in ("pdf", "png"):
            fpath = FIG_DIR / f"fig_rolling_fm_crisis.{ext}"
            fig.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: fig_rolling_fm_crisis.pdf/.png")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
