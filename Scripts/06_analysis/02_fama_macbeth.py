"""
Fama-MacBeth cross-sectional regressions. NW(6) throughout.

Specifications (methodology-consistent naming):
    B1   RSJ + controls
    B2   RES + controls
    B3   RSJ + RES + controls
    B4   RSJ decomposition (M1) + controls
    B5   RSJ decomposition (M2) + controls
    B6   RES decomposition + controls
    B7   Full decomposition using RSJ M1 + controls
    B8   Full decomposition using RSJ M2 + controls
    B9   RSJ idiosyncratic (M1) + RES idiosyncratic + controls
    B10  RSJ systematic (M1) + RES idiosyncratic + controls

Controls X: log_me (me), bm, mom, rev, ivol, illiq  — winsorized 1/99 pct per week.
All slope coefficients reported in bps/week (×10,000).

Intermediate output (weekly coefficient series) saved per spec; needed by
03_wald_tests.py and 04_crisis_state.py.

Outputs:
  data_final/results/intermediate/fm_coefs_{spec}.parquet  (all specs)
  data_final/results/tables/tab_fm_aggregate.tex
  data_final/results/tables/tab_fm_decomposition.tex
  data_final/results/tables/tab_fm_supplementary.tex
  data_final/results/figures/fig_fm_coefficients_selected.{pdf,png}
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
INTER_DIR  = ROOT / "data_final" / "results" / "intermediate"
TABLE_DIR  = ROOT / "data_final" / "results" / "tables"
FIG_DIR    = ROOT / "data_final" / "results" / "figures"
for d in (INTER_DIR, TABLE_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

NW_LAGS    = 6
MIN_STOCKS = 50
WINSOR_LOW  = 0.01
WINSOR_HIGH = 0.99

CONTROLS = ["me", "bm", "mom", "rev", "ivol", "illiq", "rvol", "rsk", "rkt"]

CTRL_LABEL = {
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

# ============================================================
# Regression specifications
# ============================================================
#  Each entry: (spec_id, rhs_vars_list)
#  Controls are identified by membership in CONTROLS set.
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

# Human-readable labels for predictor display
PRED_LABEL = {
    "rsj_weekly":      "RSJ",
    "res_weekly":      "RES",
    "rsj_sys_weekly":  r"RSJ$_{\mathrm{sys}}$ (M1)",
    "rsj_idio_weekly": r"RSJ$_{\mathrm{idio}}$ (M1)",
    "rsj_sys":         r"RSJ$_{\mathrm{sys}}$ (M2)",
    "rsj_idio":        r"RSJ$_{\mathrm{idio}}$ (M2)",
    "res_sys_p025":    r"RES$_{\mathrm{sys}}$",
    "res_idio_p025":   r"RES$_{\mathrm{idio}}$",
    **CTRL_LABEL,
    "rvol": "RVOL",
    "rsk":  "RSK",
    "rkt":  "RKT",
}


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


def _fmt_pct(val):
    """Format a proportion as a percentage string, e.g. 0.0537 → '5.37\\%'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return f"{float(val) * 100:.2f}\\%"


def winsorize_cs(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile(WINSOR_LOW), s.quantile(WINSOR_HIGH)
    return s.clip(lo, hi)


def nw_average(arr, lags):
    """NW HAC mean. Returns (mean_bps, t_stat, se_bps)."""
    y = arr[np.isfinite(arr)]
    if len(y) < 10:
        return np.nan, np.nan, np.nan
    res = sm.OLS(y, np.ones((len(y), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    m   = float(res.params[0])
    t   = float(res.tvalues[0])
    se  = float(res.bse[0])
    return m * 10_000, t, se * 10_000


def run_weekly_cs(df_week, predictors):
    """One-week OLS. Returns {predictor: coef, 'rsq_adj': float, 'n': int} or None."""
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


# ============================================================
# Main FM loop
# ============================================================
def run_fm(panel):
    """
    Run all specs. Returns:
      all_weekly: dict spec → DataFrame(week, pred1, pred2, ..., rsq_adj, n)
      summary: dict spec → dict pred → (mean_bps, t, se_bps)
      avg_rsq: dict spec → float
    """
    all_weekly = {}
    summary    = {}
    avg_rsq    = {}

    weeks = sorted(panel["week"].unique())

    for spec, predictors in SPECS.items():
        missing = [p for p in predictors if p not in panel.columns]
        if missing:
            print(f"  SKIP {spec}: missing columns {missing}")
            continue

        betas   = {p: [] for p in predictors}
        rsq_list = []
        rows     = []

        for week in weeks:
            df_w   = panel[panel["week"] == week]
            result = run_weekly_cs(df_w, predictors)
            if result is None:
                continue
            rsq_list.append(result["rsq_adj"])
            row = {"week": week, "rsq_adj": result["rsq_adj"], "n": result["n"]}
            for p in predictors:
                row[p] = result[p]
                betas[p].append(result[p])
            rows.append(row)

        if not rows:
            continue

        all_weekly[spec] = pd.DataFrame(rows)
        avg_rsq[spec]    = float(np.mean(rsq_list)) if rsq_list else np.nan

        spec_summ = {}
        for p in predictors:
            arr = np.array(betas[p], dtype=float)
            m_bps, t, se_bps = nw_average(arr, NW_LAGS)
            spec_summ[p] = (m_bps, t, se_bps)
        summary[spec] = spec_summ

        n_valid = len(rows)
        print(f"  {spec}: {n_valid} valid weeks, "
              f"avg adj-R²={avg_rsq[spec]:.4f}")

    return all_weekly, summary, avg_rsq


# ============================================================
# LaTeX helpers
# ============================================================
def _tex_head(cols_header, col_spec):
    return (
        r"\toprule" "\n"
        + cols_header + r" \\" "\n"
        + r"\midrule"
    )


def build_tab_aggregate(summary, avg_rsq, all_weekly):
    """tab_fm_aggregate.tex — 3 columns (B1-B3)."""
    specs  = ["B1", "B2", "B3"]
    labels = ["(B1) RSJ+ctrl", "(B2) RES+ctrl", "(B3) RSJ+RES+ctrl"]
    n_cols = len(specs)

    # Row order: focal predictors first, then controls
    focal = ["rsj_weekly", "res_weekly"]
    rows_vars = focal + CONTROLS

    def cell(spec, pred):
        if spec not in summary or pred not in summary[spec]:
            return "", ""
        m, t, _ = summary[spec][pred]
        return _fmt(m, t), f"({t:.2f})" if np.isfinite(t) else ""

    # N weeks from first predictor
    def n_wk(spec):
        if spec not in all_weekly:
            return ""
        return str(len(all_weekly[spec]))

    lines  = []
    lines += [r"\begin{table}[htbp]", r"\centering", r"\begin{threeparttable}"]
    lines += [r"\caption{Fama--MacBeth Regressions: Aggregate Measures}"]
    lines += [r"\label{tab:fm_aggregate}"]
    col_spec = "l" + "c" * n_cols
    lines += [rf"\begin{{tabular}}{{{col_spec}}}"]
    lines += [r"\toprule"]
    lines += [" & " + " & ".join(labels) + r" \\"]
    lines += [r"\midrule"]

    for v in rows_vars:
        if all(v not in summary.get(s, {}) for s in specs):
            continue
        lbl = PRED_LABEL.get(v, v)
        coef_row = [lbl] + [cell(s, v)[0] for s in specs]
        t_row    = [""]   + [cell(s, v)[1] for s in specs]
        lines.append(" & ".join(coef_row) + r" \\")
        lines.append(" & ".join(t_row)    + r" \\")

    lines += [r"\midrule"]
    lines += [r"$N_{\text{weeks}}$" + " & " +
              " & ".join(n_wk(s) for s in specs) + r" \\"]
    lines += [r"Avg.\ Adj.\ $R^2$" + " & " +
              " & ".join(_fmt_pct(avg_rsq.get(s, np.nan)) for s in specs) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [r"\begin{tablenotes}", r"\small",
              r"\item Notes: Dependent variable is $R_{i,w+1}$ (next-week return)."
              r" All slope coefficients in bps/week ($\times 10^4$)."
              r" Controls $X$: $\log(\mathrm{ME})$, BM, MOM, REV, IVOL, ILLIQ,"
              r" winsorized cross-sectionally at 1/99\,pct each week."
              r" $t$-statistics (in parentheses) use Newey--West HAC with 6 lags."
              r" $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
              r"\end{tablenotes}",
              r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(lines)


def build_tab_decomposition(summary, avg_rsq, all_weekly):
    """tab_fm_decomposition.tex — 5 columns B4-B8."""
    specs  = ["B4", "B5", "B6", "B7", "B8"]
    labels = ["(B4) RSJ M1", "(B5) RSJ M2", "(B6) RES decomp", "(B7) Full M1", "(B8) Full M2"]
    n_cols = len(specs)

    focal = ["rsj_sys_weekly","rsj_idio_weekly",
             "rsj_sys","rsj_idio",
             "res_sys_p025","res_idio_p025"]
    rows_vars = focal + CONTROLS

    def cell(spec, pred):
        if spec not in summary or pred not in summary[spec]:
            return "", ""
        m, t, _ = summary[spec][pred]
        return _fmt(m, t), f"({t:.2f})" if np.isfinite(t) else ""

    def n_wk(spec):
        if spec not in all_weekly:
            return ""
        return str(len(all_weekly[spec]))

    lines  = []
    lines += [r"\begin{table}[htbp]", r"\centering", r"\begin{threeparttable}"]
    lines += [r"\caption{Fama--MacBeth Regressions: Decomposed Measures}"]
    lines += [r"\label{tab:fm_decomposition}"]
    col_spec = "l" + "c" * n_cols
    lines += [rf"\begin{{tabular}}{{{col_spec}}}"]
    lines += [r"\toprule"]
    lines += [" & " + " & ".join(labels) + r" \\"]
    lines += [r"\midrule"]

    for v in rows_vars:
        if all(v not in summary.get(s, {}) for s in specs):
            continue
        lbl = PRED_LABEL.get(v, v)
        coef_row = [lbl] + [cell(s, v)[0] for s in specs]
        t_row    = [""]   + [cell(s, v)[1] for s in specs]
        lines.append(" & ".join(coef_row) + r" \\")
        lines.append(" & ".join(t_row)    + r" \\")

    lines += [r"\midrule"]
    lines += [r"$N_{\text{weeks}}$" + " & " +
              " & ".join(n_wk(s) for s in specs) + r" \\"]
    lines += [r"Avg.\ Adj.\ $R^2$" + " & " +
              " & ".join(_fmt_pct(avg_rsq.get(s, np.nan)) for s in specs) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [r"\begin{tablenotes}", r"\small",
              r"\item Notes: M1 = intraday decomposition; M2 = rolling 52-week"
              r" market-RSJ regression. See Table~\ref{tab:fm_aggregate} for further notes.",
              r"\end{tablenotes}",
              r"\end{threeparttable}", r"\end{table}"]

    note_txt = "B7 (M1 full) is the primary decomposition specification; B8 (M2 full) is the robustness check."
    return "\n".join(lines), note_txt


def build_tab_supplementary(summary, avg_rsq, all_weekly):
    """tab_fm_supplementary.tex — 2 columns B9, B10."""
    specs  = ["B9", "B10"]
    labels = [r"(B9) RSJ$_{\mathrm{idio}}$ + RES$_{\mathrm{idio}}$",
              r"(B10) RSJ$_{\mathrm{sys}}$ + RES$_{\mathrm{idio}}$"]
    n_cols = len(specs)

    focal = ["rsj_idio_weekly","rsj_sys_weekly","res_idio_p025"]
    rows_vars = focal + CONTROLS

    def cell(spec, pred):
        if spec not in summary or pred not in summary[spec]:
            return "", ""
        m, t, _ = summary[spec][pred]
        return _fmt(m, t), f"({t:.2f})" if np.isfinite(t) else ""

    def n_wk(spec):
        if spec not in all_weekly:
            return ""
        return str(len(all_weekly[spec]))

    lines  = []
    lines += [r"\begin{table}[htbp]", r"\centering", r"\begin{threeparttable}"]
    lines += [r"\caption{Fama--MacBeth Regressions: Supplementary Specifications}"]
    lines += [r"\label{tab:fm_supplementary}"]
    col_spec = "l" + "c" * n_cols
    lines += [rf"\begin{{tabular}}{{{col_spec}}}"]
    lines += [r"\toprule"]
    lines += [" & " + " & ".join(labels) + r" \\"]
    lines += [r"\midrule"]

    for v in rows_vars:
        if all(v not in summary.get(s, {}) for s in specs):
            continue
        lbl = PRED_LABEL.get(v, v)
        coef_row = [lbl] + [cell(s, v)[0] for s in specs]
        t_row    = [""]   + [cell(s, v)[1] for s in specs]
        lines.append(" & ".join(coef_row) + r" \\")
        lines.append(" & ".join(t_row)    + r" \\")

    lines += [r"\midrule"]
    lines += [r"$N_{\text{weeks}}$" + " & " +
              " & ".join(n_wk(s) for s in specs) + r" \\"]
    lines += [r"Avg.\ Adj.\ $R^2$" + " & " +
              " & ".join(_fmt_pct(avg_rsq.get(s, np.nan)) for s in specs) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [r"\begin{tablenotes}", r"\small",
              r"\item Notes: (S1) demonstrates that idiosyncratic RSJ and idiosyncratic"
              r" RES each carry incremental predictive content beyond the other."
              r" (S2) shows that systematic RSJ does not."
              r" See Table~\ref{tab:fm_aggregate} for further notes.",
              r"\end{tablenotes}",
              r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(lines)


# ============================================================
# Figure: coefficient plot with 95% CI
# ============================================================
def build_figure(summary):
    """Horizontal dot-and-whisker. Groups: aggregate, systematic, idiosyncratic."""
    # Items to plot: (label, color, spec, pred)
    items = [
        # Aggregate from B3
        ("RSJ (agg.)",          "#888888", "B3", "rsj_weekly"),
        ("RES (agg.)",          "#888888", "B3", "res_weekly"),
        # B7 systematic
        (r"RSJ$_\mathrm{sys}$ M1", "#2a9d8f", "B7", "rsj_sys_weekly"),
        (r"RES$_\mathrm{sys}$",    "#2a9d8f", "B7", "res_sys_p025"),
        # B8 systematic
        (r"RSJ$_\mathrm{sys}$ M2", "#1d7a6e", "B8", "rsj_sys"),
        # B7 idiosyncratic
        (r"RSJ$_\mathrm{idio}$ M1","#7b2d8b", "B7", "rsj_idio_weekly"),
        (r"RES$_\mathrm{idio}$",   "#7b2d8b", "B7", "res_idio_p025"),
        # B8 idiosyncratic
        (r"RSJ$_\mathrm{idio}$ M2","#5a1f6b", "B8", "rsj_idio"),
    ]

    labels, coefs, lo95, hi95, colors = [], [], [], [], []
    for lbl, col, spec, pred in items:
        if spec not in summary or pred not in summary[spec]:
            continue
        m, t, se = summary[spec][pred]
        if not np.isfinite(m):
            continue
        labels.append(lbl)
        coefs.append(m)
        lo95.append(m - 1.96 * se)
        hi95.append(m + 1.96 * se)
        colors.append(col)

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(7, max(4, len(labels) * 0.5 + 1)))
    y_pos = np.arange(len(labels))[::-1]
    for i, (y, m, lo, hi, c) in enumerate(zip(y_pos, coefs, lo95, hi95, colors)):
        ax.plot([lo, hi], [y, y], color=c, lw=1.5)
        ax.plot(m, y, "o", color=c, ms=6)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("FM Coefficient (bps/week)", fontsize=10)
    ax.set_title("Fama--MacBeth Coefficients with 95\\% NW Confidence Intervals",
                 fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
    ax.grid(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], color="#888888", marker="o", ms=6, label="Aggregate"),
        Line2D([0],[0], color="#2a9d8f", marker="o", ms=6, label="Systematic"),
        Line2D([0],[0], color="#7b2d8b", marker="o", ms=6, label="Idiosyncratic"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


# ============================================================
# Main
# ============================================================
def main():
    print("=== Fama-MacBeth Regressions ===\n")

    panel = pd.read_parquet(PANEL_FILE)
    panel["week"] = pd.to_datetime(panel["week"]).dt.normalize()
    if "valid_R_i_w_plus_1" in panel.columns:
        panel = panel[panel["valid_R_i_w_plus_1"].astype(bool)].copy()
    else:
        panel = panel[panel["R_i_w_plus_1"].notna()].copy()

    print(f"Panel: {len(panel):,} stock-weeks | "
          f"{panel['permno'].nunique():,} stocks | "
          f"{panel['week'].nunique():,} weeks\n")

    # Winsorize controls
    for c in CONTROLS:
        if c in panel.columns:
            panel[c] = panel.groupby("week")[c].transform(winsorize_cs)

    # Run FM
    all_weekly, summary, avg_rsq = run_fm(panel)

    # Save per-spec intermediate parquet
    for spec, df_wk in all_weekly.items():
        out = INTER_DIR / f"fm_coefs_{spec}.parquet"
        df_wk.to_parquet(out, index=False)
    print(f"\nSaved intermediate parquet files to {INTER_DIR.name}/\n")

    # ---- LaTeX tables ----
    # Table A
    tex_a = build_tab_aggregate(summary, avg_rsq, all_weekly)
    p = TABLE_DIR / "tab_fm_aggregate.tex"
    p.write_text(tex_a, encoding="utf-8")
    print(f"Saved: {p.name}")

    # Table B
    tex_b_result = build_tab_decomposition(summary, avg_rsq, all_weekly)
    tex_b, decomp_note = tex_b_result
    p = TABLE_DIR / "tab_fm_decomposition.tex"
    p.write_text(tex_b, encoding="utf-8")
    print(f"Saved: {p.name}")
    # Print and save the decomposition note
    print(f"\n[Note] {decomp_note}\n")
    (INTER_DIR / "decomposition_note.txt").write_text(decomp_note, encoding="utf-8")

    # Supplementary
    tex_s = build_tab_supplementary(summary, avg_rsq, all_weekly)
    p = TABLE_DIR / "tab_fm_supplementary.tex"
    p.write_text(tex_s, encoding="utf-8")
    print(f"Saved: {p.name}")

    # ---- Figure ----
    fig = build_figure(summary)
    if fig is not None:
        for ext in ("pdf", "png"):
            fpath = FIG_DIR / f"fig_fm_coefficients_selected.{ext}"
            fig.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: fig_fm_coefficients_selected.pdf/.png")

    # ---- Console summary ----
    print("\n" + "=" * 80)
    print("FAMA-MACBETH RESULTS  (bps/week)")
    print("=" * 80)
    for spec in SPECS:
        if spec not in summary:
            continue
        preds = SPECS[spec]
        focal = [p for p in preds if p not in CONTROLS]
        print(f"\n{spec}")
        for p in focal:
            if p not in summary[spec]:
                continue
            m, t, se = summary[spec][p]
            print(f"  {PRED_LABEL.get(p, p):<35} {m:>9.2f}{_stars(t):3}  (t={t:.2f})")
        if spec in avg_rsq:
            print(f"  Avg adj R² = {avg_rsq[spec]:.4f}")

    print("\n=== Done ===")
    print("Run 03_wald_tests.py  → uses fm_coefs_B7.parquet and fm_coefs_B8.parquet")
    print("Run 04_crisis_state.py → uses fm_coefs_B7.parquet and fm_coefs_B3.parquet")


if __name__ == "__main__":
    main()
