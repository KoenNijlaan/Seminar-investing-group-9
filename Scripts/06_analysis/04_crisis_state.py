"""
Crisis-state Fama-MacBeth analysis. Systematic M1 specification. NW(6) throughout.

Interaction-term approach
─────────────────────────
The FM interaction regression is:

    R_{i,w+1} = α_w + β₁ Z_{i,w} + β₃ (Z_{i,w} × nber_w) + γ_w' X_{i,w} + ε

Because nber_w is a SCALAR constant for all i in week w, the interaction term
Z × nber_w is either 0 (expansion week) or Z (recession week). This makes
nber_w and Z×nber_w collinear with the intercept and Z respectively within
any single cross-section. The FM solution is to REPARAMETRISE:

    Z_exp   = Z × (1 − nber_w)   → equals Z in expansion weeks, 0 in crisis weeks
    Z_cri   = Z × nber_w          → equals 0 in expansion weeks, Z in crisis weeks

Each week only ONE of these has cross-sectional variation, so the two columns
are orthogonal across the full time series and separately identifiable:

  • FM average of slope on Z_exp  →  β₁           (normal/expansion slope)
  • FM average of slope on Z_cri  →  β₁ + β₃      (crisis slope)
  • β₃ = crisis slope − normal slope                (interaction/crisis increment)
  • t(β₃): from OLS regression β̂_Z_cri − β̂_Z_exp ~ nber_w with NW(6) SEs

Joint crisis-state specification:
        R_{i,w+1} = α_w
                            + β1 RSJ_sys_{i,w}
                            + β2 RES_sys_{i,w}
                            + β3 C_w
                            + β4 (RSJ_sys_{i,w} × C_w)
                            + β5 (RES_sys_{i,w} × C_w)
                            + γ'_w X_{i,w} + ε_{i,w+1}

where C_w is the NBER recession indicator (week-level).

Implemented via weekly reparameterisation:
    expansion weeks: regress on RSJ_sys and RES_sys + controls
    crisis weeks:    regress on RSJ_sys and RES_sys + controls
Then recover interaction increments as crisis-minus-expansion slopes with NW(6) tests.

NBER recession periods overlapping the sample (1993–2024):
    2001-03-01 to 2001-11-30
    2007-12-01 to 2009-06-30
    2020-02-01 to 2020-04-30

Outputs:
    data_final/results/tables/tab_crisis_state.tex
    data_final/results/figures/fig_rolling_fm_decomp_idio.{pdf,png}
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

CONTROLS = ["me", "bm", "mom", "rev", "ivol", "illiq"]

SYS_PREDS = ["rsj_sys_weekly", "res_sys_p025"]
SYS_LABEL = {
    "rsj_sys_weekly": "RSJ systematic (M1)",
    "res_sys_p025": "RES systematic",
}

NBER_PERIODS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


# ============================================================
# Helpers
# ============================================================
def winsorize_cs(s):
    lo, hi = s.quantile(WINSOR_LOW), s.quantile(WINSOR_HIGH)
    return s.clip(lo, hi)


def build_nber_indicator(week_series):
    week = pd.to_datetime(week_series).dt.normalize()
    c = pd.Series(False, index=week.index)
    for start, end in NBER_PERIODS:
        c |= (week >= pd.Timestamp(start)) & (week <= pd.Timestamp(end))
    return c.astype(int)


def nw_mean_t(arr, lags):
    y = np.asarray(arr, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return np.nan, np.nan
    res = sm.OLS(y, np.ones((len(y), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    return float(res.params[0]), float(res.tvalues[0])


def crisis_increment_test(betas_exp, betas_cri, nber_w_exp, nber_w_cri, lags):
    """
    Recover β₃ = (crisis slope) − (expansion slope) and its t-stat.

    Pool the weekly slope estimates together with the nber indicator:
        slope_w = β₁ + β₃ × nber_w + u_w
    Here slope_w is the FM slope on Z, which in expansion weeks comes from
    Z_exp and in crisis weeks from Z_cri (see module docstring).

    nber_w_exp = 0 for all expansion weeks
    nber_w_cri = 1 for all crisis weeks
    """
    slopes = np.concatenate([np.asarray(betas_exp), np.asarray(betas_cri)])
    nber   = np.concatenate([np.asarray(nber_w_exp), np.asarray(nber_w_cri)])
    ok = np.isfinite(slopes) & np.isfinite(nber)
    slopes, nber = slopes[ok], nber[ok]
    if len(slopes) < 10 or nber.sum() < 3:
        return np.nan, np.nan
    X   = sm.add_constant(nber.reshape(-1, 1), has_constant="add")
    res = sm.OLS(slopes, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    return float(res.params[1]), float(res.tvalues[1])


def run_weekly_cs_interaction(df_week, pred_col, nber_w, controls):
    """
    Run one week's cross-section using the reparametrised interaction approach.

    Returns (slope_exp, slope_cri) where exactly one of them is a float and
    the other is NaN (since only one variant of the predictor has variation
    in any given week).

    Z_exp = Z * (1 - nber_w)
    Z_cri = Z * nber_w
    """
    all_preds = [pred_col] + controls
    sub = df_week[["R_i_w_plus_1"] + all_preds].dropna()
    if len(sub) < MIN_STOCKS:
        return np.nan, np.nan

    y     = pd.to_numeric(sub["R_i_w_plus_1"], errors="coerce").to_numpy(dtype=float)
    Z     = pd.to_numeric(sub[pred_col],       errors="coerce").to_numpy(dtype=float)
    ctrl  = sub[controls].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if not (np.isfinite(y).all() and np.isfinite(Z).all() and np.isfinite(ctrl).all()):
        return np.nan, np.nan

    # Z_exp and Z_cri: exactly one has cross-sectional variation
    Z_exp = Z * (1 - nber_w)   # = Z in expansion weeks, 0 in crisis weeks
    Z_cri = Z * nber_w          # = 0 in expansion weeks, Z in crisis weeks

    # Use the variant that has actual variation; include both in the spec so the
    # coefficients on the OTHER controls are estimated on the same sample.
    if nber_w == 0:
        # Expansion week: Z_exp = Z, Z_cri = 0 → include Z_exp as predictor
        X_mat = sm.add_constant(
            np.column_stack([Z_exp, ctrl]), has_constant="add"
        )
        if not np.isfinite(X_mat).all():
            return np.nan, np.nan
        try:
            res = sm.OLS(y, X_mat).fit()
        except Exception:
            return np.nan, np.nan
        return float(res.params[1]), np.nan   # slope on Z_exp = β₁
    else:
        # Crisis week: Z_cri = Z, Z_exp = 0 → include Z_cri as predictor
        X_mat = sm.add_constant(
            np.column_stack([Z_cri, ctrl]), has_constant="add"
        )
        if not np.isfinite(X_mat).all():
            return np.nan, np.nan
        try:
            res = sm.OLS(y, X_mat).fit()
        except Exception:
            return np.nan, np.nan
        return np.nan, float(res.params[1])   # slope on Z_cri = β₁+β₃


def run_weekly_joint_systematic(df_week, nber_w, controls):
    """
    Weekly cross-section for the joint systematic specification.
    Returns dict with expansion and crisis slopes for RSJ_sys and RES_sys.
    """
    needed = ["R_i_w_plus_1"] + SYS_PREDS + controls
    sub = df_week[needed].dropna()
    if len(sub) < MIN_STOCKS:
        return None

    y = pd.to_numeric(sub["R_i_w_plus_1"], errors="coerce").to_numpy(dtype=float)
    z1 = pd.to_numeric(sub["rsj_sys_weekly"], errors="coerce").to_numpy(dtype=float)
    z2 = pd.to_numeric(sub["res_sys_p025"], errors="coerce").to_numpy(dtype=float)
    ctrl = sub[controls].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(y).all() and np.isfinite(z1).all() and np.isfinite(z2).all() and np.isfinite(ctrl).all()):
        return None

    X = sm.add_constant(np.column_stack([z1, z2, ctrl]), has_constant="add")
    if not np.isfinite(X).all():
        return None
    try:
        res = sm.OLS(y, X).fit()
    except Exception:
        return None

    if nber_w == 0:
        return {
            "rsj_exp": float(res.params[1]), "rsj_cri": np.nan,
            "res_exp": float(res.params[2]), "res_cri": np.nan,
        }
    return {
        "rsj_exp": np.nan, "rsj_cri": float(res.params[1]),
        "res_exp": np.nan, "res_cri": float(res.params[2]),
    }


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


# ============================================================
# LaTeX table
# ============================================================
def build_latex(results):
    lines = []
    lines += [r"\begin{table}[htbp]", r"\centering", r"\begin{threeparttable}"]
    lines += [r"\caption{Crisis-State Fama--MacBeth: Systematic Components with NBER Interactions (M1)}"]
    lines += [r"\label{tab:crisis_state}"]
    lines += [r"\begin{tabular}{lcccc}"]
    lines += [r"\toprule"]
    lines += [r"Predictor & Normal $\hat{\beta}_1$ & Crisis $\hat{\beta}_1{+}\hat{\beta}_3$"
              r" & $\hat{\beta}_3$ (interact.) & $t(\hat{\beta}_3)$ \\"]
    lines += [r" & (bps/week) & (bps/week) & (bps/week) & \\"]
    lines += [r"\midrule"]
    for r in results:
        lbl = r["label"].replace("_", r"\_")
        n_s = _fmt(r["normal_bps"],  r["t_normal"])
        c_s = _fmt(r["crisis_bps"],  r["t_crisis"])
        d_s = _fmt(r["delta_bps"],   r["t_delta"])
        t_s = f"{r['t_delta']:.2f}"  if np.isfinite(r["t_delta"])  else ""
        lines.append(f"{lbl} & {n_s} & {c_s} & {d_s} & {t_s} \\\\")
        t_n = f"({r['t_normal']:.2f})" if np.isfinite(r["t_normal"]) else ""
        t_c = f"({r['t_crisis']:.2f})" if np.isfinite(r["t_crisis"]) else ""
        lines.append(f" & {t_n} & {t_c} & & \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [r"\begin{tablenotes}", r"\small",
              r"\item Notes: Joint systematic specification with RSJ$_{\mathrm{sys}}$ (M1),"
              r" RES$_{\mathrm{sys}}$, and their interactions with NBER crisis state."
              r" Reported normal and crisis slopes are time-averaged FM coefficients"
              r" from expansion and recession weeks, respectively."
              r" The interaction increment is computed as crisis-minus-expansion slope"
              r" with NW(6) standard errors."
              r" NBER recessions: 2001-03 to 11, 2007-12 to 2009-06, 2020-02 to 04."
              r" Full control set $X$. All coefficients in bps/week."
              r" $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
              r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(lines)


# ============================================================
# Figure: rolling 52-week FM slopes for RSJ_sys and RES_sys
# ============================================================
def build_rolling_figure(df_b1):
    df = df_b1[["week", "rsj_sys_weekly", "res_sys_p025"]].copy()
    df = df.sort_values("week").set_index("week")
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
    ax.set_ylabel("FM slope (bps/week, 52-week rolling mean)", fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_title("Rolling FM Slopes: Systematic Components", fontsize=10, pad=8)
    ax.legend(fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
    ax.grid(False)
    fig.tight_layout()
    return fig


# ============================================================
# Main
# ============================================================
def main():
    print("=== Crisis-State Fama-MacBeth [Joint Systematic Spec] ===\n")

    panel = pd.read_parquet(PANEL_FILE)
    panel["week"] = pd.to_datetime(panel["week"]).dt.normalize()
    if "valid_R_i_w_plus_1" in panel.columns:
        panel = panel[panel["valid_R_i_w_plus_1"].astype(bool)].copy()
    else:
        panel = panel[panel["R_i_w_plus_1"].notna()].copy()

    for c in CONTROLS:
        if c in panel.columns:
            panel[c] = panel.groupby("week")[c].transform(winsorize_cs)

    panel["nber"] = build_nber_indicator(panel["week"])
    nber_map = (panel[["week","nber"]]
                .drop_duplicates()
                .set_index("week")["nber"]
                .to_dict())

    n_crisis    = sum(v == 1 for v in nber_map.values())
    n_expansion = sum(v == 0 for v in nber_map.values())
    print(f"Panel: {len(panel):,} stock-weeks | {panel['week'].nunique():,} weeks")
    print(f"  Expansion weeks: {n_expansion}  |  Crisis weeks: {n_crisis}\n")

    weeks = sorted(panel["week"].unique())
    for c in SYS_PREDS:
        if c not in panel.columns:
            raise KeyError(f"Missing required systematic predictor: {c}")

    rsj_exp, rsj_cri = [], []
    res_exp, res_cri = [], []
    nber_exp_vec, nber_cri_vec = [], []

    for week in weeks:
        df_w = panel[panel["week"] == week]
        nber_w = int(nber_map.get(week, 0))
        out = run_weekly_joint_systematic(df_w, nber_w, CONTROLS)
        if out is None:
            continue

        if np.isfinite(out["rsj_exp"]):
            rsj_exp.append(out["rsj_exp"])
            res_exp.append(out["res_exp"])
            nber_exp_vec.append(0)
        if np.isfinite(out["rsj_cri"]):
            rsj_cri.append(out["rsj_cri"])
            res_cri.append(out["res_cri"])
            nber_cri_vec.append(1)

    m_rsj_exp, t_rsj_exp = nw_mean_t(rsj_exp, NW_LAGS)
    m_rsj_cri, t_rsj_cri = nw_mean_t(rsj_cri, NW_LAGS)
    d_rsj, t_d_rsj = crisis_increment_test(rsj_exp, rsj_cri, nber_exp_vec, nber_cri_vec, NW_LAGS)

    m_res_exp, t_res_exp = nw_mean_t(res_exp, NW_LAGS)
    m_res_cri, t_res_cri = nw_mean_t(res_cri, NW_LAGS)
    d_res, t_d_res = crisis_increment_test(res_exp, res_cri, nber_exp_vec, nber_cri_vec, NW_LAGS)

    print("  Joint regression components:")
    print(f"    RSJ_sys normal={m_rsj_exp*10_000:.2f} (t={t_rsj_exp:.2f}), crisis={m_rsj_cri*10_000:.2f} (t={t_rsj_cri:.2f}), delta={d_rsj*10_000:.2f} (t={t_d_rsj:.2f})")
    print(f"    RES_sys normal={m_res_exp*10_000:.2f} (t={t_res_exp:.2f}), crisis={m_res_cri*10_000:.2f} (t={t_res_cri:.2f}), delta={d_res*10_000:.2f} (t={t_d_res:.2f})\n")

    results = [
        {
            "predictor": "rsj_sys_weekly",
            "label": SYS_LABEL["rsj_sys_weekly"],
            "normal_bps": m_rsj_exp * 10_000 if np.isfinite(m_rsj_exp) else np.nan,
            "t_normal": t_rsj_exp,
            "crisis_bps": m_rsj_cri * 10_000 if np.isfinite(m_rsj_cri) else np.nan,
            "t_crisis": t_rsj_cri,
            "delta_bps": d_rsj * 10_000 if np.isfinite(d_rsj) else np.nan,
            "t_delta": t_d_rsj,
        },
        {
            "predictor": "res_sys_p025",
            "label": SYS_LABEL["res_sys_p025"],
            "normal_bps": m_res_exp * 10_000 if np.isfinite(m_res_exp) else np.nan,
            "t_normal": t_res_exp,
            "crisis_bps": m_res_cri * 10_000 if np.isfinite(m_res_cri) else np.nan,
            "t_crisis": t_res_cri,
            "delta_bps": d_res * 10_000 if np.isfinite(d_res) else np.nan,
            "t_delta": t_d_res,
        },
    ]

    # LaTeX table
    tex = build_latex(results)
    p = TABLE_DIR / "tab_crisis_state.tex"
    p.write_text(tex, encoding="utf-8")
    print(f"Saved: {p.name}")

    # Rolling figure — use B7 intermediate (full M1 decomposition) if available
    b7_path = INTER_DIR / "fm_coefs_B7.parquet"
    if b7_path.exists():
        df_b7 = pd.read_parquet(b7_path)
        df_b7["week"] = pd.to_datetime(df_b7["week"])
        needed = ["week", "rsj_sys_weekly", "res_sys_p025"]
        if all(c in df_b7.columns for c in needed):
            df_src = df_b7[needed].dropna()
        else:
            df_src = None
    else:
        df_src = None

    if df_src is not None and len(df_src) > 52:
        fig = build_rolling_figure(df_src)
        for ext in ("pdf", "png"):
            fpath = FIG_DIR / f"fig_rolling_fm_decomp_idio.{ext}"
            fig.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: fig_rolling_fm_decomp_idio.pdf/.png")
    else:
        print("Intermediate B7 coef series not found — run 02_fama_macbeth.py first.")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
