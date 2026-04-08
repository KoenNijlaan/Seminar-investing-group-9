"""
Illiquidity mechanism analysis. NW(6) throughout.

Requested regressions:
  1) RSJ idiosyncratic (M1) on illiquidity (+ controls)
  2) RES idiosyncratic on illiquidity (+ controls)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PANEL_FILE = ROOT / "data_final" / "panel" / "weekly_panel.parquet"
TABLE_DIR = ROOT / "data_final" / "results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

NW_LAGS = 6
WINSOR_LOW = 0.01
WINSOR_HIGH = 0.99
MIN_STOCKS = 30

CONTROLS_A = ["me", "bm", "mom", "rev", "ivol"]
IDIO_SIGNALS = [
    ("rsj_idio_weekly", "RSJ idio (M1)"),
    ("res_idio_p025", "RES idio"),
]


def winsorize_cs(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile(WINSOR_LOW), s.quantile(WINSOR_HIGH)
    return s.clip(lo, hi)


def nw_mean_t(arr: np.ndarray, lags: int):
    y = np.asarray(arr, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return np.nan, np.nan
    res = sm.OLS(y, np.ones((len(y), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    return float(res.params[0]), float(res.tvalues[0])


def run_ols_week(df_week: pd.DataFrame, dep: str, indep: list[str], min_n: int = MIN_STOCKS):
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
    return {indep[i]: float(res.params[i + 1]) for i in range(len(indep))}


def _stars(t):
    if not np.isfinite(t):
        return ""
    t = abs(float(t))
    if t > 3.29:
        return "***"
    if t > 2.58:
        return "**"
    if t > 1.96:
        return "*"
    return ""


def _fmt(val, t=None, dec=4):
    if not np.isfinite(val):
        return ""
    out = f"{float(val):.{dec}f}"
    if t is not None:
        out += _stars(t)
    return out


def run_mechanism(panel: pd.DataFrame):
    weeks = sorted(panel["week"].unique())
    rows = []

    for signal_col, signal_label in IDIO_SIGNALS:
        if signal_col not in panel.columns:
            continue

        delta_univ = []
        delta_ctrl = []
        n_valid = 0

        for week in weeks:
            wdf = panel[panel["week"] == week]
            r_u = run_ols_week(wdf, signal_col, ["illiq"])
            r_c = run_ols_week(wdf, signal_col, ["illiq"] + CONTROLS_A)
            if r_u is None or r_c is None:
                continue
            n_valid += 1
            delta_univ.append(r_u["illiq"])
            delta_ctrl.append(r_c["illiq"])

        m_u, t_u = nw_mean_t(np.array(delta_univ, dtype=float), NW_LAGS)
        m_c, t_c = nw_mean_t(np.array(delta_ctrl, dtype=float), NW_LAGS)

        rows.append(
            {
                "label": signal_label,
                "n_weeks": n_valid,
                "delta_univ": m_u,
                "t_delta_univ": t_u,
                "delta_ctrl": m_c,
                "t_delta_ctrl": t_c,
            }
        )

    return rows


def build_latex(rows: list[dict]) -> str:
    lines = []
    lines += [r"\begin{table}[htbp]", r"\centering", r"\begin{threeparttable}"]
    lines += [r"\caption{Illiquidity and Idiosyncratic Risk Characteristics}"]
    lines += [r"\label{tab:illiquidity}"]
    lines += [r"\begin{tabular}{lccccc}"]
    lines += [r"\toprule"]
    lines += [r"Dependent var. & Univ. $\delta$ & $t$ & Controlled $\delta$ & $t$ & $N_{weeks}$ \\"]
    lines += [r"\midrule"]

    for r in rows:
        lines.append(
            f"{r['label']} & {_fmt(r['delta_univ'], r['t_delta_univ'])} & ({r['t_delta_univ']:.2f}) & "
            f"{_fmt(r['delta_ctrl'], r['t_delta_ctrl'])} & ({r['t_delta_ctrl']:.2f}) & {r['n_weeks']} \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += [r"\begin{tablenotes}", r"\small"]
    lines += [
        r"\item Notes: Weekly cross-sectional regressions with idiosyncratic risk characteristics"
        r" as dependent variables and log-Amihud illiquidity as focal regressor."
        r" Controlled specification includes log(ME), BM, MOM, REV, and IVOL."
        r" Illiquidity is excluded from the control block because it is the focal regressor."
        r" $t$-statistics use Newey--West HAC standard errors with 6 lags."
        r" $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$."
    ]
    lines += [r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    print("=== Illiquidity Mechanism Analysis ===\n")

    panel = pd.read_parquet(PANEL_FILE)
    panel["week"] = pd.to_datetime(panel["week"]).dt.normalize()

    for c in ["illiq"] + CONTROLS_A:
        if c in panel.columns:
            panel[c] = panel.groupby("week")[c].transform(winsorize_cs)

    rows = run_mechanism(panel)
    tex = build_latex(rows)

    out = TABLE_DIR / "tab_illiquidity.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"Saved: {out.name}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
