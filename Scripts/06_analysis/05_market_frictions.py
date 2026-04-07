"""
Market Frictions and Limits-to-Arbitrage Analysis.

Two complementary Fama-MacBeth tests examining whether return predictability
and idiosyncratic risk exposures are amplified by illiquidity frictions.

Test 1 — Interaction FM regressions
    For each decomposed predictor Z, estimate the weekly cross-section:

        R_{i,w+1} = alpha + beta1*Z + beta2*illiq + beta3*(Z x illiq)
                    + gamma'*X + eps

    where X excludes illiq (already in the regression explicitly).
    The interaction term beta3 tests whether return predictability is
    amplified among stocks facing greater limits to arbitrage.

Test 2 — Signal-on-Friction regressions  (teacher's suggestion)
    For each idiosyncratic signal Z_idio, estimate weekly cross-sections:

        Z_idio_{i,w} = delta0 + delta1*illiq + gamma'*X + eps_{i,w}

    then average delta1 with Newey-West SEs.  A significant delta1 indicates
    that illiquid stocks exhibit more extreme idiosyncratic downside risk
    exposures — consistent with the limits-to-arbitrage narrative.

Output:
    data_final/market_frictions/
        friction_interaction_summary.csv   — beta1/2/3 averaged with NW t-stats
        signal_on_friction_summary.csv     — delta1 averaged with NW t-stats
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ============================================================
# Settings
# ============================================================
ROOT       = Path(__file__).resolve().parents[2]
PANEL_FILE = ROOT / "data_final" / "panel" / "weekly_panel.parquet"
OUTPUT_DIR = ROOT / "data_final" / "market_frictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NW_LAGS    = 6      # Newey-West lags (weeks) — consistent with main FM
MIN_STOCKS = 50     # minimum cross-section per week
WINSOR_LOW  = 0.01
WINSOR_HIGH = 0.99

CONTROLS = ["me", "bm", "mom", "rev", "ivol", "illiq"]

# Signals for Test 1 (interaction regressions)
INTERACTION_SIGNALS = [
    "rsj_weekly",
    "rsj_idio",
    "rsj_sys",
    "rsj_idio_weekly",
    "rsj_sys_weekly",
    "res_weekly",
    "res_idio_p025",
    "res_sys_p025",
]

# Signals for Test 2 (signal-on-friction)
IDIO_SIGNALS = [
    "rsj_idio",
    "rsj_idio_weekly",
    "res_idio_p025",
]

SIGNAL_LABELS = {
    "rsj_weekly":      "RSJ Total",
    "rsj_idio":        "RSJ Idio (M2)",
    "rsj_sys":         "RSJ Sys (M2)",
    "rsj_idio_weekly": "RSJ Idio (M1)",
    "rsj_sys_weekly":  "RSJ Sys (M1)",
    "res_weekly":      "RES Total",
    "res_idio_p025":   "RES Idio",
    "res_sys_p025":    "RES Sys",
}


# ============================================================
# Helpers
# ============================================================
def winsorize_cs(s: pd.Series, low: float = 0.01, high: float = 0.99) -> pd.Series:
    lo = s.quantile(low)
    hi = s.quantile(high)
    return s.clip(lo, hi)


def nw_mean_t(values: np.ndarray, maxlags: int) -> tuple[float, float, float]:
    """Newey-West averaged mean. Returns (mean, t_stat, se)."""
    y = values[np.isfinite(values)]
    T = len(y)
    if T < 10:
        return np.nan, np.nan, np.nan
    ones = np.ones((T, 1))
    res = sm.OLS(y, ones).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags, "use_correction": True},
    )
    return float(res.params[0]), float(res.tvalues[0]), float(res.bse[0])


def run_ols_week(
    df_week: pd.DataFrame,
    dep: str,
    indep: list[str],
) -> dict | None:
    """One-week OLS. Returns {var: coef, 'n_obs': int} or None."""
    cols = [dep] + indep
    sub = df_week[cols].dropna()
    if len(sub) < MIN_STOCKS:
        return None

    y = pd.to_numeric(sub[dep], errors="coerce").to_numpy(dtype=float)
    X_raw = sub[indep].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = sm.add_constant(X_raw, has_constant="add")

    if not (np.isfinite(y).all() and np.isfinite(X).all()):
        return None

    try:
        res = sm.OLS(y, X).fit()
    except Exception:
        return None

    out = {"n_obs": int(len(sub))}
    for i, var in enumerate(indep):
        out[var] = float(res.params[i + 1])
    return out


# ============================================================
# Test 1: Interaction FM Regressions
# ============================================================
def run_interaction_fm(panel: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Test 1: Interaction FM Regressions ===")
    weeks = sorted(panel["week"].unique())

    # controls in the interaction spec: everything except illiq
    # (illiq enters explicitly as beta2, so exclude to avoid collinearity)
    ctrl_wo_illiq = [c for c in CONTROLS if c != "illiq"]

    summary_rows = []
    for signal in INTERACTION_SIGNALS:
        if signal not in panel.columns:
            print(f"  SKIP {signal}: not in panel")
            continue

        inter_col = f"__{signal}_x_illiq"
        betas_z:     list[float] = []
        betas_illiq: list[float] = []
        betas_inter: list[float] = []

        n_valid = 0
        for week in weeks:
            wdf = panel[panel["week"] == week].copy()

            # Cross-sectional winsorize of the signal each week
            z_vals = pd.to_numeric(wdf[signal], errors="coerce")
            if z_vals.notna().sum() < MIN_STOCKS:
                continue
            q_lo = z_vals.quantile(WINSOR_LOW)
            q_hi = z_vals.quantile(WINSOR_HIGH)
            wdf = wdf.copy()
            wdf[signal] = z_vals.clip(q_lo, q_hi)

            # Interaction term
            wdf[inter_col] = wdf[signal] * wdf["illiq"]

            indep = [signal, "illiq", inter_col] + ctrl_wo_illiq
            result = run_ols_week(wdf, "R_i_w_plus_1", indep)
            if result is None:
                continue

            n_valid += 1
            betas_z.append(result[signal])
            betas_illiq.append(result["illiq"])
            betas_inter.append(result[inter_col])

        print(f"  {signal}: {n_valid} valid weeks")

        mean_z,     t_z,     se_z     = nw_mean_t(np.array(betas_z,     dtype=float), NW_LAGS)
        mean_illiq, t_illiq, se_illiq = nw_mean_t(np.array(betas_illiq, dtype=float), NW_LAGS)
        mean_inter, t_inter, se_inter = nw_mean_t(np.array(betas_inter, dtype=float), NW_LAGS)

        summary_rows.append({
            "signal":        signal,
            "label":         SIGNAL_LABELS.get(signal, signal),
            "n_weeks":       n_valid,
            "mean_z":        mean_z,
            "mean_z_bps":    mean_z * 10_000 if pd.notna(mean_z) else np.nan,
            "t_z":           t_z,
            "se_z":          se_z,
            "mean_illiq":    mean_illiq,
            "mean_illiq_bps":mean_illiq * 10_000 if pd.notna(mean_illiq) else np.nan,
            "t_illiq":       t_illiq,
            "se_illiq":      se_illiq,
            "mean_inter":    mean_inter,
            "mean_inter_bps":mean_inter * 10_000 if pd.notna(mean_inter) else np.nan,
            "t_inter":       t_inter,
            "se_inter":      se_inter,
        })

    out = pd.DataFrame(summary_rows)
    path = OUTPUT_DIR / "friction_interaction_summary.csv"
    out.to_csv(path, index=False, float_format="%.8f")
    print(f"\nSaved: {path}")
    return out


# ============================================================
# Test 2: Signal-on-Friction Regressions
# ============================================================
def run_signal_on_friction(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the full panel (not restricted to valid forward return) so we
    maximise cross-sectional variation in idiosyncratic signals.
    """
    print("\n=== Test 2: Signal-on-Friction Regressions ===")
    weeks = sorted(panel["week"].unique())

    ctrl_wo_illiq = [c for c in CONTROLS if c != "illiq"]
    indep_univ = ["illiq"]
    indep_ctrl = ["illiq"] + ctrl_wo_illiq

    summary_rows = []
    for signal in IDIO_SIGNALS:
        if signal not in panel.columns:
            print(f"  SKIP {signal}: not in panel")
            continue

        deltas_univ: list[float] = []
        deltas_ctrl: list[float] = []
        n_valid = 0

        for week in weeks:
            wdf = panel[panel["week"] == week]

            res_u = run_ols_week(wdf, signal, indep_univ)
            res_c = run_ols_week(wdf, signal, indep_ctrl)
            if res_u is None or res_c is None:
                continue

            n_valid += 1
            deltas_univ.append(res_u["illiq"])
            deltas_ctrl.append(res_c["illiq"])

        print(f"  {signal}: {n_valid} valid weeks")

        mean_u, t_u, se_u = nw_mean_t(np.array(deltas_univ, dtype=float), NW_LAGS)
        mean_c, t_c, se_c = nw_mean_t(np.array(deltas_ctrl, dtype=float), NW_LAGS)

        summary_rows.append({
            "signal":         signal,
            "label":          SIGNAL_LABELS.get(signal, signal),
            "n_weeks":        n_valid,
            "mean_delta_univ":mean_u,
            "t_delta_univ":   t_u,
            "se_delta_univ":  se_u,
            "mean_delta_ctrl":mean_c,
            "t_delta_ctrl":   t_c,
            "se_delta_ctrl":  se_c,
        })

    out = pd.DataFrame(summary_rows)
    path = OUTPUT_DIR / "signal_on_friction_summary.csv"
    out.to_csv(path, index=False, float_format="%.8f")
    print(f"\nSaved: {path}")
    return out


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=== Market Frictions and Limits-to-Arbitrage Analysis ===\n")

    panel = pd.read_parquet(PANEL_FILE)
    panel["week"] = pd.to_datetime(panel["week"]).dt.normalize()

    print(f"Full panel: {len(panel):,} rows | "
          f"{panel['permno'].nunique():,} stocks | "
          f"{panel['week'].nunique():,} weeks")

    # Winsorize controls for the full panel (used in both tests)
    print("Winsorizing controls cross-sectionally (1/99 pct per week)...")
    for ctrl in CONTROLS:
        if ctrl in panel.columns:
            panel[ctrl] = (
                panel.groupby("week")[ctrl]
                .transform(winsorize_cs)
            )

    # --- Test 1 uses only valid forward-return rows ---
    if "valid_R_i_w_plus_1" in panel.columns:
        panel_fwd = panel[panel["valid_R_i_w_plus_1"].astype(bool)].copy()
    else:
        panel_fwd = panel[panel["R_i_w_plus_1"].notna()].copy()

    print(f"Forward-return valid panel: {len(panel_fwd):,} rows")

    df_inter = run_interaction_fm(panel_fwd)

    # --- Test 2 uses the full panel (no restriction on fwd return) ---
    df_sof = run_signal_on_friction(panel)

    # Console summary
    print("\n" + "=" * 70)
    print("INTERACTION REGRESSIONS")
    print("  signal | N_wks | beta_Z (bps) t | beta_illiq (bps) t | beta_inter (bps) t")
    print("=" * 70)
    for _, r in df_inter.iterrows():
        print(
            f"  {r['label']:<20} {r['n_weeks']:>5} | "
            f"{r['mean_z_bps']:>8.2f} (t={r['t_z']:>5.2f}) | "
            f"{r['mean_illiq_bps']:>8.2f} (t={r['t_illiq']:>5.2f}) | "
            f"{r['mean_inter_bps']:>8.2f} (t={r['t_inter']:>5.2f})"
        )

    print("\n" + "=" * 70)
    print("SIGNAL-ON-FRICTION")
    print("  signal | N_wks | delta1_univ (t) | delta1_ctrl (t)")
    print("=" * 70)
    for _, r in df_sof.iterrows():
        print(
            f"  {r['label']:<22} {r['n_weeks']:>5} | "
            f"{r['mean_delta_univ']:>10.6f} (t={r['t_delta_univ']:>5.2f}) | "
            f"{r['mean_delta_ctrl']:>10.6f} (t={r['t_delta_ctrl']:>5.2f})"
        )

    print("\n=== Done. Run Scripts/results/15_table_market_frictions.py to format tables. ===")


if __name__ == "__main__":
    main()
