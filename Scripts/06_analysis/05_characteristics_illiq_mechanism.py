"""
Descriptive mechanism analysis: are RSJ-based characteristics related to Amihud illiquidity?

For each week t and characteristic C in RSJ-family measures, estimate cross-sectional OLS:
    C_{i,t} = alpha_t + beta_t * illiq_{i,t} + gamma_t' Controls_{i,t} + eps_{i,t}

Then average weekly beta_t using Newey-West t-statistics (same convention as 02_fama_macbeth.py).

This script is intentionally separate from return-predictability FM regressions.
It does not modify or replace any existing main specification.

Inputs:
  data_final/panel/weekly_panel.parquet

Outputs:
  data_final/fama_macbeth/fm_char_illiq_results.parquet
  data_final/fama_macbeth/fm_char_illiq_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================================================
# Settings
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
PANEL_FILE = ROOT / "data_final" / "panel" / "weekly_panel.parquet"
OUTPUT_DIR = ROOT / "data_final" / "fama_macbeth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINSOR_LOW = 0.01
WINSOR_HIGH = 0.99
NW_LAGS = 6
MIN_STOCKS = 50

# Controls follow the project standard, with illiq as the focal regressor.
BASE_CONTROLS = ["me", "bm", "mom", "rev", "ivol"]
FOCAL_VAR = "illiq"

# RSJ-family characteristics already used in the project.
CHARACTERISTICS = {
    "rsj_weekly": "RSJ",
    "rsj_idio": "RSJ_idio (M2)",
    "rsj_sys": "RSJ_sys (M2)",
    "rsj_idio_weekly": "RSJ_idio (M1)",
    "rsj_sys_weekly": "RSJ_sys (M1)",
}


# ============================================================
# Helpers
# ============================================================
def winsorize_cross_section(s: pd.Series) -> pd.Series:
    lo = s.quantile(WINSOR_LOW)
    hi = s.quantile(WINSOR_HIGH)
    return s.clip(lo, hi)


def nw_average(series: np.ndarray, maxlags: int) -> tuple[float, float, float]:
    y = series[~np.isnan(series)]
    t = len(y)
    if t < 10:
        return np.nan, np.nan, np.nan

    ones = np.ones((t, 1))
    res = sm.OLS(y, ones).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags, "use_correction": True},
    )
    return float(res.params[0]), float(res.tvalues[0]), float(res.bse[0])


def run_weekly_cross_section(df_week: pd.DataFrame, y_col: str, x_cols: list[str]) -> dict | None:
    cols = [y_col] + x_cols
    sub = df_week[cols].dropna()
    if len(sub) < MIN_STOCKS:
        return None

    y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
    x_raw = sub[x_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x = sm.add_constant(x_raw, has_constant="add")

    if not np.isfinite(y).all() or not np.isfinite(x).all():
        return None

    try:
        res = sm.OLS(y, x).fit()
    except Exception:
        return None

    out = {"n_obs": int(len(sub))}
    for i, var in enumerate(x_cols):
        out[var] = float(res.params[i + 1])
    return out


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=== RSJ Characteristics on Illiquidity (Descriptive FM) ===\n")

    panel = pd.read_parquet(PANEL_FILE)
    panel["week"] = pd.to_datetime(panel["week"]).dt.normalize()

    # Align with main FM sample convention where possible.
    if "valid_R_i_w_plus_1" in panel.columns:
        panel = panel[panel["valid_R_i_w_plus_1"].astype(bool)].copy()

    print(f"Panel rows before variable filter: {len(panel):,}")

    # Match control preprocessing convention from 02_fama_macbeth.py:
    # winsorize controls cross-sectionally each week.
    for col in [FOCAL_VAR] + BASE_CONTROLS:
        if col in panel.columns:
            panel[col] = panel.groupby("week")[col].transform(winsorize_cross_section)

    weeks = sorted(panel["week"].dropna().unique())

    all_weekly_rows = []
    summary_rows = []

    for char_col, char_label in CHARACTERISTICS.items():
        if char_col not in panel.columns:
            print(f"Skipping {char_col}: column not found in panel.")
            continue

        predictors = [FOCAL_VAR] + [c for c in BASE_CONTROLS if c != char_col]
        missing = [p for p in predictors if p not in panel.columns]
        if missing:
            print(f"Skipping {char_col}: missing predictors {missing}.")
            continue

        beta_illiq = []
        n_obs_by_week = []
        valid_weeks = 0

        for week in weeks:
            df_week = panel[panel["week"] == week]
            result = run_weekly_cross_section(df_week, char_col, predictors)
            if result is None:
                continue

            valid_weeks += 1
            beta_illiq.append(result[FOCAL_VAR])
            n_obs_by_week.append(result["n_obs"])

            row = {
                "analysis": "char_on_illiq",
                "characteristic": char_col,
                "label": char_label,
                "week": week,
                "n_obs": result["n_obs"],
            }
            for p in predictors:
                row[p] = result[p]
            all_weekly_rows.append(row)

        beta_arr = np.array(beta_illiq, dtype=float)
        mean_beta, t_stat, se_beta = nw_average(beta_arr, NW_LAGS)

        summary_rows.append(
            {
                "analysis": "char_on_illiq",
                "characteristic": char_col,
                "label": char_label,
                "coef_illiq": mean_beta,
                "t_stat_illiq": t_stat,
                "se_illiq": se_beta,
                "n_periods": int(valid_weeks),
                "n_obs_total": int(np.sum(n_obs_by_week)) if n_obs_by_week else 0,
                "n_obs_avg": float(np.mean(n_obs_by_week)) if n_obs_by_week else np.nan,
                "controls_included": "Yes",
                "controls": ",".join(BASE_CONTROLS),
                "estimation": "Fama-MacBeth weekly + Newey-West",
            }
        )

        print(
            f"{char_col:<16} | weeks={valid_weeks:>3} | "
            f"beta_illiq={mean_beta:>9.5f} | t={t_stat:>6.2f}"
        )

    weekly_out = pd.DataFrame(all_weekly_rows)
    summary_out = pd.DataFrame(summary_rows)

    weekly_path = OUTPUT_DIR / "fm_char_illiq_results.parquet"
    summary_path = OUTPUT_DIR / "fm_char_illiq_summary.csv"

    weekly_out.to_parquet(weekly_path, index=False)
    summary_out.to_csv(summary_path, index=False, float_format="%.6f")

    print(f"\nSaved weekly coefficients: {weekly_path}")
    print(f"Saved summary table: {summary_path}")


if __name__ == "__main__":
    main()
