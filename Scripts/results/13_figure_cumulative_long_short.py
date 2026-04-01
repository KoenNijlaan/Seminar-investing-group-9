from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from results_utils import DATA_FINAL, ensure_output_dirs, FIGURES_DIR, TABLES_DIR


MIN_STOCKS_PER_WEEK = 50

# signal, direction, label
# L-H means long lowest quintile, short highest quintile
# H-L means long highest quintile, short lowest quintile
TARGETS = [
    ("rsj_weekly", "L-H", "RSJ total (L-H)"),
    ("rsj_idio", "L-H", "RSJ idiosyncratic M2 (L-H)"),
    ("res_weekly", "H-L", "RES total (H-L)"),
    ("res_idio_p025", "H-L", "RES idiosyncratic (H-L)"),
]


def one_spread_series(panel: pd.DataFrame, signal: str, direction: str) -> pd.DataFrame:
    d = panel[["week", signal, "R_i_w_plus_1"]].copy()
    d[signal] = pd.to_numeric(d[signal], errors="coerce")
    d["R_i_w_plus_1"] = pd.to_numeric(d["R_i_w_plus_1"], errors="coerce")
    d = d.dropna(subset=["week", signal, "R_i_w_plus_1"])

    def weekly_spread(grp: pd.DataFrame) -> float:
        if len(grp) < MIN_STOCKS_PER_WEEK:
            return np.nan
        try:
            q = pd.qcut(grp[signal], 5, labels=False, duplicates="drop")
        except Exception:
            return np.nan
        if q.isna().all():
            return np.nan

        g = grp.assign(q=q + 1)
        low = g.loc[g["q"] == 1, "R_i_w_plus_1"].mean()
        high = g.loc[g["q"] == 5, "R_i_w_plus_1"].mean()

        if not np.isfinite(low) or not np.isfinite(high):
            return np.nan

        if direction == "L-H":
            return float(low - high)
        return float(high - low)

    s = d.groupby("week", sort=True).apply(weekly_spread).rename("spread_ret").reset_index()
    s = s.dropna(subset=["spread_ret"]).sort_values("week")

    # cum return from weekly simple returns
    s["cum_ret"] = (1.0 + s["spread_ret"]).cumprod() - 1.0
    return s


def main() -> None:
    ensure_output_dirs()

    panel_path = DATA_FINAL / "panel" / "weekly_panel.parquet"
    panel = pd.read_parquet(panel_path)
    panel["week"] = pd.to_datetime(panel["week"], errors="coerce")

    out_rows = []
    fig, ax = plt.subplots(figsize=(12, 6))

    for signal, direction, label in TARGETS:
        if signal not in panel.columns:
            continue

        s = one_spread_series(panel, signal, direction)
        if s.empty:
            continue

        ax.plot(s["week"], s["cum_ret"] * 100.0, linewidth=1.8, label=label)

        tmp = s.copy()
        tmp["signal"] = signal
        tmp["direction"] = direction
        tmp["label"] = label
        out_rows.append(tmp)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Cumulative Long-Short Performance (EW, Weekly Rebalanced)")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("Week")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_cumulative_long_short_performance.png", dpi=200)
    plt.close()

    if out_rows:
        out = pd.concat(out_rows, ignore_index=True)
        out.to_csv(TABLES_DIR / "series_cumulative_long_short.csv", index=False)

    print("Saved figure: fig_cumulative_long_short_performance.png")
    print("Saved series: series_cumulative_long_short.csv")


if __name__ == "__main__":
    main()
