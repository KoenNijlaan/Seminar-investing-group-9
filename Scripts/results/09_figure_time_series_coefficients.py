from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from results_utils import DATA_FINAL, ensure_output_dirs, FIGURES_DIR


NBER_RECESSION_PERIODS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


def add_recession_shading(ax) -> None:
    for start, end in NBER_RECESSION_PERIODS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color="gray")


def main() -> None:
    ensure_output_dirs()

    path = DATA_FINAL / "fama_macbeth" / "fm_results.parquet"
    df = pd.read_parquet(path)
    df["week"] = pd.to_datetime(df["week"])

    # Use B3 as baseline joint total model time-series
    d = df[df["spec"] == "B3_rsj_res_ctrl"].copy()
    if d.empty:
        print("Spec B3_rsj_res_ctrl not found in fm_results.parquet.")
        return

    d = d.sort_values("week")
    d["rsj_roll52_bps"] = d["rsj_weekly"].rolling(52, min_periods=26).mean() * 10000.0
    d["res_roll52_bps"] = d["res_weekly"].rolling(52, min_periods=26).mean() * 10000.0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d["week"], d["rsj_roll52_bps"], label="RSJ beta (52w rolling mean)")
    ax.plot(d["week"], d["res_roll52_bps"], label="RES beta (52w rolling mean)")
    add_recession_shading(ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Time-Varying FM Slopes (B3, Rolling 52-Week Mean)")
    ax.set_ylabel("Coefficient (bps)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_time_series_fm_betas_b3.png", dpi=200)
    plt.close()

    print("Saved figure: fig_time_series_fm_betas_b3.png")


if __name__ == "__main__":
    main()
