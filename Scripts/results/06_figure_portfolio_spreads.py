from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from results_utils import DATA_FINAL, ensure_output_dirs, FIGURES_DIR


DISPLAY_ORDER = [
    "rsj_weekly", "rsj_sys", "rsj_idio", "rsj_sys_weekly", "rsj_idio_weekly",
    "res_weekly", "res_sys_p025", "res_idio_p025",
]


def main() -> None:
    ensure_output_dirs()

    path = DATA_FINAL / "portfolio_sorts" / "sort_results_all.parquet"
    df = pd.read_parquet(path)
    s = df[(df["quintile"].isin(["H-L", "L-H"])) & (df["weighting"] == "EW")].copy()

    s["x"] = s["sort_var"].apply(lambda x: DISPLAY_ORDER.index(x) if x in DISPLAY_ORDER else 999)
    s = s.sort_values("x")

    means = s["mean_ret_weekly"].to_numpy() * 10000.0
    labels = s["sort_var"].tolist()
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 5))
    plt.bar(x, means)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Weekly spread return (bps)")
    plt.title("Portfolio Spread Means (EW)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_portfolio_spreads_ew_bps.png", dpi=200)
    plt.close()

    print("Saved figure: fig_portfolio_spreads_ew_bps.png")


if __name__ == "__main__":
    main()
