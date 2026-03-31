from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from results_utils import DATA_FINAL, ensure_output_dirs, FIGURES_DIR


SIGNAL_COLS = ["rsj_weekly", "res_weekly", "rsj_idio", "res_idio_p025"]


def main() -> None:
    ensure_output_dirs()

    panel_path = DATA_FINAL / "panel" / "weekly_panel.parquet"
    panel = pd.read_parquet(panel_path)

    cols = [c for c in SIGNAL_COLS if c in panel.columns]
    if not cols:
        print("No target signal columns found for distribution plots.")
        return

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for i, col in enumerate(cols):
        ax = axes[0, i]
        s = pd.to_numeric(panel[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            ax.set_title(f"{col} (no data)")
            continue

        q1, q99 = s.quantile(0.01), s.quantile(0.99)
        s_clip = s.clip(lower=q1, upper=q99)

        ax.hist(s_clip, bins=50, alpha=0.85)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(col)
        ax.set_xlabel("Value (winsorized 1%-99%)")
        ax.set_ylabel("Frequency")

    fig.suptitle("Signal Distributions in Weekly Panel", y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_signal_distributions.png", dpi=200)
    plt.close()

    print("Saved figure: fig_signal_distributions.png")


if __name__ == "__main__":
    main()
