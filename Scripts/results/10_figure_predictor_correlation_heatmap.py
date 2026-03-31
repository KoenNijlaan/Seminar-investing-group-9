from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from results_utils import DATA_FINAL, ensure_output_dirs, FIGURES_DIR


PREDICTOR_COLS = [
    "rsj_weekly",
    "rsj_sys",
    "rsj_idio",
    "rsj_sys_weekly",
    "rsj_idio_weekly",
    "res_weekly",
    "res_sys_p025",
    "res_idio_p025",
    "me",
    "bm",
    "mom",
    "rev",
    "ivol",
    "illiq",
]


def main() -> None:
    ensure_output_dirs()

    panel_path = DATA_FINAL / "panel" / "weekly_panel.parquet"
    panel = pd.read_parquet(panel_path)

    cols = [c for c in PREDICTOR_COLS if c in panel.columns]
    if len(cols) < 2:
        print("Not enough predictor columns for a correlation heatmap.")
        return

    d = panel[cols].apply(pd.to_numeric, errors="coerce")
    corr = d.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.to_numpy(), aspect="auto", vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    ax.set_title("Predictor Correlation Heatmap (Weekly Panel)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_predictor_correlation_heatmap.png", dpi=200)
    plt.close()

    print("Saved figure: fig_predictor_correlation_heatmap.png")


if __name__ == "__main__":
    main()
