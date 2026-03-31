from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from results_utils import DATA_FINAL, ensure_output_dirs, FIGURES_DIR


def main() -> None:
    ensure_output_dirs()

    panel_path = DATA_FINAL / "panel" / "weekly_panel.parquet"
    panel = pd.read_parquet(panel_path)

    if "week" not in panel.columns:
        print("Column 'week' not found in weekly_panel.parquet.")
        return

    panel["week"] = pd.to_datetime(panel["week"], errors="coerce")
    panel = panel.dropna(subset=["week"])

    n_col = "permno" if "permno" in panel.columns else None
    if n_col is None:
        n_col = "id" if "id" in panel.columns else None

    if n_col is None:
        counts = panel.groupby("week").size().rename("n_obs").reset_index()
    else:
        counts = panel.groupby("week")[n_col].nunique().rename("n_stocks").reset_index()

    counts = counts.sort_values("week")

    fig, ax = plt.subplots(figsize=(12, 5))
    y_name = [c for c in counts.columns if c != "week"][0]
    ax.plot(counts["week"], counts[y_name], linewidth=1.6)
    ax.set_title("Weekly Cross-Section Coverage")
    ax.set_ylabel("Number of stocks")
    ax.set_xlabel("Week")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_weekly_cross_section_coverage.png", dpi=200)
    plt.close()

    print("Saved figure: fig_weekly_cross_section_coverage.png")


if __name__ == "__main__":
    main()
