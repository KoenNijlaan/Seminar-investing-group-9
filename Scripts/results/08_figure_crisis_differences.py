from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from results_utils import DATA_FINAL, ensure_output_dirs, FIGURES_DIR


def main() -> None:
    ensure_output_dirs()

    path = DATA_FINAL / "crisis" / "fm_crisis_summary.csv"
    df = pd.read_csv(path)

    d = df[["spec", "risk_var", "beta_diff_crisis_minus_noncrisis_bps", "t_beta_diff"]].copy()
    d["label"] = d["spec"] + " | " + d["risk_var"]
    d = d.sort_values("beta_diff_crisis_minus_noncrisis_bps").reset_index(drop=True)

    y = np.arange(len(d))
    x = d["beta_diff_crisis_minus_noncrisis_bps"].to_numpy()

    colors = ["#1f77b4" if abs(t) >= 1.96 else "#b0b0b0" for t in d["t_beta_diff"].to_numpy()]

    plt.figure(figsize=(12, 9))
    plt.barh(y, x, color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(y, d["label"].tolist())
    plt.xlabel("Crisis minus non-crisis beta (bps)")
    plt.title("Crisis-State Differences in FM Slopes")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_crisis_beta_differences.png", dpi=200)
    plt.close()

    print("Saved figure: fig_crisis_beta_differences.png")


if __name__ == "__main__":
    main()
