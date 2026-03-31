from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from results_utils import DATA_FINAL, ensure_output_dirs, FIGURES_DIR, ci95_bps


TARGET_SPECS = ["B3_rsj_res_ctrl", "B4_decomp_m2", "B5_decomp_m1"]
TARGET_VARS = ["RSJ", "RES", "RSJ_sys (M2)", "RSJ_idio (M2)", "RES_sys", "RES_idio", "RSJ_sys (M1)", "RSJ_idio (M1)"]


def main() -> None:
    ensure_output_dirs()

    path = DATA_FINAL / "fama_macbeth" / "fm_summary.csv"
    df = pd.read_csv(path)

    d = df[df["spec"].isin(TARGET_SPECS)].copy()
    d = d[d["label"].isin(TARGET_VARS)].copy()

    if d.empty:
        print("No rows for requested specs/labels.")
        return

    d["lo"] = np.nan
    d["hi"] = np.nan
    for i, r in d.iterrows():
        lo, hi = ci95_bps(r["mean_bps"], r["se_bps"])
        d.at[i, "lo"] = lo
        d.at[i, "hi"] = hi

    labels = (d["spec"] + " | " + d["label"]).tolist()
    y = np.arange(len(d))

    plt.figure(figsize=(11, 8))
    plt.errorbar(
        d["mean_bps"],
        y,
        xerr=[d["mean_bps"] - d["lo"], d["hi"] - d["mean_bps"]],
        fmt="o",
        capsize=3,
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(y, labels)
    plt.xlabel("Coefficient (bps, 95% CI)")
    plt.title("Fama-MacBeth Coefficients (Selected Specs)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_fm_coefficients_selected.png", dpi=200)
    plt.close()

    print("Saved figure: fig_fm_coefficients_selected.png")


if __name__ == "__main__":
    main()
