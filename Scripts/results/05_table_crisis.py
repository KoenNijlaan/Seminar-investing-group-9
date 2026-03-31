from __future__ import annotations

import pandas as pd

from results_utils import DATA_FINAL, ensure_output_dirs, write_csv_and_tex, stars_from_t


SPEC_ORDER = [
    "A_rsj", "A_rsj_sys_m2", "A_rsj_idio_m2", "A_rsj_sys_m1", "A_rsj_idio_m1",
    "A_res", "A_res_sys", "A_res_idio", "B1_rsj_ctrl", "B2_res_ctrl",
    "B3_rsj_res_ctrl", "B4_decomp_m2", "B5_decomp_m1",
]


def main() -> None:
    ensure_output_dirs()

    path = DATA_FINAL / "crisis" / "fm_crisis_summary.csv"
    df = pd.read_csv(path)

    df["noncrisis_bps"] = df.apply(
        lambda r: f"{r['beta_mean_noncrisis_bps']:.2f}{stars_from_t(r['t_beta_noncrisis'])}", axis=1
    )
    df["crisis_bps"] = df.apply(
        lambda r: f"{r['beta_mean_crisis_bps']:.2f}{stars_from_t(r['t_beta_crisis'])}", axis=1
    )
    df["diff_bps"] = df.apply(
        lambda r: f"{r['beta_diff_crisis_minus_noncrisis_bps']:.2f}{stars_from_t(r['t_beta_diff'])}", axis=1
    )

    out = df[[
        "spec",
        "risk_var",
        "n_weeks_noncrisis",
        "n_weeks_crisis",
        "noncrisis_bps",
        "crisis_bps",
        "diff_bps",
        "t_beta_diff",
    ]].rename(
        columns={
            "spec": "Spec",
            "risk_var": "Predictor",
            "n_weeks_noncrisis": "N_noncrisis",
            "n_weeks_crisis": "N_crisis",
            "noncrisis_bps": "Beta_noncrisis_bps",
            "crisis_bps": "Beta_crisis_bps",
            "diff_bps": "Diff_bps",
            "t_beta_diff": "t_diff",
        }
    )

    out["order"] = out["Spec"].apply(lambda x: SPEC_ORDER.index(x) if x in SPEC_ORDER else 999)
    out = out.sort_values(["order", "Predictor"]).drop(columns=["order"]).reset_index(drop=True)

    write_csv_and_tex(out, "table_crisis_state", float_fmt="%.6f")
    print("Saved crisis-state table.")


if __name__ == "__main__":
    main()
