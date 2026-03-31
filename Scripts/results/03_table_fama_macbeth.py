from __future__ import annotations

import pandas as pd

from results_utils import DATA_FINAL, ensure_output_dirs, write_csv_and_tex, pretty_coef_bps


SPEC_ORDER = [
    "A_rsj",
    "A_rsj_sys_m2",
    "A_rsj_idio_m2",
    "A_rsj_sys_m1",
    "A_rsj_idio_m1",
    "A_res",
    "A_res_sys",
    "A_res_idio",
    "B1_rsj_ctrl",
    "B2_res_ctrl",
    "B3_rsj_res_ctrl",
    "B4_decomp_m2",
    "B5_decomp_m1",
]


def main() -> None:
    ensure_output_dirs()

    path = DATA_FINAL / "fama_macbeth" / "fm_summary.csv"
    df = pd.read_csv(path)

    df["coef_bps_sig"] = df.apply(lambda r: pretty_coef_bps(r["mean_bps"], r["t_stat"]), axis=1)

    out = df[["spec", "label", "n_weeks", "coef_bps_sig", "t_stat"]].rename(
        columns={
            "spec": "Spec",
            "label": "Predictor",
            "n_weeks": "N_weeks",
            "coef_bps_sig": "Coef_bps",
            "t_stat": "t_stat",
        }
    )

    out["order"] = out["Spec"].apply(lambda x: SPEC_ORDER.index(x) if x in SPEC_ORDER else 999)
    out = out.sort_values(["order", "Predictor"]).drop(columns=["order"]).reset_index(drop=True)

    write_csv_and_tex(out, "table_fama_macbeth_main", float_fmt="%.6f")
    print("Saved Fama-MacBeth table.")


if __name__ == "__main__":
    main()
