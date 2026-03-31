from __future__ import annotations

import pandas as pd

from results_utils import DATA_FINAL, ensure_output_dirs, write_csv_and_tex, pretty_coef_bps


DISPLAY_ORDER = [
    "rsj_weekly",
    "rsj_sys",
    "rsj_idio",
    "rsj_sys_weekly",
    "rsj_idio_weekly",
    "res_weekly",
    "res_sys_p025",
    "res_idio_p025",
]


def main() -> None:
    ensure_output_dirs()

    path = DATA_FINAL / "portfolio_sorts" / "sort_results_all.parquet"
    df = pd.read_parquet(path)

    spread = df[df["quintile"].isin(["H-L", "L-H"])].copy()
    spread = spread[[
        "sort_var",
        "label",
        "quintile",
        "weighting",
        "n_weeks",
        "mean_ret_weekly",
        "t_mean",
        "alpha_weekly",
        "t_alpha",
    ]]

    spread["mean_bps_sig"] = spread.apply(
        lambda r: pretty_coef_bps(r["mean_ret_weekly"] * 10000, r["t_mean"]), axis=1
    )
    spread["alpha_bps_sig"] = spread.apply(
        lambda r: pretty_coef_bps(r["alpha_weekly"] * 10000, r["t_alpha"]), axis=1
    )

    table = spread[[
        "sort_var",
        "label",
        "quintile",
        "weighting",
        "n_weeks",
        "mean_bps_sig",
        "alpha_bps_sig",
    ]].rename(
        columns={
            "sort_var": "Variable",
            "label": "Description",
            "quintile": "Spread",
            "weighting": "Weight",
            "n_weeks": "N_weeks",
            "mean_bps_sig": "Mean_bps",
            "alpha_bps_sig": "Alpha_bps",
        }
    )

    table["order"] = table["Variable"].apply(lambda x: DISPLAY_ORDER.index(x) if x in DISPLAY_ORDER else 999)
    table = table.sort_values(["order", "Weight"]).drop(columns=["order"]).reset_index(drop=True)

    write_csv_and_tex(table, "table_portfolio_spreads", float_fmt="%.6f")
    print("Saved portfolio spread table.")


if __name__ == "__main__":
    main()
