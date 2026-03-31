from __future__ import annotations

import pandas as pd

from results_utils import DATA_FINAL, ensure_output_dirs, write_csv_and_tex


def p_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def main() -> None:
    ensure_output_dirs()

    path = DATA_FINAL / "fama_macbeth" / "fm_wald.csv"
    df = pd.read_csv(path)

    df["W_sig"] = df.apply(lambda r: f"{r['W']:.2f}{p_stars(r['p_value'])}", axis=1)

    out = df[["spec", "hypothesis", "W_sig", "df", "p_value", "n_weeks"]].rename(
        columns={
            "spec": "Spec",
            "hypothesis": "Hypothesis",
            "W_sig": "Wald_stat",
            "df": "df",
            "p_value": "p_value",
            "n_weeks": "N_weeks",
        }
    )

    write_csv_and_tex(out, "table_wald_tests", float_fmt="%.6f")
    print("Saved Wald test table.")


if __name__ == "__main__":
    main()
