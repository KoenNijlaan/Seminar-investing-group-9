from __future__ import annotations

import pandas as pd

from results_utils import DATA_FINAL, ensure_output_dirs, write_csv_and_tex

def main() -> None:
    ensure_output_dirs()

    panel_path = DATA_FINAL / "panel" / "weekly_panel.parquet"
    panel = pd.read_parquet(panel_path)

    numeric_cols = [
        "R_i_w",
        "R_i_w_plus_1",
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
        "n_days",
        "n_obs_total",
    ]
    use_cols = [c for c in numeric_cols if c in panel.columns]

    rows = []
    for c in use_cols:
        s = pd.to_numeric(panel[c], errors="coerce")
        rows.append(
            {
                "Variable": c,
                "N": int(s.notna().sum()),
                "Mean": s.mean(),
                "Std": s.std(),
                "P25": s.quantile(0.25),
                "Median": s.median(),
                "P75": s.quantile(0.75),
                "Min": s.min(),
                "Max": s.max(),
            }
        )

    out = pd.DataFrame(rows)
    write_csv_and_tex(out, "table_summary_stats", float_fmt="%.6f")
    print("Saved summary statistics table.")


if __name__ == "__main__":
    main()
