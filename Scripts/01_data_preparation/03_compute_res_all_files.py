from pathlib import Path
import pandas as pd
import numpy as np

def compute_res(r, alpha=0.025, H=0.5):
    """
    Compute daily realized expected shortfall (RES) from intraday returns.
    """
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]

    c = len(r)
    if c == 0:
        return np.nan

    q_alpha = np.quantile(r, alpha)
    tail = r[r <= q_alpha]

    if len(tail) == 0:
        return np.nan

    tail_mean = np.mean(tail)

    # Minus sign so larger values mean more downside risk
    res = -(c ** H) * tail_mean
    return res

input_dir = Path("data_intermediate/converted_parquet")
output_dir = Path("data_intermediate/res_daily")
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(input_dir.glob("*.parquet"))

for i, file_path in enumerate(files, start=1):
    print(f"[{i}/{len(files)}] Processing {file_path.name}")

    df = pd.read_parquet(file_path)

    # Keep only stock-days with sufficient intraday observations
    df = df[df["n_obs"] >= 80].copy()

    df["res_2p5"] = df["returns_5m"].apply(
        lambda x: compute_res(x, alpha=0.025, H=0.5)
    )

    out = df[[
        "permno", "date", "sym_root",
        "ret_crsp", "ex", "n_obs",
        "res_2p5"
    ]].copy()

    out.to_parquet(output_dir / file_path.name, index=False)

print("Done.")