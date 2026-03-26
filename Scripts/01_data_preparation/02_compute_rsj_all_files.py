from pathlib import Path
import pandas as pd
import numpy as np

def compute_rsj(r):
    r = np.asarray(r, dtype=float)

    rv = np.sum(r ** 2)
    if rv == 0:
        return np.nan

    rv_pos = np.sum((r[r > 0]) ** 2)
    rv_neg = np.sum((r[r < 0]) ** 2)

    return (rv_pos - rv_neg) / rv

input_dir = Path("data_intermediate/converted_parquet")
output_dir = Path("data_intermediate/rsj_daily")
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(input_dir.glob("*.parquet"))

for i, file_path in enumerate(files, start=1):
    print(f"[{i}/{len(files)}] Processing {file_path.name}")

    df = pd.read_parquet(file_path)

    # Keep only stock-days with sufficient intraday observations
    df = df[df["n_obs"] >= 80].copy()

    df["rsj"] = df["returns_5m"].apply(compute_rsj)

    out = df[[
        "permno", "date", "sym_root",
        "ret_crsp", "ex", "n_obs",
        "rsj"
    ]].copy()

    out.to_parquet(output_dir / file_path.name, index=False)