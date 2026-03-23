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

data_path = Path("data_intermediate/converted_parquet")
file_path = sorted(data_path.glob("*.parquet"))[0]

print(f"Processing: {file_path.name}")

df = pd.read_parquet(file_path)

df["rsj"] = df["returns_5m"].apply(compute_rsj)

print("\nFirst rows:")
print(df[["permno", "date", "sym_root", "rv", "rsj"]].head(10))

print("\nRSJ summary:")
print(df["rsj"].describe())

print("\nRSJ range:")
print(df["rsj"].min(), df["rsj"].max())