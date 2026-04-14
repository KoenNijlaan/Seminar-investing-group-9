"""
Compute Weekly Stock Returns

Purpose:
  Build weekly stock returns from daily CRSP returns and create next-week return targets.

Inputs:
  - Daily parquet files in data_intermediate/converted_parquet.

Outputs:
  - weekly_stock_returns.parquet in data_intermediate/weekly_stock_returns.

Main Steps:
  - Compound daily returns to weekly returns.
  - Filter on minimum trading-day coverage.
  - Create R_i_w_plus_1 and validity flags.
"""
from pathlib import Path
import pandas as pd
import numpy as np

input_dir = Path("data_intermediate/converted_parquet")
output_dir = Path("data_intermediate/weekly_stock_returns")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "weekly_stock_returns.parquet"

files = sorted(input_dir.glob("*.parquet"))
if not files:
    raise FileNotFoundError("No parquet files found in data_intermediate/converted_parquet")

def compound_returns(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce") / 100.0
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return np.prod(1.0 + x) - 1.0

print("Reading daily CRSP returns...")

parts = []

for i, file_path in enumerate(files, start=1):
    df = pd.read_parquet(file_path, columns=["permno", "date", "ret_crsp"])

    df["date"] = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype("Int64")

    df["week"] = df["date"].dt.to_period("W-TUE").dt.end_time.dt.normalize()

    parts.append(df[["permno", "date", "week", "ret_crsp"]])

    if i % 250 == 0:
        print(f"Processed {i} daily files")

daily_df = pd.concat(parts, ignore_index=True)

print(f"\nDaily rows: {len(daily_df):,}")
print(f"Unique stocks: {daily_df['permno'].nunique():,}")
print(f"Date range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")

print("\nComputing weekly stock returns...")

weekly = (
    daily_df
    .sort_values(["permno", "date"])
    .groupby(["permno", "week"], as_index=False)
    .agg(
        R_i_w=("ret_crsp", compound_returns),
        n_days=("ret_crsp", lambda x: pd.to_numeric(x, errors="coerce").notna().sum())
    )
)

weekly = weekly.sort_values(["permno", "week"]).reset_index(drop=True)

weekly = weekly[weekly["n_days"] >= 3].copy()

print("Creating next-week return...")

weekly["R_i_w_plus_1"] = weekly.groupby("permno")["R_i_w"].shift(-1)
weekly["n_days_plus_1"] = weekly.groupby("permno")["n_days"].shift(-1)

weekly["valid_R_i_w_plus_1"] = weekly["n_days_plus_1"] >= 3

weekly = weekly[weekly["valid_R_i_w_plus_1"]].copy()

print("\nFinished weekly return construction.")
print(f"Weekly rows: {len(weekly):,}")
print(f"Non-missing R_i_w: {weekly['R_i_w'].notna().sum():,}")
print(f"Non-missing R_i_w_plus_1: {weekly['R_i_w_plus_1'].notna().sum():,}")

print("\nPreview:")
print(weekly.head())

print("\nSummary of R_i_w:")
print(weekly["R_i_w"].describe())

print("\nSummary of R_i_w_plus_1:")
print(weekly["R_i_w_plus_1"].describe())

weekly.to_parquet(output_path, index=False)

print(f"\nSaved to: {output_path}")