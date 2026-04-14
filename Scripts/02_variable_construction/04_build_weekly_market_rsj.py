"""
Build Weekly Market RSJ Series

Purpose:
  Compute weekly market-level RSJ using the market return series.

Inputs:
  - Weekly market return input data.

Outputs:
  - Weekly market RSJ file in data_intermediate/market_weekly_rsj.

Main Steps:
  - Compute RSJ from positive and negative return variation.
  - Aggregate to week level.
  - Save clean market RSJ output.
"""
from pathlib import Path
import pandas as pd
import numpy as np

def compute_rsj(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]

    rv = np.sum(r ** 2)
    if rv == 0:
        return np.nan

    rv_pos = np.sum((r[r > 0]) ** 2)
    rv_neg = np.sum((r[r < 0]) ** 2)

    return (rv_pos - rv_neg) / rv

input_dir = Path("data_intermediate/converted_parquet_etf")
output_dir = Path("data_intermediate/market_weekly_rsj")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "spy_weekly_rsj.parquet"

files = sorted(input_dir.glob("*.parquet"))
if not files:
    raise FileNotFoundError(f"No parquet files found in {input_dir}")

parts = []

print("Processing ETF parquet files...\n")

for i, file_path in enumerate(files, start=1):
    if i % 200 == 0:
        print(f"Processed {i} ETF files...")

    df = pd.read_parquet(file_path)

    df = df[df["sym_root"] == "SPY"].copy()
    if df.empty:
        continue

    df = df[df["n_obs"] >= 80].copy()
    if df.empty:
        continue

    df["date"] = pd.to_datetime(df["date"])
    df["rsj_spy_daily"] = df["returns_5m"].apply(compute_rsj)

    parts.append(df[["date", "rsj_spy_daily", "n_obs"]])

if not parts:
    raise ValueError("No SPY observations found after filtering.")

spy_daily = pd.concat(parts, ignore_index=True)

spy_daily["week"] = spy_daily["date"].dt.to_period("W-TUE").dt.end_time.dt.normalize()

spy_weekly = (
    spy_daily
    .groupby("week", as_index=False)
    .agg(
        rsj_spy_weekly=("rsj_spy_daily", "mean"),
        n_days=("rsj_spy_daily", "count"),
        n_obs_total=("n_obs", "sum")
    )
)

spy_weekly = spy_weekly[spy_weekly["n_days"] >= 3].copy()

print("\nPreview:")
print(spy_weekly.head())

print("\nSummary:")
print(spy_weekly["rsj_spy_weekly"].describe())

spy_weekly.to_parquet(output_path, index=False)

print(f"\nSaved weekly SPY RSJ to: {output_path}")
