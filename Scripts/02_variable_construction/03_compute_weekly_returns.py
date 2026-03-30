from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# PURPOSE
# =========================================================
# Construct weekly stock returns R_i_w and next-week returns R_i_w_plus_1
# from daily CRSP returns (ret_crsp).
#
# IMPORTANT:
# - ret_crsp is stored in PERCENT units in the raw files.
# - Therefore, ret_crsp must be divided by 100 before compounding.
# - Week definition matches RSJ and RES: W-TUE (includes Tuesday close).
# - We require at least 3 valid daily returns per stock-week.
# =========================================================

# =========================================================
# Paths
# =========================================================
input_dir = Path("data_intermediate/converted_parquet")
output_dir = Path("data_intermediate/weekly_stock_returns")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "weekly_stock_returns.parquet"

files = sorted(input_dir.glob("*.parquet"))
if not files:
    raise FileNotFoundError("No parquet files found in data_intermediate/converted_parquet")

# =========================================================
# Helper function
# =========================================================
def compound_returns(x: pd.Series) -> float:
    """
    Compound daily CRSP returns into a weekly return.

    ret_crsp is in percent form, so divide by 100 first:
        5.0 means 5%, not 0.05
    """
    x = pd.to_numeric(x, errors="coerce") / 100.0
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return np.prod(1.0 + x) - 1.0

# =========================================================
# Load daily returns
# =========================================================
print("Reading daily CRSP returns...")

parts = []

for i, file_path in enumerate(files, start=1):
    df = pd.read_parquet(file_path, columns=["permno", "date", "ret_crsp"])

    df["date"] = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype("Int64")

    # Week ending Tuesday (includes Tuesday close)
    df["week"] = df["date"].dt.to_period("W-TUE").dt.end_time.dt.normalize()

    parts.append(df[["permno", "date", "week", "ret_crsp"]])

    if i % 250 == 0:
        print(f"Processed {i} daily files")

daily_df = pd.concat(parts, ignore_index=True)

print(f"\nDaily rows: {len(daily_df):,}")
print(f"Unique stocks: {daily_df['permno'].nunique():,}")
print(f"Date range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")

# =========================================================
# Compute weekly stock returns R_i_w
# =========================================================
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

# Keep only stock-weeks with at least 3 valid daily returns
weekly = weekly[weekly["n_days"] >= 3].copy()

# =========================================================
# Create next-week return R_i_w_plus_1
# =========================================================
print("Creating next-week return...")

weekly["R_i_w_plus_1"] = weekly.groupby("permno")["R_i_w"].shift(-1)
weekly["n_days_plus_1"] = weekly.groupby("permno")["n_days"].shift(-1)

# Optional: validity flag for next-week return
weekly["valid_R_i_w_plus_1"] = weekly["n_days_plus_1"] >= 3

# If you want only rows with a valid next-week return:
weekly = weekly[weekly["valid_R_i_w_plus_1"]].copy()

# =========================================================
# Final diagnostics
# =========================================================
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

# =========================================================
# Save
# =========================================================
weekly.to_parquet(output_path, index=False)

print(f"\nSaved to: {output_path}")