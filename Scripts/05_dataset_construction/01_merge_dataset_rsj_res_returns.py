from pathlib import Path
import pandas as pd

# =========================================================
# PURPOSE
# =========================================================
# Construct the baseline merged dataset containing:
#   - weekly RSJ
#   - weekly RES
#   - weekly stock returns R_i_w
#   - next-week stock returns R_i_w_plus_1
#
# IMPORTANT:
#   This dataset does NOT include control variables.
#   It is the baseline dataset for initial portfolio sorts
#   and baseline predictive regressions.
# =========================================================

# =========================================================
# Paths
# =========================================================
rsj_path = Path("data_intermediate/rsj_weekly/rsj_weekly.parquet")
res_path = Path("data_intermediate/res_weekly/res_weekly.parquet")
ret_path = Path("data_intermediate/weekly_stock_returns/weekly_stock_returns.parquet")

output_dir = Path("data_final")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "dataset_rsj_res_returns.parquet"

# =========================================================
# Load datasets
# =========================================================
print("Loading datasets...")

rsj = pd.read_parquet(rsj_path)
res = pd.read_parquet(res_path)
ret = pd.read_parquet(ret_path)

# =========================================================
# Standardize key columns
# =========================================================
for df in [rsj, res, ret]:
    df["permno"] = df["permno"].astype("Int64")
    df["week"] = pd.to_datetime(df["week"])

# =========================================================
# Keep only needed columns
# =========================================================
rsj = rsj[["permno", "week", "rsj_weekly"]].copy()
res = res[["permno", "week", "res_weekly"]].copy()
ret = ret[["permno", "week", "R_i_w", "R_i_w_plus_1"]].copy()

print(f"RSJ rows: {len(rsj):,}")
print(f"RES rows: {len(res):,}")
print(f"Return rows: {len(ret):,}")

# =========================================================
# Merge RSJ and RES
# =========================================================
print("\nMerging RSJ and RES...")

df = pd.merge(
    rsj,
    res,
    on=["permno", "week"],
    how="inner"
)

print(f"Rows after RSJ+RES merge: {len(df):,}")

# =========================================================
# Merge with returns
# =========================================================
print("Merging with weekly returns...")

df = pd.merge(
    df,
    ret,
    on=["permno", "week"],
    how="inner"
)

print(f"Rows after full merge: {len(df):,}")

# =========================================================
# Drop missing baseline variables
# =========================================================
print("Dropping missing baseline variables...")

df = df.dropna(subset=[
    "rsj_weekly",
    "res_weekly",
    "R_i_w_plus_1"
])

print(f"Rows after dropping missing values: {len(df):,}")

# =========================================================
# Final formatting
# =========================================================
df = df.sort_values(["permno", "week"]).reset_index(drop=True)

# =========================================================
# Diagnostics
# =========================================================
print("\nPreview:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nSummary:")
print(df[["rsj_weekly", "res_weekly", "R_i_w", "R_i_w_plus_1"]].describe())

# =========================================================
# Save
# =========================================================
df.to_parquet(output_path, index=False)

print(f"\nSaved baseline dataset to: {output_path}")
print("This dataset contains RSJ, RES, and returns only (no control variables).")