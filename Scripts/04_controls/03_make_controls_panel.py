"""
Assemble Controls Panel

Purpose:
  Merge all weekly controls into one stock-week controls panel.

Inputs:
  - Weekly control files from data_intermediate/controls.

Outputs:
  - Merged controls panel for final dataset construction.

Main Steps:
  - Load control files.
  - Merge on stock-week keys.
  - Write consolidated controls panel.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input_path = Path("data_intermediate/controls/weekly_controls.parquet")
output_dir = Path("data_intermediate/controls/checks")
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading weekly controls...")
df = pd.read_parquet(input_path)

print("\nBasic info")
print("-" * 60)
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")
print(f"Memory usage (MB): {df.memory_usage(deep=True).sum() / (1024**2):.2f}")

print("\nColumns:")
print(df.columns.tolist())

df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
df["week"] = pd.to_datetime(df["week"], errors="coerce")

numeric_cols = ["me", "me_raw", "bm", "mom", "rev", "ivol", "illiq", "ret_weekly"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nDtypes")
print("-" * 60)
print(df.dtypes)

print("\nDuplicate key check")
print("-" * 60)
dup_count = df.duplicated(subset=["permno", "week"]).sum()
print(f"Duplicate (permno, week) rows: {dup_count:,}")

print("\nMissingness")
print("-" * 60)
missing = df.isna().mean().sort_values(ascending=False)
print(missing)

missing.to_csv(output_dir / "missing_shares.csv")

print("\nSummary statistics")
print("-" * 60)
summary = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
print(summary)

summary.to_csv(output_dir / "summary_stats.csv")

print("\nExtra percentiles")
print("-" * 60)
percentiles = pd.DataFrame({
    col: df[col].quantile([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    for col in numeric_cols if col in df.columns
}).T
print(percentiles)

percentiles.to_csv(output_dir / "percentiles.csv")

print("\nSuspicious values")
print("-" * 60)

checks = {}

if "me" in df.columns:
    checks["me_nonpositive"] = ((df["me"] <= 0) & df["me"].notna()).mean()

if "me_raw" in df.columns:
    checks["me_raw_nonpositive"] = ((df["me_raw"] <= 0) & df["me_raw"].notna()).mean()

if "bm" in df.columns:
    checks["bm_nonpositive"] = ((df["bm"] <= 0) & df["bm"].notna()).mean()
    checks["bm_gt_10"] = ((df["bm"] > 10) & df["bm"].notna()).mean()

if "mom" in df.columns:
    checks["mom_lt_-1"] = ((df["mom"] < -1) & df["mom"].notna()).mean()
    checks["mom_gt_10"] = ((df["mom"] > 10) & df["mom"].notna()).mean()

if "rev" in df.columns:
    checks["rev_lt_-1"] = ((df["rev"] < -1) & df["rev"].notna()).mean()
    checks["rev_gt_5"] = ((df["rev"] > 5) & df["rev"].notna()).mean()

if "ivol" in df.columns:
    checks["ivol_nonpositive"] = ((df["ivol"] <= 0) & df["ivol"].notna()).mean()
    checks["ivol_gt_1"] = ((df["ivol"] > 1) & df["ivol"].notna()).mean()

if "illiq" in df.columns:
    checks["illiq_nonfinite"] = (~np.isfinite(df["illiq"]) & df["illiq"].notna()).mean()

checks = pd.Series(checks).sort_index()
print(checks)

checks.to_csv(output_dir / "suspicious_value_shares.csv")

print("\nWeekly cross-sectional coverage")
print("-" * 60)

weekly_counts = (
    df.groupby("week", as_index=False)
      .agg(
          n_stocks=("permno", "nunique"),
          n_obs=("permno", "size"),
          n_me=("me", lambda x: x.notna().sum()),
          n_bm=("bm", lambda x: x.notna().sum()),
          n_mom=("mom", lambda x: x.notna().sum()),
          n_rev=("rev", lambda x: x.notna().sum()),
          n_ivol=("ivol", lambda x: x.notna().sum()),
          n_illiq=("illiq", lambda x: x.notna().sum()),
      )
)

print(weekly_counts[["n_stocks", "n_me", "n_bm", "n_mom", "n_rev", "n_ivol", "n_illiq"]].describe())

weekly_counts.to_csv(output_dir / "weekly_counts.csv", index=False)

print("\nCorrelation matrix")
print("-" * 60)

corr_cols = [c for c in ["me", "bm", "mom", "rev", "ivol", "illiq", "ret_weekly"] if c in df.columns]
corr = df[corr_cols].corr()
print(corr)

corr.to_csv(output_dir / "correlations.csv")

print("\nOutlier shares outside 1%-99% range")
print("-" * 60)

outlier_info = {}
for col in corr_cols:
    x = df[col].dropna()
    if len(x) == 0:
        continue
    p1 = x.quantile(0.01)
    p99 = x.quantile(0.99)
    share = ((df[col] < p1) | (df[col] > p99)).mean()
    outlier_info[col] = {
        "p1": p1,
        "p99": p99,
        "share_outside": share
    }

outlier_df = pd.DataFrame(outlier_info).T
print(outlier_df)

outlier_df.to_csv(output_dir / "outlier_shares.csv")

print("\nSaving diagnostic plots...")
print("-" * 60)

plot_cols = ["me", "bm", "mom", "rev", "ivol", "illiq"]

for col in plot_cols:
    if col not in df.columns:
        continue

    x = df[col].dropna()
    if len(x) == 0:
        continue

    lo = x.quantile(0.01)
    hi = x.quantile(0.99)
    x_plot = x[(x >= lo) & (x <= hi)]

    plt.figure(figsize=(8, 5))
    plt.hist(x_plot, bins=100)
    plt.title(f"{col} histogram (trimmed 1%-99%)")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / f"hist_{col}.png", dpi=150)
    plt.close()

plt.figure(figsize=(10, 5))
plt.plot(weekly_counts["week"], weekly_counts["n_stocks"])
plt.title("Number of stocks per week")
plt.xlabel("Week")
plt.ylabel("Unique stocks")
plt.tight_layout()
plt.savefig(output_dir / "weekly_stock_counts.png", dpi=150)
plt.close()

for col in ["bm", "mom", "rev", "ivol", "illiq"]:
    if col not in df.columns:
        continue

    miss_week = (
        df.groupby("week")[col]
          .apply(lambda x: x.isna().mean())
          .reset_index(name="missing_share")
    )

    plt.figure(figsize=(10, 5))
    plt.plot(miss_week["week"], miss_week["missing_share"])
    plt.title(f"Weekly missing share: {col}")
    plt.xlabel("Week")
    plt.ylabel("Missing share")
    plt.tight_layout()
    plt.savefig(output_dir / f"missing_by_week_{col}.png", dpi=150)
    plt.close()

print("\nWinsorized summary preview (1%-99%)")
print("-" * 60)

winsor_summary = {}
for col in ["bm", "mom", "rev", "ivol", "illiq"]:
    if col not in df.columns:
        continue

    x = df[col].copy()
    lo = x.quantile(0.01)
    hi = x.quantile(0.99)
    x = x.clip(lower=lo, upper=hi)

    winsor_summary[col] = {
        "mean": x.mean(),
        "std": x.std(),
        "min": x.min(),
        "p1": x.quantile(0.01),
        "median": x.median(),
        "p99": x.quantile(0.99),
        "max": x.max(),
    }

winsor_summary = pd.DataFrame(winsor_summary).T
print(winsor_summary)

winsor_summary.to_csv(output_dir / "winsorized_summary_preview.csv")

print(f"\nAll checks saved to: {output_dir}")
