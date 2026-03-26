from pathlib import Path
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
input_dir = Path("data_intermediate/res_daily")
output_dir = Path("data_intermediate/res_weekly")
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(input_dir.glob("*.parquet"))

weekly_parts = []

print("Processing daily RES files...\n")

for i, file_path in enumerate(files, start=1):
    if i % 200 == 0:
        print(f"Processed {i} files...")

    df = pd.read_parquet(file_path)

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Week ending Tuesday (includes Tuesday close)
    df["week"] = df["date"].dt.to_period("W-TUE").dt.end_time.dt.normalize()

    # Aggregate within this file
    weekly_part = (
        df.groupby(["permno", "week"], as_index=False)
          .agg(
              res_sum=("res_2p5", "sum"),
              res_count=("res_2p5", "count"),
              n_obs_total=("n_obs", "sum")
          )
    )

    weekly_parts.append(weekly_part)

print("\nCombining weekly parts...")

weekly_all = pd.concat(weekly_parts, ignore_index=True)

print("Final aggregation across all files...")

weekly = (
    weekly_all
    .groupby(["permno", "week"], as_index=False)
    .agg(
        res_sum=("res_sum", "sum"),
        res_count=("res_count", "sum"),
        n_obs_total=("n_obs_total", "sum")
    )
)

weekly["res_weekly_2p5"] = weekly["res_sum"] / weekly["res_count"]
weekly["n_days"] = weekly["res_count"]

weekly = weekly.dropna(subset=["res_weekly_2p5"])
weekly = weekly[weekly["n_days"] >= 1].copy()

weekly = weekly[["permno", "week", "res_weekly_2p5", "n_days", "n_obs_total"]]

print("\nPreview:")
print(weekly.head())

print("\nSummary:")
print(weekly["res_weekly_2p5"].describe())

output_file = output_dir / "weekly_res.parquet"
weekly.to_parquet(output_file, index=False)

print(f"\nSaved weekly dataset to: {output_file}")