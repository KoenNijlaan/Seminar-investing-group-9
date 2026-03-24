from pathlib import Path
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
input_dir = Path("data_intermediate/res_daily_split")
output_dir = Path("data_intermediate/weekly_data")
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(input_dir.glob("*.parquet"))
if not files:
    raise FileNotFoundError(f"No parquet files found in {input_dir}")

print("Processing daily decomposed RES files...\n")

# -----------------------------
# Incremental weekly accumulator
# -----------------------------
weekly_accum = {}

for i, file_path in enumerate(files, start=1):
    if i % 200 == 0 or i == 1 or i == len(files):
        print(f"Processed {i}/{len(files)} files...")

    df = pd.read_parquet(file_path)

    if df.empty:
        continue

    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()

    # aggregate within file first
    weekly_part = (
        df.groupby(["permno", "week"], as_index=False)
          .agg(
              res_sum=("res_2p5", "sum"),
              res_count=("res_2p5", "count"),

              res_sys_sum=("res_sys_2p5", "sum"),
              res_sys_count=("res_sys_2p5", "count"),

              res_idio_sum=("res_idio_2p5", "sum"),
              res_idio_count=("res_idio_2p5", "count"),

              n_obs_total=("n_obs", "sum")
          )
    )

    # add to accumulator
    for row in weekly_part.itertuples(index=False):
        key = (row.permno, row.week)

        if key not in weekly_accum:
            weekly_accum[key] = {
                "res_sum": 0.0,
                "res_count": 0,
                "res_sys_sum": 0.0,
                "res_sys_count": 0,
                "res_idio_sum": 0.0,
                "res_idio_count": 0,
                "n_obs_total": 0
            }

        weekly_accum[key]["res_sum"] += row.res_sum
        weekly_accum[key]["res_count"] += row.res_count

        weekly_accum[key]["res_sys_sum"] += row.res_sys_sum
        weekly_accum[key]["res_sys_count"] += row.res_sys_count

        weekly_accum[key]["res_idio_sum"] += row.res_idio_sum
        weekly_accum[key]["res_idio_count"] += row.res_idio_count

        weekly_accum[key]["n_obs_total"] += row.n_obs_total

print("\nBuilding final weekly DataFrame...")

rows = []
for (permno, week), vals in weekly_accum.items():
    rows.append({
        "permno": permno,
        "week": week,
        "res_sum": vals["res_sum"],
        "res_count": vals["res_count"],
        "res_sys_sum": vals["res_sys_sum"],
        "res_sys_count": vals["res_sys_count"],
        "res_idio_sum": vals["res_idio_sum"],
        "res_idio_count": vals["res_idio_count"],
        "n_obs_total": vals["n_obs_total"]
    })

weekly = pd.DataFrame(rows)

# -----------------------------
# Weekly averages
# -----------------------------
weekly["res_weekly_2p5"] = weekly["res_sum"] / weekly["res_count"]
weekly["res_sys_weekly_2p5"] = weekly["res_sys_sum"] / weekly["res_sys_count"]
weekly["res_idio_weekly_2p5"] = weekly["res_idio_sum"] / weekly["res_idio_count"]

weekly["n_days_total"] = weekly["res_count"]
weekly["n_days_sys"] = weekly["res_sys_count"]
weekly["n_days_idio"] = weekly["res_idio_count"]

# -----------------------------
# Filtering
# -----------------------------
# Keep weeks with at least 3 valid daily observations for sys and idio
# (this is usually the binding requirement once beta-based decomposition starts)
weekly = weekly[
    (weekly["n_days_sys"] >= 3) &
    (weekly["n_days_idio"] >= 3)
].copy()

weekly = weekly.dropna(
    subset=["res_sys_weekly_2p5", "res_idio_weekly_2p5"]
)

# Optional: also require total weekly RES to be present
weekly = weekly.dropna(subset=["res_weekly_2p5"])

# -----------------------------
# Final columns
# -----------------------------
weekly = weekly[[
    "permno", "week",
    "res_weekly_2p5",
    "res_sys_weekly_2p5",
    "res_idio_weekly_2p5",
    "n_days_total",
    "n_days_sys",
    "n_days_idio",
    "n_obs_total"
]]

weekly = weekly.sort_values(["permno", "week"]).reset_index(drop=True)

print("\nPreview:")
print(weekly.head())

print("\nSummary total RES:")
print(weekly["res_weekly_2p5"].describe())

print("\nSummary systematic RES:")
print(weekly["res_sys_weekly_2p5"].describe())

print("\nSummary idiosyncratic RES:")
print(weekly["res_idio_weekly_2p5"].describe())

output_file = output_dir / "weekly_res_split.parquet"
weekly.to_parquet(output_file, index=False)

print(f"\nSaved weekly dataset to: {output_file}")