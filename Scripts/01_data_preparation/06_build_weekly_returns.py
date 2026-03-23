from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

input_dir = Path("data_intermediate/converted_parquet")
output_dir = Path("data_intermediate/weekly_returns")
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(input_dir.glob("*.parquet"))

print("Building weekly returns...")

current_week = None
week_store = defaultdict(list)

# collect output per year, then flush to disk
year_buffers = defaultdict(list)

def flush_week_to_year_buffer(week, store, year_buffers):
    """Convert one completed calendar week into rows and append them to a year buffer."""
    year = pd.Timestamp(week).year

    for permno, arr_list in store.items():
        if not arr_list:
            continue

        year_buffers[year].append({
            "permno": int(permno) if pd.notna(permno) else None,
            "week": pd.Timestamp(week),
            "returns_5m_week": np.concatenate(arr_list)
        })

def flush_year_buffer_to_disk(year, year_buffers, output_dir):
    """Write one year's buffered rows to parquet and clear memory."""
    rows = year_buffers.get(year, [])
    if not rows:
        return

    df_year = pd.DataFrame(rows)
    df_year = df_year.sort_values(["permno", "week"]).reset_index(drop=True)

    out_path = output_dir / f"weekly_returns_{year}.parquet"
    df_year.to_parquet(out_path, index=False)

    print(f"Saved {len(df_year):,} rows to {out_path.name}")

    year_buffers[year] = []

for i, file_path in enumerate(files, start=1):
    df = pd.read_parquet(file_path, columns=["permno", "date", "returns_5m"])

    file_date = pd.to_datetime(df["date"].iloc[0])

    # Match RSJ convention: week ending Friday
    file_week = file_date.to_period("W-FRI").end_time.normalize()

    if current_week is None:
        current_week = file_week

    # new week begins -> flush previous week
    if file_week != current_week:
        flush_week_to_year_buffer(current_week, week_store, year_buffers)

        # if year changed, flush completed prior year to disk
        prev_year = pd.Timestamp(current_week).year
        new_year = pd.Timestamp(file_week).year
        if new_year != prev_year:
            flush_year_buffer_to_disk(prev_year, year_buffers, output_dir)

        week_store = defaultdict(list)
        current_week = file_week

    df["permno"] = df["permno"].astype("Int64")

    for permno, g in df.groupby("permno", sort=False):
        arrs = []
        for x in g["returns_5m"]:
            x = np.asarray(x, dtype=np.float64)
            x = x[np.isfinite(x)]
            if x.size:
                arrs.append(x)

        if arrs:
            week_store[permno].append(np.concatenate(arrs))

    if i % 250 == 0:
        print(f"Processed {i} daily files")

# flush final week
if current_week is not None:
    flush_week_to_year_buffer(current_week, week_store, year_buffers)

# flush remaining year buffers
for year in sorted(year_buffers.keys()):
    flush_year_buffer_to_disk(year, year_buffers, output_dir)

print("Done.")