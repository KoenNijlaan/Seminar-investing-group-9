from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# Paths and settings
# =========================================================
input_dir = Path("data_intermediate/weekly_returns")
output_dir = Path("data_intermediate/res_weekly")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "res_weekly.parquet"

ALPHA = 0.025
H = 0.5
MIN_OBS = 10
REQUIRE_FULL_4W = True  # strict 4-calendar-week window

# =========================================================
# Helper function
# =========================================================
def compute_res_from_array(x: np.ndarray, alpha: float, h: float, min_obs: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size < min_obs:
        return np.nan

    q = np.quantile(x, alpha)
    tail = x[x <= q]

    if tail.size == 0:
        return np.nan

    return -(x.size ** h) * tail.mean()

# =========================================================
# Load weekly returns
# =========================================================
print("Reading weekly returns...")

files = sorted(input_dir.glob("weekly_returns_*.parquet"))
if not files:
    raise FileNotFoundError("No files found matching weekly_returns_*.parquet")

dfs = []
for f in files:
    print(f"Reading {f.name}")
    dfs.append(pd.read_parquet(f))

df = pd.concat(dfs, ignore_index=True)

required_cols = {"permno", "week", "returns_5m_week"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[["permno", "week", "returns_5m_week"]].copy()
df["permno"] = df["permno"].astype("Int64")
df["week"] = pd.to_datetime(df["week"])
df = df.sort_values(["permno", "week"]).reset_index(drop=True)

print(f"Input rows: {len(df):,}")
print(f"Unique stocks: {df['permno'].nunique():,}")
print(f"Week range: {df['week'].min().date()} to {df['week'].max().date()}")

# =========================================================
# Compute rolling 4-week aggregate RES
# =========================================================
print("Computing weekly RES...")

results = []

for i, (permno, g) in enumerate(df.groupby("permno", sort=False), start=1):
    g = g.sort_values("week").reset_index(drop=True)

    week_to_returns = {}
    for row in g.itertuples(index=False):
        arr = np.asarray(row.returns_5m_week, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        week_to_returns[row.week] = arr

    stock_weeks = g["week"].tolist()

    for week in stock_weeks:
        required_weeks = [week - pd.Timedelta(weeks=k) for k in range(3, -1, -1)]

        REQUIRE_FULL_4W = True
        if REQUIRE_FULL_4W and not all(w in week_to_returns for w in required_weeks):
            res = np.nan
        else:
            arrays = []
            for w in required_weeks:
                if w in week_to_returns:
                    arr = week_to_returns[w]
                    if arr.size > 0:
                        arrays.append(arr)

            if len(arrays) == 0:
                res = np.nan
            else:
                pooled = np.concatenate(arrays)
                res = compute_res_from_array(
                    x=pooled,
                    alpha=ALPHA,
                    h=H,
                    min_obs=MIN_OBS
                )

        results.append({
            "permno": permno,
            "week": week,
            "res_weekly": res
        })

    if i % 1000 == 0:
        print(f"Processed {i:,} stocks...")

# =========================================================
# Finalize and save
# =========================================================
res_df = pd.DataFrame(results)
res_df = res_df.sort_values(["permno", "week"]).reset_index(drop=True)

n_total = len(res_df)
n_nonmissing = res_df["res_weekly"].notna().sum()

print("\nFinished.")
print(f"Output rows: {n_total:,}")
print(f"Non-missing RES: {n_nonmissing:,}")
print(f"Share non-missing: {n_nonmissing / n_total:.2%}")

print("\nPreview:")
print(res_df.head())

print("\nSummary:")
print(res_df["res_weekly"].describe())

res_df.to_parquet(output_path, index=False)
print(f"\nSaved to: {output_path}")