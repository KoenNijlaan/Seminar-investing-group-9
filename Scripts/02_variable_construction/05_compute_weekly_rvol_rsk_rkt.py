"""
Compute Weekly RVOL, RSK, And RKT

Purpose:
  Compute weekly higher-moment controls used in later regressions.

Inputs:
  - Weekly return data.

Outputs:
  - Weekly RVOL/RSK/RKT control files in data_intermediate.

Main Steps:
  - Calculate moments per stock-week.
  - Filter invalid observations.
  - Write control datasets.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR  = ROOT / "data_intermediate" / "converted_parquet"
OUTPUT_DIR = ROOT / "data_intermediate" / "rvol_rsk_rkt_weekly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "rvol_rsk_rkt_weekly.parquet"

MIN_OBS_PER_DAY   = 80
MIN_DAYS_PER_WEEK = 3

def compute_daily_measures(returns_5m) -> tuple[float, float, float] | None:
    if returns_5m is None:
        return None
    arr = np.asarray(returns_5m, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return None

    rv = float(np.sum(arr ** 2))
    if rv == 0:
        return None

    rsk = float(np.sqrt(n) * np.sum(arr ** 3) / rv ** 1.5)
    rkt = float(n          * np.sum(arr ** 4) / rv ** 2)

    return rv, rsk, rkt

def main():
    files = sorted(INPUT_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {INPUT_DIR}")

    print(f"Found {len(files)} daily files.\n")

    daily_parts = []

    for i, fp in enumerate(files, start=1):
        if i % 200 == 0:
            print(f"  Processed {i}/{len(files)} files...")

        df = pd.read_parquet(fp, columns=["permno", "date", "n_obs", "returns_5m"])

        df = df[df["n_obs"] >= MIN_OBS_PER_DAY].copy()
        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"])

        df["week"] = df["date"].dt.to_period("W-TUE").dt.end_time.dt.normalize()

        measures = df["returns_5m"].apply(compute_daily_measures)
        valid = measures.notna()
        df = df[valid].copy()
        measures = measures[valid]

        if df.empty:
            continue

        df["rv"]  = [m[0] for m in measures]
        df["rsk"] = [m[1] for m in measures]
        df["rkt"] = [m[2] for m in measures]

        daily_parts.append(df[["permno", "week", "rv", "rsk", "rkt"]].copy())

    print("\nCombining all daily parts...")
    daily_all = pd.concat(daily_parts, ignore_index=True)
    daily_all["permno"] = pd.to_numeric(daily_all["permno"], errors="coerce").astype("Int64")

    print("Aggregating to weekly level...")

    day_counts = (
        daily_all
        .groupby(["permno", "week"])["rv"]
        .count()
        .rename("n_days")
        .reset_index()
    )

    weekly_agg = (
        daily_all
        .groupby(["permno", "week"], as_index=False)
        .agg(
            sum_rv=("rv",  "sum"),
            rsk   =("rsk", "mean"),
            rkt   =("rkt", "mean"),
        )
    )

    weekly_agg["rvol"] = np.sqrt((252 / 5) * weekly_agg["sum_rv"])

    weekly = weekly_agg.merge(day_counts, on=["permno", "week"])

    weekly = weekly[weekly["n_days"] >= MIN_DAYS_PER_WEEK].copy()

    weekly = weekly[["permno", "week", "rvol", "rsk", "rkt", "n_days"]].copy()
    weekly = weekly.dropna(subset=["rvol", "rsk", "rkt"])
    weekly = weekly.sort_values(["week", "permno"]).reset_index(drop=True)

    print(f"\nFinal dataset: {len(weekly):,} stock-weeks")
    print(f"  Stocks : {weekly['permno'].nunique():,}")
    print(f"  Weeks  : {weekly['week'].nunique():,}")
    print(f"  Date range: {weekly['week'].min().date()} to {weekly['week'].max().date()}")
    print("\nSummary statistics:")
    print(weekly[["rvol", "rsk", "rkt"]].describe().round(6))

    weekly.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
