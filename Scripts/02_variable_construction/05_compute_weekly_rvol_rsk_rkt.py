"""
Compute weekly RVOL, RSK, RKT from 5-minute intraday returns.

Bollerslev, Li, Zhao (2020) "Good Volatility, Bad Volatility, and the
Cross-Section of Stock Returns", equations (7)-(10):

Daily realized measures (eqs. 7-8):
  RV_t   = sum_i r_{t,i}^2
  RSK_t  = sqrt(n) * sum_i r_{t,i}^3  /  RV_t^(3/2)
  RKT_t  = n       * sum_i r_{t,i}^4  /  RV_t^2
where n = number of 5-minute intervals on day t.

Weekly aggregation (eqs. 9-10):
  RVOL^week_tau = sqrt(252/5 * sum_{i=0}^{4} RV_{tau-i})   [annualized vol]
  RSK^week_tau  = (1/5) * sum_{i=0}^{4} RSK_{tau-i}        [average of daily]
  RKT^week_tau  = (1/5) * sum_{i=0}^{4} RKT_{tau-i}        [average of daily]

RSK and RKT are NOT recomputed from pooled intraday returns across the week;
they are daily measures that are averaged over the 5 trading days.

Input : data_intermediate/converted_parquet/*.parquet
Output: data_intermediate/rvol_rsk_rkt_weekly/rvol_rsk_rkt_weekly.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Settings
# ============================================================
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR  = ROOT / "data_intermediate" / "converted_parquet"
OUTPUT_DIR = ROOT / "data_intermediate" / "rvol_rsk_rkt_weekly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "rvol_rsk_rkt_weekly.parquet"

MIN_OBS_PER_DAY   = 80   # minimum intraday intervals to count a day as valid
MIN_DAYS_PER_WEEK = 3    # minimum valid trading days to keep a stock-week


# ============================================================
# Per-day measure computation
# ============================================================
def compute_daily_measures(returns_5m) -> tuple[float, float, float] | None:
    """
    Given one day's intraday returns array, return (RV, RSK, RKT) or None.

    RV  = sum r^2
    RSK = sqrt(n) * sum(r^3) / RV^(3/2)
    RKT = n       * sum(r^4) / RV^2
    """
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


# ============================================================
# Main
# ============================================================
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

        # Drop days with too few intraday observations
        df = df[df["n_obs"] >= MIN_OBS_PER_DAY].copy()
        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"])

        # Week label: W-TUE period-end timestamp (matches rest of pipeline)
        df["week"] = df["date"].dt.to_period("W-TUE").dt.end_time.dt.normalize()

        # Compute daily RV, RSK, RKT per row
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

    # Count valid trading days per (permno, week)
    day_counts = (
        daily_all
        .groupby(["permno", "week"])["rv"]
        .count()
        .rename("n_days")
        .reset_index()
    )

    # Weekly aggregation:
    #   RVOL = sqrt(252/5 * sum_daily_RV)   [annualized realized volatility]
    #   RSK  = mean of daily RSK             [eq. 10]
    #   RKT  = mean of daily RKT             [eq. 10]
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

    # Apply minimum-days filter
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
