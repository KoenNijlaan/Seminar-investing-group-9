from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =========================
# User settings
# =========================
STOCK_DIR = Path("data_intermediate/converted_parquet")
MARKET_FILE = Path("data_intermediate/market_intraday_spy.parquet")

OUTPUT_INTRADAY = Path("data_intermediate/intraday_decomposed_method1.parquet")
OUTPUT_BETAS = Path("data_intermediate/rolling_betas_method1.parquet")
OUTPUT_WEEKLY_COUNTS = Path("data_intermediate/intraday_decomposed_method1_weekly_counts.parquet")

ROLLING_WEEKS = 4
MIN_OBS_FOR_BETA = 40          # only estimate/use beta if overlap > 40
MIN_WEEKLY_OBS = 40            # keep stock-week only if n_obs_total > 40
MIN_WEEKLY_DAYS = 3            # keep stock-week only if n_days >= 3

# Candidate column names
PERMNO_CANDIDATES = ["permno"]
DATE_CANDIDATES = ["date", "trading_date"]
RETURN_CANDIDATES = ["ret", "return", "r", "intraday_ret", "ret_5m"]
INTERVAL_CANDIDATES = ["interval", "interval_id", "bar", "k"]
TIME_CANDIDATES = ["time", "timestamp", "datetime"]


def find_first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def detect_stock_columns(columns):
    return {
        "permno_col": find_first_existing(columns, PERMNO_CANDIDATES),
        "date_col": find_first_existing(columns, DATE_CANDIDATES),
        "ret_col": find_first_existing(columns, RETURN_CANDIDATES),
        "interval_col": find_first_existing(columns, INTERVAL_CANDIDATES),
        "time_col": find_first_existing(columns, TIME_CANDIDATES),
    }


def normalize_date(series):
    s = pd.to_datetime(series, errors="coerce")
    try:
        s = s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return s.dt.normalize()


def normalize_time(series):
    s = pd.to_datetime(series, errors="coerce")
    try:
        s = s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return s.dt.strftime("%H:%M:%S")


def make_week(series):
    # Keep identical across all scripts in the project
    s = pd.to_datetime(series, errors="coerce")
    return s.dt.to_period("W-SUN").dt.end_time.dt.normalize()


def ols_beta(y, x):
    valid = np.isfinite(y) & np.isfinite(x)
    y = y[valid]
    x = x[valid]

    n = len(y)
    if n <= MIN_OBS_FOR_BETA:
        return np.nan, n

    denom = np.dot(x, x)
    if denom <= 0:
        return np.nan, n

    beta = np.dot(x, y) / denom
    return float(beta), int(n)


def load_one_stock_file(file_path, detected):
    table = pq.read_table(file_path)
    df = table.to_pandas()

    permno_col = detected["permno_col"]
    date_col = detected["date_col"]
    ret_col = detected["ret_col"]
    interval_col = detected["interval_col"]
    time_col = detected["time_col"]

    if permno_col is None or date_col is None or ret_col is None:
        raise ValueError(
            f"Missing required columns in {file_path.name}. "
            f"Detected={detected}, available={list(df.columns)}"
        )

    out = pd.DataFrame()
    out["permno"] = pd.to_numeric(df[permno_col], errors="coerce")
    out["date"] = normalize_date(df[date_col])
    out["ret"] = pd.to_numeric(df[ret_col], errors="coerce")

    if interval_col is not None:
        out["interval"] = pd.to_numeric(df[interval_col], errors="coerce")
    elif time_col is not None:
        temp = pd.DataFrame({
            "date": out["date"],
            "time": normalize_time(df[time_col]),
        })
        temp = temp.sort_values(["date", "time"]).reset_index(drop=True)
        out = out.reset_index(drop=True)
        out["interval"] = temp.groupby("date").cumcount() + 1
    else:
        raise ValueError(
            f"No interval/time column found in {file_path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    out = out.dropna(subset=["permno", "date", "interval", "ret"]).copy()
    out["permno"] = out["permno"].astype("int64")
    out["interval"] = out["interval"].astype("int64")
    out["week"] = make_week(out["date"])

    return out


def load_all_stocks(stock_dir):
    stock_files = sorted(stock_dir.glob("*.parquet"))
    if not stock_files:
        raise FileNotFoundError(f"No stock parquet files found in {stock_dir}")

    first_cols = pq.read_table(stock_files[0]).schema.names
    detected = detect_stock_columns(first_cols)

    print("Detected stock columns:")
    print(detected)

    parts = []
    for i, file_path in enumerate(stock_files, start=1):
        print(f"[{i}/{len(stock_files)}] Loading {file_path.name}")
        try:
            part = load_one_stock_file(file_path, detected)
            parts.append(part)
        except Exception as e:
            print(f"  Skipping {file_path.name}: {e}")

    if not parts:
        raise ValueError("No stock files could be loaded.")

    stocks = pd.concat(parts, ignore_index=True)
    stocks = stocks.sort_values(["permno", "date", "interval"]).reset_index(drop=True)
    return stocks


def load_market(market_file):
    market = pd.read_parquet(market_file)
    market["date"] = normalize_date(market["date"])
    market["interval"] = pd.to_numeric(market["interval"], errors="coerce")
    market["market_ret"] = pd.to_numeric(market["market_ret"], errors="coerce")

    market = market.dropna(subset=["date", "interval", "market_ret"]).copy()
    market["interval"] = market["interval"].astype("int64")

    # Force one observation per date-interval
    market = (
        market.sort_values(["date", "interval"])
              .drop_duplicates(subset=["date", "interval"], keep="first")
              .reset_index(drop=True)
    )
    return market


def estimate_rolling_betas(merged):
    unique_weeks = np.sort(merged["week"].dropna().unique())
    week_to_pos = {w: i for i, w in enumerate(unique_weeks)}

    merged = merged.copy()
    merged["week_pos"] = merged["week"].map(week_to_pos)

    results = []

    for permno, g in merged.groupby("permno", sort=True):
        g = g.sort_values(["date", "interval"]).copy()
        perm_weeks = np.sort(g["week"].unique())

        for w in perm_weeks:
            w_pos = week_to_pos[w]
            start_pos = w_pos - ROLLING_WEEKS
            end_pos = w_pos - 1

            if end_pos < 0:
                continue

            window_weeks = unique_weeks[max(0, start_pos): end_pos + 1]
            hist = g[g["week"].isin(window_weeks)]

            beta_hat, n_obs_beta = ols_beta(
                hist["ret"].to_numpy(dtype=float),
                hist["market_ret"].to_numpy(dtype=float)
            )

            results.append({
                "permno": int(permno),
                "week": pd.Timestamp(w),
                "beta_hat": beta_hat,
                "n_obs_beta": int(n_obs_beta),
                "n_weeks_used": int(len(window_weeks)),
            })

    betas = pd.DataFrame(results)
    if betas.empty:
        raise ValueError("No rolling betas could be estimated.")

    # Keep only usable betas
    betas = betas[
        betas["beta_hat"].notna() &
        (betas["n_obs_beta"] > MIN_OBS_FOR_BETA)
    ].copy()

    betas = betas.sort_values(["permno", "week"]).reset_index(drop=True)
    return betas


def main():
    if not STOCK_DIR.exists():
        raise FileNotFoundError(f"Stock input directory does not exist: {STOCK_DIR}")
    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Market file does not exist: {MARKET_FILE}")

    print("Loading stock data...")
    stocks = load_all_stocks(STOCK_DIR)

    print("Loading market data...")
    market = load_market(MARKET_FILE)

    # Exact alignment on stock grid
    merged = stocks.merge(
        market,
        on=["date", "interval"],
        how="inner",
        validate="many_to_one"
    )

    merged = merged.dropna(subset=["permno", "date", "interval", "ret", "market_ret", "week"]).copy()
    merged = merged.sort_values(["permno", "date", "interval"]).reset_index(drop=True)

    print("Estimating rolling betas...")
    betas = estimate_rolling_betas(merged)

    print("Merging betas back to intraday panel...")
    decomp = merged.merge(
        betas[["permno", "week", "beta_hat", "n_obs_beta"]],
        on=["permno", "week"],
        how="inner",
        validate="many_to_one"
    )

    # Decomposition
    decomp["ret_sys"] = decomp["beta_hat"] * decomp["market_ret"]
    decomp["ret_idio"] = decomp["ret"] - decomp["ret_sys"]

    # Weekly quality filter
    weekly_counts = (
        decomp.groupby(["permno", "week"], as_index=False)
              .agg(
                  n_days=("date", "nunique"),
                  n_obs_total=("ret", lambda x: int(np.isfinite(pd.to_numeric(x, errors="coerce")).sum())),
                  beta_hat=("beta_hat", "first"),
                  n_obs_beta=("n_obs_beta", "first"),
              )
    )

    weekly_counts["keep_week"] = (
        (weekly_counts["n_obs_total"] > MIN_WEEKLY_OBS) &
        (weekly_counts["n_days"] >= MIN_WEEKLY_DAYS)
    )

    decomp = decomp.merge(
        weekly_counts[["permno", "week", "n_days", "n_obs_total", "keep_week"]],
        on=["permno", "week"],
        how="left",
        validate="many_to_one"
    )

    decomp = decomp[decomp["keep_week"]].copy()

    # Final column order
    keep_cols = [
        "permno", "date", "week", "interval",
        "ret", "market_ret", "beta_hat",
        "ret_sys", "ret_idio",
        "n_days", "n_obs_total"
    ]
    decomp = decomp[keep_cols].sort_values(["permno", "date", "interval"]).reset_index(drop=True)

    # Keep only retained weekly summaries
    weekly_counts = weekly_counts[weekly_counts["keep_week"]].copy()
    weekly_counts = weekly_counts.sort_values(["permno", "week"]).reset_index(drop=True)

    # Save
    OUTPUT_INTRADAY.parent.mkdir(parents=True, exist_ok=True)
    decomp.to_parquet(OUTPUT_INTRADAY, index=False)
    betas.to_parquet(OUTPUT_BETAS, index=False)
    weekly_counts.to_parquet(OUTPUT_WEEKLY_COUNTS, index=False)

    print("\nDone.")
    print(f"Intraday decomposed rows: {len(decomp):,}")
    print(f"Rolling betas rows: {len(betas):,}")
    print(f"Retained stock-weeks: {len(weekly_counts):,}")
    print(f"Intraday output: {OUTPUT_INTRADAY}")
    print(f"Betas output: {OUTPUT_BETAS}")
    print(f"Weekly counts output: {OUTPUT_WEEKLY_COUNTS}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)