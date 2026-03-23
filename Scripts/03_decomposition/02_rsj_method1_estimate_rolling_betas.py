from pathlib import Path
import sys
import pandas as pd
import numpy as np
import pyarrow.parquet as pq


# =========================
# User settings
# =========================
STOCK_DIR = Path("data_intermediate/converted_parquet")
MARKET_FILE = Path("data_intermediate/market_intraday_spy.parquet")
OUTPUT_FILE = Path("data_intermediate/rolling_betas_method1.parquet")

MIN_OBS_FOR_BETA = 50
ROLLING_WEEKS = 4

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
    return pd.to_datetime(series, errors="coerce").dt.tz_localize(None).dt.normalize()


def normalize_time(series):
    x = pd.to_datetime(series, errors="coerce")
    if hasattr(x.dt, "tz_localize"):
        try:
            x = x.dt.tz_localize(None)
        except TypeError:
            pass
    return x.dt.strftime("%H:%M:%S")


def make_week(series):
    s = pd.to_datetime(series, errors="coerce")
    return s.dt.to_period("W-SUN").dt.end_time.dt.normalize()


def ols_beta(y, x):
    valid = np.isfinite(y) & np.isfinite(x)
    y = y[valid]
    x = x[valid]

    n = len(y)
    if n < MIN_OBS_FOR_BETA:
        return np.nan, n

    denom = np.dot(x, x)
    if denom == 0:
        return np.nan, n

    beta = np.dot(x, y) / denom
    return beta, n


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
            f"Detected: {detected}, available: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["permno"] = pd.to_numeric(df[permno_col], errors="coerce")
    out["date"] = normalize_date(df[date_col])
    out["ret"] = pd.to_numeric(df[ret_col], errors="coerce")

    if interval_col is not None:
        out["interval"] = pd.to_numeric(df[interval_col], errors="coerce")
    elif time_col is not None:
        tmp = pd.DataFrame({
            "date": out["date"],
            "time": normalize_time(df[time_col]),
        })
        tmp = tmp.sort_values(["date", "time"]).reset_index(drop=True)
        out = out.reset_index(drop=True)
        out["interval"] = tmp.groupby("date").cumcount() + 1
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


def main():
    if not STOCK_DIR.exists():
        raise FileNotFoundError(f"Stock input directory does not exist: {STOCK_DIR}")
    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Market file does not exist: {MARKET_FILE}")

    stock_files = sorted(STOCK_DIR.glob("*.parquet"))
    if not stock_files:
        raise FileNotFoundError(f"No stock parquet files found in: {STOCK_DIR}")

    first_cols = pq.read_table(stock_files[0]).schema.names
    detected = detect_stock_columns(first_cols)

    print("Detected stock columns:")
    print(detected)

    market = pd.read_parquet(MARKET_FILE)
    market["date"] = normalize_date(market["date"])
    market["interval"] = pd.to_numeric(market["interval"], errors="coerce").astype("Int64")
    market["market_ret"] = pd.to_numeric(market["market_ret"], errors="coerce")
    market = market.dropna(subset=["date", "interval", "market_ret"]).copy()
    market["interval"] = market["interval"].astype("int64")

    stock_parts = []
    for i, file_path in enumerate(stock_files, start=1):
        print(f"[{i}/{len(stock_files)}] Loading {file_path.name}")
        try:
            part = load_one_stock_file(file_path, detected)
            stock_parts.append(part)
        except Exception as e:
            print(f"  Skipping {file_path.name}: {e}")

    if not stock_parts:
        raise ValueError("No stock files could be loaded.")

    stocks = pd.concat(stock_parts, ignore_index=True)

    df = stocks.merge(
        market,
        on=["date", "interval"],
        how="inner",
        validate="many_to_one"
    )

    df = df.dropna(subset=["permno", "date", "interval", "ret", "market_ret", "week"]).copy()
    df = df.sort_values(["permno", "date", "interval"]).reset_index(drop=True)

    unique_weeks = np.sort(df["week"].dropna().unique())
    week_to_pos = {w: i for i, w in enumerate(unique_weeks)}
    df["week_pos"] = df["week"].map(week_to_pos)

    results = []

    for permno, g in df.groupby("permno", sort=True):
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

            beta_hat, n_obs = ols_beta(
                hist["ret"].to_numpy(dtype=float),
                hist["market_ret"].to_numpy(dtype=float)
            )

            results.append({
                "permno": int(permno),
                "week": pd.Timestamp(w),
                "beta_hat": beta_hat,
                "n_obs_beta": int(n_obs),
                "n_weeks_used": int(len(window_weeks)),
            })

    betas = pd.DataFrame(results)
    betas = betas.sort_values(["permno", "week"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    betas.to_parquet(OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Rows written: {len(betas):,}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)