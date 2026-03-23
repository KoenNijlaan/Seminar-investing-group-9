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
BETAS_FILE = Path("data_intermediate/rolling_betas_method1.parquet")

DAILY_OUTPUT = Path("data_intermediate/rsj_method1_daily.parquet")
WEEKLY_OUTPUT = Path("data_intermediate/rsj_method1_weekly.parquet")

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


def semivariance_pos(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    pos = x[x > 0]
    if len(pos) == 0:
        return 0.0
    return float(np.sum(pos ** 2))


def semivariance_neg(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    neg = x[x < 0]
    if len(neg) == 0:
        return 0.0
    return float(np.sum(neg ** 2))


def safe_rsj(rv_pos, rv_neg):
    if pd.isna(rv_pos) or pd.isna(rv_neg):
        return np.nan
    denom = rv_pos + rv_neg
    if denom <= 0:
        return np.nan
    return (rv_pos - rv_neg) / denom


def main():
    if not STOCK_DIR.exists():
        raise FileNotFoundError(f"Stock input directory does not exist: {STOCK_DIR}")
    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Market file does not exist: {MARKET_FILE}")
    if not BETAS_FILE.exists():
        raise FileNotFoundError(f"Betas file does not exist: {BETAS_FILE}")

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

    betas = pd.read_parquet(BETAS_FILE)
    betas["permno"] = pd.to_numeric(betas["permno"], errors="coerce").astype("Int64")
    betas["week"] = normalize_date(betas["week"])
    betas["beta_hat"] = pd.to_numeric(betas["beta_hat"], errors="coerce")
    betas = betas.dropna(subset=["permno", "week", "beta_hat"]).copy()
    betas["permno"] = betas["permno"].astype("int64")

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

    df = df.merge(
        betas[["permno", "week", "beta_hat"]],
        on=["permno", "week"],
        how="left",
        validate="many_to_one"
    )

    df = df.dropna(subset=["permno", "date", "interval", "ret", "market_ret", "week", "beta_hat"]).copy()

    df["ret_sys"] = df["beta_hat"] * df["market_ret"]
    df["ret_idio"] = df["ret"] - df["ret_sys"]

    daily = (
        df.groupby(["permno", "date", "week"], as_index=False)
          .agg(
              rv_pos_sys=("ret_sys", semivariance_pos),
              rv_neg_sys=("ret_sys", semivariance_neg),
              rv_pos_idio=("ret_idio", semivariance_pos),
              rv_neg_idio=("ret_idio", semivariance_neg),
              n_obs=("ret", lambda x: int(np.isfinite(pd.to_numeric(x, errors="coerce")).sum())),
              beta_hat=("beta_hat", "first"),
          )
    )

    daily["rsj_sys_d"] = [
        safe_rsj(p, n) for p, n in zip(daily["rv_pos_sys"], daily["rv_neg_sys"])
    ]
    daily["rsj_idio_d"] = [
        safe_rsj(p, n) for p, n in zip(daily["rv_pos_idio"], daily["rv_neg_idio"])
    ]

    weekly = (
        daily.groupby(["permno", "week"], as_index=False)
             .agg(
                 rsj_sys_weekly=("rsj_sys_d", "mean"),
                 rsj_idio_weekly=("rsj_idio_d", "mean"),
                 n_days=("date", "nunique"),
                 n_obs_total=("n_obs", "sum"),
             )
    )

    DAILY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DAILY_OUTPUT, index=False)
    weekly.to_parquet(WEEKLY_OUTPUT, index=False)

    print("\nDone.")
    print(f"Daily rows written: {len(daily):,}")
    print(f"Weekly rows written: {len(weekly):,}")
    print(f"Daily output: {DAILY_OUTPUT}")
    print(f"Weekly output: {WEEKLY_OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)