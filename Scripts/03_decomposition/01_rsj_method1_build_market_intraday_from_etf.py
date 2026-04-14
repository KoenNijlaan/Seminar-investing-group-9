"""
Build Method-1 Market Intraday Input

Purpose:
  Create the market intraday reference series from ETF data for RSJ method 1.

Inputs:
  - Converted ETF intraday files.

Outputs:
  - Market intraday files in data_intermediate/decomposition.

Main Steps:
  - Load ETF intraday files.
  - Build market return series.
  - Store method-1 market input files.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

INPUT_DIR = Path("data_intermediate/converted_parquet_etf")
OUTPUT_FILE = Path("data_intermediate/market_returns/market_intraday_spy.parquet")

TARGET_SYMBOL = "SPY"
MIN_DAILY_OBS = 80

SYMBOL_CANDIDATES = ["sym_root", "symbol", "ticker", "tic"]
DATE_CANDIDATES = ["date", "trading_date"]
RETURNS_VECTOR_CANDIDATES = ["returns_5m", "ret_5m_series", "intraday_returns"]
NOBS_CANDIDATES = ["n_obs", "n_intervals"]

def find_first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

def normalize_date(series):
    s = pd.to_datetime(series, errors="coerce")
    try:
        s = s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return s.dt.normalize()

def extract_return_vector(x):
    if x is None:
        return []

    if isinstance(x, np.ndarray):
        vals = x.tolist()
    elif isinstance(x, (list, tuple)):
        vals = list(x)
    else:
        return []

    out = []
    for v in vals:
        try:
            fv = float(v)
            out.append(fv if np.isfinite(fv) else np.nan)
        except Exception:
            out.append(np.nan)
    return out

def expand_one_row(row, date_col, returns_col, nobs_col=None):
    date_val = row[date_col]
    returns_vec = extract_return_vector(row[returns_col])

    if len(returns_vec) == 0:
        return None

    df = pd.DataFrame({
        "date": [date_val] * len(returns_vec),
        "interval": np.arange(1, len(returns_vec) + 1, dtype=int),
        "market_ret": returns_vec
    })
    if nobs_col is not None and nobs_col in row.index:
        df["n_obs"] = row[nobs_col]
    return df

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    files = sorted(INPUT_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {INPUT_DIR}")

    first_df = pd.read_parquet(files[0])
    first_cols = list(first_df.columns)

    symbol_col = find_first_existing(first_cols, SYMBOL_CANDIDATES)
    date_col = find_first_existing(first_cols, DATE_CANDIDATES)
    returns_col = find_first_existing(first_cols, RETURNS_VECTOR_CANDIDATES)
    nobs_col = find_first_existing(first_cols, NOBS_CANDIDATES)

    if symbol_col is None:
        raise ValueError(f"Could not find symbol column. Found columns: {first_cols}")
    if date_col is None:
        raise ValueError(f"Could not find date column. Found columns: {first_cols}")
    if returns_col is None:
        raise ValueError(
            f"Could not find intraday returns vector column. "
            f"Expected one of {RETURNS_VECTOR_CANDIDATES}. Found columns: {first_cols}"
        )

    print("Detected ETF columns:")
    print({
        "symbol_col": symbol_col,
        "date_col": date_col,
        "returns_col": returns_col,
        "nobs_col": nobs_col,
    })

    parts = []

    for i, file_path in enumerate(files, start=1):
        if i % 200 == 0 or i == 1:
            print(f"[{i}/{len(files)}] Processing {file_path.name}")

        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"  Skipping {file_path.name}: {e}")
            continue

        if symbol_col not in df.columns or date_col not in df.columns or returns_col not in df.columns:
            continue

        df = df[df[symbol_col].astype(str).str.upper() == TARGET_SYMBOL].copy()
        if df.empty:
            continue

        df[date_col] = normalize_date(df[date_col])

        expanded_rows = []
        for _, row in df.iterrows():
            expanded = expand_one_row(row, date_col, returns_col, nobs_col)
            if expanded is not None and not expanded.empty:
                expanded_rows.append(expanded)

        if not expanded_rows:
            continue

        out = pd.concat(expanded_rows, ignore_index=True)
        out = out.dropna(subset=["date", "interval", "market_ret"]).copy()

        if not out.empty:
            parts.append(out)

    if not parts:
        raise ValueError(
            "No SPY intraday observations found after filtering. "
            "Check whether converted_parquet_etf contains SPY rows with returns_5m."
        )

    market = pd.concat(parts, ignore_index=True)

    market["date"] = normalize_date(market["date"])
    market["interval"] = pd.to_numeric(market["interval"], errors="coerce")
    market["market_ret"] = pd.to_numeric(market["market_ret"], errors="coerce")

    market = market.dropna(subset=["date", "interval", "market_ret"]).copy()
    market["interval"] = market["interval"].astype("int64")

    market = (
        market.sort_values(["date", "interval"])
              .drop_duplicates(subset=["date", "interval"], keep="first")
              .reset_index(drop=True)
    )

    if "n_obs" in market.columns:
        valid_dates = market.groupby("date")["n_obs"].first()
        valid_dates = valid_dates[valid_dates >= MIN_DAILY_OBS].index
        market = market[market["date"].isin(valid_dates)].copy()

    market = market.sort_values(["date", "interval"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    market.to_parquet(OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Rows written: {len(market):,}")
    print(f"Unique dates: {market['date'].nunique():,}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nPreview:")
    print(market.head())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
