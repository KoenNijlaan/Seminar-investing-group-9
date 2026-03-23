from pathlib import Path
import sys
import pandas as pd
import pyarrow.parquet as pq


# =========================
# User settings
# =========================
INPUT_DIR = Path("data_intermediate/converted_parquet_etf")
OUTPUT_FILE = Path("data_intermediate/market_intraday_spy.parquet")

TARGET_SYMBOL = "SPY"

# Candidate column names
SYMBOL_CANDIDATES = ["sym_root", "symbol", "ticker", "tic"]
DATE_CANDIDATES = ["date", "trading_date"]
RETURN_CANDIDATES = ["ret", "return", "r", "intraday_ret", "ret_5m"]
INTERVAL_CANDIDATES = ["interval", "interval_id", "bar", "k"]
TIME_CANDIDATES = ["time", "timestamp", "datetime"]


def find_first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def detect_columns(columns):
    symbol_col = find_first_existing(columns, SYMBOL_CANDIDATES)
    date_col = find_first_existing(columns, DATE_CANDIDATES)
    ret_col = find_first_existing(columns, RETURN_CANDIDATES)
    interval_col = find_first_existing(columns, INTERVAL_CANDIDATES)
    time_col = find_first_existing(columns, TIME_CANDIDATES)

    return {
        "symbol_col": symbol_col,
        "date_col": date_col,
        "ret_col": ret_col,
        "interval_col": interval_col,
        "time_col": time_col,
    }


def normalize_date(series):
    # Handles strings, timestamps, timezone-aware timestamps
    return pd.to_datetime(series, errors="coerce").dt.date


def normalize_time(series):
    x = pd.to_datetime(series, errors="coerce")
    return x.dt.strftime("%H:%M:%S")


def process_one_file(file_path, detected):
    table = pq.read_table(file_path)
    df = table.to_pandas()

    symbol_col = detected["symbol_col"]
    date_col = detected["date_col"]
    ret_col = detected["ret_col"]
    interval_col = detected["interval_col"]
    time_col = detected["time_col"]

    # Keep only SPY
    df = df[df[symbol_col].astype(str).str.upper() == TARGET_SYMBOL].copy()
    if df.empty:
        return None

    # Date
    df["date"] = normalize_date(df[date_col])

    # Interval logic
    if interval_col is not None:
        df["interval"] = df[interval_col]
    elif time_col is not None:
        df["time"] = normalize_time(df[time_col])
        # dense rank of intraday timestamps within day
        df = df.sort_values(["date", "time"])
        df["interval"] = df.groupby("date").cumcount() + 1
    else:
        raise ValueError(
            f"No interval/time column found in {file_path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    # Return
    df["market_ret"] = pd.to_numeric(df[ret_col], errors="coerce")

    # Keep clean output
    keep_cols = ["date", "interval", "market_ret"]
    if time_col is not None and "time" in df.columns:
        keep_cols.append("time")

    out = df[keep_cols].copy()
    out = out.dropna(subset=["date", "interval", "market_ret"])

    return out


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    files = sorted(INPUT_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {INPUT_DIR}")

    # Detect columns from first file
    first_table = pq.read_table(files[0])
    first_columns = first_table.schema.names
    detected = detect_columns(first_columns)

    if detected["symbol_col"] is None:
        raise ValueError(
            f"Could not find symbol column. Expected one of {SYMBOL_CANDIDATES}. "
            f"Found columns: {first_columns}"
        )

    if detected["date_col"] is None:
        raise ValueError(
            f"Could not find date column. Expected one of {DATE_CANDIDATES}. "
            f"Found columns: {first_columns}"
        )

    if detected["ret_col"] is None:
        raise ValueError(
            f"Could not find return column. Expected one of {RETURN_CANDIDATES}. "
            f"Found columns: {first_columns}\n\n"
            "This usually means your ETF parquet files contain daily aggregates "
            "(like rv/res/rquantile) instead of true intraday returns."
        )

    if detected["interval_col"] is None and detected["time_col"] is None:
        raise ValueError(
            f"Could not find interval/time column. Expected one of "
            f"{INTERVAL_CANDIDATES + TIME_CANDIDATES}. "
            f"Found columns: {first_columns}"
        )

    print("Detected columns:")
    print(detected)

    all_parts = []
    n_kept_files = 0

    for i, file_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Processing {file_path.name}")
        try:
            part = process_one_file(file_path, detected)
            if part is not None and not part.empty:
                all_parts.append(part)
                n_kept_files += 1
        except Exception as e:
            print(f"  Skipping {file_path.name}: {e}")

    if not all_parts:
        raise ValueError(
            "No SPY intraday observations were found. "
            "Check whether the ETF parquet files actually contain SPY and intraday returns."
        )

    market = pd.concat(all_parts, ignore_index=True)

    # Final cleaning
    market = market.drop_duplicates(subset=["date", "interval"])
    market = market.sort_values(["date", "interval"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    market.to_parquet(OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Files with SPY data used: {n_kept_files}")
    print(f"Rows written: {len(market):,}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)