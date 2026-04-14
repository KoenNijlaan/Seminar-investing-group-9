"""
RES Decomposition From Log Return Splits

Purpose:
  Decompose RES into total, systematic, and idiosyncratic components.

Inputs:
  - Log-return split intermediates and market series.

Outputs:
  - RES decomposition outputs for panel construction.

Main Steps:
  - Load split components.
  - Run decomposition formulas.
  - Save total/systematic/idiosyncratic outputs.
"""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data_intermediate" / "converted_parquet"
SPY_FILE = ROOT / "data_intermediate" / "market_returns" / "market_intraday_spy.parquet"
OUTPUT_DIR = ROOT / "data_intermediate" / "decomposition" / "log_returns_split_spy_daily"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_INTERVALS = 78
LOOKBACK_WEEKS = 4
MIN_TRANSACTIONS_PER_DAY = 80
MIN_VALID_DAYS_PER_WEEK = 3
MIN_TRAIN_OBS = 78

MAX_WEEKS = None

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def to_1d_float_array(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    try:
        arr = np.array(x, dtype="float64", copy=True).reshape(-1)
    except Exception:
        return None
    arr[~np.isfinite(arr)] = np.nan
    return arr

def simple_to_log_array(x, expected_len=None):
    arr = to_1d_float_array(x)
    if arr is None:
        return None

    if expected_len is not None and len(arr) != expected_len:
        return None

    if np.any(~np.isfinite(arr)):
        return None

    if np.any(arr <= -1):
        return None

    return np.log1p(arr)

def fit_market_model(stock_arrays, spy_arrays):
    y_list = []
    x_list = []

    for y_arr, x_arr in zip(stock_arrays, spy_arrays):
        if y_arr is None or x_arr is None:
            continue
        if len(y_arr) != len(x_arr):
            continue

        mask = np.isfinite(y_arr) & np.isfinite(x_arr)
        if mask.sum() == 0:
            continue

        y_list.append(y_arr[mask])
        x_list.append(x_arr[mask])

    if not y_list:
        return None

    y = np.concatenate(y_list)
    x = np.concatenate(x_list)

    if len(y) < MIN_TRAIN_OBS:
        return None

    if np.nanstd(x) == 0:
        return None

    X = np.column_stack([np.ones(len(x)), x])

    try:
        alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        return None

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "n_train_obs": int(len(y)),
    }

def load_spy_data(spy_file: Path) -> pd.DataFrame:
    print(f"Loading SPY file from: {spy_file.resolve()}")

    if not spy_file.exists():
        raise FileNotFoundError(f"SPY file not found: {spy_file}")

    spy = pd.read_parquet(spy_file)
    spy = clean_columns(spy)

    required_cols = {"date", "interval", "market_ret"}
    missing = required_cols - set(spy.columns)
    if missing:
        raise ValueError(
            f"SPY file is missing columns: {missing}. "
            f"Available columns: {spy.columns.tolist()}"
        )

    spy = spy.copy()
    spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()
    spy["interval"] = pd.to_numeric(spy["interval"], errors="coerce")
    spy["market_ret"] = pd.to_numeric(spy["market_ret"], errors="coerce")

    spy = spy.dropna(subset=["date", "interval", "market_ret"]).copy()
    spy = spy.sort_values(["date", "interval"])

    spy_daily = (
        spy.groupby("date", sort=True)["market_ret"]
        .apply(list)
        .reset_index(name="market_ret_5m")
    )

    spy_daily["spy_log_5m"] = spy_daily["market_ret_5m"].apply(
        lambda x: simple_to_log_array(x, expected_len=EXPECTED_INTERVALS)
    )

    bad_dates = spy_daily["spy_log_5m"].isna().sum()
    if bad_dates > 0:
        print(f"Dropping {bad_dates} SPY dates without exactly {EXPECTED_INTERVALS} valid intervals.")

    spy_daily = spy_daily.loc[spy_daily["spy_log_5m"].notna()].copy()
    spy_daily = spy_daily.sort_values("date").reset_index(drop=True)

    print(f"Loaded SPY dates: {len(spy_daily)}")
    return spy_daily[["date", "spy_log_5m"]]

def discover_stock_files(input_dir: Path):
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    records = []
    for path in files:
        try:
            dt = pd.to_datetime(path.stem).normalize()
        except Exception:
            continue
        week = dt.to_period("W-TUE")
        records.append({
            "date": dt,
            "path": path,
            "week": week,
            "week_end": week.end_time.normalize(),
        })

    if not records:
        raise ValueError(f"No date-named parquet files found in {input_dir}")

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)

@lru_cache(maxsize=40)
def load_stock_day_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    df = pd.read_parquet(path)
    df = clean_columns(df)

    required_cols = {"date", "permno", "sym_root", "n_obs", "returns_5m"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    keep_cols = ["date", "permno", "sym_root", "n_obs", "returns_5m"]
    for optional_col in ["ret_crsp", "ex"]:
        if optional_col in df.columns:
            keep_cols.append(optional_col)

    df = df[keep_cols].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["n_obs"] = pd.to_numeric(df["n_obs"], errors="coerce")

    df["stock_log_5m"] = df["returns_5m"].apply(
        lambda x: simple_to_log_array(x, expected_len=EXPECTED_INTERVALS)
    )

    return df

def process_one_week(
    current_week,
    current_week_end,
    file_map: pd.DataFrame,
    spy_df: pd.DataFrame,
):
    week_start = current_week.start_time.normalize()

    load_start = week_start - pd.Timedelta(weeks=LOOKBACK_WEEKS)
    load_end = current_week_end

    week_files = file_map[
        (file_map["date"] >= load_start) & (file_map["date"] <= load_end)
    ].copy()

    if week_files.empty:
        print(f"Week ending {current_week_end.date()}: no files found.")
        return

    daily_frames = []
    for _, row in week_files.iterrows():
        try:
            day_df = load_stock_day_cached(str(row["path"])).copy()
            daily_frames.append(day_df)
        except Exception as e:
            print(f"Could not load {row['path'].name}: {e}")

    if not daily_frames:
        print(f"Week ending {current_week_end.date()}: no daily data could be loaded.")
        return

    df = pd.concat(daily_frames, ignore_index=True)

    df = df.merge(spy_df, on="date", how="left")

    df["week"] = df["date"].dt.to_period("W-TUE")
    df["week_end"] = df["week"].dt.end_time.dt.normalize()

    df["is_valid_day"] = (
        (df["n_obs"] >= MIN_TRANSACTIONS_PER_DAY)
        & df["stock_log_5m"].notna()
        & df["spy_log_5m"].notna()
    )

    current_valid = df[(df["week"] == current_week) & (df["is_valid_day"])].copy()
    if current_valid.empty:
        print(f"Week ending {current_week_end.date()}: no valid current-week rows.")
        return

    valid_counts = current_valid.groupby("permno").size()
    eligible_permnos = set(valid_counts[valid_counts >= MIN_VALID_DAYS_PER_WEEK].index)

    if not eligible_permnos:
        print(f"Week ending {current_week_end.date()}: no permnos with >= {MIN_VALID_DAYS_PER_WEEK} valid days.")
        return

    current_valid = current_valid[current_valid["permno"].isin(eligible_permnos)].copy()

    train_df = df[
        (df["date"] >= load_start) &
        (df["date"] < week_start) &
        (df["is_valid_day"]) &
        (df["permno"].isin(eligible_permnos))
    ].copy()

    current_groups = {permno: grp.sort_values("date") for permno, grp in current_valid.groupby("permno")}
    train_groups = {permno: grp.sort_values("date") for permno, grp in train_df.groupby("permno")}

    out_rows = []

    for permno, curr_grp in current_groups.items():
        train_grp = train_groups.get(permno)

        if train_grp is None or train_grp.empty:
            continue

        model = fit_market_model(
            stock_arrays=train_grp["stock_log_5m"],
            spy_arrays=train_grp["spy_log_5m"],
        )

        if model is None:
            continue

        alpha = model["alpha"]
        beta = model["beta"]
        n_train_obs = model["n_train_obs"]

        for _, row in curr_grp.iterrows():
            y = row["stock_log_5m"]
            x = row["spy_log_5m"]

            if y is None or x is None or len(y) != len(x):
                continue

            systematic = alpha + beta * x
            idiosyncratic = y - systematic

            out_row = {
                "date": row["date"],
                "week_end": row["week_end"],
                "permno": row["permno"],
                "sym_root": row["sym_root"],
                "n_obs": row["n_obs"],
                "alpha_spy": alpha,
                "beta_spy": beta,
                "n_train_obs": n_train_obs,
                "systematic_log_returns_5m": systematic.tolist(),
                "idiosyncratic_log_returns_5m": idiosyncratic.tolist(),
            }

            if "ret_crsp" in row.index:
                out_row["ret_crsp"] = row["ret_crsp"]
            if "ex" in row.index:
                out_row["ex"] = row["ex"]

            out_rows.append(out_row)

    if not out_rows:
        print(f"Week ending {current_week_end.date()}: no output rows after regression.")
        return

    out_df = pd.DataFrame(out_rows).sort_values(["date", "permno"])

    for out_date, day_df in out_df.groupby("date"):
        out_path = OUTPUT_DIR / f"{out_date.strftime('%Y-%m-%d')}.parquet"
        day_df.to_parquet(out_path, index=False)

    print(
        f"Week ending {current_week_end.date()}: saved "
        f"{len(out_df)} rows across {out_df['date'].nunique()} daily files."
    )

def main():
    print("Project root:", ROOT)
    print("Input dir:", INPUT_DIR)
    print("SPY file:", SPY_FILE)
    print("Output dir:", OUTPUT_DIR)
    print()

    spy_df = load_spy_data(SPY_FILE)
    file_map = discover_stock_files(INPUT_DIR)

    unique_weeks = (
        file_map[["week", "week_end"]]
        .drop_duplicates()
        .sort_values("week_end")
        .reset_index(drop=True)
    )

    if MAX_WEEKS is not None:
        unique_weeks = unique_weeks.head(MAX_WEEKS).copy()

    print(f"Found {len(file_map)} daily stock files")
    print(f"Processing {len(unique_weeks)} weeks\n")

    for i, row in unique_weeks.iterrows():
        current_week = row["week"]
        current_week_end = row["week_end"]

        print(f"[{i+1}/{len(unique_weeks)}] Processing week ending {current_week_end.date()}")

        try:
            process_one_week(
                current_week=current_week,
                current_week_end=current_week_end,
                file_map=file_map,
                spy_df=spy_df,
            )
        except Exception as e:
            print(f"Error in week ending {current_week_end.date()}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
