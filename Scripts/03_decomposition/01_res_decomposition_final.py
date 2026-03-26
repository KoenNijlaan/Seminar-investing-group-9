from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Settings
# ============================================================
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data_intermediate" / "decomposition" / "log_returns_split_spy_daily"
OUTPUT_DIR = ROOT / "data_intermediate" / "weekly_data_res_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "weekly_res_sys_idio.parquet"

P = 0.025
H = 0.5
EXPECTED_INTERVALS = 78
MIN_VALID_DAYS_PER_WEEK = 3


# ============================================================
# Helpers
# ============================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_float_array(x, expected_len=None):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    try:
        arr = np.array(x, dtype="float64", copy=True).reshape(-1)
    except Exception:
        return None
    if expected_len is not None and len(arr) != expected_len:
        return None
    arr[~np.isfinite(arr)] = np.nan
    return arr


def empirical_es(x: np.ndarray, p: float):
    if x is None or len(x) == 0:
        return np.nan, np.nan, 0

    x = np.array(x, dtype="float64", copy=True)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan, 0

    q = np.quantile(x, p)
    tail = x[x <= q]

    if len(tail) == 0:
        return q, np.nan, 0

    es = tail.mean()
    return q, es, len(tail)


def compute_weekly_res_from_arrays(arrays, p=P, h=H):
    valid = []

    for arr in arrays:
        a = to_float_array(arr, expected_len=EXPECTED_INTERVALS)
        if a is None:
            continue
        a = a[np.isfinite(a)]
        if len(a) == 0:
            continue
        valid.append(a)

    if not valid:
        return {
            "c": 0,
            "quantile": np.nan,
            "es_intraday": np.nan,
            "n_tail": 0,
            "res_raw": np.nan,
            "res_pos": np.nan,
        }

    pooled = np.concatenate(valid)
    c = len(pooled)

    q, es, n_tail = empirical_es(pooled, p)

    res_raw = np.nan if not np.isfinite(es) else (c ** h) * es

    return {
        "c": int(c),
        "quantile": float(q) if np.isfinite(q) else np.nan,
        "es_intraday": float(es) if np.isfinite(es) else np.nan,
        "n_tail": int(n_tail),
        "res_raw": float(res_raw) if np.isfinite(res_raw) else np.nan,
        "res_pos": float(-res_raw) if np.isfinite(res_raw) else np.nan,
    }


def discover_files(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    rows = []
    for f in files:
        try:
            d = pd.to_datetime(f.stem).normalize()
        except Exception:
            continue
        week = d.to_period("W-FRI")
        rows.append({
            "date": d,
            "week": week,
            "week_end": week.end_time.normalize(),
            "path": f,
        })

    if not rows:
        raise ValueError("No date-named parquet files found.")

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def read_daily_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = clean_columns(df)

    required = {
        "date",
        "week_end",
        "permno",
        "sym_root",
        "systematic_log_returns_5m",
        "idiosyncratic_log_returns_5m",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    keep_cols = [
        "date",
        "week_end",
        "permno",
        "sym_root",
        "systematic_log_returns_5m",
        "idiosyncratic_log_returns_5m",
    ]

    for c in ["ret_crsp", "ex", "n_obs", "alpha_spy", "beta_spy", "n_train_obs"]:
        if c in df.columns:
            keep_cols.append(c)

    df = df[keep_cols].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["week_end"] = pd.to_datetime(df["week_end"]).dt.normalize()
    return df


# ============================================================
# Weekly aggregation
# ============================================================
def aggregate_one_week(week_files: pd.DataFrame, p=P, h=H) -> pd.DataFrame:
    parts = []

    for _, row in week_files.iterrows():
        try:
            parts.append(read_daily_file(row["path"]))
        except Exception as e:
            print(f"Skipping {row['path'].name}: {e}")

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    results = []
    grouped = df.groupby(["permno", "sym_root", "week_end"], sort=True)

    for (permno, sym_root, week_end), grp in grouped:
        grp = grp.sort_values("date")
        n_valid_days = grp["date"].nunique()

        if n_valid_days < MIN_VALID_DAYS_PER_WEEK:
            continue

        total_arrays = []
        for sys_arr, idio_arr in zip(
            grp["systematic_log_returns_5m"],
            grp["idiosyncratic_log_returns_5m"],
        ):
            sys_a = to_float_array(sys_arr, expected_len=EXPECTED_INTERVALS)
            idio_a = to_float_array(idio_arr, expected_len=EXPECTED_INTERVALS)

            if sys_a is None or idio_a is None:
                continue
            if len(sys_a) != len(idio_a):
                continue

            total_arrays.append(sys_a + idio_a)

        sys_stats = compute_weekly_res_from_arrays(grp["systematic_log_returns_5m"], p=p, h=h)
        idio_stats = compute_weekly_res_from_arrays(grp["idiosyncratic_log_returns_5m"], p=p, h=h)
        total_stats = compute_weekly_res_from_arrays(total_arrays, p=p, h=h)

        row = {
            "permno": permno,
            "sym_root": sym_root,
            "week_end": week_end,
            "n_valid_days": int(n_valid_days),

            "c_total": total_stats["c"],
            "q_total_p025": total_stats["quantile"],
            "es_total_intraday_p025": total_stats["es_intraday"],
            "n_tail_total_p025": total_stats["n_tail"],
            "res_total_raw_p025": total_stats["res_raw"],
            "res_total_p025": total_stats["res_pos"],

            "c_sys": sys_stats["c"],
            "q_sys_p025": sys_stats["quantile"],
            "es_sys_intraday_p025": sys_stats["es_intraday"],
            "n_tail_sys_p025": sys_stats["n_tail"],
            "res_sys_raw_p025": sys_stats["res_raw"],
            "res_sys_p025": sys_stats["res_pos"],

            "c_idio": idio_stats["c"],
            "q_idio_p025": idio_stats["quantile"],
            "es_idio_intraday_p025": idio_stats["es_intraday"],
            "n_tail_idio_p025": idio_stats["n_tail"],
            "res_idio_raw_p025": idio_stats["res_raw"],
            "res_idio_p025": idio_stats["res_pos"],
        }

        if "ret_crsp" in grp.columns:
            daily_ret = pd.to_numeric(grp["ret_crsp"], errors="coerce")
            if daily_ret.notna().any():
                row["weekly_ret_crsp"] = float(np.prod(1 + daily_ret.dropna()) - 1)

        if "ex" in grp.columns:
            ex_vals = grp["ex"].dropna()
            row["ex"] = ex_vals.iloc[0] if len(ex_vals) > 0 else None

        if "alpha_spy" in grp.columns:
            row["alpha_spy_week"] = pd.to_numeric(grp["alpha_spy"], errors="coerce").mean()

        if "beta_spy" in grp.columns:
            row["beta_spy_week"] = pd.to_numeric(grp["beta_spy"], errors="coerce").mean()

        if "n_train_obs" in grp.columns:
            row["avg_n_train_obs"] = pd.to_numeric(grp["n_train_obs"], errors="coerce").mean()

        results.append(row)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(["week_end", "permno"]).reset_index(drop=True)


# ============================================================
# Main
# ============================================================
def main():
    file_map = discover_files(INPUT_DIR)

    unique_weeks = (
        file_map[["week", "week_end"]]
        .drop_duplicates()
        .sort_values("week_end")
        .reset_index(drop=True)
    )

    print(f"Found {len(file_map)} daily files")
    print(f"Found {len(unique_weeks)} weeks\n")

    weekly_parts = []

    for i, row in unique_weeks.iterrows():
        week = row["week"]
        week_end = row["week_end"]

        week_files = file_map[file_map["week"] == week].copy()

        print(f"[{i+1}/{len(unique_weeks)}] Processing week ending {week_end.date()}")

        weekly_df = aggregate_one_week(week_files, p=P, h=H)

        if weekly_df.empty:
            print("  No weekly observations created.")
            continue

        weekly_parts.append(weekly_df)
        print(f"  Added {len(weekly_df)} stock-weeks.")

    if not weekly_parts:
        raise ValueError("No weekly RES observations were created.")

    weekly = pd.concat(weekly_parts, ignore_index=True)
    weekly = weekly.sort_values(["week_end", "permno"]).reset_index(drop=True)

    weekly.to_parquet(OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(weekly):,}")
    print(f"Weeks: {weekly['week_end'].nunique():,}")
    print(f"Stocks: {weekly['permno'].nunique():,}")


if __name__ == "__main__":
    main()