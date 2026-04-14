"""
RSJ Method-1 Decomposition

Purpose:
  Compute method-1 systematic and idiosyncratic RSJ components.

Inputs:
  - Method-1 stock and market intraday inputs.

Outputs:
  - Method-1 decomposition files in data_intermediate/decomposition.

Main Steps:
  - Load method-1 inputs.
  - Run decomposition calculations.
  - Save weekly component files.
"""
from pathlib import Path
import sys
import sqlite3
import gc
import numpy as np
import pandas as pd

STOCK_DIR = Path("data_intermediate/converted_parquet")
MARKET_FILE = Path("data_intermediate/market_returns/market_intraday_spy.parquet")

DAILY_OUTPUT_DIR = Path("data_intermediate/decomposition/method1/rsj_daily")
WEEKLY_OUTPUT_FILE = Path("data_intermediate/decomposition/method1/rsj_method1_weekly.parquet")
BETAS_OUTPUT_FILE = Path("data_intermediate/decomposition/method1/rolling_betas_method1.parquet")

WORK_DB = Path("data_intermediate/decomposition/method1/rsj_method1_working.sqlite")

RET_COL = "returns_5m"
NOBS_COL = "n_obs"

LOOKBACK_WEEKS = 4
MIN_DAY_OBS = 80
MIN_BETA_OBS = 160
MIN_WEEKLY_OBS = 40
MIN_WEEKLY_DAYS = 3

def normalize_date_scalar(x):
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    try:
        ts = ts.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return ts.normalize()

def make_week_from_date(ts):
    return pd.Timestamp(ts).to_period("W-TUE")

def extract_return_vector(x):
    if x is None:
        return np.array([], dtype=float)

    if isinstance(x, np.ndarray):
        vals = x.tolist()
    elif isinstance(x, (list, tuple)):
        vals = list(x)
    else:
        return np.array([], dtype=float)

    out = []
    for v in vals:
        try:
            fv = float(v)
            out.append(fv if np.isfinite(fv) else np.nan)
        except Exception:
            out.append(np.nan)
    return np.asarray(out, dtype=float)

def align_vectors(stock_r, market_r):
    n = min(len(stock_r), len(market_r))
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    s = np.asarray(stock_r[:n], dtype=float)
    m = np.asarray(market_r[:n], dtype=float)

    mask = np.isfinite(s) & np.isfinite(m)
    return s[mask], m[mask]

def compute_alpha_beta_sufficient_stats(stock_r, market_r):
    s, m = align_vectors(stock_r, market_r)
    if len(s) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    sx = float(np.sum(m))
    sy = float(np.sum(s))
    sxy = float(np.dot(m, s))
    sxx = float(np.dot(m, m))
    n = int(len(s))
    return sx, sy, sxy, sxx, n

def split_returns(stock_r, market_r, alpha, beta):
    stock_r = np.asarray(stock_r, dtype=float)
    market_r = np.asarray(market_r, dtype=float)

    n = min(len(stock_r), len(market_r))
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    stock_r = stock_r[:n]
    market_r = market_r[:n]

    if pd.isna(alpha) or pd.isna(beta):
        nan_arr = np.full(n, np.nan)
        return nan_arr, nan_arr

    r_sys = alpha + beta * market_r
    r_idio = stock_r - r_sys

    invalid = ~np.isfinite(stock_r) | ~np.isfinite(market_r)
    r_sys[invalid] = np.nan
    r_idio[invalid] = np.nan

    return r_sys, r_idio

def semivar_pos(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan
    x = r[r > 0]
    if len(x) == 0:
        return 0.0
    return float(np.sum(x ** 2))

def semivar_neg(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan
    x = r[r < 0]
    if len(x) == 0:
        return 0.0
    return float(np.sum(x ** 2))

def compute_rsj_from_returns(r):
    rv_pos = semivar_pos(r)
    rv_neg = semivar_neg(r)

    if pd.isna(rv_pos) or pd.isna(rv_neg):
        return np.nan, rv_pos, rv_neg

    denom = rv_pos + rv_neg
    if denom <= 0:
        return np.nan, rv_pos, rv_neg

    rsj = (rv_pos - rv_neg) / denom
    return float(rsj), float(rv_pos), float(rv_neg)

def init_db(conn):
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS weekly_stats (
            permno    INTEGER NOT NULL,
            week_idx  INTEGER NOT NULL,
            sx        REAL NOT NULL,
            sy        REAL NOT NULL,
            sxy       REAL NOT NULL,
            sxx       REAL NOT NULL,
            n_valid   INTEGER NOT NULL,
            PRIMARY KEY (permno, week_idx)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS betas (
            permno    INTEGER NOT NULL,
            week_idx  INTEGER NOT NULL,
            alpha_hf  REAL,
            beta_hf   REAL,
            PRIMARY KEY (permno, week_idx)
        )
    """)

    conn.commit()

def upsert_weekly_stats(conn, records):
    if not records:
        return

    conn.executemany("""
        INSERT INTO weekly_stats (permno, week_idx, sx, sy, sxy, sxx, n_valid)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(permno, week_idx) DO UPDATE SET
            sx = weekly_stats.sx + excluded.sx,
            sy = weekly_stats.sy + excluded.sy,
            sxy = weekly_stats.sxy + excluded.sxy,
            sxx = weekly_stats.sxx + excluded.sxx,
            n_valid = weekly_stats.n_valid + excluded.n_valid
    """, records)
    conn.commit()

def insert_betas(conn, records):
    if not records:
        return

    conn.executemany("""
        INSERT OR REPLACE INTO betas (permno, week_idx, alpha_hf, beta_hf)
        VALUES (?, ?, ?, ?)
    """, records)
    conn.commit()

def load_market_map():
    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Market file does not exist: {MARKET_FILE}")

    market = pd.read_parquet(MARKET_FILE)
    required = {"date", "interval", "market_ret"}
    if not required.issubset(market.columns):
        raise KeyError(f"Market file missing columns: {required - set(market.columns)}")

    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market["interval"] = pd.to_numeric(market["interval"], errors="coerce")
    market["market_ret"] = pd.to_numeric(market["market_ret"], errors="coerce")

    market = market.dropna(subset=["date", "interval", "market_ret"]).copy()
    market["interval"] = market["interval"].astype(int)

    market = market.sort_values(["date", "interval"])

    market_map = {}
    for dt, g in market.groupby("date"):
        market_map[pd.Timestamp(dt)] = g["market_ret"].to_numpy(dtype=float)

    return market_map

def main():
    if not STOCK_DIR.exists():
        raise FileNotFoundError(f"Stock directory does not exist: {STOCK_DIR}")

    stock_files = sorted(STOCK_DIR.glob("*.parquet"))
    if not stock_files:
        raise FileNotFoundError(f"No parquet files found in {STOCK_DIR}")

    market_map = load_market_map()

    DAILY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEEKLY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    BETAS_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if WORK_DB.exists():
        WORK_DB.unlink()

    conn = sqlite3.connect(WORK_DB)
    init_db(conn)

    common_dates = []
    stock_file_meta = []

    for f in stock_files:
        date_val = normalize_date_scalar(f.stem)
        if pd.isna(date_val):
            continue
        if date_val in market_map:
            common_dates.append(pd.Timestamp(date_val))
            stock_file_meta.append((f, pd.Timestamp(date_val)))

    if not stock_file_meta:
        raise ValueError("No overlapping stock dates with the market file.")

    all_weeks = sorted({make_week_from_date(dt) for _, dt in stock_file_meta})
    week_to_idx = {w: i for i, w in enumerate(all_weeks)}

    print("\nPASS 1: Building weekly alpha/beta statistics...")

    for i, (stock_file, trade_date) in enumerate(stock_file_meta, start=1):
        if i % 250 == 0 or i == 1 or i == len(stock_file_meta):
            print(f"[PASS 1] {i}/{len(stock_file_meta)}: {stock_file.name}")

        stock_df = pd.read_parquet(stock_file)

        required_cols = {"permno", "date", NOBS_COL, RET_COL}
        missing = required_cols - set(stock_df.columns)
        if missing:
            raise KeyError(f"{stock_file.name} missing columns: {missing}")

        stock_df[NOBS_COL] = pd.to_numeric(stock_df[NOBS_COL], errors="coerce")
        stock_df = stock_df[stock_df[NOBS_COL] >= MIN_DAY_OBS].copy()
        if stock_df.empty:
            continue

        market_returns = market_map[trade_date]
        week = make_week_from_date(trade_date)
        week_idx = week_to_idx[week]

        records = []

        for row in stock_df.itertuples(index=False):
            permno = int(row.permno)
            stock_returns = extract_return_vector(getattr(row, RET_COL))
            sx, sy, sxy, sxx, n_valid = compute_alpha_beta_sufficient_stats(stock_returns, market_returns)
            records.append((permno, week_idx, sx, sy, sxy, sxx, n_valid))

        upsert_weekly_stats(conn, records)

        del stock_df, records
        if i % 250 == 0:
            gc.collect()

    print("\nComputing rolling weekly alpha/beta...")

    weekly_stats_df = pd.read_sql_query(
        "SELECT permno, week_idx, sx, sy, sxy, sxx, n_valid FROM weekly_stats ORDER BY permno, week_idx",
        conn
    )

    beta_records = []
    unique_permnos = weekly_stats_df["permno"].nunique()

    for j, (permno, g) in enumerate(weekly_stats_df.groupby("permno"), start=1):
        if j % 5000 == 0 or j == 1 or j == unique_permnos:
            print(f"Alpha/Beta progress: {j}/{unique_permnos} permnos")

        wk_dict = {
            int(r.week_idx): (float(r.sx), float(r.sy), float(r.sxy), float(r.sxx), int(r.n_valid))
            for r in g.itertuples(index=False)
        }

        for week_idx in sorted(wk_dict.keys()):
            if week_idx < LOOKBACK_WEEKS:
                alpha = np.nan
                beta = np.nan
            else:
                prev_idx = range(week_idx - LOOKBACK_WEEKS, week_idx)

                sum_sx = 0.0
                sum_sy = 0.0
                sum_sxy = 0.0
                sum_sxx = 0.0
                sum_n = 0

                for pidx in prev_idx:
                    if pidx in wk_dict:
                        sx, sy, sxy, sxx, n_valid = wk_dict[pidx]
                        sum_sx += sx
                        sum_sy += sy
                        sum_sxy += sxy
                        sum_sxx += sxx
                        sum_n += n_valid

                if sum_n < MIN_BETA_OBS:
                    alpha = np.nan
                    beta = np.nan
                else:
                    denom = sum_sxx - (sum_sx * sum_sx) / sum_n
                    if denom == 0:
                        alpha = np.nan
                        beta = np.nan
                    else:
                        beta = (sum_sxy - (sum_sx * sum_sy) / sum_n) / denom
                        alpha = (sum_sy - beta * sum_sx) / sum_n

            beta_records.append((
                int(permno),
                int(week_idx),
                float(alpha) if pd.notna(alpha) else None,
                float(beta) if pd.notna(beta) else None
            ))

        if j % 5000 == 0:
            insert_betas(conn, beta_records)
            beta_records = []
            gc.collect()

    insert_betas(conn, beta_records)

    betas_df = pd.read_sql_query(
        "SELECT permno, week_idx, alpha_hf, beta_hf FROM betas ORDER BY permno, week_idx",
        conn
    )
    betas_df["week"] = betas_df["week_idx"].map(lambda i: all_weeks[int(i)].end_time.normalize())
    betas_df = betas_df[["permno", "week", "alpha_hf", "beta_hf"]]
    betas_df.to_parquet(BETAS_OUTPUT_FILE, index=False)

    del weekly_stats_df, betas_df
    gc.collect()

    print("\nPASS 2: Computing daily Method 1 RSJ...")

    weekly_agg = {}
    current_week_idx = None
    beta_map_for_week = {}

    for i, (stock_file, trade_date) in enumerate(stock_file_meta, start=1):
        if i % 250 == 0 or i == 1 or i == len(stock_file_meta):
            print(f"[PASS 2] {i}/{len(stock_file_meta)}: {stock_file.name}")

        stock_df = pd.read_parquet(stock_file)

        stock_df[NOBS_COL] = pd.to_numeric(stock_df[NOBS_COL], errors="coerce")
        stock_df = stock_df[stock_df[NOBS_COL] >= MIN_DAY_OBS].copy()
        if stock_df.empty:
            continue

        market_returns = market_map[trade_date]
        week = make_week_from_date(trade_date)
        week_idx = week_to_idx[week]

        if current_week_idx != week_idx:
            beta_week_df = pd.read_sql_query(
                f"SELECT permno, alpha_hf, beta_hf FROM betas WHERE week_idx = {week_idx}",
                conn
            )
            beta_map_for_week = {
                int(r.permno): (r.alpha_hf, r.beta_hf)
                for r in beta_week_df.itertuples(index=False)
            }
            current_week_idx = week_idx

        daily_rows = []

        for row in stock_df.itertuples(index=False):
            permno = int(row.permno)
            stock_returns = extract_return_vector(getattr(row, RET_COL))
            alpha, beta = beta_map_for_week.get(permno, (np.nan, np.nan))

            r_sys, r_idio = split_returns(stock_returns, market_returns, alpha, beta)

            n_obs = int(np.sum(np.isfinite(r_sys))) if len(r_sys) > 0 else 0

            rsj_sys_d, rv_pos_sys, rv_neg_sys = compute_rsj_from_returns(r_sys)
            rsj_idio_d, rv_pos_idio, rv_neg_idio = compute_rsj_from_returns(r_idio)

            daily_rows.append({
                "permno": permno,
                "date": trade_date,
                "week": week.end_time.normalize(),
                "alpha_hf": alpha,
                "beta_hf": beta,
                "n_obs": n_obs,
                "rv_pos_sys": rv_pos_sys,
                "rv_neg_sys": rv_neg_sys,
                "rv_pos_idio": rv_pos_idio,
                "rv_neg_idio": rv_neg_idio,
                "rsj_sys_d": rsj_sys_d,
                "rsj_idio_d": rsj_idio_d,
            })

            key = (permno, week_idx)
            if key not in weekly_agg:
                weekly_agg[key] = {
                    "sum_rsj_sys": 0.0,
                    "sum_rsj_idio": 0.0,
                    "days_seen": set(),
                    "n_obs_total": 0,
                }

            rec = weekly_agg[key]

            if pd.notna(rsj_sys_d) and pd.notna(rsj_idio_d):
                rec["sum_rsj_sys"] += rsj_sys_d
                rec["sum_rsj_idio"] += rsj_idio_d
                rec["days_seen"].add(pd.Timestamp(trade_date))
                rec["n_obs_total"] += n_obs

        day_out = pd.DataFrame(daily_rows)

        optional_cols = ["sym_root", "ret_crsp", "ex", "n_obs"]
        for col in optional_cols:
            if col in stock_df.columns and col != "n_obs":
                day_out[col] = stock_df[col].values

        day_out.to_parquet(DAILY_OUTPUT_DIR / stock_file.name, index=False)

        del stock_df, daily_rows, day_out
        if i % 250 == 0:
            gc.collect()

    conn.close()

    weekly_rows = []

    for (permno, week_idx), rec in weekly_agg.items():
        n_unique_days = len(rec["days_seen"])

        if rec["n_obs_total"] <= MIN_WEEKLY_OBS:
            continue
        if n_unique_days < MIN_WEEKLY_DAYS:
            continue
        if n_unique_days == 0:
            continue

        week = all_weeks[week_idx].end_time.normalize()

        weekly_rows.append({
            "permno": permno,
            "week": week,
            "rsj_sys_weekly": rec["sum_rsj_sys"] / n_unique_days,
            "rsj_idio_weekly": rec["sum_rsj_idio"] / n_unique_days,
            "n_days": n_unique_days,
            "n_obs_total": rec["n_obs_total"],
        })

    weekly_df = pd.DataFrame(weekly_rows).sort_values(["permno", "week"]).reset_index(drop=True)
    weekly_df.to_parquet(WEEKLY_OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Daily files written to: {DAILY_OUTPUT_DIR}")
    print(f"Weekly output: {WEEKLY_OUTPUT_FILE}")
    print(f"Betas output: {BETAS_OUTPUT_FILE}")
    print(f"Temporary DB: {WORK_DB}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
