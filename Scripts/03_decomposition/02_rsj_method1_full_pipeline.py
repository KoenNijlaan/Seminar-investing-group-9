from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================
STOCK_DIR = Path("data_intermediate/converted_parquet")
ETF_DIR = Path("data_intermediate/converted_parquet_etf")

DAILY_OUTPUT = Path("data_intermediate/rsj_daily/rsj_method1_daily.parquet")
WEEKLY_OUTPUT = Path("data_intermediate/rsj_weekly/rsj_method1_weekly.parquet")
BETAS_OUTPUT = Path("data_intermediate/rolling_betas_method1.parquet")

TARGET_SYMBOL = "SPY"
ROLLING_WEEKS = 4
MIN_OBS_FOR_BETA = 40
MIN_WEEKLY_OBS = 40
MIN_WEEKLY_DAYS = 3


# =========================================================
# HELPERS
# =========================================================
def make_week(series):
    s = pd.to_datetime(series, errors="coerce")
    return s.dt.to_period("W-FRI").dt.end_time.dt.normalize()


def normalize_date_scalar(x):
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


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


def align_vectors(stock_vec, market_vec):
    n = min(len(stock_vec), len(market_vec))
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    s = stock_vec[:n]
    m = market_vec[:n]

    valid = np.isfinite(s) & np.isfinite(m)
    return s[valid], m[valid]


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


def semivar_pos(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    pos = x[x > 0]
    if len(pos) == 0:
        return 0.0
    return float(np.sum(pos ** 2))


def semivar_neg(x):
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
    return float((rv_pos - rv_neg) / denom)


# =========================================================
# LOAD ONE DAY OF SPY
# =========================================================
def load_spy_returns_for_day(etf_file):
    df = pd.read_parquet(etf_file)

    if "sym_root" not in df.columns or "returns_5m" not in df.columns or "date" not in df.columns:
        return None, None

    df = df[df["sym_root"].astype(str).str.upper() == TARGET_SYMBOL].copy()
    if df.empty:
        return None, None

    if "n_obs" in df.columns:
        df["n_obs"] = pd.to_numeric(df["n_obs"], errors="coerce")
        df = df[df["n_obs"] >= 1].copy()
        if df.empty:
            return None, None

    row = df.iloc[0]
    date_val = normalize_date_scalar(row["date"])
    ret_vec = extract_return_vector(row["returns_5m"])

    if pd.isna(date_val) or len(ret_vec) == 0:
        return None, None

    return date_val, ret_vec


# =========================================================
# MAIN
# =========================================================
def main():
    if not STOCK_DIR.exists():
        raise FileNotFoundError(f"Stock input directory does not exist: {STOCK_DIR}")
    if not ETF_DIR.exists():
        raise FileNotFoundError(f"ETF input directory does not exist: {ETF_DIR}")

    stock_files = sorted(STOCK_DIR.glob("*.parquet"))
    etf_files = sorted(ETF_DIR.glob("*.parquet"))

    if not stock_files:
        raise FileNotFoundError(f"No stock parquet files found in {STOCK_DIR}")
    if not etf_files:
        raise FileNotFoundError(f"No ETF parquet files found in {ETF_DIR}")

    etf_map = {f.name: f for f in etf_files}

    # History per permno: list of {"week": week_ts, "ret": np.array, "mkt": np.array}
    history = defaultdict(list)

    # Cache beta per (permno, week)
    beta_cache = {}

    daily_rows = []
    beta_rows = []

    unique_weeks_seen = []

    for i, stock_file in enumerate(stock_files, start=1):
        if i % 200 == 0 or i == 1:
            print(f"[{i}/{len(stock_files)}] Processing {stock_file.name}")

        if stock_file.name not in etf_map:
            continue

        # Load market day
        date_mkt, market_vec = load_spy_returns_for_day(etf_map[stock_file.name])
        if date_mkt is None or market_vec is None or len(market_vec) == 0:
            continue

        # Load stock day
        df = pd.read_parquet(stock_file)

        required = {"permno", "date", "returns_5m"}
        if not required.issubset(df.columns):
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["n_obs"] = pd.to_numeric(df.get("n_obs", np.nan), errors="coerce")

        date_val = normalize_date_scalar(df["date"].iloc[0])
        if pd.isna(date_val):
            continue

        week_val = make_week(pd.Series([date_val])).iloc[0]

        if week_val not in unique_weeks_seen:
            unique_weeks_seen.append(week_val)

        current_week_idx = unique_weeks_seen.index(week_val)
        prev_weeks = unique_weeks_seen[max(0, current_week_idx - ROLLING_WEEKS): current_week_idx]

        for _, row in df.iterrows():
            permno = row["permno"]
            if pd.isna(permno):
                continue
            permno = int(permno)

            stock_vec = extract_return_vector(row["returns_5m"])
            s_aligned, m_aligned = align_vectors(stock_vec, market_vec)
            n_obs_day = len(s_aligned)

            if n_obs_day == 0:
                continue

            # Estimate beta once per permno-week, using only previous weeks
            beta_key = (permno, week_val)

            if beta_key not in beta_cache:
                hist_records = history[permno]
                hist_records = [r for r in hist_records if r["week"] in prev_weeks]

                if hist_records:
                    y_hist = np.concatenate([r["ret"] for r in hist_records if len(r["ret"]) > 0])
                    x_hist = np.concatenate([r["mkt"] for r in hist_records if len(r["mkt"]) > 0])
                else:
                    y_hist = np.array([], dtype=float)
                    x_hist = np.array([], dtype=float)

                beta_hat, n_obs_beta = ols_beta(y_hist, x_hist)

                beta_cache[beta_key] = {
                    "beta_hat": beta_hat,
                    "n_obs_beta": n_obs_beta,
                    "n_weeks_used": len(prev_weeks),
                }

                beta_rows.append({
                    "permno": permno,
                    "week": week_val,
                    "beta_hat": beta_hat,
                    "n_obs_beta": n_obs_beta,
                    "n_weeks_used": len(prev_weeks),
                })

            beta_info = beta_cache[beta_key]
            beta_hat = beta_info["beta_hat"]
            n_obs_beta = beta_info["n_obs_beta"]

            if pd.isna(beta_hat) or n_obs_beta <= MIN_OBS_FOR_BETA:
                # still update history for future weeks
                history[permno].append({
                    "week": week_val,
                    "ret": s_aligned,
                    "mkt": m_aligned,
                })
                continue

            ret_sys = beta_hat * m_aligned
            ret_idio = s_aligned - ret_sys

            rv_pos_sys = semivar_pos(ret_sys)
            rv_neg_sys = semivar_neg(ret_sys)
            rv_pos_idio = semivar_pos(ret_idio)
            rv_neg_idio = semivar_neg(ret_idio)

            rsj_sys_d = safe_rsj(rv_pos_sys, rv_neg_sys)
            rsj_idio_d = safe_rsj(rv_pos_idio, rv_neg_idio)

            daily_rows.append({
                "permno": permno,
                "date": date_val,
                "week": week_val,
                "beta_hat": beta_hat,
                "n_obs_beta": n_obs_beta,
                "n_obs": n_obs_day,
                "rv_pos_sys": rv_pos_sys,
                "rv_neg_sys": rv_neg_sys,
                "rv_pos_idio": rv_pos_idio,
                "rv_neg_idio": rv_neg_idio,
                "rsj_sys_d": rsj_sys_d,
                "rsj_idio_d": rsj_idio_d,
            })

            # Update history after using fixed beta for current week
            history[permno].append({
                "week": week_val,
                "ret": s_aligned,
                "mkt": m_aligned,
            })

    if not daily_rows:
        raise ValueError("No daily Method 1 RSJ rows were created.")

    daily = pd.DataFrame(daily_rows)
    betas = pd.DataFrame(beta_rows).drop_duplicates(subset=["permno", "week"]).sort_values(["permno", "week"])

    weekly = (
        daily.groupby(["permno", "week"], as_index=False)
             .agg(
                 rsj_sys_weekly=("rsj_sys_d", "mean"),
                 rsj_idio_weekly=("rsj_idio_d", "mean"),
                 n_days=("date", "nunique"),
                 n_obs_total=("n_obs", "sum"),
             )
    )

    weekly = weekly[
        (weekly["n_obs_total"] > MIN_WEEKLY_OBS) &
        (weekly["n_days"] >= MIN_WEEKLY_DAYS)
    ].copy()

    valid_weeks = weekly[["permno", "week"]].copy()
    daily = daily.merge(valid_weeks, on=["permno", "week"], how="inner", validate="many_to_one")

    DAILY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    WEEKLY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    BETAS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    daily.to_parquet(DAILY_OUTPUT, index=False)
    weekly.to_parquet(WEEKLY_OUTPUT, index=False)
    betas.to_parquet(BETAS_OUTPUT, index=False)

    print("\nDone.")
    print(f"Daily RSJ rows: {len(daily):,}")
    print(f"Weekly RSJ rows: {len(weekly):,}")
    print(f"Betas rows: {len(betas):,}")
    print(f"Daily output: {DAILY_OUTPUT}")
    print(f"Weekly output: {WEEKLY_OUTPUT}")
    print(f"Betas output: {BETAS_OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)