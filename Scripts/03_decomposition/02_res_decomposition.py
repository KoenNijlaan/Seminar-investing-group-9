from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# Settings
# =========================================================
STOCK_DIR = Path("data_intermediate/converted_parquet")
ETF_DIR = Path("data_intermediate/converted_parquet_etf")
OUTPUT_DIR = Path("data_intermediate/decomposition/res_decomposition/res_daily_split")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.025
H = 0.5
LOOKBACK_WEEKS = 4
MIN_DAY_OBS = 80
MIN_BETA_OBS = 320   # 4 weeks * 80 aligned observations

RET_COL = "returns_5m"
NOBS_COL = "n_obs"

ETF_SYMBOL = "SPY"
ETF_SYMBOL_COL = "sym_root"


# =========================================================
# Helpers
# =========================================================
def compute_res(r, alpha=0.025, H=0.5):
    """
    Compute daily realized expected shortfall (RES) from intraday returns.
    """
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]

    c = len(r)
    if c == 0:
        return np.nan

    q_alpha = np.quantile(r, alpha)
    tail = r[r <= q_alpha]

    if len(tail) == 0:
        return np.nan

    tail_mean = np.mean(tail)

    # Minus sign so larger values mean more downside risk
    return -(c ** H) * tail_mean


def extract_market_returns(etf_df, file_name):
    """
    Extract the ETF intraday return series for one day.
    """
    if len(etf_df) == 0:
        raise ValueError(f"No ETF rows found in {file_name}")

    if ETF_SYMBOL_COL in etf_df.columns:
        subset = etf_df[etf_df[ETF_SYMBOL_COL] == ETF_SYMBOL].copy()

        if len(subset) == 1:
            return subset.iloc[0][RET_COL]

        if len(subset) > 1:
            raise ValueError(f"Multiple ETF rows for {ETF_SYMBOL} in {file_name}")

        # fallback if file has just one row
        if len(etf_df) == 1:
            return etf_df.iloc[0][RET_COL]

        available = sorted(etf_df[ETF_SYMBOL_COL].dropna().unique().tolist())
        raise ValueError(
            f"ETF symbol {ETF_SYMBOL} not found in {file_name}. "
            f"Available symbols (first 10): {available[:10]}"
        )

    if len(etf_df) == 1:
        return etf_df.iloc[0][RET_COL]

    raise ValueError(
        f"{file_name} has multiple ETF rows but no symbol column '{ETF_SYMBOL_COL}'."
    )


def compute_beta_sufficient_stats(stock_r, market_r):
    """
    For no-intercept regression r_i = beta * r_M + eps,
    return sufficient statistics:
        sxy = sum(r_M * r_i)
        sxx = sum(r_M^2)
        n   = number of aligned non-missing observations
    """
    stock_r = np.asarray(stock_r, dtype=float)
    market_r = np.asarray(market_r, dtype=float)

    n = min(len(stock_r), len(market_r))
    if n == 0:
        return 0.0, 0.0, 0

    stock_r = stock_r[:n]
    market_r = market_r[:n]

    mask = ~(np.isnan(stock_r) | np.isnan(market_r))
    if mask.sum() == 0:
        return 0.0, 0.0, 0

    stock_valid = stock_r[mask]
    market_valid = market_r[mask]

    sxy = float(np.dot(market_valid, stock_valid))
    sxx = float(np.dot(market_valid, market_valid))
    n_valid = int(mask.sum())

    return sxy, sxx, n_valid


def split_returns(stock_r, market_r, beta):
    """
    Split intraday returns into:
        systematic    = beta * market
        idiosyncratic = stock - beta * market
    """
    stock_r = np.asarray(stock_r, dtype=float)
    market_r = np.asarray(market_r, dtype=float)

    n = min(len(stock_r), len(market_r))
    if n == 0:
        return np.array([]), np.array([])

    stock_r = stock_r[:n]
    market_r = market_r[:n]

    if pd.isna(beta):
        nan_arr = np.full(n, np.nan)
        return nan_arr, nan_arr

    r_sys = beta * market_r
    r_idio = stock_r - r_sys

    invalid = np.isnan(stock_r) | np.isnan(market_r)
    r_sys[invalid] = np.nan
    r_idio[invalid] = np.nan

    return r_sys, r_idio


# =========================================================
# Match overlapping stock and ETF files
# =========================================================
stock_files = sorted(STOCK_DIR.glob("*.parquet"))
etf_files = sorted(ETF_DIR.glob("*.parquet"))

if not stock_files:
    raise FileNotFoundError(f"No stock parquet files found in {STOCK_DIR}")
if not etf_files:
    raise FileNotFoundError(f"No ETF parquet files found in {ETF_DIR}")

stock_map = {f.name: f for f in stock_files}
etf_map = {f.name: f for f in etf_files}
common_names = sorted(set(stock_map) & set(etf_map))

if not common_names:
    raise ValueError("No overlapping parquet filenames between stock and ETF folders.")

print(f"Stock files: {len(stock_files)}")
print(f"ETF files:   {len(etf_files)}")
print(f"Overlap:     {len(common_names)}")
print(f"First overlap: {common_names[0]}")
print(f"Last overlap:  {common_names[-1]}")

# Global week index based on overlapping file dates
all_weeks = sorted({
    pd.Timestamp(Path(fname).stem).to_period("W-TUE")
    for fname in common_names
})
week_to_idx = {w: i for i, w in enumerate(all_weeks)}

# =========================================================
# PASS 1: Build weekly sufficient statistics for beta
# =========================================================
print("\nPASS 1: Building weekly beta statistics...")

# key = (permno, week), value = [sum_xy, sum_xx, n_aligned]
weekly_stats = {}

for i, fname in enumerate(common_names, start=1):
    if i % 250 == 0 or i == 1 or i == len(common_names):
        print(f"[PASS 1] {i}/{len(common_names)}: {fname}")

    stock_file = stock_map[fname]
    etf_file = etf_map[fname]

    stock_df = pd.read_parquet(stock_file)
    etf_df = pd.read_parquet(etf_file)

    required_cols = ["permno", "date", NOBS_COL, RET_COL]
    missing = [c for c in required_cols if c not in stock_df.columns]
    if missing:
        raise KeyError(f"{stock_file.name} is missing required columns: {missing}")

    if RET_COL not in etf_df.columns:
        raise KeyError(f"{etf_file.name} is missing required column: {RET_COL}")

    stock_df = stock_df[stock_df[NOBS_COL] >= MIN_DAY_OBS].copy()
    if stock_df.empty:
        continue

    market_returns = extract_market_returns(etf_df, etf_file.name)

    # all rows in a daily file should share the same trading date
    trade_date = pd.to_datetime(stock_df["date"].iloc[0])
    week = trade_date.to_period("W-TUE")

    for row in stock_df.itertuples(index=False):
        permno = row.permno
        stock_returns = getattr(row, RET_COL)

        sxy, sxx, n_valid = compute_beta_sufficient_stats(stock_returns, market_returns)

        key = (permno, week)
        if key not in weekly_stats:
            weekly_stats[key] = [0.0, 0.0, 0]

        weekly_stats[key][0] += sxy
        weekly_stats[key][1] += sxx
        weekly_stats[key][2] += n_valid

print(f"Weekly stat entries: {len(weekly_stats):,}")

# Organize stats by permno
stats_by_permno = {}
for (permno, week), (sxy, sxx, n_valid) in weekly_stats.items():
    if permno not in stats_by_permno:
        stats_by_permno[permno] = {}
    stats_by_permno[permno][week] = (sxy, sxx, n_valid)

# =========================================================
# Compute beta lookup: beta for week w uses previous 4 weeks
# =========================================================
print("\nComputing rolling weekly betas...")

beta_lookup = {}

permnos = list(stats_by_permno.keys())
for j, permno in enumerate(permnos, start=1):
    if j % 5000 == 0 or j == 1 or j == len(permnos):
        print(f"Beta progress: {j}/{len(permnos)} permnos")

    wk_dict = stats_by_permno[permno]
    stock_weeks = sorted(wk_dict.keys())

    for w in stock_weeks:
        idx = week_to_idx[w]

        # Need full 4-week lookback
        if idx < LOOKBACK_WEEKS:
            beta_lookup[(permno, w)] = np.nan
            continue

        prev_weeks = all_weeks[idx - LOOKBACK_WEEKS:idx]

        sum_sxy = 0.0
        sum_sxx = 0.0
        sum_n = 0

        for pw in prev_weeks:
            if pw in wk_dict:
                sxy, sxx, n_valid = wk_dict[pw]
                sum_sxy += sxy
                sum_sxx += sxx
                sum_n += n_valid

        if sum_n < MIN_BETA_OBS or sum_sxx == 0:
            beta = np.nan
        else:
            beta = sum_sxy / sum_sxx

        beta_lookup[(permno, w)] = beta

print(f"Beta entries: {len(beta_lookup):,}")

# free memory no longer needed
del weekly_stats
del stats_by_permno

# =========================================================
# PASS 2: Compute daily total/sys/idio RES and save per day
# =========================================================
print("\nPASS 2: Computing daily RES decomposition...")

for i, fname in enumerate(common_names, start=1):
    if i % 250 == 0 or i == 1 or i == len(common_names):
        print(f"[PASS 2] {i}/{len(common_names)}: {fname}")

    stock_file = stock_map[fname]
    etf_file = etf_map[fname]

    stock_df = pd.read_parquet(stock_file)
    etf_df = pd.read_parquet(etf_file)

    stock_df = stock_df[stock_df[NOBS_COL] >= MIN_DAY_OBS].copy()

    if stock_df.empty:
        # optional: skip empty output files
        continue

    market_returns = extract_market_returns(etf_df, etf_file.name)

    trade_date = pd.to_datetime(stock_df["date"].iloc[0])
    week = trade_date.to_period("W-TUE")

    beta_vals = []
    res_total_vals = []
    res_sys_vals = []
    res_idio_vals = []

    for row in stock_df.itertuples(index=False):
        permno = row.permno
        stock_returns = getattr(row, RET_COL)

        beta = beta_lookup.get((permno, week), np.nan)
        r_sys, r_idio = split_returns(stock_returns, market_returns, beta)

        beta_vals.append(beta)
        res_total_vals.append(compute_res(stock_returns, alpha=ALPHA, H=H))
        res_sys_vals.append(compute_res(r_sys, alpha=ALPHA, H=H))
        res_idio_vals.append(compute_res(r_idio, alpha=ALPHA, H=H))

    out = pd.DataFrame({
        "permno": stock_df["permno"].values,
        "date": stock_df["date"].values,
        "beta_hf": beta_vals,
        "res_2p5": res_total_vals,
        "res_sys_2p5": res_sys_vals,
        "res_idio_2p5": res_idio_vals,
    })

    # add optional columns if they exist
    optional_cols = ["sym_root", "ret_crsp", "ex", "n_obs"]
    for col in optional_cols:
        if col in stock_df.columns:
            out[col] = stock_df[col].values

    # reorder columns more nicely
    preferred_order = [
        "permno", "date", "sym_root", "ret_crsp", "ex", "n_obs",
        "beta_hf", "res_2p5", "res_sys_2p5", "res_idio_2p5"
    ]
    out = out[[c for c in preferred_order if c in out.columns]]

    out.to_parquet(OUTPUT_DIR / fname, index=False)

print("\nDone.")