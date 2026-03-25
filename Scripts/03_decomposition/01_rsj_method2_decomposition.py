from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# PURPOSE
# =========================================================
# Method 2 decomposition of weekly stock RSJ using weekly SPY RSJ:
#
#   RSJ_i,w = alpha_i,w-1 + beta_i,w-1 * RSJ_SPY,w + epsilon_i,w
#
# Rolling window: 52 weeks
#
# Output:
#   data_intermediate/decomposition/rsj_method2_spy.parquet
# =========================================================

def parse_week_column(s: pd.Series) -> pd.Series:
    """
    Parse week column safely.
    Handles both unix milliseconds and already-formatted datetimes.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="ms")
    return pd.to_datetime(s)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
rsj_path = Path("data_intermediate/rsj_weekly/rsj_weekly.parquet")
market_path = Path("data_intermediate/market_weekly_rsj/spy_weekly_rsj.parquet")

output_dir = Path("data_intermediate/decomposition")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "rsj_method2_spy.parquet"

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
print("Loading weekly stock RSJ and weekly SPY RSJ...")

rsj = pd.read_parquet(rsj_path)
market = pd.read_parquet(market_path)

rsj["permno"] = rsj["permno"].astype("Int64")
rsj["week"] = parse_week_column(rsj["week"])
market["week"] = parse_week_column(market["week"])

# Keep only needed columns from market file
market = market[["week", "rsj_spy_weekly"]].copy()

print(f"RSJ rows: {len(rsj):,}")
print(f"Market rows: {len(market):,}")

# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------
df = pd.merge(rsj, market, on="week", how="inner")
df = df.sort_values(["permno", "week"]).reset_index(drop=True)

print(f"Merged rows: {len(df):,}")
print(f"Unique stocks: {df['permno'].nunique():,}")
print(f"Unique weeks: {df['week'].nunique():,}")

# ---------------------------------------------------------
# Rolling regression settings
# ---------------------------------------------------------
window = 52

results = []

print("\nRunning rolling Method 2 decomposition...")

for i, (permno, group) in enumerate(df.groupby("permno"), start=1):
    if i % 1000 == 0:
        print(f"Processed {i} stocks...")

    group = group.sort_values("week").copy()
    n = len(group)

    group["alpha_rsj"] = np.nan
    group["beta_rsj"] = np.nan

    y = group["rsj_weekly"].to_numpy(dtype=float)
    x = group["rsj_spy_weekly"].to_numpy(dtype=float)

    for t in range(window, n):
        y_window = y[t - window:t]
        x_window = x[t - window:t]

        if np.isnan(y_window).any() or np.isnan(x_window).any():
            continue

        X = np.column_stack([np.ones(len(x_window)), x_window])

        try:
            coef = np.linalg.lstsq(X, y_window, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        group.iloc[t, group.columns.get_loc("alpha_rsj")] = coef[0]
        group.iloc[t, group.columns.get_loc("beta_rsj")] = coef[1]

    results.append(group)

out = pd.concat(results, ignore_index=True)

# ---------------------------------------------------------
# Construct decomposition
# ---------------------------------------------------------
# Exact statistical decomposition (matches your Method 2 text)
out["rsj_sys"] = out["alpha_rsj"] + out["beta_rsj"] * out["rsj_spy_weekly"]
out["rsj_idio"] = out["rsj_weekly"] - out["rsj_sys"]

# Optional pure market-driven component (without intercept)
out["rsj_sys_no_alpha"] = out["beta_rsj"] * out["rsj_spy_weekly"]

# Keep rows with valid rolling estimates
out = out.dropna(subset=["alpha_rsj", "beta_rsj", "rsj_sys", "rsj_idio"]).copy()

# ---------------------------------------------------------
# Select columns
# ---------------------------------------------------------
out = out[
    [
        "permno",
        "week",
        "rsj_weekly",
        "n_days",
        "n_obs_total",
        "rsj_spy_weekly",
        "alpha_rsj",
        "beta_rsj",
        "rsj_sys",
        "rsj_idio",
        "rsj_sys_no_alpha",
    ]
].sort_values(["permno", "week"]).reset_index(drop=True)

# ---------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------
print("\nPreview:")
print(out.head())

print("\nSummary:")
print(
    out[
        [
            "rsj_weekly",
            "rsj_spy_weekly",
            "alpha_rsj",
            "beta_rsj",
            "rsj_sys",
            "rsj_idio",
        ]
    ].describe()
)

print("\nCorrelations:")
print(
    out[
        [
            "rsj_weekly",
            "rsj_spy_weekly",
            "rsj_sys",
            "rsj_idio",
        ]
    ].corr()
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
out.to_parquet(output_path, index=False)

print(f"\nSaved Method 2 decomposition to: {output_path}")