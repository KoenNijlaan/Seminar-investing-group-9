from pathlib import Path
import pandas as pd

# =========================================================
# PURPOSE
# =========================================================
# Build regression dataset for:
#
#   R_{i,w+1} = alpha + beta1 * RSJ_{i,w} + beta2 * RES_{i,w} + gamma' X_{i,w} + e_{i,w+1}
#
# using:
#   - RSJ weekly
#   - RES weekly
#   - weekly controls
#   - weekly next-week returns
#
# Output:
#   data_intermediate/regression_panels/panel_step3.parquet
# =========================================================

# ---------------------------------------------------------
# Base path
# ---------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------
# Correct paths
# ---------------------------------------------------------
rsj_path = BASE / "data_intermediate/rsj_weekly/rsj_weekly.parquet"
res_path = BASE / "data_intermediate/res_weekly/weekly_res.parquet"
controls_path = BASE / "data_intermediate/controls_variables/weekly_controls.parquet"
returns_path = BASE / "data_intermediate/weekly_stock_returns/weekly_stock_returns.parquet"

output_path = BASE / "data_intermediate/regression_panels/panel_step3.parquet"
output_path.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Check files
# ---------------------------------------------------------
print("Checking files...\n")
for p in [rsj_path, res_path, controls_path, returns_path]:
    print(p, "->", p.exists())

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
print("\nLoading data...")

rsj = pd.read_parquet(rsj_path)
res = pd.read_parquet(res_path)
controls = pd.read_parquet(controls_path)
ret = pd.read_parquet(returns_path)

# ---------------------------------------------------------
# Standardize keys
# ---------------------------------------------------------
for df in [rsj, res, controls, ret]:
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    df["week"] = pd.to_datetime(df["week"], errors="coerce")

# ---------------------------------------------------------
# RSJ
# ---------------------------------------------------------
if "rsj_weekly" not in rsj.columns:
    raise KeyError(f"'rsj_weekly' not found in RSJ file. Columns are: {list(rsj.columns)}")

rsj = rsj[["permno", "week", "rsj_weekly"]].drop_duplicates()

# ---------------------------------------------------------
# RES: detect column automatically
# ---------------------------------------------------------
preferred_res_names = [
    "res_weekly",
    "weekly_res",
    "res",
    "RES_weekly",
    "RES"
]

res_col = None

for c in preferred_res_names:
    if c in res.columns:
        res_col = c
        break

if res_col is None:
    candidates = [
        c for c in res.columns
        if c not in {"permno", "week"} and "res" in c.lower()
    ]
    if len(candidates) == 1:
        res_col = candidates[0]
    else:
        raise KeyError(
            "Could not identify RES column automatically.\n"
            f"Available columns: {list(res.columns)}\n"
            f"Candidates found: {candidates}"
        )

print(f"\nUsing RES column: {res_col}")
res = res[["permno", "week", res_col]].rename(columns={res_col: "res_weekly"}).drop_duplicates()

# ---------------------------------------------------------
# Controls
# ---------------------------------------------------------
needed_controls = ["permno", "week", "me", "bm", "mom", "rev", "ivol", "illiq"]
missing_controls = [c for c in needed_controls if c not in controls.columns]
if missing_controls:
    raise KeyError(
        f"Missing control columns: {missing_controls}\n"
        f"Available control columns: {list(controls.columns)}"
    )

controls = controls[needed_controls].drop_duplicates()

# ---------------------------------------------------------
# Returns
# ---------------------------------------------------------
if "R_i_w_plus_1" not in ret.columns:
    raise KeyError(
        f"'R_i_w_plus_1' not found in returns file.\n"
        f"Available columns: {list(ret.columns)}"
    )

ret = ret[["permno", "week", "R_i_w_plus_1"]].drop_duplicates()

# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------
print("\nMerging datasets...")

df = rsj.merge(res, on=["permno", "week"], how="inner", validate="one_to_one")
df = df.merge(controls, on=["permno", "week"], how="left", validate="one_to_one")
df = df.merge(ret, on=["permno", "week"], how="inner", validate="one_to_one")

df = df.sort_values(["permno", "week"]).reset_index(drop=True)

print(f"\nRows after merge: {len(df):,}")
print("\nMissing shares after merge:")
print(df.isna().mean())

# ---------------------------------------------------------
# Keep regression-complete sample
# ---------------------------------------------------------
reg_cols = ["R_i_w_plus_1", "rsj_weekly", "res_weekly", "me", "bm", "mom", "rev", "ivol", "illiq"]
panel = df.dropna(subset=reg_cols).copy()

print(f"\nRows after dropna for regression: {len(panel):,}")
print(f"Unique permnos: {panel['permno'].nunique():,}")
print(f"Unique weeks: {panel['week'].nunique():,}")

print("\nPreview:")
print(panel.head())

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
panel.to_parquet(output_path, index=False)
print(f"\nSaved panel to: {output_path}")