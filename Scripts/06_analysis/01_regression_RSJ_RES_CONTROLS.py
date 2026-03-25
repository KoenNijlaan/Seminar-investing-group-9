from pathlib import Path
import numpy as np
import pandas as pd

# =========================================================
# PURPOSE
# =========================================================
# Run Fama-MacBeth regression for STEP 3:
#
#   R_{i,w+1} = alpha + RSJ + RES + controls
# =========================================================

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]

panel_path = BASE / "data_intermediate/regression_panels/panel_step3.parquet"

output_dir = BASE / "data_intermediate/regression_results"
output_dir.mkdir(parents=True, exist_ok=True)

betas_path = output_dir / "fmb_step3_weekly_betas.parquet"
summary_path = output_dir / "fmb_step3_summary.csv"

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
min_obs = 30

variables = [
    "rsj_weekly",
    "res_weekly",
    "me",
    "bm",
    "mom",
    "rev",
    "ivol",
    "illiq"
]

coef_names = ["alpha"] + variables

# ---------------------------------------------------------
# Load panel
# ---------------------------------------------------------
print("Loading panel...")

df = pd.read_parquet(panel_path)
df["week"] = pd.to_datetime(df["week"])

print(f"Rows: {len(df):,}")
print(f"Weeks: {df['week'].nunique():,}")

# ---------------------------------------------------------
# Run weekly regressions
# ---------------------------------------------------------
print("\nRunning weekly regressions...")

results = []

for i, (week, g) in enumerate(df.groupby("week"), start=1):

    if len(g) < min_obs:
        continue

    y = g["R_i_w_plus_1"].to_numpy(dtype=float)

    X = np.column_stack([
        np.ones(len(g)),  # alpha
        *[g[v].to_numpy(dtype=float) for v in variables]
    ])

    # remove missing
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask]
    X = X[mask]

    if len(y) < min_obs:
        continue

    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        continue

    row = {"week": week}
    for j, name in enumerate(coef_names):
        row[name] = beta[j]

    results.append(row)

    if i % 100 == 0:
        print(f"Processed {i} weeks...")

betas = pd.DataFrame(results)

print("\nFinished weekly regressions.")
print(f"Estimated weeks: {len(betas):,}")

# Save betas immediately so results are not lost
betas.to_parquet(betas_path, index=False)
print(f"Saved weekly betas to: {betas_path}")

# ---------------------------------------------------------
# Newey-West
# ---------------------------------------------------------
def newey_west(x):
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    T = len(x)

    if T == 0:
        return np.nan, np.nan

    mean = x.mean()
    u = x - mean

    L = int(4 * (T / 100) ** (2 / 9))

    var = (u @ u) / T

    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        gamma = (u[l:] @ u[:-l]) / T
        var += 2 * w * gamma

    se = np.sqrt(var / T)
    t = mean / se if se > 0 else np.nan

    return mean, t

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\nComputing summary...")

summary = []

for col in coef_names:
    mean, t = newey_west(betas[col])
    summary.append({
        "variable": col,
        "coef": mean,
        "t_stat": t
    })

summary = pd.DataFrame(summary)

print("\nFama-MacBeth Results:")
print(summary)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
summary.to_csv(summary_path, index=False)

print("\nSaved to:")
print(betas_path)
print(summary_path)