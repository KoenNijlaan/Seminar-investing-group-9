from pathlib import Path
import gc
import numpy as np
import pandas as pd

# =========================================================
# PURPOSE
# =========================================================
# Build weekly control variables aligned to the RSJ weekly sample.
#
# Controls:
#   - ME      : log market equity
#   - BM      : book-to-market
#   - MOM     : momentum, day t-252 to t-21
#   - REV     : lagged 1-week return
#   - IVOL    : idiosyncratic volatility from daily FF3 residuals over t-20:t
#   - ILLIQ   : log Amihud illiquidity over t-4:t
#
# INPUTS
#   data_intermediate/rsj_weekly/rsj_weekly.parquet
#   data_raw/wrds/crsp_daily_rsj_sample_1992_2024.parquet
#   data_raw/wrds/crsp_delist_rsj_sample_1992_2024.parquet
#   data_raw/wrds/ff_daily_factors_1992_2024.parquet
#   data_raw/wrds/comp_funda_1991_2024.parquet
#   data_raw/wrds/ccm_linktable.parquet
#
# OUTPUT
#   data_intermediate/controls/weekly_controls.parquet
#
# INTERMEDIATE OUTPUTS
#   data_intermediate/controls/_daily_controls_base.parquet
#   data_intermediate/controls/_daily_ivol.parquet
# =========================================================

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
rsj_path = Path("data_intermediate/rsj_weekly/rsj_weekly.parquet")
crsp_path = Path("data_raw/wrds/crsp_daily_rsj_sample_1992_2024.parquet")
delist_path = Path("data_raw/wrds/crsp_delist_rsj_sample_1992_2024.parquet")
ff_path = Path("data_raw/wrds/ff_daily_factors_1992_2024.parquet")
comp_path = Path("data_raw/wrds/comp_funda_1991_2024.parquet")
ccm_path = Path("data_raw/wrds/ccm_linktable.parquet")

output_dir = Path("data_intermediate/controls")
output_dir.mkdir(parents=True, exist_ok=True)

daily_base_path = output_dir / "_daily_controls_base.parquet"
daily_ivol_path = output_dir / "_daily_ivol.parquet"
output_path = output_dir / "weekly_controls.parquet"

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
mom_skip_days = 21
mom_lookback_days = 252
illiq_window = 5
ivol_window = 21
ivol_progress_every = 500

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def parse_date_col(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)
    return pd.to_datetime(s, errors="coerce")

def make_week_from_date(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.to_period("W-FRI").dt.end_time.dt.normalize()

def safe_to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def rolling_cumret_ex_skip(ret: pd.Series, lookback: int, skip: int) -> pd.Series:
    """
    Momentum at date t:
    compound gross return from t-lookback through t-skip.
    """
    vals = pd.to_numeric(ret, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)

    gross = 1.0 + np.where(np.isfinite(vals), vals, np.nan)

    for t in range(len(vals)):
        start = t - lookback
        end = t - skip
        if start < 0 or end < start:
            continue

        window = gross[start:end + 1]
        if np.any(~np.isfinite(window)):
            continue

        out[t] = np.prod(window) - 1.0

    return pd.Series(out, index=ret.index, dtype=float)

def compute_ivol_array(ret_adj, rf, mktrf, smb, hml, window=21):
    """
    Compute daily IVOL as rolling std of residuals from FF3 regression.
    """
    y_all = ret_adj - rf
    n = len(y_all)
    out = np.full(n, np.nan)

    for t in range(window - 1, n):
        y = y_all[t - window + 1:t + 1]
        X = np.column_stack([
            np.ones(window),
            mktrf[t - window + 1:t + 1],
            smb[t - window + 1:t + 1],
            hml[t - window + 1:t + 1],
        ])

        if np.any(~np.isfinite(y)) or np.any(~np.isfinite(X)):
            continue

        try:
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ coef
            out[t] = np.std(resid, ddof=1)
        except np.linalg.LinAlgError:
            continue

    return out

# ---------------------------------------------------------
# 1) Load RSJ keys
# ---------------------------------------------------------
print("Loading RSJ weekly sample...")
rsj = pd.read_parquet(rsj_path)
rsj["permno"] = pd.to_numeric(rsj["permno"], errors="coerce").astype("Int64")
rsj["week"] = parse_date_col(rsj["week"])
rsj = rsj.dropna(subset=["permno", "week"]).copy()

rsj_keys = rsj[["permno", "week"]].drop_duplicates().copy()

print(f"RSJ stock-week rows: {len(rsj_keys):,}")
print(f"RSJ unique permnos : {rsj_keys['permno'].nunique():,}")
print(f"RSJ unique weeks   : {rsj_keys['week'].nunique():,}")

del rsj
gc.collect()

# ---------------------------------------------------------
# 2) Build daily base controls from CRSP
# ---------------------------------------------------------
print("\nLoading CRSP daily...")
crsp = pd.read_parquet(crsp_path)

needed_cols = [
    "permno", "date", "ret", "retx", "prc", "vol", "shrout",
    "me", "dollar_vol"
]
existing_cols = [c for c in needed_cols if c in crsp.columns]
crsp = crsp[existing_cols].copy()

crsp = safe_to_numeric(crsp, ["permno", "ret", "retx", "prc", "vol", "shrout", "me", "dollar_vol"])
crsp["date"] = parse_date_col(crsp["date"])
crsp["permno"] = crsp["permno"].astype("Int64")

print("Loading CRSP delist...")
delist = pd.read_parquet(delist_path)
if len(delist) > 0:
    delist = safe_to_numeric(delist, ["permno", "dlret"])
    delist["dlstdt"] = parse_date_col(delist["dlstdt"])
    delist["permno"] = delist["permno"].astype("Int64")
    delist = delist.rename(columns={"dlstdt": "date"})
    delist = delist[["permno", "date", "dlret"]].copy()
else:
    delist = pd.DataFrame(columns=["permno", "date", "dlret"])

print("Merging delisting returns...")
crsp = crsp.merge(delist, on=["permno", "date"], how="left")
del delist
gc.collect()

# Adjusted return
crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce").astype(float)
crsp["dlret"] = pd.to_numeric(crsp["dlret"], errors="coerce").astype(float)

crsp["ret_adj"] = np.nan
mask_ret = crsp["ret"].notna()
mask_dl = crsp["dlret"].notna()

crsp.loc[mask_ret & ~mask_dl, "ret_adj"] = crsp.loc[mask_ret & ~mask_dl, "ret"]
crsp.loc[~mask_ret & mask_dl, "ret_adj"] = crsp.loc[~mask_ret & mask_dl, "dlret"]
crsp.loc[mask_ret & mask_dl, "ret_adj"] = (
    (1.0 + crsp.loc[mask_ret & mask_dl, "ret"]) *
    (1.0 + crsp.loc[mask_ret & mask_dl, "dlret"]) - 1.0
)

crsp = crsp.dropna(subset=["permno", "date"]).copy()
crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True)

print("Constructing daily ME, ILLIQ, MOM...")

# Market equity
if "me" not in crsp.columns or crsp["me"].isna().all():
    crsp["me"] = crsp["prc"].abs() * crsp["shrout"]

crsp["me"] = pd.to_numeric(crsp["me"], errors="coerce").astype(float)
crsp["log_me_daily"] = np.nan
mask_me = crsp["me"].notna() & (crsp["me"] > 0)
crsp.loc[mask_me, "log_me_daily"] = np.log(crsp.loc[mask_me, "me"])

# Dollar volume
if "dollar_vol" not in crsp.columns or crsp["dollar_vol"].isna().all():
    crsp["dollar_vol"] = crsp["prc"].abs() * crsp["vol"]

crsp["dollar_vol"] = pd.to_numeric(crsp["dollar_vol"], errors="coerce").astype(float)
crsp["ret_adj"] = pd.to_numeric(crsp["ret_adj"], errors="coerce").astype(float)

crsp["amihud_daily"] = np.nan
mask_illiq = crsp["dollar_vol"].notna() & (crsp["dollar_vol"] > 0) & crsp["ret_adj"].notna()
crsp.loc[mask_illiq, "amihud_daily"] = (
    np.abs(crsp.loc[mask_illiq, "ret_adj"]) / crsp.loc[mask_illiq, "dollar_vol"]
)

# Rolling illiquidity
crsp["illiq_daily"] = (
    crsp.groupby("permno")["amihud_daily"]
        .rolling(illiq_window, min_periods=illiq_window)
        .mean()
        .reset_index(level=0, drop=True)
)

crsp["illiq_daily"] = pd.to_numeric(crsp["illiq_daily"], errors="coerce").astype(float)
crsp["log_illiq_daily"] = np.nan
mask_log_illiq = crsp["illiq_daily"].notna() & (crsp["illiq_daily"] > 0)
crsp.loc[mask_log_illiq, "log_illiq_daily"] = np.log(crsp.loc[mask_log_illiq, "illiq_daily"])

# Momentum
print("Computing daily momentum...")
mom_parts = []
n_stocks = crsp["permno"].nunique()

for i, (permno, g) in enumerate(crsp.groupby("permno", sort=False), start=1):
    if i % ivol_progress_every == 0:
        print(f"Momentum: {i:,} / {n_stocks:,} stocks...")

    g = g.sort_values("date").copy()
    g["mom_daily"] = rolling_cumret_ex_skip(
        g["ret_adj"],
        lookback=mom_lookback_days,
        skip=mom_skip_days
    )
    mom_parts.append(g[["permno", "date", "mom_daily"]])

mom_daily = pd.concat(mom_parts, ignore_index=True)
del mom_parts
gc.collect()

crsp = crsp.merge(mom_daily, on=["permno", "date"], how="left", validate="one_to_one")
del mom_daily
gc.collect()

# Keep only what is needed before IVOL
crsp = crsp[[
    "permno",
    "date",
    "ret_adj",
    "me",
    "log_me_daily",
    "log_illiq_daily",
    "mom_daily"
]].copy()

crsp.to_parquet(daily_base_path, index=False)
print(f"Saved daily base controls: {daily_base_path}")

# ---------------------------------------------------------
# 3) Compute daily IVOL separately
# ---------------------------------------------------------
print("\nLoading FF daily factors...")
ff = pd.read_parquet(ff_path)
ff = safe_to_numeric(ff, ["mktrf", "smb", "hml", "rf"])
ff["date"] = parse_date_col(ff["date"])
ff = ff[["date", "mktrf", "smb", "hml", "rf"]].copy()

print("Computing daily IVOL...")
daily_base = pd.read_parquet(daily_base_path)
daily_base["date"] = parse_date_col(daily_base["date"])
daily_base["permno"] = pd.to_numeric(daily_base["permno"], errors="coerce").astype("Int64")

daily_base = daily_base.merge(ff, on="date", how="left", validate="many_to_one")
del ff
gc.collect()

ivol_parts = []
n_stocks = daily_base["permno"].nunique()

for i, (permno, g) in enumerate(daily_base.groupby("permno", sort=False), start=1):
    if i % ivol_progress_every == 0:
        print(f"IVOL: {i:,} / {n_stocks:,} stocks...")

    g = g.sort_values("date").copy()

    ret_adj = pd.to_numeric(g["ret_adj"], errors="coerce").to_numpy(dtype=float)
    rf = pd.to_numeric(g["rf"], errors="coerce").to_numpy(dtype=float)
    mktrf = pd.to_numeric(g["mktrf"], errors="coerce").to_numpy(dtype=float)
    smb = pd.to_numeric(g["smb"], errors="coerce").to_numpy(dtype=float)
    hml = pd.to_numeric(g["hml"], errors="coerce").to_numpy(dtype=float)

    ivol = compute_ivol_array(
        ret_adj=ret_adj,
        rf=rf,
        mktrf=mktrf,
        smb=smb,
        hml=hml,
        window=ivol_window
    )

    ivol_parts.append(
        pd.DataFrame({
            "permno": g["permno"].to_numpy(),
            "date": g["date"].to_numpy(),
            "ivol_daily": ivol
        })
    )

ivol_daily = pd.concat(ivol_parts, ignore_index=True)
ivol_daily.to_parquet(daily_ivol_path, index=False)
print(f"Saved daily IVOL: {daily_ivol_path}")

del ivol_parts
gc.collect()

# ---------------------------------------------------------
# 4) Merge daily base + IVOL, aggregate to weekly
# ---------------------------------------------------------
print("\nAggregating daily controls to weekly...")
daily = pd.read_parquet(daily_base_path)
daily["date"] = parse_date_col(daily["date"])
daily["permno"] = pd.to_numeric(daily["permno"], errors="coerce").astype("Int64")

ivol_daily = pd.read_parquet(daily_ivol_path)
ivol_daily["date"] = parse_date_col(ivol_daily["date"])
ivol_daily["permno"] = pd.to_numeric(ivol_daily["permno"], errors="coerce").astype("Int64")

daily = daily.merge(ivol_daily, on=["permno", "date"], how="left", validate="one_to_one")
del ivol_daily
gc.collect()

daily["week"] = make_week_from_date(daily["date"])

weekly_ret = (
    daily.groupby(["permno", "week"], as_index=False)
        .agg(
            ret_weekly=("ret_adj", lambda x: np.prod(1.0 + x.dropna()) - 1.0 if x.dropna().size > 0 else np.nan)
        )
)

daily_last = (
    daily.sort_values(["permno", "date"])
        .groupby(["permno", "week"], as_index=False)
        .tail(1)
        .copy()
)

weekly_controls = daily_last[[
    "permno",
    "week",
    "log_me_daily",
    "mom_daily",
    "log_illiq_daily",
    "ivol_daily",
    "me",
    "date"
]].copy()

weekly_controls = weekly_controls.rename(columns={
    "log_me_daily": "log_me",
    "mom_daily": "mom",
    "log_illiq_daily": "illiq",
    "ivol_daily": "ivol",
    "me": "me_raw",
    "date": "week_last_trading_date",
})

weekly_controls = weekly_controls.merge(weekly_ret, on=["permno", "week"], how="left")
weekly_controls = weekly_controls.sort_values(["permno", "week"]).reset_index(drop=True)
weekly_controls["rev"] = weekly_controls.groupby("permno")["ret_weekly"].shift(1)

del daily
del daily_last
del weekly_ret
gc.collect()

# ---------------------------------------------------------
# 5) Build BM from Compustat + CCM + December ME
# ---------------------------------------------------------
print("Constructing BM from Compustat and CCM...")

comp = pd.read_parquet(comp_path)
ccm = pd.read_parquet(ccm_path)

comp["datadate"] = parse_date_col(comp["datadate"])
ccm["linkdt"] = parse_date_col(ccm["linkdt"])
ccm["linkenddt"] = parse_date_col(ccm["linkenddt"])

comp = safe_to_numeric(comp, ["fyear", "fyr", "seq", "txditc", "pstkrv", "pstkl", "pstk"])
ccm = safe_to_numeric(ccm, ["permno"])

comp["gvkey"] = comp["gvkey"].astype(str)
ccm["gvkey"] = ccm["gvkey"].astype(str)
ccm["permno"] = pd.to_numeric(ccm["permno"], errors="coerce").astype("Int64")

pref = comp["pstkrv"].copy()
pref = pref.fillna(comp["pstkl"])
pref = pref.fillna(comp["pstk"])
pref = pref.fillna(0.0)

txditc = comp["txditc"].fillna(0.0)
comp["book_equity"] = comp["seq"] + txditc - pref
comp = comp[comp["book_equity"].notna() & (comp["book_equity"] > 0)].copy()

ccm = ccm[ccm["linktype"].isin(["LU", "LC", "LS"])].copy()
ccm = ccm[ccm["linkprim"].isin(["P", "C"])].copy()
ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2100-12-31"))

bm_base = comp.merge(ccm, on="gvkey", how="inner")
bm_base = bm_base[
    (bm_base["datadate"] >= bm_base["linkdt"]) &
    (bm_base["datadate"] <= bm_base["linkenddt"])
].copy()

# December ME from weekly_controls me_raw is not enough; use daily base / raw daily file
crsp_for_dec = pd.read_parquet(daily_base_path)
crsp_for_dec["date"] = parse_date_col(crsp_for_dec["date"])
crsp_for_dec["permno"] = pd.to_numeric(crsp_for_dec["permno"], errors="coerce").astype("Int64")
crsp_for_dec["year"] = crsp_for_dec["date"].dt.year
crsp_for_dec["month"] = crsp_for_dec["date"].dt.month

dec_me = (
    crsp_for_dec[crsp_for_dec["month"] == 12]
        .sort_values(["permno", "date"])
        .groupby(["permno", "year"], as_index=False)
        .tail(1)
        .copy()
)

dec_me = dec_me[["permno", "year", "me"]].rename(columns={"year": "dec_year", "me": "dec_me"})
del crsp_for_dec
gc.collect()

bm_base["dec_year"] = bm_base["fyear"]
bm_base["bm_year"] = bm_base["fyear"] + 1

bm_base = bm_base.merge(dec_me, on=["permno", "dec_year"], how="left")
bm_base["bm"] = bm_base["book_equity"] / (bm_base["dec_me"] / 1000)
bm_base = bm_base[(bm_base["bm"] > 0) & np.isfinite(bm_base["bm"])].copy()

bm_yearly = (
    bm_base.sort_values(["permno", "bm_year", "datadate"])
        .groupby(["permno", "bm_year"], as_index=False)
        .tail(1)
        .copy()
)

bm_yearly = bm_yearly[["permno", "bm_year", "bm"]].copy()
bm_yearly["permno"] = pd.to_numeric(bm_yearly["permno"], errors="coerce").astype("Int64")

weekly_controls["year"] = weekly_controls["week"].dt.year
weekly_controls["month"] = weekly_controls["week"].dt.month
weekly_controls["bm_year"] = np.where(
    weekly_controls["month"] >= 7,
    weekly_controls["year"],
    weekly_controls["year"] - 1
).astype(int)

weekly_controls = weekly_controls.merge(
    bm_yearly,
    on=["permno", "bm_year"],
    how="left"
)

# ---------------------------------------------------------
# 6) Align to RSJ sample and save
# ---------------------------------------------------------
print("Aligning final controls to RSJ weekly sample...")

final = rsj_keys.merge(
    weekly_controls,
    on=["permno", "week"],
    how="left",
    validate="one_to_one"
)

final = final[[
    "permno",
    "week",
    "log_me",
    "me_raw",
    "bm",
    "mom",
    "rev",
    "ivol",
    "illiq",
    "ret_weekly",
    "week_last_trading_date",
]].copy()

final = final.rename(columns={"log_me": "me"})

print("\nPreview:")
print(final.head())

print("\nMissing shares:")
print(final[["me", "bm", "mom", "rev", "ivol", "illiq"]].isna().mean())

print("\nSummary:")
print(final[["me", "bm", "mom", "rev", "ivol", "illiq"]].describe())

final.to_parquet(output_path, index=False)
print(f"\nSaved weekly controls to: {output_path}")