"""
Build weekly analysis panel.

Merges all weekly data sources onto the RSJ universe (permno x week).
Applies:
  - Share code filter: shrcd in [10, 11] (ordinary common shares)
  - Price filter: 5 <= abs(prc) <= 1000, using end-of-week price

Join key: (permno, week) where week is the W-TUE period-end timestamp.

Output: data_intermediate/panel/weekly_panel.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Settings
# ============================================================
ROOT = Path(__file__).resolve().parents[2]

# Inputs
RSJ_WEEKLY          = ROOT / "data_intermediate" / "rsj_weekly" / "rsj_weekly.parquet"
RSJ_METHOD1_WEEKLY  = ROOT / "data_intermediate" / "decomposition" / "method1" / "rsj_method1_weekly.parquet"
RSJ_METHOD2_WEEKLY  = ROOT / "data_intermediate" / "decomposition" / "method2" / "rsj_method2_spy.parquet"
RES_WEEKLY          = ROOT / "data_intermediate" / "res_weekly" / "res_weekly.parquet"
RES_SPLIT_WEEKLY    = ROOT / "data_intermediate" / "weekly_data_res_split" / "weekly_res_sys_idio.parquet"
WEEKLY_RETURNS      = ROOT / "data_intermediate" / "weekly_stock_returns" / "weekly_stock_returns.parquet"
WEEKLY_CONTROLS     = ROOT / "data_intermediate" / "controls" / "weekly_controls.parquet"
RVOL_RSK_RKT_WEEKLY = ROOT / "data_intermediate" / "rvol_rsk_rkt_weekly" / "rvol_rsk_rkt_weekly.parquet"

# Output
OUTPUT_DIR  = ROOT / "data_final" / "panel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "weekly_panel.parquet"

# Filters
SHRCD_KEEP  = {10, 11}
PRICE_MIN   = 5.0
PRICE_MAX   = 1000.0


# ============================================================
# Helpers
# ============================================================
def read(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  WARNING: {label} not found at {path.name} — skipping.")
        return None
    df = pd.read_parquet(path)
    print(f"  Loaded {label}: {len(df):,} rows, cols: {df.columns.tolist()}")
    return df


def to_int64_permno(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    return df


def normalize_week(df: pd.DataFrame, col: str = "week") -> pd.DataFrame:
    """Ensure week column is datetime64[ns] (normalize to midnight)."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col]).dt.normalize()
    return df


# ============================================================
# Main
# ============================================================
def main():
    print("=== Build Weekly Panel ===\n")

    # ----------------------------------------------------------
    # 1) Base universe: RSJ weekly
    # ----------------------------------------------------------
    print("Loading base universe (RSJ weekly)...")
    base = read(RSJ_WEEKLY, "rsj_weekly")
    if base is None:
        raise FileNotFoundError(f"Base file not found: {RSJ_WEEKLY}")

    base = to_int64_permno(base)
    base = normalize_week(base, "week")
    base = base[["permno", "week", "rsj_weekly", "n_days", "n_obs_total"]].copy()
    print(f"  Base universe: {len(base):,} stock-weeks, "
          f"{base['permno'].nunique():,} stocks, "
          f"{base['week'].nunique():,} weeks\n")

    # ----------------------------------------------------------
    # 2) Weekly stock returns (R_i_w_plus_1 is the forward return)
    # ----------------------------------------------------------
    print("Loading weekly returns...")
    ret = read(WEEKLY_RETURNS, "weekly_returns")
    if ret is not None:
        ret = to_int64_permno(ret)
        ret = normalize_week(ret, "week")
        ret = ret[["permno", "week", "R_i_w", "R_i_w_plus_1", "valid_R_i_w_plus_1"]].copy()
        base = base.merge(ret, on=["permno", "week"], how="left")
    print()

    # ----------------------------------------------------------
    # 3) Weekly controls
    # ----------------------------------------------------------
    print("Loading weekly controls...")
    ctrl = read(WEEKLY_CONTROLS, "weekly_controls")
    if ctrl is not None:
        ctrl = to_int64_permno(ctrl)
        ctrl = normalize_week(ctrl, "week")
        ctrl_cols = ["permno", "week", "me", "me_raw", "bm", "mom", "rev",
                     "ivol", "illiq", "week_last_trading_date"]
        for c in ["prc", "shrcd", "exchcd"]:
            if c in ctrl.columns:
                ctrl_cols.append(c)
        ctrl = ctrl[ctrl_cols].copy()
        ctrl["week_last_trading_date"] = pd.to_datetime(ctrl["week_last_trading_date"]).dt.normalize()
        base = base.merge(ctrl, on=["permno", "week"], how="left")
    print()

    # ----------------------------------------------------------
    # 4) RSJ Method 1 decomposition
    # ----------------------------------------------------------
    print("Loading RSJ Method 1 decomposition...")
    m1 = read(RSJ_METHOD1_WEEKLY, "rsj_method1_weekly")
    if m1 is not None:
        m1 = to_int64_permno(m1)
        m1 = normalize_week(m1, "week")
        m1 = m1[["permno", "week", "rsj_sys_weekly", "rsj_idio_weekly"]].copy()
        base = base.merge(m1, on=["permno", "week"], how="left")
    print()

    # ----------------------------------------------------------
    # 5) RSJ Method 2 decomposition
    # ----------------------------------------------------------
    print("Loading RSJ Method 2 decomposition...")
    m2 = read(RSJ_METHOD2_WEEKLY, "rsj_method2_weekly")
    if m2 is not None:
        m2 = to_int64_permno(m2)
        m2 = normalize_week(m2, "week")
        m2 = m2[["permno", "week", "rsj_sys", "rsj_idio", "rsj_sys_no_alpha",
                  "beta_rsj", "alpha_rsj"]].copy()
        base = base.merge(m2, on=["permno", "week"], how="left")
    print()

    # ----------------------------------------------------------
    # 6) RES weekly (total)
    # ----------------------------------------------------------
    print("Loading RES weekly (total)...")
    res = read(RES_WEEKLY, "res_weekly")
    if res is not None:
        res = to_int64_permno(res)
        # RES uses 'week' column (renamed from week_end in 02_res_decomposition_final.py)
        if "week" in res.columns:
            res = normalize_week(res, "week")
        elif "week_end" in res.columns:
            res = res.rename(columns={"week_end": "week"})
            res = normalize_week(res, "week")
        res = res[["permno", "week", "res_weekly"]].copy()
        base = base.merge(res, on=["permno", "week"], how="left")
    print()

    # ----------------------------------------------------------
    # 7) RES split weekly (systematic + idiosyncratic)
    # ----------------------------------------------------------
    print("Loading RES split weekly...")
    res_split = read(RES_SPLIT_WEEKLY, "weekly_res_sys_idio")
    if res_split is not None:
        res_split = to_int64_permno(res_split)
        if "week_end" in res_split.columns and "week" not in res_split.columns:
            res_split = res_split.rename(columns={"week_end": "week"})
        res_split = normalize_week(res_split, "week")
        keep_cols = ["permno", "week"]
        for c in ["res_sys_p025", "res_idio_p025", "res_total_p025"]:
            if c in res_split.columns:
                keep_cols.append(c)
        res_split = res_split[keep_cols].copy()
        base = base.merge(res_split, on=["permno", "week"], how="left")
    print()

    # ----------------------------------------------------------
    # 8) RVOL, RSK, RKT weekly (Bollerslev et al. 2020)
    # ----------------------------------------------------------
    print("Loading RVOL/RSK/RKT weekly...")
    rvol_rsk_rkt = read(RVOL_RSK_RKT_WEEKLY, "rvol_rsk_rkt_weekly")
    if rvol_rsk_rkt is not None:
        rvol_rsk_rkt = to_int64_permno(rvol_rsk_rkt)
        rvol_rsk_rkt = normalize_week(rvol_rsk_rkt, "week")
        rvol_rsk_rkt = rvol_rsk_rkt[["permno", "week", "rvol", "rsk", "rkt"]].copy()
        base = base.merge(rvol_rsk_rkt, on=["permno", "week"], how="left")
    print()

    # prc, shrcd, exchcd come from controls (added in 02_build_weekly_controls.py)
    for c in ["prc", "shrcd", "exchcd"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")
    print()

    # ----------------------------------------------------------
    # 9) Apply filters
    # ----------------------------------------------------------
    n_before = len(base)
    print(f"Rows before filtering: {n_before:,}")

    # Share code filter
    if "shrcd" in base.columns:
        valid_shrcd = base["shrcd"].isin(SHRCD_KEEP)
        missing_shrcd = base["shrcd"].isna()
        # Keep rows with valid shrcd; drop missing shrcd (no info available)
        base = base[valid_shrcd].copy()
        print(f"  After shrcd filter ({SHRCD_KEEP}): {len(base):,} rows "
              f"(dropped {n_before - len(base):,})")
        n_before = len(base)

    # Price filter: abs(prc) in [5, 1000]
    if "prc" in base.columns:
        abs_prc = base["prc"].abs()
        valid_price = (abs_prc >= PRICE_MIN) & (abs_prc <= PRICE_MAX)
        base = base[valid_price].copy()
        print(f"  After price filter (${PRICE_MIN}–${PRICE_MAX}): {len(base):,} rows "
              f"(dropped {n_before - len(base):,})")

    # Drop prc/shrcd after filtering; keep exchcd for NYSE breakpoints in portfolio sorts
    base = base.drop(columns=["prc", "shrcd"], errors="ignore")

    # ----------------------------------------------------------
    # 10) Stock-week filter: keep only stock-weeks where ALL
    #     main (non-control) variables are available.
    #
    #     Main variables required (all must be non-null):
    #       rsj_weekly, res_weekly, rvol, rsk, rkt,
    #       rsj_sys_weekly, rsj_idio_weekly,
    #       rsj_sys, rsj_idio,
    #       res_sys_p025, res_idio_p025
    #
    #     Control variables (me, bm, mom, rev, ivol, illiq,
    #     R_i_w, R_i_w_plus_1) are allowed to be missing.
    # ----------------------------------------------------------
    print("\nApplying stock-week filter (all main variables must be non-null per row)...")
    n_before = len(base)

    MAIN_VARS = [
        "rsj_weekly",
        "res_weekly",
        "rvol", "rsk", "rkt",
        "rsj_sys_weekly", "rsj_idio_weekly",
        "rsj_sys", "rsj_idio",
        "res_sys_p025", "res_idio_p025",
    ]

    present_main_vars = [c for c in MAIN_VARS if c in base.columns]
    missing_main_vars = [c for c in MAIN_VARS if c not in base.columns]
    if missing_main_vars:
        print(f"  WARNING: main variable columns not in panel (skipped): {missing_main_vars}")

    if present_main_vars:
        mask = base[present_main_vars].notna().all(axis=1)
        base = base[mask].copy()
        print(f"  Required columns: {present_main_vars}")
        print(f"  Rows after stock-week filter: {len(base):,} "
              f"(dropped {n_before - len(base):,})")
    else:
        print("  WARNING: no main variable columns found; filter skipped.")

    # ----------------------------------------------------------
    # 11) Save
    # ----------------------------------------------------------
    base = base.sort_values(["week", "permno"]).reset_index(drop=True)

    print(f"\nFinal panel: {len(base):,} stock-weeks")
    print(f"  Stocks : {base['permno'].nunique():,}")
    print(f"  Weeks  : {base['week'].nunique():,}")
    print(f"  Date range: {base['week'].min().date()} to {base['week'].max().date()}")
    print(f"\nColumns: {base.columns.tolist()}")

    missing = base.isna().mean().sort_values(ascending=False)
    print("\nMissing rate per column:")
    print(missing[missing > 0].round(4).to_string())

    base.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
