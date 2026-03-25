from pathlib import Path
import math
import pandas as pd
import wrds

# =========================================================
# PURPOSE
# =========================================================
# Download WRDS inputs needed for control variables:
#
#   - CRSP daily stock data      -> ME, MOM, REV, ILLIQ, IVOL inputs
#   - CRSP delisting returns     -> adjusted returns if needed later
#   - Compustat annual funda     -> BM inputs
#   - CCM link table             -> CRSP-Compustat merge for BM
#   - Fama-French daily factors  -> IVOL
#
# SPACE-SAVING APPROACH:
#   Restrict CRSP downloads to PERMNOs that appear in:
#       data_intermediate/rsj_weekly/rsj_weekly.parquet
#
# OUTPUTS:
#   data_raw/wrds/crsp_daily_rsj_sample_1992_2024.parquet
#   data_raw/wrds/crsp_delist_rsj_sample_1992_2024.parquet
#   data_raw/wrds/ff_daily_factors_1992_2024.parquet
#   data_raw/wrds/comp_funda_1991_2024.parquet
#   data_raw/wrds/ccm_linktable.parquet
# =========================================================

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
RSJ_WEEKLY_PATH = Path("data_intermediate/rsj_weekly/rsj_weekly.parquet")
OUT_DIR = Path("data_raw/wrds")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CRSP_START = "1992-01-01"   # one extra year for lagged controls
COMP_START = "1991-01-01"   # one extra year for BM timing
END_DATE = "2024-12-31"

CRSP_OUT = OUT_DIR / "crsp_daily_rsj_sample_1992_2024.parquet"
DELIST_OUT = OUT_DIR / "crsp_delist_rsj_sample_1992_2024.parquet"
FF_OUT = OUT_DIR / "ff_daily_factors_1992_2024.parquet"
COMP_OUT = OUT_DIR / "comp_funda_1991_2024.parquet"
CCM_OUT = OUT_DIR / "ccm_linktable.parquet"

# Fama-French library fallback order
FF_LIBRARY_CANDIDATES = ["ff_all", "ff"]

# Query chunk size for PERMNO filtering
CHUNK_SIZE = 800

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def chunked(seq, size):
    seq = list(seq)
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def sql_in_list(values):
    return "(" + ",".join(str(int(v)) for v in values) + ")"

def find_ff_table(db):
    """
    Find the daily Fama-French factors table on this WRDS account.
    Returns (library, table_name).
    """
    preferred_names = [
        "factors_daily",
        "ff_factors_daily",
        "fama_french_daily"
    ]

    for lib in FF_LIBRARY_CANDIDATES:
        try:
            tables = set(db.list_tables(library=lib))
        except Exception:
            continue

        for name in preferred_names:
            if name in tables:
                return lib, name

        # Fallback: look for any daily factor-like table
        for t in sorted(tables):
            tl = t.lower()
            if "daily" in tl and ("factor" in tl or "ff" in tl):
                return lib, t

    raise RuntimeError(
        "Could not locate a daily Fama-French factor table. "
        "Run Scripts/04_controls/00_check_wrds_tables.py first."
    )

# ---------------------------------------------------------
# Load RSJ sample PERMNOs
# ---------------------------------------------------------
print("Loading RSJ weekly file...")
rsj = pd.read_parquet(RSJ_WEEKLY_PATH)

if "permno" not in rsj.columns:
    raise KeyError("Expected column 'permno' in rsj_weekly.parquet")

permnos = (
    pd.Series(rsj["permno"])
    .dropna()
    .astype(int)
    .sort_values()
    .unique()
    .tolist()
)

if not permnos:
    raise ValueError("No PERMNOs found in RSJ weekly file.")

print(f"Found {len(permnos):,} unique PERMNOs in RSJ sample.")

# ---------------------------------------------------------
# Connect WRDS
# ---------------------------------------------------------
print("\nConnecting to WRDS...")
db = wrds.Connection()

# ---------------------------------------------------------
# 1) CRSP daily stock data, restricted to RSJ PERMNOs
# ---------------------------------------------------------
print("\nDownloading CRSP daily stock data in chunks...")

crsp_parts = []

for idx, chunk in enumerate(chunked(permnos, CHUNK_SIZE), start=1):
    print(f"  Chunk {idx}: {len(chunk)} permnos")
    permno_filter = sql_in_list(chunk)

    sql = f"""
    select
        a.permno,
        a.permco,
        a.date,
        a.ret,
        a.retx,
        a.prc,
        a.vol,
        a.shrout,
        a.cfacpr,
        a.cfacshr,
        a.bid,
        a.ask,
        b.shrcd,
        b.exchcd,
        b.ticker,
        b.ncusip
    from crsp.dsf as a
    left join crsp.dsenames as b
      on a.permno = b.permno
     and b.namedt <= a.date
     and a.date <= b.nameendt
    where a.date between '{CRSP_START}' and '{END_DATE}'
      and a.permno in {permno_filter}
      and b.shrcd in (10, 11)
      and b.exchcd in (1, 2, 3)
    order by a.permno, a.date
    """

    part = db.raw_sql(sql, date_cols=["date"])
    crsp_parts.append(part)

crsp = pd.concat(crsp_parts, ignore_index=True)
crsp = crsp.drop_duplicates(subset=["permno", "date"]).copy()

# CRSP cleanups
crsp["prc"] = crsp["prc"].abs()
crsp["me"] = crsp["prc"] * crsp["shrout"]            # market equity in $ thousands
crsp["dollar_vol"] = crsp["prc"] * crsp["vol"]       # daily dollar trading volume proxy

crsp.to_parquet(CRSP_OUT, index=False)
print(f"Saved: {CRSP_OUT}")
print(f"CRSP rows: {len(crsp):,}")

# ---------------------------------------------------------
# 2) CRSP delisting returns, restricted to RSJ PERMNOs
# ---------------------------------------------------------
print("\nDownloading CRSP delisting data in chunks...")

delist_parts = []

for idx, chunk in enumerate(chunked(permnos, CHUNK_SIZE), start=1):
    print(f"  Chunk {idx}: {len(chunk)} permnos")
    permno_filter = sql_in_list(chunk)

    sql = f"""
    select
        permno,
        dlstdt,
        dlret,
        dlstcd
    from crsp.dsedelist
    where dlstdt between '{CRSP_START}' and '{END_DATE}'
      and permno in {permno_filter}
    order by permno, dlstdt
    """

    part = db.raw_sql(sql, date_cols=["dlstdt"])
    delist_parts.append(part)

delist = pd.concat(delist_parts, ignore_index=True) if delist_parts else pd.DataFrame()
if not delist.empty:
    delist = delist.drop_duplicates(subset=["permno", "dlstdt"]).copy()

delist.to_parquet(DELIST_OUT, index=False)
print(f"Saved: {DELIST_OUT}")
print(f"Delist rows: {len(delist):,}")

# ---------------------------------------------------------
# 3) Fama-French daily factors
# ---------------------------------------------------------
print("\nLocating Fama-French daily factor table...")
ff_lib, ff_table = find_ff_table(db)
print(f"Using {ff_lib}.{ff_table}")

sql_ff = f"""
select
    date,
    mktrf,
    smb,
    hml,
    rf
from {ff_lib}.{ff_table}
where date between '{CRSP_START}' and '{END_DATE}'
order by date
"""

ff = db.raw_sql(sql_ff, date_cols=["date"])

# WRDS Fama-French factors are typically stored in percent units; convert to decimals.
for col in ["mktrf", "smb", "hml", "rf"]:
    ff[col] = ff[col] / 100.0

ff.to_parquet(FF_OUT, index=False)
print(f"Saved: {FF_OUT}")
print(f"FF rows: {len(ff):,}")

# ---------------------------------------------------------
# 4) Compustat annual fundamentals
# ---------------------------------------------------------
print("\nDownloading Compustat annual fundamentals...")

sql_comp = f"""
select
    gvkey,
    datadate,
    fyear,
    fyr,
    cusip,
    conm,
    at,
    ceq,
    seq,
    txdb,
    txditc,
    pstkrv,
    pstkl,
    pstk
from comp.funda
where datadate between '{COMP_START}' and '{END_DATE}'
  and indfmt = 'INDL'
  and datafmt = 'STD'
  and popsrc = 'D'
  and consol = 'C'
order by gvkey, datadate
"""

comp = db.raw_sql(sql_comp, date_cols=["datadate"])
comp.to_parquet(COMP_OUT, index=False)
print(f"Saved: {COMP_OUT}")
print(f"Compustat rows: {len(comp):,}")

# ---------------------------------------------------------
# 5) CCM link table
# ---------------------------------------------------------
print("\nDownloading CCM link table...")

sql_ccm = """
select
    gvkey,
    lpermno as permno,
    lpermco as permco,
    linktype,
    linkprim,
    linkdt,
    linkenddt
from crsp.ccmxpf_linktable
where lpermno is not null
order by gvkey, lpermno, linkdt
"""

ccm = db.raw_sql(sql_ccm, date_cols=["linkdt", "linkenddt"])
ccm.to_parquet(CCM_OUT, index=False)
print(f"Saved: {CCM_OUT}")
print(f"CCM rows: {len(ccm):,}")

# ---------------------------------------------------------
# Done
# ---------------------------------------------------------
db.close()

print("\nAll WRDS downloads complete.")
print("Files created:")
print(f"  - {CRSP_OUT}")
print(f"  - {DELIST_OUT}")
print(f"  - {FF_OUT}")
print(f"  - {COMP_OUT}")
print(f"  - {CCM_OUT}")