# Seminar-investing-group-9

This repository builds high-frequency downside-risk measures for US equities, decomposes them into systematic and idiosyncratic components, and prepares weekly panels for return predictability tests.

Main research objects:
- RSJ (Relative Signed Jump): asymmetry in intraday semivariance.
- RES (Realized Expected Shortfall): downside tail risk from intraday returns.
- Systematic vs idiosyncratic decompositions of RSJ and RES.
- Weekly prediction tests (portfolio sorts and Fama-MacBeth style panels).

## 1) Repository Data Flow

```
data_raw/intraday_stocks          data_raw/intraday_etfs
        |                                  |
  (Stage 01)                         (Stage 01)
        |                                  |
data_intermediate/converted_parquet   data_intermediate/converted_parquet_etf
        |                                  |
  (Stage 01)                         (Stage 03)
        |                                  |
data_intermediate/rsj_daily        data_intermediate/market_returns/market_intraday_spy.parquet
        |                                  |
  (Stage 02)                    (Stage 03 decompositions)
        |
data_intermediate/rsj_weekly/rsj_weekly.parquet
data_intermediate/weekly_stock_returns/weekly_stock_returns.parquet
        |
  (Stage 03 + Stage 04)
        |
data_intermediate/decomposition/        data_intermediate/res_weekly/
data_intermediate/controls/             data_intermediate/weekly_data_res_split/
```

All scripts use weeks ending **Tuesday** (`W-TUE`), so each week covers through Tuesday market close. Stock-days require at least **80 intraday 5-minute observations** (transactions) to be included.

## 2) Script Stages

### Stage 01: Data Preparation
Folder: `Scripts/01_data_preparation/`

| Script | Input | Output |
|--------|-------|--------|
| `00_convert_rds_to_parquet_etf.R` | `data_raw/intraday_etfs` | `data_intermediate/converted_parquet_etf` |
| `01_convert_rds_to_parquet.R` | `data_raw/intraday_stocks` | `data_intermediate/converted_parquet` |
| `02_compute_rsj_all_files.py` | `data_intermediate/converted_parquet` | `data_intermediate/rsj_daily` |

`02_compute_rsj_all_files.py` filters stock-days with `n_obs >= 80` before computing RSJ.

### Stage 02: Variable Construction
Folder: `Scripts/02_variable_construction/`

| Script | Input | Output |
|--------|-------|--------|
| `01_aggregate_weekly_rsj.py` | `data_intermediate/rsj_daily` | `data_intermediate/rsj_weekly/rsj_weekly.parquet` |
| `03_compute_weekly_returns.py` | `data_intermediate/converted_parquet` | `data_intermediate/weekly_stock_returns/weekly_stock_returns.parquet` |
| `04_build_weekly_market_rsj.py` | `data_intermediate/converted_parquet_etf` | `data_intermediate/market_weekly_rsj/spy_weekly_rsj.parquet` |

`04_build_weekly_market_rsj.py` filters ETF days with `n_obs >= 80`.

Weekly RSJ format: `permno`, `week`, `rsj_weekly`, `n_days`, `n_obs_total`.

### Stage 03: Decomposition
Folder: `Scripts/03_decomposition/`

**RSJ Method 1 (intraday-level decomposition):**

| Script | Input | Output |
|--------|-------|--------|
| `01_rsj_method1_build_market_intraday_from_etf.py` | `data_intermediate/converted_parquet_etf` | `data_intermediate/market_returns/market_intraday_spy.parquet` |
| `02_rsj_method1_decomposition.py` | `converted_parquet` + `market_intraday_spy.parquet` | `data_intermediate/decomposition/method1/rsj_daily/`, `rsj_method1_weekly.parquet`, `rolling_betas_method1.parquet` |

Method 1 decomposes at the intraday return level before RSJ construction, using a 4-week rolling window.

**RSJ Method 2 (weekly-level decomposition):**

| Script | Input | Output |
|--------|-------|--------|
| `01_rsj_method2_decomposition.py` | `rsj_weekly.parquet` + `spy_weekly_rsj.parquet` | `data_intermediate/decomposition/rsj_method2_spy.parquet` |

Method 2 runs a rolling 52-week regression of stock RSJ on market RSJ to extract systematic/idiosyncratic components.

**RES Decomposition:**

| Script | Input | Output |
|--------|-------|--------|
| `01_res_decomposition_log_return_split.py` | `converted_parquet` + `market_intraday_spy.parquet` | `data_intermediate/decomposition/log_returns_split_spy_daily/` (one file per day) |
| `02_res_decomposition_final.py` | `log_returns_split_spy_daily/` | `data_intermediate/weekly_data_res_split/weekly_res_sys_idio.parquet` + `data_intermediate/res_weekly/res_weekly.parquet` |

`01_res_decomposition_log_return_split.py` fits a rolling 4-week OLS of stock log returns on SPY log returns to decompose each day's 5-minute returns into systematic and idiosyncratic components. Requires `n_obs >= 80` transactions per day.

`02_res_decomposition_final.py` aggregates the daily arrays into weekly RES. Saves two outputs:
- `weekly_res_sys_idio.parquet`: full split with systematic, idiosyncratic, and total RES columns.
- `res_weekly/res_weekly.parquet`: total weekly RES only, in RSJ format (`permno`, `week`, `res_weekly`, `n_days`, `n_obs_total`).

### Stage 04: Controls
Folder: `Scripts/04_controls/`

| Script | Input | Output |
|--------|-------|--------|
| `01_download_wrds_input.py` | WRDS API | `data_raw/wrds/` |
| `02_build_weekly_controls.py` | `rsj_weekly.parquet` + WRDS files | `data_intermediate/controls/weekly_controls.parquet` |
| `03_make_controls_panel.py` | `weekly_controls.parquet` | `data_intermediate/controls/checks/` |

Controls constructed: ME (log market equity), BM (book-to-market), MOM (momentum t-252 to t-21), REV (1-week reversal), IVOL (idiosyncratic volatility from FF3 residuals), ILLIQ (Amihud illiquidity).

## 3) Key Output Files

| File | Description |
|------|-------------|
| `data_intermediate/rsj_weekly/rsj_weekly.parquet` | Weekly RSJ per stock |
| `data_intermediate/res_weekly/res_weekly.parquet` | Weekly RES (total) per stock |
| `data_intermediate/weekly_data_res_split/weekly_res_sys_idio.parquet` | Weekly RES split into systematic and idiosyncratic |
| `data_intermediate/decomposition/method1/rsj_method1_weekly.parquet` | RSJ Method 1 systematic/idiosyncratic split |
| `data_intermediate/decomposition/rsj_method2_spy.parquet` | RSJ Method 2 systematic/idiosyncratic split |
| `data_intermediate/weekly_stock_returns/weekly_stock_returns.parquet` | Weekly returns and one-week-ahead return target |
| `data_intermediate/controls/weekly_controls.parquet` | Weekly control variables |

## 4) Full Run Order

Run from the repository root in this order:

```
# Stage 01: convert raw data and compute daily RSJ
Rscript Scripts/01_data_preparation/00_convert_rds_to_parquet_etf.R
Rscript Scripts/01_data_preparation/01_convert_rds_to_parquet.R
python Scripts/01_data_preparation/02_compute_rsj_all_files.py

# Stage 02: weekly aggregation
python Scripts/02_variable_construction/01_aggregate_weekly_rsj.py
python Scripts/02_variable_construction/03_compute_weekly_returns.py
python Scripts/02_variable_construction/04_build_weekly_market_rsj.py

# Stage 03: decomposition
python Scripts/03_decomposition/01_rsj_method1_build_market_intraday_from_etf.py
python Scripts/03_decomposition/02_rsj_method1_decomposition.py
python Scripts/03_decomposition/01_rsj_method2_decomposition.py
python Scripts/03_decomposition/01_res_decomposition_log_return_split.py
python Scripts/03_decomposition/02_res_decomposition_final.py

# Stage 04: controls
python Scripts/04_controls/01_download_wrds_input.py
python Scripts/04_controls/02_build_weekly_controls.py
python Scripts/04_controls/03_make_controls_panel.py
```

If only re-running from the weekly aggregation step onward (e.g. after a parameter change), start from Stage 02.

## 5) Important Notes

- **n_obs filter**: all scripts filter stock-days with `n_obs >= 80` intraday transactions. This ensures RSJ and RES are computed from sufficiently active trading days.
- **Week definition**: all weekly variables use `W-TUE` (week ending Tuesday). Do not mix with old Friday-ending files.
- **Log vs simple returns**: RES is computed from log returns (converted from simple returns in the decomposition pipeline). RSJ is computed from simple returns directly.
- **RES total**: `res_total_p025` in `weekly_res_sys_idio.parquet` equals the non-decomposed weekly RES (systematic + idiosyncratic log returns sum back to the total by construction). This is also saved separately as `res_weekly.parquet`.

