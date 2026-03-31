# Seminar-investing-group-9

This repository builds and tests high-frequency downside-risk measures for US equities.

Core objects:
- RSJ (realized semivariance-of-jumps asymmetry)
- RES (realized expected shortfall, p = 2.5%)
- Systematic and idiosyncratic decompositions of RSJ and RES
- Weekly return predictability tests (portfolio sorts and Fama-MacBeth)

## 1) End-to-End Pipeline

```
data_raw/intraday_stocks + data_raw/intraday_etfs
      -> (R conversion scripts)
data_intermediate/converted_parquet + converted_parquet_etf
      -> (daily RSJ + weekly market/returns + decompositions)
data_intermediate/rsj_weekly + res_weekly + controls + decomposition outputs
      -> (panel construction)
data_final/panel/weekly_panel.parquet
      -> (analysis)
data_final/portfolio_sorts/*
data_final/fama_macbeth/*
```

## 2) Script Inventory by Stage

### Stage 01: Data Preparation (`Scripts/01_data_preparation`)

| Script | Main input | Main output |
|---|---|---|
| `00_convert_rds_to_parquet_etf.R` | `data_raw/intraday_etfs/*.rds` | `data_intermediate/converted_parquet_etf/*.parquet` |
| `01_convert_rds_to_parquet.R` | `data_raw/intraday_stocks/*.rds` | `data_intermediate/converted_parquet/*.parquet` |
| `02_compute_rsj_all_files.py` | `data_intermediate/converted_parquet/*.parquet` | `data_intermediate/rsj_daily/*.parquet` |

Notes:
- Daily RSJ keeps only rows with `n_obs >= 80`.

### Stage 02: Weekly Variable Construction (`Scripts/02_variable_construction`)

| Script | Main input | Main output |
|---|---|---|
| `01_aggregate_weekly_rsj.py` | `data_intermediate/rsj_daily/*.parquet` | `data_intermediate/rsj_weekly/rsj_weekly.parquet` |
| `03_compute_weekly_returns.py` | `data_intermediate/converted_parquet/*.parquet` | `data_intermediate/weekly_stock_returns/weekly_stock_returns.parquet` |
| `04_build_weekly_market_rsj.py` | `data_intermediate/converted_parquet_etf/*.parquet` (SPY only) | `data_intermediate/market_weekly_rsj/spy_weekly_rsj.parquet` |

Notes:
- These scripts use week ending Tuesday (`W-TUE`).
- Weekly filters require at least 3 valid days.

### Stage 03: Decomposition (`Scripts/03_decomposition`)

#### RSJ Method 1 (intraday decomposition)

| Script | Main input | Main output |
|---|---|---|
| `01_rsj_method1_build_market_intraday_from_etf.py` | `data_intermediate/converted_parquet_etf/*.parquet` | `data_intermediate/market_returns/market_intraday_spy.parquet` |
| `02_rsj_method1_decomposition.py` | stock intraday + SPY intraday | `data_intermediate/decomposition/method1/rsj_daily/*.parquet` + `data_intermediate/decomposition/method1/rsj_method1_weekly.parquet` + `data_intermediate/decomposition/method1/rolling_betas_method1.parquet` |

#### RSJ Method 2 (weekly rolling regression)

| Script | Main input | Main output |
|---|---|---|
| `01_rsj_method2_decomposition.py` | `data_intermediate/rsj_weekly/rsj_weekly.parquet` + `data_intermediate/market_weekly_rsj/spy_weekly_rsj.parquet` | `data_intermediate/decomposition/method2/rsj_method2_spy.parquet` |

#### RES decomposition and aggregation

| Script | Main input | Main output |
|---|---|---|
| `01_res_decomposition_log_return_split.py` | stock intraday + SPY intraday | `data_intermediate/decomposition/log_returns_split_spy_daily/*.parquet` |
| `02_res_decomposition_final.py` | `data_intermediate/decomposition/log_returns_split_spy_daily/*.parquet` | `data_intermediate/weekly_data_res_split/weekly_res_sys_idio.parquet` + `data_intermediate/res_weekly/res_weekly.parquet` |

### Stage 04: Controls (`Scripts/04_controls`)

| Script | Main input | Main output |
|---|---|---|
| `01_download_wrds_input.py` | WRDS connection + RSJ sample | `data_raw/wrds/*.parquet` |
| `02_build_weekly_controls.py` | RSJ sample + WRDS files | `data_intermediate/controls/weekly_controls.parquet` |
| `03_make_controls_panel.py` | weekly controls | `data_intermediate/controls/checks/*` |

Controls included in weekly controls:
- `me` (log market equity)
- `bm` (book-to-market)
- `mom` (momentum, t-252 to t-21)
- `rev` (1-week reversal)
- `ivol` (FF3 residual volatility, 21-day window)
- `illiq` (Amihud illiquidity, log)

### Stage 05: Panel Construction (`Scripts/05_dataset_construction`)

| Script | Main input | Main output |
|---|---|---|
| `01_build_data_panel.py` | weekly RSJ/RES/decompositions/controls/returns | `data_final/panel/weekly_panel.parquet` |

Applied filters in panel build:
- `shrcd` in `{10, 11}`
- `5 <= abs(prc) <= 1000`

### Stage 06: Analysis (`Scripts/06_analysis`)

| Script | Main input | Main output |
|---|---|---|
| `01_portfolio_sorts.py` | `data_final/panel/weekly_panel.parquet` + FF factors | `data_final/portfolio_sorts/sort_results_all.parquet` + per-variable CSV files |
| `02_fama_macbeth.py` | `data_final/panel/weekly_panel.parquet` | `data_final/fama_macbeth/fm_results.parquet` + `data_final/fama_macbeth/fm_summary.csv` |
| `03_wald_tests.py` | `data_final/fama_macbeth/fm_results.parquet` | `data_final/fama_macbeth/fm_wald.csv` |
| `04_crisis_state_fm.py` | `data_final/panel/weekly_panel.parquet` | `data_final/crisis/fm_crisis_weekly_betas.parquet` + `data_final/crisis/fm_crisis_summary.csv` |

### Stage 07 folders

`Scripts/07_latex_tables` and `Scripts/07_output` currently exist but are empty in this branch.

## 3) Full Run Order

Run from repository root:

```bash
# Stage 01
Rscript Scripts/01_data_preparation/00_convert_rds_to_parquet_etf.R
Rscript Scripts/01_data_preparation/01_convert_rds_to_parquet.R
python Scripts/01_data_preparation/02_compute_rsj_all_files.py

# Stage 02
python Scripts/02_variable_construction/01_aggregate_weekly_rsj.py
python Scripts/02_variable_construction/03_compute_weekly_returns.py
python Scripts/02_variable_construction/04_build_weekly_market_rsj.py

# Stage 03
python Scripts/03_decomposition/01_rsj_method1_build_market_intraday_from_etf.py
python Scripts/03_decomposition/02_rsj_method1_decomposition.py
python Scripts/03_decomposition/01_rsj_method2_decomposition.py
python Scripts/03_decomposition/01_res_decomposition_log_return_split.py
python Scripts/03_decomposition/02_res_decomposition_final.py

# Stage 04
python Scripts/04_controls/01_download_wrds_input.py
python Scripts/04_controls/02_build_weekly_controls.py
python Scripts/04_controls/03_make_controls_panel.py

# Stage 05
python Scripts/05_dataset_construction/01_build_data_panel.py

# Stage 06
python Scripts/06_analysis/01_portfolio_sorts.py
python Scripts/06_analysis/02_fama_macbeth.py
python Scripts/06_analysis/03_wald_tests.py
python Scripts/06_analysis/04_crisis_state_fm.py
```

## 4) Key Data Products

| Path | Description |
|---|---|
| `data_intermediate/rsj_weekly/rsj_weekly.parquet` | Weekly total RSJ |
| `data_intermediate/market_weekly_rsj/spy_weekly_rsj.parquet` | Weekly SPY RSJ factor |
| `data_intermediate/decomposition/method1/rsj_method1_weekly.parquet` | Weekly RSJ Method 1 (`rsj_sys_weekly`, `rsj_idio_weekly`) |
| `data_intermediate/decomposition/method2/rsj_method2_spy.parquet` | Weekly RSJ Method 2 (`rsj_sys`, `rsj_idio`, `beta_rsj`, `alpha_rsj`) |
| `data_intermediate/weekly_data_res_split/weekly_res_sys_idio.parquet` | Weekly RES split (`res_total_p025`, `res_sys_p025`, `res_idio_p025`) |
| `data_intermediate/res_weekly/res_weekly.parquet` | Weekly total RES (`res_weekly`) |
| `data_intermediate/weekly_stock_returns/weekly_stock_returns.parquet` | Weekly returns (`R_i_w`) and next-week return (`R_i_w_plus_1`) |
| `data_intermediate/controls/weekly_controls.parquet` | Weekly controls |
| `data_final/panel/weekly_panel.parquet` | Final merged analysis panel |
| `data_final/portfolio_sorts/*` | Portfolio sort outputs |
| `data_final/fama_macbeth/*` | Fama-MacBeth and Wald outputs |

## 5) Calendar and Filtering Conventions

- Main weekly key throughout the final pipeline is `W-TUE` (week ending Tuesday).
- Minimum daily intraday observations is generally `n_obs >= 80` for RSJ/market/decomposition entry.
- Weekly stock-level outputs usually enforce `n_days >= 3`.
- RES split script (`01_res_decomposition_log_return_split.py`) initially labels weeks as `W-FRI`, but the final aggregator (`02_res_decomposition_final.py`) recomputes and stores week endpoints using `W-TUE`.
- `ret_crsp` from intraday parquet is in percent units; weekly return compounding converts by dividing by 100 first.

## 6) Dependencies

No single environment file is currently provided in this branch.

From script imports, you need at least:
- R: `arrow`
- Python: `pandas`, `numpy`, `pyarrow`, `statsmodels`, `scipy`, `matplotlib`, `wrds`

If you plan to run WRDS download scripts, ensure WRDS credentials are configured for your environment.

