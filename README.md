# Seminar-investing-group-9

This repository builds high-frequency downside-risk measures for US equities, decomposes them into systematic and idiosyncratic components, and prepares weekly panels for return predictability tests.

Main research objects:
- RSJ (Relative Signed Jump): asymmetry in intraday semivariance.
- RES (Realized Expected Shortfall): downside tail risk from intraday returns.
- Systematic vs idiosyncratic decompositions of RSJ and RES.
- Weekly prediction tests (portfolio sorts and Fama-MacBeth style panels).

## 1) Repository Data Flow

The project follows a staged pipeline:

1. Raw intraday data
	 - Stocks: data_raw/intraday_stocks
	 - ETFs (for market proxy, mainly SPY): data_raw/intraday_etfs
2. Daily measures from intraday returns
	 - RSJ daily: data_intermediate/rsj_daily
	 - RES daily: data_intermediate/res_daily
3. Weekly aggregation
	 - Stock weekly RSJ: data_intermediate/rsj_weekly/rsj_weekly.parquet
	 - Stock weekly RES: data_intermediate/res_weekly/weekly_res.parquet
	 - Weekly stock returns and lead return: data_intermediate/weekly_stock_returns/weekly_stock_returns.parquet
	 - Market weekly RSJ (SPY): data_intermediate/market_weekly_rsj/spy_weekly_rsj.parquet
4. Decomposition layer
	 - RSJ Method 1 outputs: data_intermediate/rsj_daily/decomposition/method1, data_intermediate/rsj_weekly/decomposition/method1
	 - RSJ Method 2 output: data_intermediate/decomposition/method2/rsj_method2_spy.parquet
	 - RES decomposition outputs: data_intermediate/decomposition/res_decomposition
5. Controls and diagnostics
	 - Weekly controls: data_intermediate/controls_variables/weekly_controls.parquet
	 - Control diagnostics: data_intermediate/controls_variables/checks
6. Final analysis panel
	 - data_intermediate/regression_panels/panel_step3.parquet

In short: raw intraday files -> daily RSJ/RES -> weekly variables -> decomposition + controls -> final regression panel.

## 2) Script Stages (What Happens and Where It Is Saved)

### Stage 01: Data Preparation
Folder: Scripts/01_data_preparation

- 00_convert_rds_to_parquet_etf.R
	- Converts ETF intraday RDS files to parquet.
	- Input: data_raw/intraday_etfs
	- Output: data_intermediate/converted_parquet_etf

- 01_convert_rds_to_parquet.R
	- Converts stock intraday RDS files to parquet.
	- Input: data_raw/intraday_stocks
	- Output: data_intermediate/converted_parquet

- 02_compute_rsj_all_files.py
	- Computes daily RSJ per stock-day from intraday returns.
	- Input: data_intermediate/converted_parquet
	- Output: data_intermediate/rsj_daily

- 03_compute_res_all_files.py
	- Computes daily RES per stock-day from intraday returns.
	- Input: data_intermediate/converted_parquet
	- Output: data_intermediate/res_daily

### Stage 02: Variable Construction
Folder: Scripts/02_variable_construction

- 01_aggregate_weekly_rsj.py
	- Aggregates daily RSJ to stock-week level.
	- Output: data_intermediate/rsj_weekly/rsj_weekly.parquet

- 02_aggregate_weekly_res.py
	- Aggregates daily RES to stock-week level.
	- Output target in current data layout: data_intermediate/res_weekly/weekly_res.parquet

- 03_compute_weekly_returns.py
	- Computes weekly compounded stock returns and next-week return.
	- Output: data_intermediate/weekly_stock_returns/weekly_stock_returns.parquet

- 04_build_weekly_market_rsj.py
	- Builds weekly market RSJ (SPY proxy).
	- Output: data_intermediate/market_weekly_rsj/spy_weekly_rsj.parquet

### Stage 03: Decomposition
Folder: Scripts/03_decomposition

- 01_rsj_method1_build_market_intraday_from_etf.py
	- Builds intraday market return series from SPY ETF files.
	- Output: data_intermediate/market_returns/market_intraday_spy.parquet

- 02_rsj_method1_full_pipeline.py
	- Method 1 RSJ decomposition (decompose at intraday return level before RSJ construction).
	- Outputs:
		- data_intermediate/rsj_daily/decomposition/method1/rsj_method1_daily.parquet
		- data_intermediate/rsj_weekly/decomposition/method1/rsj_method1_weekly.parquet
		- data_intermediate/decomposition/method1/rolling_betas_method1.parquet

- 01_rsj_method2_decomposition.py
	- Method 2 RSJ decomposition (decompose the final weekly RSJ directly via rolling weekly regression on market RSJ).
	- Output: data_intermediate/decomposition/method2/rsj_method2_spy.parquet

- 01_res_decomposition.py and 01_res_decomposition_aggregate_weekly.py
	- RES systematic/idiosyncratic decomposition from intraday level and weekly aggregation.
	- Outputs under: data_intermediate/decomposition/res_decomposition

### Stage 04: Controls
Folder: Scripts/04_controls

- 01_download_wrds_input.py
	- Downloads WRDS inputs for the RSJ sample.
	- Outputs: data_raw/wrds

- 02_build_weekly_controls.py
	- Constructs weekly controls (size, book-to-market, momentum, reversal, idiosyncratic volatility, illiquidity).
	- Current saved location in this repository data layout: data_intermediate/controls_variables/weekly_controls.parquet
	- Intermediate files: _daily_controls_base.parquet and _daily_ivol.parquet in the same controls directory.

- 03_make_controls_panel.py
	- Produces control quality diagnostics (missingness, summaries, plots).
	- Output: data_intermediate/controls_variables/checks

### Stage 05: Dataset Construction
Folder: Scripts/05_dataset_construction

- 01_dataset_regression_RSJ_RES_CONTROLS.py
	- Merges weekly RSJ, weekly RES, weekly controls, and next-week returns.
	- Output: data_intermediate/regression_panels/panel_step3.parquet

### Stage 06: Analysis
Folder: Scripts/06_analysis

- Intended for estimation scripts and regression tests.
- Currently empty in this repository snapshot.

### Stage 07: Output
Folder: Scripts/07_output

- Intended for final tables/figures for reporting.
- Currently empty in this repository snapshot.

## 3) Most Important Data Products

If you only need the key files for empirical analysis, focus on these:

- data_intermediate/rsj_weekly/rsj_weekly.parquet
	- Weekly RSJ by stock.

- data_intermediate/res_weekly/weekly_res.parquet
	- Weekly RES by stock.

- data_intermediate/market_weekly_rsj/spy_weekly_rsj.parquet
	- Market (SPY) weekly RSJ factor used in decomposition.

- data_intermediate/decomposition/method2/rsj_method2_spy.parquet
	- Method 2 RSJ split into systematic and idiosyncratic components.

- data_intermediate/rsj_weekly/decomposition/method1/rsj_method1_weekly.parquet
	- Method 1 weekly RSJ decomposition output.

- data_intermediate/decomposition/res_decomposition/res_weekly_split/weekly_res_split.parquet
	- Weekly RES systematic/idiosyncratic split.

- data_intermediate/controls_variables/weekly_controls.parquet
	- Weekly control variables.

- data_intermediate/weekly_stock_returns/weekly_stock_returns.parquet
	- Weekly returns and one-week-ahead return target.

- data_intermediate/regression_panels/panel_step3.parquet
	- Final merged panel for predictive regressions.

## 4) Practical Run Order

A practical full run order is:

1. Scripts/01_data_preparation
2. Scripts/02_variable_construction
3. Scripts/03_decomposition
4. Scripts/04_controls
5. Scripts/05_dataset_construction
6. Scripts/06_analysis (when populated)
7. Scripts/07_output (when populated)

## 5) Important Notes on Paths and Reproducibility

Most scripts use hardcoded input/output paths, so consistency matters.

In this repository snapshot, some path names differ between script internals/comments and currently saved folders. The most relevant examples are:

- Weekly RES naming/location differences (for example weekly_data vs res_weekly references).
- Controls folder naming differences (for example controls vs controls_variables references).
- Some older script headers/comments may not match current destination folders.

If you rerun the full pipeline from scratch, verify path definitions inside each script before running large batches. This avoids writing outputs to unintended folders and prevents downstream merge failures.