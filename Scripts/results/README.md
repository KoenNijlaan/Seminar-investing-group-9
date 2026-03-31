# Results Generation Scripts

This folder contains scripts to build paper-ready tables and figures from existing analysis outputs.

## Inputs expected

- `data_final/panel/weekly_panel.parquet`
- `data_final/portfolio_sorts/sort_results_all.parquet`
- `data_final/fama_macbeth/fm_results.parquet`
- `data_final/fama_macbeth/fm_summary.csv`
- `data_final/fama_macbeth/fm_wald.csv`
- `data_final/crisis/fm_crisis_summary.csv`

## Outputs generated

- `data_final/results/tables/*.csv`
- `data_final/results/tables/*.tex`
- `data_final/results/figures/*.png`

## Scripts

- `01_table_summary_stats.py`
- `02_table_portfolio_sorts.py`
- `03_table_fama_macbeth.py`
- `04_table_wald.py`
- `05_table_crisis.py`
- `06_figure_portfolio_spreads.py`
- `07_figure_fm_coefficients.py`
- `08_figure_crisis_differences.py`
- `09_figure_time_series_coefficients.py`
- `10_figure_predictor_correlation_heatmap.py`
- `11_figure_cross_section_coverage.py`
- `12_figure_signal_distributions.py`

## One-command run

```bash
python Scripts/results/run_all_results.py
```
