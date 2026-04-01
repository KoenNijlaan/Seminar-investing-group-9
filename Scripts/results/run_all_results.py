from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    "01_table_summary_stats.py",
    "02_table_portfolio_sorts.py",
    "03_table_fama_macbeth.py",
    "04_table_wald.py",
    "05_table_crisis.py",
    "06_figure_portfolio_spreads.py",
    "07_figure_fm_coefficients.py",
    "08_figure_crisis_differences.py",
    "09_figure_time_series_coefficients.py",
    "10_figure_predictor_correlation_heatmap.py",
    "11_figure_cross_section_coverage.py",
    "12_figure_signal_distributions.py",
    "13_figure_cumulative_long_short.py",
]


def run_script(script_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "main"):
        raise AttributeError(f"Script has no main(): {script_path}")
    mod.main()


def main() -> None:
    print("=== Build All Results Tables/Figures ===")
    for s in SCRIPTS:
        p = ROOT / s
        print(f"\nRunning: {s}")
        run_script(p)
    print("\nDone. All result artifacts generated.")


if __name__ == "__main__":
    main()
