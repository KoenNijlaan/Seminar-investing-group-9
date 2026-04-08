"""
Master script: run all analysis scripts in the correct execution order.

Execution order (as specified):
  1. Scripts/06_analysis/01_portfolio_sorts.py
  2. Scripts/06_analysis/02_fama_macbeth.py       ← saves intermediate coefs
  3. Scripts/06_analysis/03_wald_tests.py          ← needs intermediate from step 2
  4. Scripts/06_analysis/04_crisis_state.py        ← needs intermediate from step 2
  5. Scripts/06_analysis/05_illiquidity.py
  6. Scripts/06_analysis/06_verify_and_regenerate_figures.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "Scripts" / "06_analysis"

SCRIPTS = [
    ANALYSIS_DIR / "01_portfolio_sorts.py",
    ANALYSIS_DIR / "02_fama_macbeth.py",
    ANALYSIS_DIR / "03_wald_tests.py",
    ANALYSIS_DIR / "04_crisis_state.py",
    ANALYSIS_DIR / "05_illiquidity.py",
    ANALYSIS_DIR / "06_verify_and_regenerate_figures.py",
]


def run_script(script_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "main"):
        raise AttributeError(f"No main() in: {script_path}")
    mod.main()


def main() -> None:
    print("=== Run All Analysis Scripts ===\n")
    for p in SCRIPTS:
        if not p.exists():
            print(f"\n[SKIP] Not found: {p.name}")
            continue
        print(f"\n{'=' * 60}")
        print(f"Running: {p.name}")
        print('=' * 60)
        run_script(p)

    print("\n=== All done. Outputs in data_final/results/ ===")


if __name__ == "__main__":
    main()
