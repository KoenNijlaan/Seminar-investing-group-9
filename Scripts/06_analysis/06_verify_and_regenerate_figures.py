"""
Verify and regenerate figures.

Checks whether each of the 6 required figures exists and is fresh relative to the
weekly panel. Regenerates stale or missing figures directly from saved intermediate
data where possible; otherwise prints which analysis script to re-run.

Figures checked:
  1  fig_portfolio_spreads_ew_bps       ← from 01_portfolio_sorts.py
  2  fig_fm_coefficients_selected       ← from 02_fama_macbeth.py
  3  fig_cumulative_long_short_performance ← generated here from spread_returns.parquet
    4  fig_time_series_fm_betas_b3        ← generated here from fm_coefs_B3.parquet
  5  fig_predictor_correlation_heatmap  ← generated here from weekly panel
  6  fig_weekly_cross_section_coverage  ← generated here from weekly panel
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT       = Path(__file__).resolve().parents[2]
PANEL_FILE = ROOT / "data_final" / "panel" / "weekly_panel.parquet"
INTER_DIR  = ROOT / "data_final" / "results" / "intermediate"
FIG_DIR    = ROOT / "data_final" / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NBER_PERIODS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

# Figures that require re-running the parent analysis script (not regenerated here)
PARENT_SCRIPTS = {
    "fig_portfolio_spreads_ew_bps": "01_portfolio_sorts.py",
    "fig_fm_coefficients_selected": "02_fama_macbeth.py",
}


def _add_nber(ax):
    for s, e in NBER_PERIODS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.15, color="gray", lw=0)


def _style_ax(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")


def _save(fig, stem):
    for ext in ("pdf", "png"):
        fpath = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {stem}.pdf/.png")


def is_stale(fig_path, panel_mtime):
    """Figure is stale if it doesn't exist or is older than the panel."""
    if not fig_path.exists():
        return True
    return fig_path.stat().st_mtime < panel_mtime


# ============================================================
# Figure 3: Cumulative long-short performance
# ============================================================
def make_cumulative_figure():
    path = INTER_DIR / "spread_returns.parquet"
    if not path.exists():
        print("  SKIP fig_cumulative_long_short_performance: "
              "spread_returns.parquet missing — run 01_portfolio_sorts.py first.")
        return

    df = pd.read_parquet(path)
    df["week"] = pd.to_datetime(df["week"]).dt.normalize()
    df = df.sort_values("week").set_index("week")

    signals = {
        "rsj_weekly":      ("RSJ total",             "#1f77b4"),
        "rsj_idio":        ("RSJ idio (M2)",          "#aec7e8"),
        "res_weekly":      ("RES total",              "#ff7f0e"),
        "res_idio_p025":   ("RES idio",               "#ffbb78"),
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    _add_nber(ax)

    for col, (label, color) in signals.items():
        if col not in df.columns:
            continue
        ts = df[col].dropna()
        cumret = (1 + ts).cumprod() - 1
        ax.plot(cumret.index, cumret.values * 100, lw=1.4, label=label, color=color)

    ax.axhline(0, color="black", lw=0.6, linestyle="--")
    ax.set_ylabel("Cumulative Return (%)", fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_title("Cumulative Long--Short Performance", fontsize=10, pad=8)
    ax.legend(fontsize=8, loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, "fig_cumulative_long_short_performance")


# ============================================================
# Figure 4: Rolling FM betas for RSJ and RES (spec 5)
# ============================================================
def make_rolling_betas_figure():
    path = INTER_DIR / "fm_coefs_B3.parquet"
    if not path.exists():
        print("  SKIP fig_time_series_fm_betas_b3: "
              "fm_coefs_B3.parquet missing — run 02_fama_macbeth.py first.")
        return

    df = pd.read_parquet(path)
    df["week"] = pd.to_datetime(df["week"]).dt.normalize()
    df = df.sort_values("week").set_index("week")

    roll_rsj = df["rsj_weekly"].rolling(52, min_periods=26).mean() * 10_000
    roll_res = df["res_weekly"].rolling(52, min_periods=26).mean() * 10_000

    fig, ax = plt.subplots(figsize=(9, 4))
    _add_nber(ax)
    ax.plot(roll_rsj.index, roll_rsj.values, color="#1f77b4", lw=1.5, label="RSJ (agg.)")
    ax.plot(roll_res.index, roll_res.values, color="#ff7f0e", lw=1.5, label="RES (agg.)")
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_ylabel("FM slope (bps/week, 52-week rolling mean)", fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_title("Rolling FM Slopes: RSJ and RES (Specification 5)", fontsize=10, pad=8)
    ax.legend(fontsize=9)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, "fig_time_series_fm_betas_b3")


# ============================================================
# Figure 5: Predictor correlation heatmap
# ============================================================
def make_heatmap_figure(panel):
    cols = [
        "rsj_weekly","rsj_sys_weekly","rsj_idio_weekly",
        "rsj_sys","rsj_idio",
        "res_weekly","res_sys_p025","res_idio_p025",
        "me","bm","mom","rev","ivol","illiq","rvol","rsk","rkt",
    ]
    available = [c for c in cols if c in panel.columns]
    if not available:
        print("  SKIP fig_predictor_correlation_heatmap: no predictor columns found.")
        return

    labels = {
        "rsj_weekly":      "RSJ", "rsj_sys_weekly": "RSJ$_{sys}$ M1",
        "rsj_idio_weekly": "RSJ$_{idio}$ M1", "rsj_sys": "RSJ$_{sys}$ M2",
        "rsj_idio":        "RSJ$_{idio}$ M2", "res_weekly": "RES",
        "res_sys_p025":    "RES$_{sys}$", "res_idio_p025": "RES$_{idio}$",
        "me": "log(ME)", "bm": "BM", "mom": "MOM",
        "rev": "REV", "ivol": "IVOL", "illiq": "ILLIQ",
        "rvol": "RVOL", "rsk": "RSK", "rkt": "RKT",
    }

    sub = panel[available].dropna(how="all")
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(max(8, len(available) * 0.6),
                                    max(7, len(available) * 0.6)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = [labels.get(c, c) for c in available]
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)

    for i in range(len(available)):
        for j in range(len(available)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(v) > 0.6 else "black")

    ax.set_title("Predictor Correlation Matrix", fontsize=10, pad=8)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, "fig_predictor_correlation_heatmap")


# ============================================================
# Figure 6: Weekly cross-section coverage
# ============================================================
def make_coverage_figure(panel):
    weekly_n = panel.groupby("week").size().reset_index(name="n_stocks")
    weekly_n["week"] = pd.to_datetime(weekly_n["week"])
    weekly_n = weekly_n.sort_values("week")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    _add_nber(ax)
    ax.fill_between(weekly_n["week"], weekly_n["n_stocks"],
                    color="#4a90d9", alpha=0.6)
    ax.plot(weekly_n["week"], weekly_n["n_stocks"],
            color="#4a90d9", lw=0.8)
    ax.set_ylabel("Number of stocks", fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_title("Weekly Cross-Sectional Coverage", fontsize=10, pad=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, "fig_weekly_cross_section_coverage")


# ============================================================
# Main
# ============================================================
def main():
    print("=== Verify and Regenerate Figures ===\n")

    panel_mtime = PANEL_FILE.stat().st_mtime

    FIGURES = [
        "fig_portfolio_spreads_ew_bps",
        "fig_fm_coefficients_selected",
        "fig_cumulative_long_short_performance",
        "fig_time_series_fm_betas_b3",
        "fig_predictor_correlation_heatmap",
        "fig_weekly_cross_section_coverage",
    ]

    print("Status check:")
    stale = []
    for stem in FIGURES:
        png = FIG_DIR / f"{stem}.png"
        pdf = FIG_DIR / f"{stem}.pdf"
        if not png.exists() and not pdf.exists():
            status = "MISSING"
            stale.append(stem)
        elif is_stale(png if png.exists() else pdf, panel_mtime):
            status = "STALE"
            stale.append(stem)
        else:
            status = "OK"
        print(f"  {status:<8} {stem}")

    if not stale:
        print("\nAll figures are up-to-date. Nothing to regenerate.")
        return

    print(f"\nRegenerating {len(stale)} figure(s)...\n")

    # Load panel once (needed for heatmap and coverage)
    panel = None

    for stem in stale:
        print(f"[{stem}]")

        if stem in PARENT_SCRIPTS:
            print(f"  → Re-run Scripts/06_analysis/{PARENT_SCRIPTS[stem]}")
            continue

        if stem == "fig_cumulative_long_short_performance":
            make_cumulative_figure()

        elif stem == "fig_time_series_fm_betas_b3":
            make_rolling_betas_figure()

        elif stem in ("fig_predictor_correlation_heatmap",
                      "fig_weekly_cross_section_coverage"):
            if panel is None:
                print("  Loading panel...")
                panel = pd.read_parquet(PANEL_FILE)
                panel["week"] = pd.to_datetime(panel["week"]).dt.normalize()

            if stem == "fig_predictor_correlation_heatmap":
                make_heatmap_figure(panel)
            else:
                make_coverage_figure(panel)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
