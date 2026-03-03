"""
L2 Conditional Ecosystem Lift vs Entry Mass Threshold.

For each metric, plots mean delta_eco (with 95% CI bands) as a function
of the minimum entry-mass threshold.  Each line is a strategy.  Strategies
drop off the chart when their sample count falls below MIN_N.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
PKL_PATH = Path("data/analysis/iterative_analysis_results.pkl")
FIG_DIR = Path("data/analysis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

AGENTS = ["ppo", "psro", "mappo", "openai_5.2_none", "nfsp"]
LABELS = {"openai_5.2_none": "openai", "ppo": "ppo", "psro": "psro",
          "mappo": "mappo", "nfsp": "nfsp"}
COLORS = {"ppo": "#1f77b4", "psro": "#ff7f0e", "mappo": "#2ca02c",
          "openai_5.2_none": "#d62728", "nfsp": "#9467bd"
          }

THRESHOLDS = [i / 100 for i in range(1, 51)]  # 1% to 50% in 1% steps
MIN_N = 1 

METRICS = [
    ("uw", "delta_eco", "Utilitarian Welfare (UW)",  "raw utility"),
    ("nw",  "delta_eco",  "Nash Welfare (NW)",  "raw utility"),
    ("nw_plus", "delta_eco",     "NW+ (surplus above BATNA)", "raw utility"),
    ("ef1",     "ef1_lift",      "EF1 Frequency",  "pp"),
    ("ef1_plus", "ef1_plus_lift", "EF1+ Frequency", "pp"),
]


def load_data():
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


def compute_conditional(raw, agent, metric_key, metric_source, threshold,
                        trim=True):
    """Return (mean, lo, hi, n) for delta conditional on entry_mass > threshold.

    If trim=True, discard the upper and lower quartiles of the delta values
    and compute stats from the middle 50% (IQR).
    """
    entry_masses = [r["l2"][agent]["entry_mass"] for r in raw]
    mask = [em > threshold for em in entry_masses]
    n = sum(mask)
    if n < MIN_N:
        return None, None, None, n

    if metric_source == "delta_eco":
        vals = [r["l2"][agent]["delta_eco"][metric_key]
                for r, m in zip(raw, mask) if m]
    else:
        # ef1_lift or ef1_plus_lift — stored as fractions, convert to pp
        vals = [r["l2"][agent][metric_source] * 100
                for r, m in zip(raw, mask) if m]

    a = np.sort(np.array(vals))

    if trim and len(a) >= 4:
        # Keep only the middle 50%
        lo_cut = int(len(a) * 0.25)
        hi_cut = int(len(a) * 0.75)
        a = a[lo_cut:hi_cut]

    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(np.mean(a)), float(lo), float(hi), n


def make_plots(raw):
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.5), sharey=False)
    fig.suptitle(
        "L2 Conditional Ecosystem Lift vs Entry-Mass Threshold",
        fontsize=14, fontweight="bold", y=1.02,
    )

    x_pct = [t * 100 for t in THRESHOLDS]  # for display as %

    for ax, (metric_key, metric_source, title, unit) in zip(axes, METRICS):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Min Entry Mass (%)", fontsize=9)
        ax.set_ylabel(f"Mean Δ ({unit})", fontsize=9)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)

        # Collect all means to set y-limits based on means, not CIs
        all_means = []

        for agent in AGENTS:
            means, los, his, xs, ns = [], [], [], [], []
            for thresh in THRESHOLDS:
                mean, lo, hi, n = compute_conditional(
                    raw, agent, metric_key, metric_source, thresh
                )
                if mean is not None:
                    means.append(mean)
                    los.append(lo)
                    his.append(hi)
                    xs.append(thresh * 100)
                    ns.append(n)

            if not xs:
                continue

            all_means.extend(means)
            label = LABELS[agent]
            color = COLORS[agent]

            # Plot mean line (no markers — 50 points is too dense)
            ax.plot(xs, means, "-", color=color, label=label,
                    linewidth=2.0, zorder=3)

            # Light CI shading
            ax.fill_between(xs, los, his, color=color, alpha=0.08, zorder=1)

            # Annotate N at last point
            ax.annotate(
                f"n={ns[-1]}",
                xy=(xs[-1], means[-1]),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=7,
                color=color,
                fontweight="bold",
                alpha=0.9,
            )

        # Set y-limits to focus on means with some padding
        if all_means:
            ymin = min(all_means)
            ymax = max(all_means)
            pad = max((ymax - ymin) * 0.3, 2)
            ax.set_ylim(ymin - pad, ymax + pad)

        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_xticklabels(["1%", "10%", "20%", "30%", "40%", "50%"],
                           fontsize=8)
        ax.set_xlim(0, 52)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.2)

    # Single legend at the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=10,
               bbox_to_anchor=(1.02, 0.5), framealpha=0.9)

    fig.tight_layout()
    out = FIG_DIR / "l2_conditional_threshold_sweep.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    results = load_data()
    raw = results["raw"]
    print(f"Loaded {len(raw)} bootstrap samples")
    make_plots(raw)
