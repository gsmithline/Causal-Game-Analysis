"""Visualizations for equilibrium-consistent credit analysis.

Generates:
1. First-order credit bar charts (one per metric)
2. Second-order influence diverging stacked bar charts (one per metric)
3. Greedy add/remove path line charts (one each per metric)

Usage:
    python visuals/eq_credit_visuals.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from visuals.visualize_analysis import (
    DEFAULT_DPI,
    DISPLAY_NAMES,
    L3_NORM,
    STRATEGY_COLORS,
    STRATEGY_ORDER,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "analysis"
INPUT_PKL = DATA_DIR / "eq_credit_results.pkl"
FIG_DIR = DATA_DIR / "figures"

METRIC_LABELS = {
    "uw": "Utilitarian Welfare",
    "nw": "Nash Welfare",
    "nw_plus": "NW+",
    "ef1": "EF1",
    "ef1_plus": "EF1+",
}


def _dn(strategy: str) -> str:
    return DISPLAY_NAMES.get(strategy, strategy)


def _norm_scale(metric):
    """Return (divisor, scale_factor, unit_label) for a metric."""
    div = L3_NORM[metric]
    if metric in ("ef1", "ef1_plus"):
        return div, 100.0, "pp"
    return div, 100.0, "% of max"


# ---------------------------------------------------------------------------
# 1. First-order credit bar charts
# ---------------------------------------------------------------------------


def plot_credits(results):
    """One bar chart per metric showing C_i(S) for all strategies."""
    credits = results["credits"]
    policies = results["policies"]

    for metric in results["metrics"]:
        div, scale, unit = _norm_scale(metric)

        means = [credits[p][metric]["mean"] / div * scale for p in policies]
        ci_lo = [credits[p][metric]["ci_lower"] / div * scale for p in policies]
        ci_hi = [credits[p][metric]["ci_upper"] / div * scale for p in policies]
        errors_lo = [max(0, m - lo) for m, lo in zip(means, ci_lo)]
        errors_hi = [max(0, hi - m) for m, hi in zip(means, ci_hi)]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(policies))
        colors = [STRATEGY_COLORS[p] for p in policies]
        ax.bar(
            x, means, yerr=[errors_lo, errors_hi],
            color=colors, capsize=3, edgecolor="black", linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([_dn(p) for p in policies], rotation=45, ha="right")
        ax.set_ylabel(f"Credit ({unit})")
        ax.set_title(f"First-Order Credit: {METRIC_LABELS[metric]}")
        ax.axhline(0, color="black", linewidth=0.5)

        fig.tight_layout()
        fig.savefig(FIG_DIR / f"eq_credit_{metric}.png",
                    dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved eq_credit_{metric}.png")


# ---------------------------------------------------------------------------
# 2. Second-order influence diverging stacked bars
# ---------------------------------------------------------------------------


def plot_influences(results):
    """Diverging stacked bar chart: for each strategy i, show Infl_{j→i}."""
    influences = results["influences"]
    policies = results["policies"]

    for metric in results["metrics"]:
        div, scale, unit = _norm_scale(metric)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(policies))

        for i_idx, i in enumerate(policies):
            pos_bottom = 0.0
            neg_bottom = 0.0
            for j in policies:
                if j == i:
                    continue
                val = influences[(i, j)][metric]["mean"] / div * scale
                color = STRATEGY_COLORS[j]
                if val >= 0:
                    ax.bar(i_idx, val, bottom=pos_bottom, color=color,
                           edgecolor="white", linewidth=0.3, width=0.7)
                    pos_bottom += val
                else:
                    ax.bar(i_idx, val, bottom=neg_bottom, color=color,
                           edgecolor="white", linewidth=0.3, width=0.7)
                    neg_bottom += val

        ax.set_xticks(x)
        ax.set_xticklabels([_dn(p) for p in policies], rotation=45, ha="right")
        ax.set_ylabel(f"Influence ({unit})")
        ax.set_title(f"Second-Order Influence on Credit: {METRIC_LABELS[metric]}")
        ax.axhline(0, color="black", linewidth=0.8)

        # Legend: one entry per strategy
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=STRATEGY_COLORS[p])
            for p in policies
        ]
        ax.legend(handles, [_dn(p) for p in policies],
                  loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

        fig.tight_layout()
        fig.savefig(FIG_DIR / f"eq_influence_{metric}.png",
                    dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved eq_influence_{metric}.png")


# ---------------------------------------------------------------------------
# 3. Greedy path charts
# ---------------------------------------------------------------------------


def plot_greedy_paths(results):
    """Line charts for greedy-add and greedy-remove paths."""
    for direction in ("add", "remove"):
        key = f"greedy_{direction}"
        path_data = results[key]

        for metric in results["metrics"]:
            div, scale, unit = _norm_scale(metric)
            steps = path_data[metric]

            xs = list(range(len(steps)))
            means = [s["value"]["mean"] / div * scale for s in steps]
            ci_lo = [s["value"]["ci_lower"] / div * scale for s in steps]
            ci_hi = [s["value"]["ci_upper"] / div * scale for s in steps]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(xs, means, "o-", color="#333333", linewidth=2, markersize=6)
            ax.fill_between(xs, ci_lo, ci_hi, alpha=0.2, color="#888888")

            # Label each step with the strategy added/removed
            for idx, step in enumerate(steps):
                strat = step["most_common_strategy"]
                if strat is None:
                    label = "S (all)"
                else:
                    label = _dn(strat)
                ax.annotate(
                    label, (idx, means[idx]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=7, ha="center", rotation=30,
                )

            if direction == "add":
                ax.set_xlabel("Strategies added")
                ax.set_title(
                    f"Greedy Add Path: {METRIC_LABELS[metric]}")
            else:
                ax.set_xlabel("Strategies removed")
                ax.set_title(
                    f"Greedy Remove Path: {METRIC_LABELS[metric]}")

            ax.set_ylabel(f"Coalition Value ({unit})")
            ax.set_xticks(xs)

            fig.tight_layout()
            fig.savefig(
                FIG_DIR / f"eq_greedy_{direction}_{metric}.png",
                dpi=DEFAULT_DPI, bbox_inches="tight",
            )
            plt.close(fig)
            print(f"  Saved eq_greedy_{direction}_{metric}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Loading {INPUT_PKL} ...")
    with open(INPUT_PKL, "rb") as f:
        results = pickle.load(f)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Plotting first-order credits ...")
    plot_credits(results)

    print("Plotting second-order influences ...")
    plot_influences(results)

    print("Plotting greedy paths ...")
    plot_greedy_paths(results)

    print("Done.")


if __name__ == "__main__":
    main()
