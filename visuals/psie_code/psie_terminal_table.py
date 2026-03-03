"""PSIE terminal summary table.

Shows all terminal support sets appearing in >=30% of bootstrap samples,
with the most frequent seed that reaches each terminal.

Usage:
    python visuals/psie_terminal_table.py
    python visuals/psie_terminal_table.py --min-freq 0.20
"""

import argparse
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from visualize_analysis import DISPLAY_NAMES, STRATEGY_ORDER, DEFAULT_DPI


def _dn(name):
    return DISPLAY_NAMES.get(name, name)


def _support_label(names):
    """Format a support set using display names, ordered by STRATEGY_ORDER."""
    ordered = sorted(names, key=lambda n: STRATEGY_ORDER.index(n)
                     if n in STRATEGY_ORDER else 999)
    return ", ".join(_dn(n) for n in ordered)


def compute_terminal_stats(psie_data):
    """Compute per-terminal statistics from bootstrap analyses.

    Returns list of dicts sorted by sample frequency (descending).
    """
    names = psie_data["config"]["strategy_names"]
    analyses = psie_data["bootstrap"]["analyses"]
    n_bootstrap = len(analyses)
    n_seeds = len(names)

    terminal_by_seed = {}
    terminal_sample_counts = Counter()

    for b_idx, a in enumerate(analyses):
        eo = a["entry_order"]
        seen = set()
        for seed_idx in range(n_seeds):
            terminal = frozenset(
                strat for strat in names
                if eo[strat][seed_idx] is not None
            )
            if terminal not in terminal_by_seed:
                terminal_by_seed[terminal] = Counter()
            terminal_by_seed[terminal][names[seed_idx]] += 1
            seen.add(terminal)
        for t in seen:
            terminal_sample_counts[t] += 1

    rows = []
    for t, count in terminal_sample_counts.most_common():
        sample_freq = count / n_bootstrap
        top_seed, top_count = terminal_by_seed[t].most_common(1)[0]
        seed_freq = top_count / n_bootstrap
        rows.append({
            "support": t,
            "label": _support_label(t),
            "size": len(t),
            "sample_freq": sample_freq,
            "top_seed": top_seed,
            "seed_freq": seed_freq,
        })

    return rows, n_bootstrap


def draw_table(rows, n_bootstrap, save_path, min_freq=0.30):
    """Draw a publication-quality table of frequent terminals."""
    filtered = [r for r in rows if r["sample_freq"] >= min_freq]
    if not filtered:
        print(f"No terminals above {min_freq:.0%} threshold.")
        return

    col_labels = ["Terminal Support", "Size", "Frequency", "Top Seed", "Seed Freq"]
    cell_text = []
    cell_colors = []

    for r in filtered:
        # Color by attractor type
        has_psro = "psro" in r["support"]
        has_ppo = "ppo" in r["support"]
        if has_psro and not has_ppo:
            bg = "#FDEBD0"
        elif has_ppo and not has_psro:
            bg = "#D5F5E3"
        elif has_psro and has_ppo:
            bg = "#E8DAEF"
        else:
            bg = "#D6EAF8"

        cell_text.append([
            r["label"],
            str(r["size"]),
            f"{r['sample_freq']:.1%}",
            _dn(r["top_seed"]),
            f"{r['seed_freq']:.1%}",
        ])
        cell_colors.append([bg] * 5)

    fig_height = max(2.5, 0.5 * len(filtered) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", color="white")
        cell.set_facecolor("#2C3E50")

    # Widen the first column
    for i in range(len(filtered) + 1):
        table[i, 0].set_width(0.35)

    pct = int(min_freq * 100)
    ax.set_title(
        f"PSIE Frequent Terminals ({n_bootstrap} bootstrap samples, "
        f"\u2265{pct}% frequency)\n"
        f"Each row: terminal support set with its most common seed",
        fontsize=14, fontweight="bold", pad=18, color="#2C3E50",
    )

    # Legend annotation
    legend_text = (
        "Orange = PSRO-anchored    "
        "Green = PPO-anchored    "
        "Purple = Mixed"
    )
    ax.text(0.5, -0.02, legend_text, transform=ax.transAxes,
            ha="center", va="top", fontsize=9, fontstyle="italic",
            color="#666666")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PSIE terminal summary table",
    )
    parser.add_argument("--min-freq", type=float, default=0.30,
                        help="Minimum bootstrap frequency (default 0.30)")
    args = parser.parse_args()

    data_path = (Path(__file__).parent.parent
                 / "data" / "analysis" / "psie_results.pkl")
    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        psie_data = pickle.load(f)

    rows, n_bootstrap = compute_terminal_stats(psie_data)
    print(f"  {n_bootstrap} samples, {len(rows)} unique terminals, "
          f"{sum(1 for r in rows if r['sample_freq'] >= args.min_freq)} "
          f"above {args.min_freq:.0%}")

    save_path = (Path(__file__).parent.parent
                 / "data" / "analysis" / "figures"
                 / "psie_terminal_table.png")
    draw_table(rows, n_bootstrap, str(save_path), min_freq=args.min_freq)


if __name__ == "__main__":
    main()
