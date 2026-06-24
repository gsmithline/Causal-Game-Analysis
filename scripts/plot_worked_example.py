"""Worked-example figure for §2.1 / §5.1 of the paper.

Two panels:
  Left  — equilibrium mixture shift across 4 scenarios (Full / -5.2_low / -PPO / -PSRO),
          with the UW delta annotated beneath each bar.
  Right — Level-3 CURB-conditional UW intervals for the 4 strategies referenced
          in the worked example, with the Level-2 (full-game) LOO marked.

Data comes from the bargaining "average game" worked example documented in README.md
(§ "Worked Example (Bargaining Domain, Average Game)").

Usage:
    uv run python scripts/plot_worked_example.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "worked_example.png"


# ----------------------------------------------------------------------------
# Data (from README.md worked example — bargaining average game, MENE)
# ----------------------------------------------------------------------------

# Strategy family colors (LLM = orange, RL = blue, heuristic = grey)
FAMILY = {
    "5.2-low":    "LLM",
    "5.2-none":   "LLM",
    "5.2-medium": "LLM",
    "5.4-low":    "LLM",
    "5.4-medium": "LLM",
    "PPO":        "RL",
    "PSRO":       "RL",
    "MAPPO":      "RL",
}
FAMILY_PALETTE = {
    "LLM":       "#E69F00",   # orange
    "LLM-light": "#F5C66E",
    "LLM-dark":  "#B97900",
    "LLM-2":     "#FFB74D",
    "LLM-3":     "#D88A00",
    "RL":        "#0072B2",   # blue
    "RL-light":  "#66A8D6",
    "RL-dark":   "#005580",
    "heuristic": "#999999",
}

# Per-strategy color (so the same strategy gets the same shade across bars)
COLORS = {
    "PPO":        "#0072B2",   # blue
    "PSRO":       "#56B4E9",   # lighter blue
    "MAPPO":      "#003F5C",   # dark blue
    "5.2-low":    "#E69F00",   # orange
    "5.2-none":   "#D55E00",   # red-orange
    "5.2-medium": "#F0C57A",   # tan
    "5.4-low":    "#B97900",   # dark orange
    "5.4-medium": "#FFB74D",   # light orange
}

# Scenarios: (label, [(strategy, share)], welfare_delta_label)
SCENARIOS = [
    ("Full game",
     [("PPO", 0.79), ("PSRO", 0.21)],
     "baseline"),
    ("Remove 5.2-low",
     [("PPO", 0.68), ("5.4-medium", 0.30), ("PSRO", 0.02)],
     r"$\Delta$UW = $-10.87$"),
    ("Remove PPO",
     [("MAPPO", 0.80), ("PSRO", 0.20)],
     r"$\Delta$UW $\approx 0$"),
    ("Remove PSRO",
     [("5.4-medium", 0.44), ("5.2-low", 0.32),
      ("5.2-medium", 0.17), ("5.2-none", 0.07)],
     r"$\Delta$UW = $-39.08$"),
]

# Right panel: CURB-conditional UW intervals + full-game (Level-2) LOO
# from README worked example. Order: most-stable on top.
CURB_INTERVALS = [
    # (name, curb_min, curb_max, level2_loo, classification)
    ("PPO",     +6.26,  +6.26,   +0.00, "helpful"),         # +0 because full-game ΔUW ≈ 0
    ("MAPPO",   -3.57,   0.00,   +0.00, "harmful"),         # full-game LOO ≈ 0 but CURB shows negative
    ("PSRO",   -45.34,  +6.67,  -39.08, "CURB-dependent"),
    ("5.2-low",-46.92,   0.00,  -10.87, "harmful"),
]

CLASS_COLOR = {
    "helpful":         "#2CA02C",   # green
    "harmful":         "#D62728",   # red
    "CURB-dependent":  "#E5B100",   # amber
}


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def make_figure():
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.2, 1.0]}
    )

    # --- LEFT: Regime-shift stacked bars ---------------------------------
    n_scen = len(SCENARIOS)
    ypos = np.arange(n_scen)[::-1]   # top-down

    seen_in_legend = set()
    for y, (label, mix, delta_label) in zip(ypos, SCENARIOS):
        left = 0.0
        for strat, share in mix:
            color = COLORS.get(strat, "#888888")
            axL.barh(y, share, left=left, color=color,
                     edgecolor="white", linewidth=1.2, height=0.65)
            # In-bar percentage label if wide enough
            if share >= 0.08:
                axL.text(left + share / 2, y,
                         f"{strat}\n{share*100:.0f}%",
                         ha="center", va="center",
                         fontsize=8, color="white", fontweight="bold")
            elif share >= 0.04:
                axL.text(left + share / 2, y, f"{share*100:.0f}%",
                         ha="center", va="center",
                         fontsize=7, color="white", fontweight="bold")
            seen_in_legend.add(strat)
            left += share

        # Welfare-delta annotation to the right
        axL.text(1.02, y, delta_label,
                 ha="left", va="center", fontsize=9,
                 transform=axL.get_yaxis_transform())

    axL.set_yticks(ypos)
    axL.set_yticklabels([s[0] for s in SCENARIOS], fontsize=10)
    axL.set_xlim(0, 1.0)
    axL.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axL.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    axL.set_xlabel("Equilibrium support (MENE)")
    axL.set_title("(a)  Equilibrium regime shift under strategy removal",
                  fontsize=11, loc="left", pad=10)
    axL.spines["top"].set_visible(False)
    axL.spines["right"].set_visible(False)
    axL.grid(axis="x", alpha=0.3, linestyle=":")
    axL.set_axisbelow(True)

    # --- RIGHT: CURB-conditional intervals -------------------------------
    n_agents = len(CURB_INTERVALS)
    ypos_r = np.arange(n_agents)[::-1]

    for y, (name, cmin, cmax, l2, cls) in zip(ypos_r, CURB_INTERVALS):
        color = CLASS_COLOR[cls]
        # Interval bar
        axR.plot([cmin, cmax], [y, y],
                 color=color, linewidth=4, solid_capstyle="round", alpha=0.55)
        # End-caps
        axR.plot([cmin, cmin], [y - 0.18, y + 0.18], color=color, linewidth=2)
        axR.plot([cmax, cmax], [y - 0.18, y + 0.18], color=color, linewidth=2)
        # Level-2 LOO marker
        axR.scatter([l2], [y], s=80, color=color, edgecolor="black",
                    linewidth=1.0, zorder=4)
        # Classification label
        axR.text(1.02, y, cls,
                 ha="left", va="center", fontsize=8.5,
                 color=color, fontweight="bold",
                 transform=axR.get_yaxis_transform())

    axR.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    axR.set_yticks(ypos_r)
    axR.set_yticklabels([c[0] for c in CURB_INTERVALS], fontsize=10)
    axR.set_xlabel(r"$\Delta$UW under removal")
    axR.set_title("(b)  Level-3 CURB-conditional LOO (UW)",
                  fontsize=11, loc="left", pad=10)
    axR.spines["top"].set_visible(False)
    axR.spines["right"].set_visible(False)
    axR.grid(axis="x", alpha=0.3, linestyle=":")
    axR.set_axisbelow(True)
    axR.set_xlim(-55, 18)

    # Right-panel legend
    legend_handles = [
        Patch(facecolor=CLASS_COLOR["helpful"], label="Consistently helpful"),
        Patch(facecolor=CLASS_COLOR["harmful"], label="Consistently harmful"),
        Patch(facecolor=CLASS_COLOR["CURB-dependent"], label="CURB-dependent"),
    ]
    axR.legend(
        handles=legend_handles, loc="lower right", frameon=False,
        fontsize=8, handlelength=1.2,
    )
    # Marker key
    axR.text(0.02, 0.02, "● = full-game (Level-2) LOO",
             transform=axR.transAxes, fontsize=8, color="black",
             style="italic")

    fig.tight_layout(rect=[0, 0, 0.98, 1])
    return fig


def main():
    fig = make_figure()
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved → {OUT}")
    print(f"saved → {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
