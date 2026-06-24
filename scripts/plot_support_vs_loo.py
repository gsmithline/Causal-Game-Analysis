"""Two-panel "support vs LOO" figure for §2.1 / §5.1.

Top panel:    NE support % on the bargaining average game (MENE).
Bottom panel: Signed ΔUW = W_full − W_LOO under leave-one-out
              (positive → presence helps welfare; negative → presence hurts).

Headline: support ≠ welfare importance.
  - PPO has 80% NE support but only +6.26 ΔUW.
  - PSRO has 20% NE support but −39.08 ΔUW (presence hurts welfare).
  - 5.2-low has 0% NE support but −10.87 ΔUW (larger effect than PPO).

Data is loaded from scripts/avg_game_loo_13.json
(13-strategy MENE on the average game).

Usage:
    .venv/bin/python scripts/plot_support_vs_loo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "scripts" / "avg_game_loo_13.json"
OUT = ROOT / "notebooks" / "support_vs_loo.png"

DISPLAY_NAME = {
    "walk": "walk", "tough": "tough", "soft": "soft",
    "nfsp": "NFSP", "mappo": "MAPPO", "ppo": "PPO", "psro": "PSRO",
    "ef1_bargainer": "EF1-barg.",
    "openai_5.2_none":   "5.2-none",
    "openai_5.2_low":    "5.2-low",
    "openai_5.2_medium": "5.2-medium",
    "openai_5.4_low":    "5.4-low",
    "openai_5.4_medium": "5.4-medium",
}

FAMILY = {
    "walk": "heuristic", "tough": "heuristic", "soft": "heuristic",
    "nfsp": "RL", "mappo": "RL", "ppo": "RL", "psro": "RL",
    "ef1_bargainer": "heuristic",
    "openai_5.2_none":   "LLM",
    "openai_5.2_low":    "LLM",
    "openai_5.2_medium": "LLM",
    "openai_5.4_low":    "LLM",
    "openai_5.4_medium": "LLM",
}

POS_COLOR = "#2CA02C"   # green: presence helps welfare
NEG_COLOR = "#D62728"   # red: presence hurts welfare
ZERO_COLOR = "#BBBBBB"  # grey: no effect


def load_data():
    d = json.loads(DATA.read_text())
    names = d["strategy_names"]
    support = np.array(d["full_support"]) * 100
    d_uw = np.array([r["d_uw"] if r["d_uw"] is not None else 0.0 for r in d["loo"]])
    return names, support, d_uw


def ordered(names, support, d_uw):
    """Sort: in-support agents first (by support desc), then by |ΔUW| desc."""
    order = []
    paired = list(zip(range(len(names)), support, d_uw))
    in_supp = sorted([p for p in paired if p[1] > 1e-2], key=lambda x: -x[1])
    out_supp = sorted([p for p in paired if p[1] <= 1e-2], key=lambda x: -abs(x[2]))
    return [p[0] for p in in_supp + out_supp]


def make_figure():
    names, support, d_uw = load_data()
    order = ordered(names, support, d_uw)
    names = [names[i] for i in order]
    support = support[order]
    d_uw = d_uw[order]
    labels = [DISPLAY_NAME.get(n, n) for n in names]
    x = np.arange(len(names))
    bar_w = 0.65

    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(9.5, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.30},
    )

    # --- Top: NE support % --------------------------------------------------
    # Use a single accent color for visual clarity; emphasize what's tall.
    top_colors = ["#0072B2" if s > 1e-2 else "#E0E0E0" for s in support]
    axT.bar(x, support, width=bar_w, color=top_colors, edgecolor="white", linewidth=1.0)
    for xi, s in zip(x, support):
        if s > 1:
            axT.text(xi, s + 2.5, f"{s:.0f}%", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
        else:
            axT.text(xi, 1.5, "0%", ha="center", va="bottom",
                     fontsize=8, color="#888")
    axT.set_ylabel("NE support  (MENE)")
    axT.set_ylim(0, 110)
    axT.set_yticks([0, 25, 50, 75, 100])
    axT.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axT.set_title("(a)  Who's in the equilibrium?", loc="left",
                  fontsize=11, pad=8, fontweight="bold")
    axT.spines["top"].set_visible(False)
    axT.spines["right"].set_visible(False)
    axT.grid(axis="y", alpha=0.3, linestyle=":")
    axT.set_axisbelow(True)

    # --- Bottom: signed ΔUW -------------------------------------------------
    bot_colors = [
        ZERO_COLOR if abs(v) < 0.5 else (POS_COLOR if v > 0 else NEG_COLOR)
        for v in d_uw
    ]
    axB.bar(x, d_uw, width=bar_w, color=bot_colors, edgecolor="white", linewidth=1.0)
    axB.axhline(0, color="black", linewidth=0.7)
    ymax = max(max(d_uw.max(), 0) * 1.4, 8)
    ymin = min(d_uw.min(), 0) * 1.18
    for xi, v in zip(x, d_uw):
        if abs(v) < 0.5:
            continue
        va = "bottom" if v > 0 else "top"
        offset = (ymax - ymin) * 0.025
        axB.text(xi, v + offset if v > 0 else v - offset,
                 f"{v:+.1f}",
                 ha="center", va=va, fontsize=9, fontweight="bold",
                 color="black")
    axB.set_ylim(ymin, ymax)
    axB.set_ylabel(r"$\Delta$UW  $=W_{\rm full}-W_{\rm LOO}$")
    axB.set_title("(b)  Whose presence shifts welfare?", loc="left",
                  fontsize=11, pad=8, fontweight="bold")
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.grid(axis="y", alpha=0.3, linestyle=":")
    axB.set_axisbelow(True)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=9.5, rotation=25, ha="right")

    # Sign-of-ΔUW key (placed at far right to avoid overlapping bars)
    axB.text(0.99, 0.96,
             r"$\Delta$UW > 0  $\Rightarrow$  presence helps welfare",
             transform=axB.transAxes, fontsize=8, color=POS_COLOR,
             va="top", ha="right")
    axB.text(0.99, 0.04,
             r"$\Delta$UW < 0  $\Rightarrow$  presence hurts welfare",
             transform=axB.transAxes, fontsize=8, color=NEG_COLOR,
             va="bottom", ha="right")

    # --- Annotations highlighting the mismatch ------------------------------
    # PSRO: 20% support, huge negative ΔUW — annotate in upper area to avoid overlap
    psro_i = names.index("psro")
    axB.annotate(
        "individually strong, but\npresence blocks the\nhigher-welfare LLM NE",
        xy=(psro_i, d_uw[psro_i] * 0.55),
        xytext=(psro_i + 2.0, ymax * 0.45),
        fontsize=8.5, color=NEG_COLOR, style="italic",
        arrowprops=dict(arrowstyle="->", color=NEG_COLOR, lw=1.0),
    )
    # 5.2-low: 0% support but nontrivial negative ΔUW
    low_i = names.index("openai_5.2_low")
    axB.annotate(
        "0% NE support,\n$|\\Delta$UW$|$ > PPO's",
        xy=(low_i, d_uw[low_i]),
        xytext=(low_i + 2.0, d_uw[low_i] - 4),
        fontsize=8.5, color=NEG_COLOR, style="italic",
        arrowprops=dict(arrowstyle="->", color=NEG_COLOR, lw=1.0),
    )

    fig.suptitle(
        "Bargaining average game — equilibrium support ≠ welfare importance",
        fontsize=12.5, fontweight="bold", y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    fig = make_figure()
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved → {OUT}")
    print(f"saved → {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
