"""
Publication-quality visualization of PSIE robust terminal analysis.

Shows four regions:
  - PSRO-anchored cluster (12 robust terminals)
  - PPO-anchored cluster (5 robust terminals)
  - Mixed psro+ppo crossover (2 robust terminals)
  - Singleton / fragile zone (1 singleton + walk as fragile)

Expansion flow is rendered as a directed graph: seeds on the left,
intermediaries in the middle, attractors (PSRO, PPO) on the right.

Data comes from 1,000-sample bootstrap at the 10% frequency threshold.

Usage:
    python visuals/psie_robust_terminals.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

PSRO_EDGE_C = "#C0501D"
PSRO_FILL = "#FDEBD0"
PSRO_BG = "#FDF2E9"

PPO_EDGE_C = "#117A65"
PPO_FILL = "#D5F5E3"
PPO_BG = "#E8F8F5"

MIXED_EDGE_C = "#6C3483"
MIXED_FILL = "#E8DAEF"
MIXED_BG = "#F5EEF8"

SINGLETON_EDGE_C = "#2471A3"
SINGLETON_FILL = "#D6EAF8"

FRAGILE_NODE = "#AAAAAA"
FRAGILE_FILL = "#F0F0F0"

ATTRACTOR_GLOW_PSRO = "#E59866"
ATTRACTOR_GLOW_PPO = "#76D7C4"

DISPLAY = {
    "psro": "PSRO",
    "ppo": "PPO",
    "openai_5.2_none": "OA-None",
    "openai_5.2_low": "OA-Low",
    "ef1_bargainer": "EF1",
    "mappo": "MAPPO",
    "nfsp": "NFSP",
    "walk": "Walk",
    "tough": "Tough",
    "soft": "Soft",
}


def _dn(name):
    """Short display name."""
    return DISPLAY.get(name, name)


# ---------------------------------------------------------------------------
# 10% threshold data  (1,000 bootstrap samples)
# ---------------------------------------------------------------------------

PSRO_PATHS = [
    ("psro",    ["psro"],                                              0.94),
    ("oa_none", ["openai_5.2_none", "psro"],                           0.36),
    ("oa_low",  ["openai_5.2_low", "psro"],                            0.32),
    ("ef1",     ["ef1_bargainer", "openai_5.2_none", "psro"],          0.26),
    ("mappo",   ["mappo", "openai_5.2_none", "psro"],                  0.25),
    ("tough",   ["tough", "openai_5.2_low", "psro"],                   0.18),
    ("soft",    ["soft", "tough", "openai_5.2_low", "psro"],           0.18),
    ("ef1",     ["ef1_bargainer", "openai_5.2_low",
                 "openai_5.2_none", "psro"],                           0.15),
    ("tough",   ["tough", "openai_5.2_none", "psro"],                  0.12),
    ("soft",    ["soft", "tough", "openai_5.2_none", "psro"],          0.12),
    ("oa_none", ["openai_5.2_none", "openai_5.2_low", "psro"],        0.12),
    ("mappo",   ["mappo", "openai_5.2_low", "openai_5.2_none", "psro"], 0.11),
]

# Transition frequencies that differ from terminal frequencies.
# soft→tough is 100% (every bootstrap sample), not 18% (terminal freq).
EDGE_FREQ_OVERRIDES = {
    ("soft", "tough"): 1.0,
}

PPO_PATHS = [
    ("ppo",     ["ppo"],                                               0.72),
    ("nfsp",    ["nfsp", "ppo"],                                       0.64),
    ("oa_low",  ["openai_5.2_low", "nfsp", "ppo"],                    0.17),
    ("oa_none", ["openai_5.2_none", "nfsp", "ppo"],                   0.15),
    ("ef1",     ["ef1_bargainer", "openai_5.2_none", "nfsp", "ppo"],  0.13),
]

MIXED_PATHS = [
    ("ppo",  ["ppo", "psro"],                                         0.15),
    ("nfsp", ["nfsp", "ppo", "psro"],                                 0.13),
]

SINGLETON_FREQ = 0.13   # {openai_5.2_low} alone
FRAGILE_SEEDS = ["walk"]  # 134 terminals, none above 10%


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _collect_edges(paths, cluster_tag):
    """Yield (src, dst, freq, cluster) for consecutive pairs in each path."""
    for _seed, path, freq in paths:
        for i in range(len(path) - 1):
            yield path[i], path[i + 1], freq, cluster_tag


def build_graph():
    """Build directed graph with max-frequency aggregation per edge."""
    G = nx.DiGraph()

    # Node frequencies: every node in every path gets credit.
    node_freqs = {}
    for path_list in (PSRO_PATHS, PPO_PATHS, MIXED_PATHS):
        for _seed, path, freq in path_list:
            for node in path:
                node_freqs[node] = max(node_freqs.get(node, 0), freq)

    # Edge frequencies: keep max freq per (u,v).
    edge_data = {}
    all_edges = (
        list(_collect_edges(PSRO_PATHS, "psro"))
        + list(_collect_edges(PPO_PATHS, "ppo"))
        + list(_collect_edges(MIXED_PATHS, "mixed"))
    )
    for src, dst, freq, cluster in all_edges:
        key = (src, dst)
        if key not in edge_data:
            edge_data[key] = {}
        edge_data[key][cluster] = max(edge_data[key].get(cluster, 0), freq)

    # Assign each node to its primary cluster.
    cluster_map = {
        "psro": "psro", "ppo": "ppo", "nfsp": "ppo",
        "openai_5.2_none": "psro", "openai_5.2_low": "psro",
        "ef1_bargainer": "psro", "mappo": "psro",
        "tough": "psro", "soft": "psro",
    }
    for node, freq in node_freqs.items():
        G.add_node(node, cluster=cluster_map.get(node, "psro"), freq=freq)

    # Edges: pick dominant cluster colour.
    for (u, v), cdict in edge_data.items():
        dominant = max(cdict, key=cdict.get)
        freq = max(cdict.values())
        # Apply transition-frequency overrides where terminal freq != step freq.
        freq = EDGE_FREQ_OVERRIDES.get((u, v), freq)
        G.add_edge(u, v, freq=freq, cluster=dominant)

    # Fragile seeds.
    for name in FRAGILE_SEEDS:
        G.add_node(name, cluster="fragile", freq=0.0)

    return G


# ---------------------------------------------------------------------------
# Hand-tuned layout
# ---------------------------------------------------------------------------

def compute_layout():
    """Left-to-right flow: seeds | intermediaries | attractors."""
    return {
        # Attractors (right)
        "psro":           (14.0, 7.0),
        "ppo":            (14.0, -2.0),
        # Key intermediaries (middle)
        "openai_5.2_none": (8.0, 10.5),
        "openai_5.2_low":  (8.0, 3.5),
        "nfsp":            (8.0, -2.0),
        # Secondary intermediary
        "tough":           (3.5, 7.0),
        # Outer seeds (left)
        "ef1_bargainer":   (1.5, 12.0),
        "mappo":           (1.5, 9.5),
        "soft":            (0.0, 3.5),
        # Fragile (bottom)
        "walk":            (4.0, -6.0),
    }


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_glow(ax, x, y, color, radius=0.7):
    """Soft radial glow behind an attractor node."""
    for r_mult, alpha in [(2.0, 0.06), (1.6, 0.10), (1.3, 0.16)]:
        ax.add_patch(plt.Circle(
            (x, y), radius * r_mult,
            facecolor=color, edgecolor="none", alpha=alpha, zorder=1,
        ))


def _edge_style(freq):
    """(linestyle, alpha) by frequency bracket."""
    if freq > 0.60:
        return "solid", 0.90
    elif freq > 0.30:
        return (0, (8, 4)), 0.80
    else:
        return (0, (3, 3)), 0.70


def _edge_color(cluster):
    return {"psro": PSRO_EDGE_C, "ppo": PPO_EDGE_C}.get(cluster, MIXED_EDGE_C)


# ---------------------------------------------------------------------------
# Main drawing
# ---------------------------------------------------------------------------

def draw_figure(G, pos, save_path):
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    ax.set_xlim(-3.5, 18.5)
    ax.set_ylim(-9.0, 15.0)

    # ── 1. Cluster backgrounds ──────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (-1.5, 2.0), 17.5, 12.0,
        boxstyle="round,pad=0.6",
        facecolor=PSRO_BG, edgecolor=PSRO_EDGE_C,
        linewidth=1.5, linestyle="--", alpha=0.28, zorder=0,
    ))
    ax.text(15.5, 13.6, "PSRO-anchored cluster",
            fontsize=13, fontstyle="italic", color=PSRO_EDGE_C,
            fontweight="bold", ha="right", va="top", alpha=0.55)
    ax.text(15.5, 13.0, "12 robust terminals",
            fontsize=10, fontstyle="italic", color=PSRO_EDGE_C,
            ha="right", va="top", alpha=0.45)

    ax.add_patch(FancyBboxPatch(
        (6.0, -4.0), 10.0, 4.5,
        boxstyle="round,pad=0.6",
        facecolor=PPO_BG, edgecolor=PPO_EDGE_C,
        linewidth=1.5, linestyle="--", alpha=0.28, zorder=0,
    ))
    ax.text(15.5, 0.2, "PPO-anchored cluster",
            fontsize=13, fontstyle="italic", color=PPO_EDGE_C,
            fontweight="bold", ha="right", va="top", alpha=0.55)
    ax.text(15.5, -0.4, "5 robust terminals",
            fontsize=10, fontstyle="italic", color=PPO_EDGE_C,
            ha="right", va="top", alpha=0.45)

    # ── 2. Attractor glows ──────────────────────────────────────────────
    _draw_glow(ax, *pos["psro"], ATTRACTOR_GLOW_PSRO, radius=0.85)
    _draw_glow(ax, *pos["ppo"], ATTRACTOR_GLOW_PPO, radius=0.85)

    # ── 3. Edges ────────────────────────────────────────────────────────
    # Per-edge label offsets (dx, dy from midpoint) to prevent collisions.
    label_offsets = {
        ("ef1_bargainer", "openai_5.2_none"):   ( 0.3,  0.65),
        ("ef1_bargainer", "openai_5.2_low"):    (-0.8, -0.35),
        ("mappo", "openai_5.2_none"):           ( 0.3, -0.65),
        ("mappo", "openai_5.2_low"):            (-0.8, -0.40),
        ("openai_5.2_none", "psro"):            ( 0.3,  0.65),
        ("openai_5.2_low", "psro"):             ( 0.3, -0.65),
        ("openai_5.2_none", "openai_5.2_low"):  (-0.85, 0.0),
        ("openai_5.2_low", "openai_5.2_none"):  ( 0.85, 0.0),
        ("tough", "openai_5.2_low"):            (-0.55, -0.55),
        ("tough", "openai_5.2_none"):           (-0.55,  0.65),
        ("soft", "tough"):                      ( 0.0,  0.60),
        ("nfsp", "ppo"):                        ( 0.0,  0.55),
        ("ppo", "psro"):                        ( 1.0, 0.0),
        ("openai_5.2_low", "nfsp"):             (-0.5, -0.65),
        ("openai_5.2_none", "nfsp"):            ( 0.6, -0.65),
    }

    # Per-edge curvature overrides.
    curvature = {
        ("ppo", "psro"): 0.15,
        ("openai_5.2_none", "openai_5.2_low"): 0.22,
        ("openai_5.2_low", "openai_5.2_none"): 0.22,
        ("ef1_bargainer", "openai_5.2_low"): 0.15,
        ("mappo", "openai_5.2_low"): 0.15,
        ("tough", "openai_5.2_none"): 0.12,
    }

    for u, v, edata in G.edges(data=True):
        freq = edata["freq"]
        cluster = edata["cluster"]
        color = _edge_color(cluster)
        width = 1.2 + 4.0 * freq
        style, alpha = _edge_style(freq)
        rad = curvature.get((u, v), 0.08)

        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            arrowstyle="-|>", arrowsize=20,
            edge_color=[color], width=width, style=style,
            connectionstyle=f"arc3,rad={rad}",
            alpha=alpha, ax=ax,
            min_source_margin=28, min_target_margin=28,
        )

        # Label
        mx = (pos[u][0] + pos[v][0]) / 2
        my = (pos[u][1] + pos[v][1]) / 2
        if (u, v) in label_offsets:
            ox, oy = label_offsets[(u, v)]
        else:
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]
            L = max(np.hypot(dx, dy), 0.01)
            ox, oy = -dy / L * 0.45, dx / L * 0.45

        ax.text(
            mx + ox, my + oy, f"{freq:.0%}",
            fontsize=10, fontweight="bold", color=color,
            ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.92),
        )

    # ── 4. Nodes ────────────────────────────────────────────────────────

    # 4a. Fragile (gray square)
    for node in FRAGILE_SEEDS:
        x, y = pos[node]
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node],
            node_color=FRAGILE_FILL, node_size=2200,
            linewidths=2.0, edgecolors=FRAGILE_NODE,
            node_shape="s", ax=ax, alpha=0.80,
        )
        s = 0.48
        ax.add_patch(plt.Rectangle(
            (x - s, y - s), 2 * s, 2 * s,
            facecolor="none", edgecolor=FRAGILE_NODE,
            linewidth=1.8, linestyle="--", zorder=3,
        ))
        ax.text(x, y, f"{_dn(node)}\n(fragile)",
                ha="center", va="center", fontsize=10,
                color="#777777", zorder=5, linespacing=1.3)

    # 4b. Cluster nodes (circles)
    for node in G.nodes():
        data = G.nodes[node]
        if data.get("cluster") == "fragile":
            continue

        freq = data.get("freq", 0)
        x, y = pos[node]
        is_attractor = node in ("psro", "ppo")

        fill_c = PPO_FILL if node in ("ppo", "nfsp") else PSRO_FILL
        edge_c = PPO_EDGE_C if node in ("ppo", "nfsp") else PSRO_EDGE_C

        if is_attractor:
            lw, size = 4.0, 3800 + 3000 * freq
        else:
            lw, size = 2.5, 1800 + 2800 * freq

        nx.draw_networkx_nodes(
            G, pos, nodelist=[node],
            node_color=fill_c, node_size=size,
            linewidths=lw, edgecolors=edge_c,
            node_shape="o", ax=ax, alpha=1.0,
        )

        # Double ring for attractors
        if is_attractor:
            r = np.sqrt(size / np.pi) / 56
            ax.add_patch(plt.Circle(
                (x, y), r * 0.82,
                facecolor="none", edgecolor=edge_c,
                linewidth=1.5, alpha=0.40, zorder=3,
            ))

        label = _dn(node)
        fs = 14 if is_attractor else 11
        fw = "bold" if is_attractor else "semibold"
        txt_str = f"{label}\n({freq:.0%})" if freq > 0 else label

        txt = ax.text(
            x, y, txt_str,
            ha="center", va="center",
            fontsize=fs, fontweight=fw,
            color="#2C3E50", zorder=7, linespacing=1.3,
        )
        txt.set_path_effects([
            pe.Stroke(linewidth=3.5, foreground="white"),
            pe.Normal(),
        ])

    # ── 5. Singleton bubble ─────────────────────────────────────────────
    sx, sy = -1.5, -0.5
    ax.add_patch(plt.Circle(
        (sx, sy), 0.65,
        facecolor=SINGLETON_FILL, edgecolor=SINGLETON_EDGE_C,
        linewidth=2.0, alpha=0.75, zorder=2,
    ))
    ax.text(sx, sy + 0.10, f"{_dn('openai_5.2_low')}\nsingleton",
            ha="center", va="center", fontsize=9, fontweight="semibold",
            color=SINGLETON_EDGE_C, zorder=5, linespacing=1.3)
    ax.text(sx, sy - 0.55, "13%",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=SINGLETON_EDGE_C, zorder=5)
    ax.text(sx, sy + 1.05, "Neither cluster",
            fontsize=10, fontstyle="italic", color=SINGLETON_EDGE_C,
            ha="center", va="bottom", alpha=0.65, fontweight="bold")

    # ── 6. Mixed psro+ppo annotation ────────────────────────────────────
    mx, my = 16.5, 3.0
    ax.text(mx, my, "Mixed psro+ppo\n2 terminals",
            fontsize=10, fontstyle="italic", color=MIXED_EDGE_C,
            ha="center", va="center", fontweight="bold", alpha=0.70,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=MIXED_BG,
                      edgecolor=MIXED_EDGE_C, linewidth=1.2, alpha=0.55))
    emx = (pos["ppo"][0] + pos["psro"][0]) / 2
    emy = (pos["ppo"][1] + pos["psro"][1]) / 2
    ax.annotate("", xy=(emx + 0.5, emy), xytext=(mx - 0.6, my - 0.5),
                arrowprops=dict(arrowstyle="-|>", color=MIXED_EDGE_C,
                                lw=1.2, alpha=0.55), zorder=4)

    # ── 7. Fragile separator ────────────────────────────────────────────
    ax.plot([-3.0, 17.5], [-5.0, -5.0],
            color="#CCCCCC", linewidth=1.0, linestyle=":", zorder=0)
    ax.text(4.0, -7.3,
            "Fragile seed -- 134 terminals, none above 10%",
            fontsize=10, fontstyle="italic", color="#888888",
            ha="center", va="top")

    # ── 8. Multi-terminal note (lower left) ─────────────────────────────
    ax.text(
        0.02, 0.02,
        "Note: some seeds reach multiple robust terminals\n"
        "(e.g. EF1 \u2192 psro-path at 26% and ppo-path at 13%;\n"
        " Tough \u2192 via OA-Low at 18% and via OA-None at 12%)",
        transform=ax.transAxes,
        fontsize=9, fontstyle="italic", color="#666666",
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FAFAFA",
                  edgecolor="#DDDDDD", alpha=0.90),
    )

    # ── 9. Legend (upper right, outside the cluster boxes) ──────────────
    legend_elements = [
        mpatches.Patch(facecolor=PSRO_FILL, edgecolor=PSRO_EDGE_C,
                       linewidth=2.0, label="PSRO-cluster node"),
        mpatches.Patch(facecolor=PPO_FILL, edgecolor=PPO_EDGE_C,
                       linewidth=2.0, label="PPO-cluster node"),
        mpatches.Patch(facecolor=SINGLETON_FILL, edgecolor=SINGLETON_EDGE_C,
                       linewidth=1.5, label="Singleton terminal"),
        mpatches.Patch(facecolor=FRAGILE_FILL, edgecolor=FRAGILE_NODE,
                       linewidth=1.5, linestyle="--",
                       label="Fragile (no robust terminal)"),
        plt.Line2D([], [], color="none", label=""),
        plt.Line2D([0], [0], color="#555555", linewidth=3.5,
                   linestyle="solid", label="Edge freq > 60%"),
        plt.Line2D([0], [0], color="#555555", linewidth=2.5,
                   linestyle="dashed", label="Edge freq 30\u201360%"),
        plt.Line2D([0], [0], color="#555555", linewidth=1.5,
                   linestyle="dotted", label="Edge freq 10\u201330%"),
    ]
    legend = ax.legend(
        handles=legend_elements, loc="lower left",
        fontsize=9.5, framealpha=0.95, edgecolor="#CCCCCC",
        title="Legend", title_fontsize=10,
        borderpad=0.8, labelspacing=0.55,
        bbox_to_anchor=(0.50, 0.0),
    )
    legend.get_frame().set_linewidth(1.0)

    # ── 10. Interpretive annotation (bottom right) ──────────────────────
    ax.text(
        0.98, 0.02,
        "Edges: seed \u2192 recruit \u2192 attractor\n"
        "Labels: bootstrap frequency of that terminal\n"
        "Node size \u221d importance (freq-weighted presence)\n"
        "20 robust terminals at \u226510% threshold",
        transform=ax.transAxes,
        fontsize=9, fontstyle="italic", color="#777777",
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#DDDDDD", alpha=0.90),
    )

    # ── 11. Title ───────────────────────────────────────────────────────
    ax.set_title(
        "PSIE Robust Terminals "
        "(\u226510% bootstrap frequency, 1000 samples)\n"
        "Expansion flow from seeds to attractor equilibria",
        fontsize=17, fontweight="bold", pad=22, color="#2C3E50",
    )

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    G = build_graph()
    pos = compute_layout()
    save_path = (
        Path(__file__).parent.parent
        / "data" / "analysis" / "figures" / "psie_robust_terminals.png"
    )
    draw_figure(G, pos, str(save_path))


if __name__ == "__main__":
    main()
