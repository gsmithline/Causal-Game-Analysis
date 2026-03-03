"""PSIE expansion graph: bootstrap-averaged directed graph.

Builds the graph from saved PSIE results (psie_results.pkl), then draws
directed graphs at multiple frequency thresholds.

Usage:
    python visuals/psie_sankey.py                          # 10% and 20%
    python visuals/psie_sankey.py --min-freq 0.10 0.20 0.33
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib import cm, colors
import networkx as nx
import numpy as np

# Add project root so we can import codebase constants
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse codebase constants
from visualize_analysis import DISPLAY_NAMES, STRATEGY_ORDER, DEFAULT_DPI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PPO_COLOR = "#4a90d9"
PSRO_COLOR = "#e07b54"
OTHER_COLOR = "#999999"

PPO_EDGE = "#2b6cb0"
PSRO_EDGE = "#c05621"

SHORT_NAMES = {
    "walk": "Walk", "tough": "Tough", "nfsp": "NFSP", "mappo": "MAPPO",
    "soft": "Soft", "ppo": "PPO", "psro": "PSRO",
    "openai_5.2_none": "OA-5.2", "openai_5.2_low": "OA-Low",
    "ef1_bargainer": "EF1",
}


def _lighten(hex_color, factor=0.5):
    """Blend *hex_color* toward white by *factor* (0 = unchanged, 1 = white)."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _basin_color(terminal_support):
    has_ppo = "ppo" in terminal_support
    has_psro = "psro" in terminal_support
    if has_ppo and not has_psro:
        return PPO_COLOR
    if has_psro and not has_ppo:
        return PSRO_COLOR
    return OTHER_COLOR


def _basin_edge_color(terminal_support):
    has_ppo = "ppo" in terminal_support
    has_psro = "psro" in terminal_support
    if has_ppo and not has_psro:
        return PPO_EDGE
    if has_psro and not has_ppo:
        return PSRO_EDGE
    return "#666666"


def _dn(name):
    return DISPLAY_NAMES.get(name, name)


def _short(name):
    return SHORT_NAMES.get(name, name[:3])


def _support_label(names):
    """Build a compact multi-line label for a support set."""
    ordered = sorted(names, key=lambda n: STRATEGY_ORDER.index(n)
                     if n in STRATEGY_ORDER else 999)
    if len(ordered) == 1:
        return _dn(ordered[0])
    # For 2+ strategies, use short names to fit inside nodes
    parts = [_short(n) for n in ordered]
    if len(parts) == 2:
        return ", ".join(parts)
    # 3+: two per line
    lines = []
    for i in range(0, len(parts), 2):
        lines.append(", ".join(parts[i:i+2]))
    return "\n".join(lines)


def _support_key(names):
    return frozenset(names)


# ---------------------------------------------------------------------------
# 1. Build graph from saved PSIE results (no re-running)
# ---------------------------------------------------------------------------


def _reconstruct_paths_from_entry_order(entry_order, strategy_names, n_seeds):
    """Reconstruct per-seed expansion paths from entry_order dict.

    entry_order[strategy] = list of n_seeds values, where each value is the
    step at which that strategy entered the seed's expansion (or None).

    Yields (seed_idx, path) where path is a list of (step, frozenset) tuples.
    """
    for seed_idx in range(n_seeds):
        entries = []
        for strat in strategy_names:
            step = entry_order[strat][seed_idx]
            if step is not None:
                entries.append((step, strat))
        entries.sort()

        # Build cumulative support at each step
        path = []
        current = frozenset()
        for step, strat in entries:
            current = current | {strat}
            if not path or path[-1][0] != step:
                path.append((step, current))
            else:
                # Multiple strategies entered at same step — update last
                path[-1] = (step, current)
        yield seed_idx, path


def compute_bootstrap_graph_from_pkl(psie_data):
    """Build frequency-weighted graph from saved PSIE results.

    Args:
        psie_data: loaded psie_results.pkl dict

    Returns:
        G: networkx DiGraph with frequency-weighted edges/nodes
        n_samples: number of bootstrap samples used
    """
    strategy_names = psie_data["config"]["strategy_names"]
    analyses = psie_data["bootstrap"]["analyses"]
    n_bootstrap = len(analyses)
    n_seeds = len(strategy_names)
    total_traces = n_bootstrap * n_seeds

    # Aggregate across all bootstrap samples
    edge_sample_sets = {}   # (src_key, dst_key) -> set of sample indices
    edge_basins = {}        # (src_key, dst_key) -> list of basin colors
    node_sample_sets = {}   # sup_key -> set of sample indices
    node_step = {}          # sup_key -> min step seen
    node_terminal_count = {}  # sup_key -> count of times it's terminal
    node_visit_count = {}     # sup_key -> total visits across all traces
    node_basins = {}        # sup_key -> list of basin colors

    for b_idx, analysis in enumerate(analyses):
        eo = analysis["entry_order"]
        for seed_idx, path in _reconstruct_paths_from_entry_order(
                eo, strategy_names, n_seeds):
            if not path:
                continue

            # Terminal support = last in path
            terminal_support = path[-1][1]
            basin = _basin_color(terminal_support)

            prev_key = None
            for step, sup_key in path:
                # Node tracking
                node_sample_sets.setdefault(sup_key, set()).add(b_idx)
                if sup_key not in node_step:
                    node_step[sup_key] = step
                else:
                    node_step[sup_key] = min(node_step[sup_key], step)
                is_terminal = (sup_key == terminal_support)
                node_terminal_count[sup_key] = (
                    node_terminal_count.get(sup_key, 0) + (1 if is_terminal else 0)
                )
                node_visit_count[sup_key] = node_visit_count.get(sup_key, 0) + 1
                node_basins.setdefault(sup_key, []).append(basin)

                # Edge tracking
                if prev_key is not None and prev_key != sup_key:
                    ekey = (prev_key, sup_key)
                    edge_sample_sets.setdefault(ekey, set()).add(b_idx)
                    edge_basins.setdefault(ekey, []).append(basin)
                prev_key = sup_key

    # Build graph
    G = nx.DiGraph()

    for sup_key in node_sample_sets:
        basins = node_basins[sup_key]
        dominant_basin = max(set(basins), key=basins.count)
        freq = len(node_sample_sets[sup_key]) / n_bootstrap
        visits = node_visit_count.get(sup_key, 1)
        terminal_freq = node_terminal_count.get(sup_key, 0) / visits

        G.add_node(sup_key,
                   label=_support_label(sup_key),
                   basin_color=dominant_basin,
                   step=node_step[sup_key],
                   freq=freq,
                   terminal_freq=terminal_freq,
                   is_terminal=(terminal_freq > 0.01))

    for (src, dst), sample_set in edge_sample_sets.items():
        freq = len(sample_set) / n_bootstrap
        added = dst - src
        basins = edge_basins[(src, dst)]
        dominant_basin = max(set(basins), key=basins.count)

        added_label = ", ".join(_dn(s) for s in sorted(added,
                                key=lambda n: STRATEGY_ORDER.index(n)
                                if n in STRATEGY_ORDER else 999))

        G.add_edge(src, dst,
                   freq=freq,
                   count=len(sample_set),
                   added=added_label,
                   basin_color=(
                       PPO_EDGE if dominant_basin == PPO_COLOR else
                       PSRO_EDGE if dominant_basin == PSRO_COLOR else "#666666"
                   ))

    print(f"Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"({n_bootstrap} samples)")

    return G, n_bootstrap


def filter_graph(G, min_freq):
    """Return a copy of G with low-frequency edges/nodes removed."""
    H = G.copy()

    remove_edges = [(u, v) for u, v in H.edges()
                    if H.edges[u, v]["freq"] < min_freq]
    H.remove_edges_from(remove_edges)

    # Remove orphan non-seed nodes
    remove_nodes = [n for n in H.nodes()
                    if H.degree(n) == 0 and H.nodes[n]["step"] > 0]
    H.remove_nodes_from(remove_nodes)

    # Remove low-frequency non-seed nodes
    remove_nodes = [n for n in H.nodes()
                    if H.nodes[n]["freq"] < min_freq and H.nodes[n]["step"] > 0]
    H.remove_nodes_from(remove_nodes)

    print(f"Filtered graph (>={min_freq:.0%}): "
          f"{H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    return H


# ---------------------------------------------------------------------------
# 2. Layout
# ---------------------------------------------------------------------------


def _compute_layout(G):
    """Position nodes left-to-right by step, basins separated vertically.

    Vertical spacing adapts to the densest column so nodes don't overlap.
    """
    by_step = {}
    for node, data in G.nodes(data=True):
        by_step.setdefault(data["step"], []).append(node)

    # Determine per-column counts to pick a spacing that avoids overlap
    max_column = max(len(nl) for nl in by_step.values()) if by_step else 1
    # Base spacing 2.0 works for <=8 nodes; scale up for denser columns
    if max_column <= 8:
        y_spacing = 2.0
    else:
        y_spacing = 2.0 + 0.15 * (max_column - 8)
    y_spacing = min(y_spacing, 3.0)  # cap

    x_spacing = 5.0
    pos = {}

    for step, node_list in by_step.items():
        x = step * x_spacing

        ppo_nodes = [n for n in node_list
                     if G.nodes[n]["basin_color"] == PPO_COLOR]
        psro_nodes = [n for n in node_list
                      if G.nodes[n]["basin_color"] != PPO_COLOR]

        ppo_nodes.sort(key=lambda n: -G.nodes[n]["freq"])
        psro_nodes.sort(key=lambda n: -G.nodes[n]["freq"])

        basin_gap = 2.5

        for i, node in enumerate(ppo_nodes):
            pos[node] = (x, -(i * y_spacing) - basin_gap)

        for i, node in enumerate(psro_nodes):
            pos[node] = (x, i * y_spacing)

    return pos


# ---------------------------------------------------------------------------
# 3. Draw — styled like the BR graph
# ---------------------------------------------------------------------------


def draw_psie_graph(G, n_samples, save_path, min_freq=0.20):
    """Draw the bootstrap PSIE expansion graph.

    Styled to match the bootstrap best-response graph:
    - Blues colormap for edge color based on frequency
    - Edge width scales with frequency
    - Dashed/dotted line styles by frequency threshold (matching BR graph)
    - Basin-colored nodes with navy borders, white-haloed labels
    """
    pos = _compute_layout(G)
    if not pos:
        print("No nodes to draw.")
        return

    n_nodes = G.number_of_nodes()
    fig_h = max(14, min(24, 14 + (n_nodes - 30) * 0.25))
    fig, ax = plt.subplots(figsize=(20, fig_h))
    ax.axis("off")

    # Set axis limits with margin
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    margin = 2.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # --- Paper conventions (Section 5, lines 469-475) ---
    # Blues colormap, vmin=0 vmax=1
    # Dotted if p in [0.01, 0.33], dashed if p in (0.33, 0.66], solid if p > 0.66
    # Darker & thicker = higher frequency: width = 3 + 6*p, color = Blues(p)
    # arrowsize = 70, edges drawn before nodes
    cmap = cm.Blues
    norm = colors.Normalize(vmin=0, vmax=1)

    # --- Draw edges first (same order as BR graph) ---
    for u, v, edata in G.edges(data=True):
        freq = edata["freq"]
        color = cmap(norm(freq))
        width = 3 + 6 * freq
        # Low-freq edges get reduced alpha to cut visual noise
        alpha = 1.0 if freq > 0.33 else 0.45
        style = ("solid" if freq > 0.66
                 else "dashed" if freq > 0.33
                 else "dotted")
        rad = 0.15 if G.has_edge(v, u) else 0.0
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            arrowstyle="-|>", arrowsize=70,
            edge_color=[color], width=width, style=style,
            connectionstyle=f"arc3,rad={rad}",
            alpha=alpha,
            ax=ax,
        )

    # --- Classify nodes ---
    node_list = list(G.nodes())
    node_sizes = {}
    node_styles = {}  # 'terminal', 'filtered_leaf', or 'intermediate'
    for n in node_list:
        sz = len(n)
        node_sizes[n] = 5000 if sz == 1 else 6500 if sz == 2 else 8000
        terminal_freq = G.nodes[n].get("terminal_freq", 0)
        is_true_terminal = terminal_freq >= 0.10
        is_filtered_leaf = (G.out_degree(n) == 0) and not is_true_terminal
        if is_true_terminal:
            node_styles[n] = "terminal"
        elif is_filtered_leaf:
            node_styles[n] = "filtered_leaf"
        else:
            node_styles[n] = "intermediate"

    # Pass 1 — Gold halo behind terminal nodes
    terminal_nodes = [n for n in node_list if node_styles[n] == "terminal"]
    if terminal_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=terminal_nodes,
            node_color="#FFD700",
            node_size=[int(node_sizes[n] * 1.55) for n in terminal_nodes],
            linewidths=0, edgecolors="none",
            ax=ax,
        )

    # Pass 2 — All nodes cyan, filtered leaves faded
    fill_colors = []
    for n in node_list:
        if node_styles[n] == "filtered_leaf":
            fill_colors.append(_lighten("#00FFFF", 0.55))
        else:
            fill_colors.append("#00FFFF")

    nx.draw_networkx_nodes(
        G, pos, nodelist=node_list,
        node_color=fill_colors,
        node_size=[node_sizes[n] for n in node_list],
        linewidths=2, edgecolors="#333333",
        ax=ax,
    )


    # --- Node labels with halo ---
    for n in node_list:
        label = G.nodes[n]["label"]
        sz = len(n)
        font_size = 13 if sz == 1 else 11 if sz == 2 else 9
        txt = ax.text(
            pos[n][0], pos[n][1], label,
            ha="center", va="center",
            fontsize=font_size, fontweight="bold",
            fontfamily="DejaVu Sans", color="black",
            zorder=5,
        )
        txt.set_path_effects([
            pe.Stroke(linewidth=4, foreground="white"),
            pe.Normal(),
        ])

        # Show terminal frequency below true-terminal nodes
        if node_styles[n] == "terminal":
            tf = G.nodes[n]["terminal_freq"]
            freq_txt = ax.text(
                pos[n][0], pos[n][1] - 0.6, f"{tf:.0%}",
                ha="center", va="top",
                fontsize=11, fontweight="bold", color="#8B0000",
                zorder=5,
            )
            freq_txt.set_path_effects([
                pe.Stroke(linewidth=4, foreground="white"),
                pe.Normal(),
            ])

    # --- Step column headers ---
    by_step = {}
    for node, data in G.nodes(data=True):
        by_step.setdefault(data["step"], []).append(node)
    max_y = max(p[1] for p in pos.values())
    for step in sorted(by_step.keys()):
        x = step * 5.0
        label = "Seed" if step == 0 else f"Step {step}"
        ax.text(x, max_y + 1.5, label, ha="center", va="bottom",
                fontsize=15, fontweight="bold")

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor="#00FFFF", edgecolor="#333333",
                       linewidth=1.5, label="Intermediate"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#FFD700", markeredgecolor="#333333",
                   markersize=14, linewidth=0,
                   label="Terminal (gold halo)"),
        mpatches.Patch(facecolor=_lighten("#00FFFF", 0.55),
                       edgecolor="#333333", linewidth=1.5,
                       label="Filtered leaf (faded)"),
        plt.Line2D([0], [0], color=cmap(0.85), linewidth=4,
                   linestyle="solid", label="Freq > 66%"),
        plt.Line2D([0], [0], color=cmap(0.5), linewidth=3,
                   linestyle="dashed", label="Freq 33\u201366%"),
        plt.Line2D([0], [0], color=cmap(0.2), linewidth=2,
                   linestyle="dotted", label="Freq 1\u201333%"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=11, framealpha=0.9, edgecolor="gray")

    pct = int(min_freq * 100)
    ax.set_title(f"PSIE: Bootstrap Expansion Graph ({n_samples} samples, "
                 f"\u2265{pct}% frequency)",
                 fontsize=18, fontweight="bold", pad=20)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def draw_hidden_terminals_table(G_full, H_filtered, n_samples, save_path,
                                min_tf=0.50, min_node_freq=0.10):
    """Draw a table of high-frequency terminals hidden by the edge filter.

    Shows nodes with terminal_freq >= min_tf and node_freq >= min_node_freq
    that were excluded from the filtered graph.
    """
    h_nodes = set(H_filtered.nodes())
    rows = []
    for n in G_full.nodes():
        d = G_full.nodes[n]
        if (n not in h_nodes
                and d["terminal_freq"] >= min_tf
                and d["freq"] >= min_node_freq):
            label = ", ".join(
                _short(s) for s in sorted(
                    n, key=lambda s: STRATEGY_ORDER.index(s)
                    if s in STRATEGY_ORDER else 999))
            # Determine basin attractor
            if "psro" in n and "ppo" not in n:
                attractor = "PSRO"
            elif "ppo" in n and "psro" not in n:
                attractor = "PPO"
            elif "ppo" in n and "psro" in n:
                attractor = "PSRO+PPO"
            else:
                attractor = "OA-Low"
            rows.append((label, d["terminal_freq"], d["freq"],
                         d["step"], attractor))

    if not rows:
        print("No hidden terminals to display.")
        return

    rows.sort(key=lambda r: -r[1])

    fig, ax = plt.subplots(figsize=(14, 0.45 * len(rows) + 2.0))
    ax.axis("off")

    col_labels = ["Support", "Terminal %", "Node Freq", "Step", "Attractor"]
    cell_text = []
    cell_colors = []
    for label, tf, freq, step, attractor in rows:
        cell_text.append([label, f"{tf:.0%}", f"{freq:.0%}",
                          str(step), attractor])
        if attractor == "PSRO":
            c = "#e8d4cc"
        elif attractor == "PPO":
            c = "#ccdae8"
        elif attractor == "PSRO+PPO":
            c = "#d8cce8"
        else:
            c = "#e0e0e0"
        cell_colors.append([c] * 5)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellColours=cell_colors, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#333333")
        cell.set_text_props(color="white", fontweight="bold")

    ax.set_title(
        f"Hidden Terminals: tf \u2265 {min_tf:.0%}, node freq \u2265 {min_node_freq:.0%}\n"
        f"(not shown in graph due to edge filter)",
        fontsize=13, fontweight="bold", pad=15)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="PSIE bootstrap expansion graph")
    parser.add_argument("--min-freq", type=float, nargs="+",
                        default=[0.10, 0.20],
                        help="Frequency thresholds to generate (default: 0.10 0.20)")
    args = parser.parse_args()

    # Load saved PSIE results (no re-running needed)
    data_path = (Path(__file__).parent.parent
                 / "data" / "analysis" / "psie_results.pkl")
    print(f"Loading PSIE results from {data_path}...")
    with open(data_path, "rb") as f:
        psie_data = pickle.load(f)

    n_bootstrap = len(psie_data["bootstrap"]["analyses"])
    n_strats = len(psie_data["config"]["strategy_names"])
    print(f"  {n_bootstrap} bootstrap samples, {n_strats} strategies")

    # Build full graph once from saved data
    G, n_samples = compute_bootstrap_graph_from_pkl(psie_data)

    # Draw at each frequency threshold
    fig_dir = Path(__file__).parent.parent / "data" / "analysis" / "figures"
    for mf in args.min_freq:
        pct = int(mf * 100)
        H = filter_graph(G, mf)
        save_path = fig_dir / f"psie_sankey_filter_{pct}.png"
        draw_psie_graph(H, n_samples, str(save_path), min_freq=mf)

        # Companion table of hidden terminals
        table_path = fig_dir / f"psie_hidden_terminals_{pct}.png"
        draw_hidden_terminals_table(G, H, n_samples, str(table_path))


if __name__ == "__main__":
    main()
