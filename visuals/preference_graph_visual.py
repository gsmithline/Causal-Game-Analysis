import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm, colors
import networkx as nx
import os


def _short_name(s):
    """Short display name for a strategy."""
    if s == "openai_5.2_low":
        return "5.2:L"
    elif s == "openai_5.2_none":
        return "5.2:N"
    elif s == "openai_5.2_medium":
        return "5.2:M"
    elif s == "ef1_bargainer":
        return "EF1"
    return s[:5]


def create_preference_graph(
    avg_p1,
    avg_p2,
    strategy_names,
    filename='average_preference_graph',
    save_dir=None,
    dpi=300,
    edge_threshold=0.01,
):
    """
    Visualize the average preference graph over N^2 joint strategy profiles.

    Scales node size, font, spacing, and figure size based on the number of
    strategies so the graph looks good at any n.

    Args:
        avg_p1: np.ndarray, shape (n^2, n^2). P1 BR deviation edge frequencies.
        avg_p2: np.ndarray, shape (n^2, n^2). P2 BR deviation edge frequencies.
        strategy_names: List[str], length n.
        filename: Base name for output file.
        save_dir: Directory to save PNG.
        dpi: Resolution.
        edge_threshold: Minimum frequency to draw an edge.
    """
    n = len(strategy_names)
    n2 = n * n

    # --- Adaptive sizing ---
    cell_size = max(2.5, 4.0 - 0.15 * n)  # spacing between nodes
    margin = cell_size * 0.8
    fig_side = n * cell_size + 2 * margin
    figsize = (fig_side, fig_side)

    node_size = max(400, int(3500 / (1 + 0.3 * n)))
    font_size = max(5, int(14 - 0.7 * n))
    axis_font_size = max(8, int(14 - 0.5 * n))
    title_font_size = max(12, int(20 - 0.5 * n))
    edge_arrow_size = max(15, int(30 - n))
    edge_width_base = max(1, 3 - 0.1 * n)
    edge_width_scale = max(2, 6 - 0.3 * n)
    edge_rad = max(0.1, 0.3 - 0.01 * n)

    short_names = [_short_name(s) for s in strategy_names]

    # Build profile labels
    profile_labels = {}
    for a in range(n):
        for b in range(n):
            idx = a * n + b
            profile_labels[idx] = f"{short_names[a]},\n{short_names[b]}"

    # Build networkx graph
    G = nx.DiGraph()
    for idx in range(n2):
        G.add_node(idx)

    # Collect edges with player attribution
    p1_edges = []
    p2_edges = []
    for src in range(n2):
        for dst in range(n2):
            w1 = float(avg_p1[src, dst])
            w2 = float(avg_p2[src, dst])
            if w1 >= edge_threshold:
                G.add_edge(src, dst, weight=w1)
                p1_edges.append((src, dst, w1))
            if w2 >= edge_threshold:
                G.add_edge(src, dst, weight=w2)
                p2_edges.append((src, dst, w2))

    # Identify sinks (pure NE): no outgoing edges from either player
    sinks = set()
    for idx in range(n2):
        if avg_p1[idx, :].sum() + avg_p2[idx, :].sum() < edge_threshold:
            sinks.add(idx)

    # Grid layout scaled by cell_size
    pos = {}
    for a in range(n):
        for b in range(n):
            idx = a * n + b
            pos[idx] = (b * cell_size, -a * cell_size)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.set_title("Average Preference Graph\n(BR Deviations over Joint Profiles)",
                 fontsize=title_font_size, fontweight='bold', pad=20)

    # Set axis limits to frame the grid tightly
    ax.set_xlim(-margin, (n - 1) * cell_size + margin)
    ax.set_ylim(-(n - 1) * cell_size - margin, margin)
    ax.set_aspect('equal')

    # Axis labels
    ax.text((n - 1) * cell_size / 2, margin * 0.7,
            "Meta-game Player 2 Strategy",
            ha='center', fontsize=axis_font_size, fontweight='bold')
    ax.text(-margin * 0.7, -(n - 1) * cell_size / 2,
            "Meta-game Player 1 Strategy",
            ha='center', fontsize=axis_font_size, fontweight='bold', rotation=90)

    # Strategy labels on axes
    for b in range(n):
        ax.text(b * cell_size, margin * 0.45, short_names[b],
                ha='center', fontsize=axis_font_size - 1, fontstyle='italic')
    for a in range(n):
        ax.text(-margin * 0.45, -a * cell_size, short_names[a],
                ha='center', fontsize=axis_font_size - 1, fontstyle='italic')

    # Colormaps: Blues for P1, Reds for P2
    cmap_p1 = cm.Blues
    cmap_p2 = cm.Reds
    norm = colors.Normalize(vmin=0, vmax=1)

    def _draw_edges(edges, cmap, rad_sign):
        for src, dst, w in edges:
            color = cmap(norm(w))
            width = edge_width_base + edge_width_scale * w
            style = 'solid' if w > 1.0 else 'dashed' if w > 0.33 else 'dotted'

            nx.draw_networkx_edges(
                G, pos, edgelist=[(src, dst)],
                arrowstyle='-|>',
                arrowsize=edge_arrow_size,
                edge_color=[color],
                style=style,
                width=width,
                connectionstyle=f'arc3,rad={rad_sign * edge_rad}',
                ax=ax,
            )

    # P1 deviations curve one way, P2 the other
    _draw_edges(p1_edges, cmap_p1, rad_sign=1)
    _draw_edges(p2_edges, cmap_p2, rad_sign=-1)

    # Draw nodes
    node_colors = ['gold' if idx in sinks else 'lightblue' for idx in range(n2)]
    node_borders = ['darkred' if idx in sinks else 'navy' for idx in range(n2)]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_colors,
        linewidths=2,
        edgecolors=node_borders,
        ax=ax,
    )

    # Draw labels
    labels_drawn = nx.draw_networkx_labels(
        G, pos,
        labels=profile_labels,
        font_size=font_size,
        font_color='black',
        font_weight='bold',
        ax=ax,
    )
    for txt in labels_drawn.values():
        txt.set_path_effects([
            pe.Stroke(linewidth=2, foreground='white'),
            pe.Normal()
        ])


    out_path = os.path.join(save_dir, filename + '.png') if save_dir else filename + '.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved preference graph to: {out_path}")

    # Print summary
    total_edges = len(p1_edges) + len(p2_edges)
    print(f"\nPreference Graph Summary:")
    print(f"  Profiles (nodes): {n2}")
    print(f"  P1 edges (freq >= {edge_threshold}): {len(p1_edges)}")
    print(f"  P2 edges (freq >= {edge_threshold}): {len(p2_edges)}")
    print(f"  Total edges: {total_edges}")
    print(f"  Sinks (pure NE): {len(sinks)}")
    for idx in sorted(sinks):
        a, b = divmod(idx, n)
        print(f"    ({strategy_names[a]}, {strategy_names[b]})")
