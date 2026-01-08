import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from collections import defaultdict
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import networkx as nx
import matplotlib.patheffects as pe
from matplotlib import cm, colors
from matplotlib.patches import FancyArrowPatch
import re

def create_average_best_response_graph(
    avg_br_matrix,
    agent_names,
    filename='average_best_response_graph',
    save_dir=None,
    figsize=(12, 12),
    dpi=300
):
    """
    Create a best-response graph using NetworkX + Matplotlib with enhanced label readability.
    Saves a PNG to disk.

    Args:
        avg_br_matrix: np.ndarray, shape (n,n)
            avg_br_matrix[j, i] = frequency with which agent j is BR to agent i.
        agent_names: List[str], length n
            Names of the agents, in the same order as avg_br_matrix indices.
        filename: str
            Base name (no extension) for the output file.
        save_dir: Optional[str]
            Directory to save the PNG. If None, uses cwd.
        figsize: tuple
            Figure size.
        dpi: int
            Resolution for saving.
    """
    G = nx.DiGraph()
    G.add_nodes_from(agent_names)

    br_edges = []
    n = len(agent_names)
    for i, ag_i in enumerate(agent_names): 
        for j, ag_j in enumerate(agent_names): 
            p_val = float(avg_br_matrix[j, i]) 
            if p_val >= 0.01:
                br_edges.append((ag_i, ag_j, p_val)) 

    for u_node, v_node, p_weight in br_edges:
        G.add_edge(u_node, v_node, weight=p_weight)

    high_freq_in_degrees = defaultdict(int)
    for _, target_node, p_val in br_edges: 
        if p_val >= 0.33:
            high_freq_in_degrees[target_node] += 1
    
    max_high_freq_in_degree_nodes = []
    max_high_freq_in_degree = 0
    if high_freq_in_degrees: 
        max_high_freq_in_degree = max(high_freq_in_degrees.values())
        max_high_freq_in_degree_nodes = [node for node, degree in high_freq_in_degrees.items() if degree == max_high_freq_in_degree]
    
    print(f"\n--- In-degree Analysis (count of incoming BRs, min p => 0.33) ---")
    if max_high_freq_in_degree_nodes:
        print(f"Node(s) with highest in-degree (p => 0.5) ({max_high_freq_in_degree}): {', '.join(max_high_freq_in_degree_nodes)}")
    else:
        print("No nodes with in-degrees (p => 0.33) found.")

    # 2. Node with the highest sum of empirical frequencies of their in-degrees (min p >= 0.01)
    weighted_in_degrees = dict(G.in_degree(weight='weight'))
    max_weighted_in_degree_nodes = []
    max_weighted_in_degree_sum = 0.0
    if weighted_in_degrees: # Check if dictionary is not empty
        max_weighted_in_degree_sum = max(weighted_in_degrees.values())
        # Handle potential floating point inaccuracies when comparing for max
        max_weighted_in_degree_nodes = [node for node, degree_sum in weighted_in_degrees.items() if abs(degree_sum - max_weighted_in_degree_sum) < 1e-9]

    print(f"\n--- Sum of In-edge Frequencies Analysis (sum of p for incoming BRs, min p >= 0.01) ---")
    if max_weighted_in_degree_nodes:
        print(f"Node(s) with highest sum of in-edge frequencies ({max_weighted_in_degree_sum:.4f}): {', '.join(max_weighted_in_degree_nodes)}")
        # print("All sums of in-edge frequencies (nthode: sum_p):")
        # for node, degree_sum in sorted(weighted_in_degrees.items(), key=lambda item: item[1], reverse=True):
        #     print(f"  {node}: {degree_sum:.4f}")
    else:
        print("No nodes with weighted in-degrees found (or no qualifying edges)." )
    print("--- End of Degree Analysis ---\n")
    # --- END REQUESTED ANALYSIS ---

    # 3) Circular layout
    pos = nx.circular_layout(G)

    # 4) Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # 5) Colormap
    cmap = cm.Blues
    norm = colors.Normalize(vmin=0, vmax=1)

    # Prepare reciprocal lookup
    br_set = {(u, v) for u, v, _ in br_edges}

    # 6) Draw edges - separate self-loops from normal edges
    non_self_edges = [(u, v, p) for u, v, p in br_edges if u != v]
    self_loop_edges = [(u, v, p) for u, v, p in br_edges if u == v]
    
    # Draw non-self edges first
    for u, v, p in non_self_edges:
        color = cmap(norm(p))
        width = 3 + 6 * p
        style = 'solid' if p > 1.0 else 'dashed' if p > 0.33 else 'dotted'
        arrow_size = 70

        if (v, u) in br_set:
            rad = 0.15 if u < v else -0.15
            connectionstyle = f'arc3,rad={rad}'
        else:
            connectionstyle = 'arc3,rad=0.0'
        
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               arrowstyle='-|>',
                               arrowsize=arrow_size,
                               edge_color=[color],
                               style=style,
                               width=width,
                               connectionstyle=connectionstyle)

    # 7) Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_size=6500,
                           node_color='cyan',
                           linewidths=2.5,
                           edgecolors='navy')

    # 8) Draw self-loops AFTER nodes with large radius so they're visible
    for u, v, p in self_loop_edges:
        color = cmap(norm(p))
        width = 3 + 6 * p
        style = 'solid' if p > 1.0 else 'dashed' if p > 0.33 else 'dotted'
        
        # Use FancyArrowPatch for better self-loop control
        x, y = pos[u]
        arrow = FancyArrowPatch(
            posA=(x - 0.08, y - 0.12),
            posB=(x + 0.08, y - 0.12),
            connectionstyle="arc3,rad=1.5",
            arrowstyle="-|>",
            mutation_scale=30,
            color=color,
            linewidth=width,
            linestyle='solid' if style == 'solid' else ('dashed' if style == 'dashed' else 'dotted'),
            zorder=1,
        )
        ax.add_patch(arrow)

    # 9) Draw labels with halo
    labels = {}
    pat = re.compile(r'(-c-\d+)$')
    for name in agent_names:
        m = pat.search(name)
        if m:
            head = name[:m.start()]
            tail = m.group(1)
            # two lines: "head" on top, "-c-#" underneath
            labels[name] = f"{head}\n{tail}"
        else:
            labels[name] = name

    labels = nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=18,
        font_color='black',
        font_weight='bold',
        font_family='DejaVu Sans'
    )
    for txt in labels.values():
        txt.set_path_effects([
            pe.Stroke(linewidth=4, foreground='white'),
            pe.Normal()
        ])

    # 9) Save figure
    out_path = os.path.join(save_dir, filename + '.png') if save_dir else filename + '.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
