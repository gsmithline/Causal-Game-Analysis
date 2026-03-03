import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from collections import defaultdict, Counter
import os


SHORT_MAP = {
    'openai_5.2_low': '5.2:L',
    'openai_5.2_none': '5.2:N',
    'openai_5.2_medium': '5.2:M',
    'ef1_bargainer': 'EF1',
    'psro': 'PSRO',
    'ppo': 'PPO',
    'mappo': 'MAPPO',
    'nfsp': 'NFSP',
    'walk': 'Walk',
    'soft': 'Soft',
    'tough': 'Tough',
}


def _load_and_aggregate(curb_results_path, min_minimal_rate=0.5, min_bootstraps=10):
    """Load curb results and aggregate bootstrap welfare per CURB set."""
    with open(curb_results_path, 'rb') as f:
        results = pickle.load(f)

    strategy_names = results['config']['strategy_names']
    bootstraps = results['bootstrap']['analyses']
    N = len(bootstraps)

    metrics_keys = ['uw', 'nw', 'nw_plus', 'ef1', 'ef1_plus']

    # Collect bootstrap welfare per CURB set
    boot_welfare = defaultdict(lambda: {k: [] for k in metrics_keys})
    boot_freq = Counter()
    boot_minimal_freq = Counter()

    for b in bootstraps:
        minimal_sets = set(frozenset(c) for c in b['minimal_curb_sets'])
        for cset, m in b['metrics'].items():
            boot_freq[cset] += 1
            for k in metrics_keys:
                boot_welfare[cset][k].append(m[k])
        for ms in minimal_sets:
            boot_minimal_freq[ms] += 1

    # Build ecology records for sets that are frequently minimal
    ecologies = []
    for cset, freq in boot_freq.items():
        minimal_count = boot_minimal_freq.get(cset, 0)
        minimal_rate = minimal_count / freq if freq > 0 else 0
        is_min = minimal_rate >= min_minimal_rate and minimal_count >= min_bootstraps

        names = sorted([SHORT_MAP.get(strategy_names[i], strategy_names[i])
                        for i in cset])
        label = ', '.join(names) if len(names) <= 3 else f'{", ".join(names[:2])}, +{len(names)-2}'

        rec = {
            'cset': cset,
            'size': len(cset),
            'freq': freq / N,
            'minimal_rate': minimal_rate,
            'is_minimal': is_min,
            'label': label,
            'names': names,
        }
        for k in metrics_keys:
            vals = boot_welfare[cset][k]
            rec[f'{k}_mean'] = np.mean(vals)
            rec[f'{k}_std'] = np.std(vals)
            rec[f'{k}_lo'] = np.percentile(vals, 5)
            rec[f'{k}_hi'] = np.percentile(vals, 95)
            rec[f'{k}_vals'] = np.array(vals)

        ecologies.append(rec)

    return ecologies, strategy_names, N


def _classify_ecology(rec, strategy_names):
    """Classify an ecology for coloring."""
    names_set = set(rec['names'])
    rl = {'PSRO', 'PPO', 'MAPPO', 'NFSP'}
    has_rl = bool(names_set & rl)
    has_nonrl = bool(names_set - rl - {'Walk', 'Soft', 'Tough'})

    if rec['size'] == 1 and names_set <= rl:
        return 'rl_singleton'
    if rec['size'] == 1 and 'Walk' in names_set:
        return 'degenerate'
    if rec['size'] == 1:
        return 'nonrl_singleton'
    if not has_rl:
        return 'nonrl_coalition'
    if has_rl and has_nonrl:
        return 'mixed_coalition'
    return 'rl_coalition'


ECOLOGY_COLORS = {
    'rl_singleton': '#666666',
    'degenerate': '#999999',
    'nonrl_singleton': '#e6550d',
    'nonrl_coalition': '#2171b5',
    'mixed_coalition': '#807dba',
    'rl_coalition': '#a1a1a1',
}

ECOLOGY_LABELS = {
    'rl_singleton': 'RL Singleton',
    'degenerate': 'Degenerate',
    'nonrl_singleton': 'Non-RL Singleton',
    'nonrl_coalition': 'Non-RL Coalition',
    'mixed_coalition': 'Mixed Coalition',
    'rl_coalition': 'RL Coalition',
}


def create_attractor_landscape(
    curb_results_path,
    filename='attractor_landscape',
    save_dir=None,
    dpi=300,
):
    """
    Figure 1: Attractor Landscape — bubble scatter with confidence ellipses.

    X = EF1 (fairness), Y = UW (welfare).
    Bubble size = bootstrap frequency.
    Color = ecology type.
    Ellipses = 90% bootstrap confidence.
    Only shows minimal CURB sets.
    """
    ecologies, strategy_names, N = _load_and_aggregate(curb_results_path)

    # Only minimal ecologies
    minimals = [e for e in ecologies if e['is_minimal']]
    minimals.sort(key=lambda e: e['freq'], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw confidence ellipses and bubbles
    for rec in minimals:
        cat = _classify_ecology(rec, strategy_names)
        color = ECOLOGY_COLORS[cat]

        x_mean = rec['ef1_mean']
        y_mean = rec['uw_mean']
        x_vals = rec['ef1_vals']
        y_vals = rec['uw_vals']

        # Confidence ellipse from bootstrap samples
        if len(x_vals) > 2:
            cov = np.cov(x_vals, y_vals)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # 90% chi2 with 2 dof = 4.605
            chi2_90 = 4.605
            width = 2 * np.sqrt(eigvals[1] * chi2_90)
            height = 2 * np.sqrt(eigvals[0] * chi2_90)
            angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

            ellipse = Ellipse(
                (x_mean, y_mean), width, height, angle=angle,
                facecolor=color, alpha=0.15, edgecolor=color,
                linewidth=1.5, linestyle='--', zorder=2,
            )
            ax.add_patch(ellipse)

        # Bubble: size proportional to frequency
        bubble_size = 100 + 800 * rec['freq']
        ax.scatter(
            x_mean, y_mean,
            s=bubble_size, c=color, edgecolors='white',
            linewidths=1.5, zorder=3, alpha=0.9,
        )

        # Label
        offset_x, offset_y = 8, 8
        # Nudge overlapping labels
        if rec['label'] == 'PPO':
            offset_y = -14
        if rec['label'] == 'MAPPO':
            offset_x = -45
            offset_y = -14

        ax.annotate(
            rec['label'],
            (x_mean, y_mean),
            textcoords='offset points',
            xytext=(offset_x, offset_y),
            fontsize=9, fontweight='bold', color=color,
            zorder=4,
        )

    ax.set_xlabel('Fairness (EF1 at Restricted Equilibrium)', fontsize=12)
    ax.set_ylabel('Utilitarian Welfare (UW at Restricted Equilibrium)', fontsize=12)
    ax.set_title('Attractor Landscape: Welfare vs Fairness of Minimal CURB Sets',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # Legend for ecology types
    from matplotlib.lines import Line2D
    seen_cats = set()
    handles = []
    for rec in minimals:
        cat = _classify_ecology(rec, strategy_names)
        if cat not in seen_cats:
            seen_cats.add(cat)
            handles.append(Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor=ECOLOGY_COLORS[cat], markersize=10,
                label=ECOLOGY_LABELS[cat],
            ))
    ax.legend(handles=handles, loc='upper left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(save_dir, filename + '.png') if save_dir else filename + '.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved attractor landscape: {out_path}')
    print(f'  Plotted {len(minimals)} minimal ecologies')


def create_raincloud_plots(
    curb_results_path,
    filename='curb_raincloud',
    save_dir=None,
    dpi=300,
):
    """
    Figure 2: Raincloud plots — bootstrap distributions per minimal ecology.

    5 panels (UW, NW, NW+, EF1, EF1+), each showing half-violin + strip
    for each minimal CURB set, sorted by median.
    """
    ecologies, strategy_names, N = _load_and_aggregate(curb_results_path)

    minimals = [e for e in ecologies if e['is_minimal']]

    metrics_keys = ['uw', 'nw', 'nw_plus', 'ef1', 'ef1_plus']
    metric_labels = {
        'uw': 'Utilitarian Welfare',
        'nw': 'Nash Welfare',
        'nw_plus': 'Nash Welfare+',
        'ef1': 'EF1',
        'ef1_plus': 'EF1+',
    }

    fig, axes = plt.subplots(1, 5, figsize=(30, 7))
    fig.suptitle(
        'Bootstrap Welfare Distributions of Minimal CURB Sets',
        fontsize=16, fontweight='bold', y=1.02,
    )

    for ax, k in zip(axes, metrics_keys):
        # Sort by median for this metric
        sorted_minimals = sorted(minimals, key=lambda e: np.median(e[f'{k}_vals']))

        labels_list = [e['label'] for e in sorted_minimals]
        data = [e[f'{k}_vals'] for e in sorted_minimals]
        cats = [_classify_ecology(e, strategy_names) for e in sorted_minimals]
        colors = [ECOLOGY_COLORS[c] for c in cats]
        freqs = [e['freq'] for e in sorted_minimals]

        positions = list(range(len(sorted_minimals)))

        # Half violin (right side only)
        parts = ax.violinplot(
            data, positions=positions, vert=False,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for i, body in enumerate(parts['bodies']):
            # Clip to right half (above center line)
            m = np.mean(body.get_paths()[0].vertices[:, 1])
            body.get_paths()[0].vertices[:, 1] = np.clip(
                body.get_paths()[0].vertices[:, 1], m, None
            )
            body.set_facecolor(colors[i])
            body.set_alpha(0.5)
            body.set_edgecolor(colors[i])

        # Jittered strip plot (below center line)
        rng = np.random.RandomState(42)
        for i, vals in enumerate(data):
            # Subsample if too many points
            if len(vals) > 100:
                idx = rng.choice(len(vals), 100, replace=False)
                plot_vals = vals[idx]
            else:
                plot_vals = vals
            jitter = rng.uniform(-0.25, -0.05, len(plot_vals))
            ax.scatter(
                plot_vals, i + jitter,
                s=8, alpha=0.3, color=colors[i],
                edgecolors='none', zorder=2,
            )

        # Median + IQR lines
        for i, vals in enumerate(data):
            med = np.median(vals)
            q25, q75 = np.percentile(vals, [25, 75])
            ax.plot([q25, q75], [i, i], color=colors[i], linewidth=2.5, zorder=3)
            ax.plot(med, i, 'D', color='white', markersize=5,
                    markeredgecolor=colors[i], markeredgewidth=1.5, zorder=4)

        # Labels with frequency
        label_strs = [f'{l}  ({f*100:.0f}%)' for l, f in zip(labels_list, freqs)]
        ax.set_yticks(positions)
        ax.set_yticklabels(label_strs, fontsize=9)
        ax.set_xlabel(metric_labels[k], fontsize=11)
        ax.grid(True, axis='x', alpha=0.2)

    plt.tight_layout()
    out_path = os.path.join(save_dir, filename + '.png') if save_dir else filename + '.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved raincloud plots: {out_path}')
    print(f'  Plotted {len(minimals)} minimal ecologies across 5 metrics')


def create_coalition_network(
    curb_results_path,
    filename='coalition_network',
    save_dir=None,
    dpi=300,
    min_edge_count=5,
):
    """
    Network graph of strategy coalition structure from minimal CURB sets.

    Nodes = strategies. Size = singleton min-CURB frequency.
    Edges = co-membership in minimal CURB sets, thickness = bootstrap frequency.
    Color = RL (grey) vs non-RL (orange/blue).
    """
    import networkx as nx
    import matplotlib.patheffects as pe

    with open(curb_results_path, 'rb') as f:
        results = pickle.load(f)

    strategy_names = results['config']['strategy_names']
    bootstraps = results['bootstrap']['analyses']
    n = len(strategy_names)
    N = len(bootstraps)

    # Collect frequencies
    singleton_freq = Counter()
    pair_freq = Counter()
    any_minimal_freq = Counter()  # how often i appears in ANY min-CURB

    for b in bootstraps:
        for m in b['minimal_curb_sets']:
            s = frozenset(m)
            if len(s) == 1:
                singleton_freq[list(s)[0]] += 1
            for i in s:
                any_minimal_freq[i] += 1
                for j in s:
                    if i < j:
                        pair_freq[(i, j)] += 1

    # Build graph
    G = nx.Graph()
    short_names = [SHORT_MAP.get(s, s) for s in strategy_names]

    # Classify strategies
    rl_set = {'PSRO', 'PPO', 'MAPPO', 'NFSP'}
    degenerate_set = {'Walk', 'Soft', 'Tough'}

    for i in range(n):
        sn = short_names[i]
        freq = any_minimal_freq.get(i, 0) / N
        if freq < 0.001 and singleton_freq.get(i, 0) == 0:
            continue  # skip strategies never in any min-CURB
        G.add_node(sn, freq=freq, singleton_freq=singleton_freq.get(i, 0) / N)

    for (i, j), count in pair_freq.items():
        if count >= min_edge_count:
            sn_i, sn_j = short_names[i], short_names[j]
            if sn_i in G and sn_j in G:
                G.add_edge(sn_i, sn_j, weight=count / N)

    # Layout: spring with some tuning
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    ax.set_title(
        'Coalition Network: Strategy Co-membership in Minimal CURB Sets',
        fontsize=15, fontweight='bold', pad=20,
    )

    # Node colors
    node_colors = []
    for node in G.nodes():
        if node in rl_set:
            node_colors.append('#666666')
        elif node in degenerate_set:
            node_colors.append('#bbbbbb')
        else:
            node_colors.append('#e6550d')

    # Node sizes: based on any_minimal_freq (how often in any min-CURB)
    node_sizes = []
    for node in G.nodes():
        f = G.nodes[node]['freq']
        node_sizes.append(300 + 3000 * f)

    # Edge widths and colors
    edge_widths = []
    edge_colors = []
    for u, v, d in G.edges(data=True):
        w = d['weight']
        edge_widths.append(1 + 15 * w)
        # Color by whether edge connects RL-nonRL or nonRL-nonRL
        u_rl = u in rl_set
        v_rl = v in rl_set
        if not u_rl and not v_rl:
            edge_colors.append('#2171b5')
        elif u_rl and v_rl:
            edge_colors.append('#999999')
        else:
            edge_colors.append('#807dba')

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.6,
        style='solid',
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='white',
        linewidths=2,
        alpha=0.9,
    )

    # Labels
    labels_drawn = nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=12,
        font_weight='bold',
        font_color='white',
    )
    for txt in labels_drawn.values():
        txt.set_path_effects([
            pe.Stroke(linewidth=3, foreground='black'),
            pe.Normal(),
        ])

    # Annotate singleton frequency below each node
    for node in G.nodes():
        x, y = pos[node]
        sf = G.nodes[node]['singleton_freq']
        if sf > 0:
            ax.text(
                x, y - 0.08, f'{sf*100:.0f}%',
                ha='center', va='top', fontsize=9,
                color='darkred', fontweight='bold',
                path_effects=[
                    pe.Stroke(linewidth=2, foreground='white'),
                    pe.Normal(),
                ],
            )

    # Edge weight labels
    for u, v, d in G.edges(data=True):
        w = d['weight']
        if w >= 0.01:
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            ax.text(
                x, y, f'{w*100:.1f}%',
                ha='center', va='center', fontsize=7,
                color='#333333', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.8),
            )

    # Legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#666666',
               markersize=12, label='RL Strategy'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e6550d',
               markersize=12, label='Non-RL Strategy'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#bbbbbb',
               markersize=12, label='Degenerate'),
        Line2D([0], [0], color='#2171b5', linewidth=3, label='Non-RL coalition'),
        Line2D([0], [0], color='#807dba', linewidth=3, label='Mixed coalition'),
        Line2D([0], [0], color='#999999', linewidth=3, label='RL coalition'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=10, framealpha=0.9)

    # Note
    ax.text(
        0.5, -0.02,
        'Node size = frequency in any minimal CURB set. '
        'Red % = singleton min-CURB frequency. '
        'Edge thickness = co-membership frequency.',
        ha='center', va='top', transform=ax.transAxes,
        fontsize=9, fontstyle='italic', color='#555555',
    )

    plt.tight_layout()
    out_path = os.path.join(save_dir, filename + '.png') if save_dir else filename + '.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved coalition network: {out_path}')
    print(f'  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')


def _compute_absorption(curb_results_path):
    """Compute absorption rates and basin sizes for each strategy and minimal CURB set."""
    with open(curb_results_path, 'rb') as f:
        results = pickle.load(f)

    strategy_names = results['config']['strategy_names']
    bootstraps = results['bootstrap']['analyses']
    n = len(strategy_names)
    N = len(bootstraps)

    # Per-strategy: in_curb, in_core counts
    in_curb = Counter()
    in_core = Counter()

    # Per minimal CURB set: how often is it the core of a larger set?
    # basin_size[m] = count of (bootstrap, CURB set) pairs where m is a core
    basin_count = Counter()  # just bootstrap count where m is a core of ANYTHING

    for b in bootstraps:
        minimal_sets = [frozenset(c) for c in b['minimal_curb_sets']]
        minimal_set_of = set(frozenset(c) for c in b['minimal_curb_sets'])

        # Track which minimal sets are active this bootstrap
        active_minimals = set()
        for m in minimal_sets:
            active_minimals.add(m)

        for c in [frozenset(cs) for cs in b['all_curb_sets']]:
            cores = [m for m in minimal_sets if m.issubset(c)]
            core_union = frozenset().union(*cores) if cores else frozenset()

            for i in c:
                in_curb[i] += 1
                if i in core_union:
                    in_core[i] += 1

        # Basin: each minimal set's basin = it appears as a core in this bootstrap
        for m in active_minimals:
            basin_count[m] += 1

    # Compute absorption rates
    absorption = {}
    for i in range(n):
        sn = SHORT_MAP.get(strategy_names[i], strategy_names[i])
        ic = in_curb[i]
        icr = in_core[i]
        absorption[sn] = {
            'in_curb': ic,
            'in_core': icr,
            'absorbed': ic - icr,
            'absorption_rate': (ic - icr) / ic if ic > 0 else 0,
        }

    return absorption, basin_count, strategy_names, N


def create_absorption_chart(
    curb_results_path,
    filename='absorption_rates',
    save_dir=None,
    dpi=300,
):
    """
    Bar chart: absorption rate per strategy.

    Absorption = how often a strategy is in a CURB set but NOT in any
    minimal core inside that set. High absorption = passenger/transient.
    """
    absorption, _, strategy_names, N = _compute_absorption(curb_results_path)

    # Sort by absorption rate
    sorted_strats = sorted(absorption.keys(), key=lambda s: absorption[s]['absorption_rate'])

    names = sorted_strats
    rates = [absorption[s]['absorption_rate'] for s in names]
    in_curb = [absorption[s]['in_curb'] for s in names]
    in_core = [absorption[s]['in_core'] for s in names]

    rl_set = {'PSRO', 'PPO', 'MAPPO', 'NFSP'}
    degenerate_set = {'Walk', 'Soft', 'Tough'}
    colors = []
    for s in names:
        if s in rl_set:
            colors.append('#666666')
        elif s in degenerate_set:
            colors.append('#bbbbbb')
        else:
            colors.append('#e6550d')

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(len(names)), rates, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11, fontweight='bold')
    ax.set_xlabel('Absorption Rate', fontsize=12)
    ax.set_title(
        'Strategy Absorption: Passengers vs Drivers\n'
        '(Fraction of CURB set appearances where strategy is NOT in the minimal core)',
        fontsize=13, fontweight='bold',
    )
    ax.set_xlim(0, 1.08)
    ax.grid(True, axis='x', alpha=0.2)

    # Annotate bars
    for i, (rate, s) in enumerate(zip(rates, names)):
        ic = absorption[s]['in_core']
        itot = absorption[s]['in_curb']
        ax.text(
            rate + 0.01, i,
            f'{rate*100:.1f}%  (core {ic}/{itot})',
            va='center', fontsize=9, color='#333333',
        )

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor='#666666', label='RL'),
        Patch(facecolor='#e6550d', label='Non-RL'),
        Patch(facecolor='#bbbbbb', label='Degenerate'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(save_dir, filename + '.png') if save_dir else filename + '.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved absorption chart: {out_path}')


def create_collapse_diagram(
    curb_results_path,
    filename='collapse_diagram',
    save_dir=None,
    dpi=300,
    min_freq_pct=15,
):
    """
    Hasse DAG: formation collapse from large CURB sets down to minimal cores.

    Nodes = CURB sets appearing in >= min_freq_pct% of bootstraps.
    Edges = direct subset (S ⊂ T, |T| = |S| + k, no intermediate CURB set).
    Layout = top-down by size.
    """
    import networkx as nx
    import matplotlib.patheffects as pe

    with open(curb_results_path, 'rb') as f:
        results = pickle.load(f)

    strategy_names = results['config']['strategy_names']
    bootstraps = results['bootstrap']['analyses']
    N = len(bootstraps)

    boot_freq = Counter()
    boot_minimal_freq = Counter()
    for b in bootstraps:
        for c in b['all_curb_sets']:
            boot_freq[frozenset(c)] += 1
        for m in b['minimal_curb_sets']:
            boot_minimal_freq[frozenset(m)] += 1

    # Filter to frequent sets
    threshold = min_freq_pct / 100
    frequent = {c for c, f in boot_freq.items() if f / N >= threshold}

    # Also include minimal sets that appear >= 5%
    for c, f in boot_minimal_freq.items():
        if f / N >= 0.05:
            frequent.add(c)

    frequent = sorted(frequent, key=lambda c: (len(c), sorted(c)))

    def set_label(cset):
        names = sorted([SHORT_MAP.get(strategy_names[i], strategy_names[i]) for i in cset])
        if len(names) <= 4:
            return ', '.join(names)
        return ', '.join(names[:3]) + f'\n+{len(names)-3} more'

    # Build Hasse diagram: edge from A to B if A ⊂ B and no C with A ⊂ C ⊂ B in frequent
    G = nx.DiGraph()
    for c in frequent:
        freq = boot_freq[c] / N
        mrate = boot_minimal_freq.get(c, 0) / boot_freq[c] if boot_freq[c] > 0 else 0
        G.add_node(
            id(c), label=set_label(c), size=len(c), freq=freq,
            is_minimal=mrate >= 0.5, cset=c,
        )

    node_map = {id(c): c for c in frequent}
    node_ids = list(node_map.keys())

    for i, nid_a in enumerate(node_ids):
        a = node_map[nid_a]
        for j, nid_b in enumerate(node_ids):
            if i == j:
                continue
            b = node_map[nid_b]
            if a < b:  # a is proper subset of b
                # Check no intermediate: no c in frequent with a ⊂ c ⊂ b
                has_intermediate = False
                for k, nid_c in enumerate(node_ids):
                    if k == i or k == j:
                        continue
                    c = node_map[nid_c]
                    if a < c < b:
                        has_intermediate = True
                        break
                if not has_intermediate:
                    # Edge label: strategies added
                    added = b - a
                    added_names = sorted([SHORT_MAP.get(strategy_names[s], strategy_names[s])
                                          for s in added])
                    G.add_edge(nid_a, nid_b, added=', '.join(added_names))

    # Layout: position by size (y) with horizontal spread
    size_groups = defaultdict(list)
    for nid in node_ids:
        size_groups[G.nodes[nid]['size']].append(nid)

    pos = {}
    max_size = max(size_groups.keys())
    for size, nids in size_groups.items():
        y = -size  # bigger sets at the top (inverted so minimal at bottom)
        n_in_row = len(nids)
        for idx, nid in enumerate(sorted(nids, key=lambda n: sorted(node_map[n]))):
            x = (idx - (n_in_row - 1) / 2) * 2.0
            pos[nid] = (x, y)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    ax.set_title(
        'Formation Collapse: CURB Set Hierarchy',
        fontsize=16, fontweight='bold', pad=20,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#aaaaaa',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=15,
        width=1.5,
        connectionstyle='arc3,rad=0.05',
    )

    # Draw nodes
    node_colors = []
    node_sizes = []
    for nid in G.nodes():
        freq = G.nodes[nid]['freq']
        is_min = G.nodes[nid]['is_minimal']
        node_sizes.append(400 + 2000 * freq)
        if is_min:
            node_colors.append('crimson')
        else:
            node_colors.append('steelblue')

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='white',
        linewidths=1.5,
        alpha=0.9,
    )

    # Node labels: set name + frequency
    label_dict = {}
    for nid in G.nodes():
        lbl = G.nodes[nid]['label']
        freq = G.nodes[nid]['freq']
        label_dict[nid] = f'{lbl}\n{freq*100:.0f}%'

    labels_drawn = nx.draw_networkx_labels(
        G, pos, labels=label_dict, ax=ax,
        font_size=7, font_weight='bold', font_color='white',
    )
    for txt in labels_drawn.values():
        txt.set_path_effects([
            pe.Stroke(linewidth=2, foreground='black'),
            pe.Normal(),
        ])

    # Edge labels (which strategy was added)
    edge_labels = {(u, v): d['added'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=6, font_color='#555555',
        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.7),
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson',
               markersize=12, label='Minimal CURB (attractor)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markersize=12, label='Non-minimal CURB (formation)'),
    ]
    ax.legend(handles=legend_items, loc='upper left', fontsize=10, framealpha=0.9)

    ax.text(
        0.5, -0.01,
        'Node size = bootstrap frequency. Edges = direct subset relation. '
        'Labels show which strategies are added.',
        ha='center', va='top', transform=ax.transAxes,
        fontsize=9, fontstyle='italic', color='#555555',
    )

    plt.tight_layout()
    out_path = os.path.join(save_dir, filename + '.png') if save_dir else filename + '.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved collapse diagram: {out_path}')
    print(f'  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')


# ---------------------------------------------------------------------------
# CURB-selected Banzhaf visualizations
# ---------------------------------------------------------------------------

# Match the existing Banzhaf beeswarm style from visualize_analysis.py
POS_COLOR = "#d62728"   # SHAP-style red for positive
NEG_COLOR = "#1f77b4"   # SHAP-style blue for negative

# Normalization: maps metric -> divisor (same as visualize_analysis.L3_NORM)
_MAX_UW = 805.9
_MAX_NW = 378.7
_MAX_NW_PLUS = 81.7
CURB_L3_NORM = {
    'uw': _MAX_UW,
    'nw': _MAX_NW,
    'nw_plus': _MAX_NW_PLUS,
    'ef1': 1.0,
    'ef1_plus': 1.0,
}

METRIC_LABELS = {
    'uw': 'Utilitarian Welfare',
    'nw': 'Nash Welfare',
    'nw_plus': 'NW+ Welfare',
    'ef1': 'EF1 Frequency',
    'ef1_plus': 'EF1+ Frequency',
}


def _load_curb_banzhaf(banzhaf_path):
    """Load curb_banzhaf_results.pkl."""
    with open(banzhaf_path, 'rb') as f:
        data = pickle.load(f)
    return (
        data.get('point_estimate', {}),
        data.get('bootstrap'),
        data['config']['strategy_names'],
    )


def _dn_curb(strategy):
    """Display name for a strategy (uses SHORT_MAP)."""
    return SHORT_MAP.get(strategy, strategy)


def _curb_label(cset, strategy_names):
    """Short label for a CURB set."""
    names = sorted(
        [SHORT_MAP.get(strategy_names[i], strategy_names[i]) for i in cset]
    )
    if len(names) <= 4:
        return '{' + ', '.join(names) + '}'
    return '{' + ', '.join(names[:3]) + f', +{len(names)-3}' + '}'


def create_curb_banzhaf_beeswarm(
    banzhaf_path,
    regular_results_path=None,
    metrics=None,
    filename='curb_banzhaf_beeswarm',
    save_dir=None,
    dpi=200,
):
    """Beeswarm plot of CURB-Banzhaf per strategy, matching existing L3 style.

    One panel per metric. Each dot = one bootstrap sample's CURB-Banzhaf
    for that strategy. Red = positive, blue = negative. Sorted by mean
    absolute value. Normalized same as regular Banzhaf beeswarm.

    Args:
        banzhaf_path: Path to curb_banzhaf_results.pkl.
        regular_results_path: If provided, path to iterative_analysis_results.pkl
            to load per-bootstrap raw CURB-Banzhaf values. Otherwise uses the
            aggregated summary from the pkl.
        metrics: Metrics to plot. Defaults to ['uw', 'nw', 'nw_plus', 'ef1', 'ef1_plus'].
        filename: Output filename stem.
        save_dir: Directory to save figure.
        dpi: Figure resolution (default 200 to match existing plots).
    """
    point_cb, boot_agg, strategy_names = _load_curb_banzhaf(banzhaf_path)
    if metrics is None:
        metrics = ['uw', 'nw', 'nw_plus', 'ef1', 'ef1_plus']

    # Check for raw per-bootstrap values (needed for true beeswarm)
    has_raw = (boot_agg is not None
               and 'raw_values' in boot_agg
               and boot_agg['raw_values'])

    n_panels = len(metrics)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        div = CURB_L3_NORM[metric]
        scale = 100.0

        if has_raw:
            # True beeswarm: scatter each bootstrap sample's value
            raw_vals = boot_agg['raw_values'][metric]
            strats = list(strategy_names)

            # Sort by mean absolute value
            strats = sorted(
                strats,
                key=lambda s: np.mean(np.abs(
                    np.array(raw_vals.get(s, [0])) / div * scale)),
                reverse=True,
            )

            y_positions = np.arange(len(strats))
            rng = np.random.default_rng(42)
            for i, s in enumerate(strats):
                vals = np.array(raw_vals.get(s, [0])) / div * scale
                jitter = rng.uniform(-0.2, 0.2, size=len(vals))
                dot_colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]
                ax.scatter(vals, i + jitter, c=dot_colors, s=40, alpha=0.7,
                           edgecolors='black', linewidths=0.3, zorder=3)

            ax.axvline(0, color='black', lw=0.8, ls='--')
            ax.set_yticks(y_positions)
            ax.set_yticklabels([_dn_curb(s) for s in strats])
            ax.invert_yaxis()

        elif boot_agg is not None:
            # Fallback: mean + CI (no raw data)
            strats = list(strategy_names)
            means = []
            for s in strats:
                stats = boot_agg['banzhaf'][metric].get(s, {})
                means.append(stats.get('mean', 0) / div * scale)
            order = sorted(range(len(strats)),
                           key=lambda i: abs(means[i]), reverse=True)
            strats = [strats[i] for i in order]
            means = [means[i] for i in order]

            y_positions = np.arange(len(strats))
            colors = [POS_COLOR if m >= 0 else NEG_COLOR for m in means]
            ax.barh(y_positions, means, color=colors,
                    edgecolor='white', alpha=0.85)
            ax.axvline(0, color='black', lw=0.8, ls='--')
            ax.set_yticks(y_positions)
            ax.set_yticklabels([_dn_curb(s) for s in strats])
            ax.invert_yaxis()

        else:
            # Point estimate only
            strats = list(strategy_names)
            vals = [point_cb['banzhaf'][metric].get(s, 0) / div * scale
                    for s in strats]
            order = sorted(range(len(strats)),
                           key=lambda i: abs(vals[i]), reverse=True)
            strats = [strats[i] for i in order]
            vals = [vals[i] for i in order]

            y_positions = np.arange(len(strats))
            colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]
            ax.barh(y_positions, vals, color=colors,
                    edgecolor='white', alpha=0.85)
            ax.axvline(0, color='black', lw=0.8, ls='--')
            ax.set_yticks(y_positions)
            ax.set_yticklabels([_dn_curb(s) for s in strats])
            ax.invert_yaxis()

        unit = 'pp' if metric in ('ef1', 'ef1_plus') else '% of max'
        ax.set_title(f'CURB-Banzhaf – {METRIC_LABELS[metric]}')
        ax.set_xlabel(f'CURB-Banzhaf Value ({unit})')

    fig.tight_layout()
    out_path = os.path.join(save_dir, filename + '.png') if save_dir else filename + '.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved CURB-Banzhaf beeswarm: {out_path}')


def create_marginal_heatmap(
    banzhaf_path,
    metric='uw',
    filename='curb_marginal_heatmap',
    save_dir=None,
    dpi=300,
):
    """Heatmap of per-coalition marginal contributions (point estimate).

    Rows = strategies, columns = CURB sets. Cell = v(C) - v(C \\ {i}).
    Shows which ecologies each strategy matters most in.

    Args:
        banzhaf_path: Path to curb_banzhaf_results.pkl.
        metric: Which metric to show.
        filename: Output filename stem.
        save_dir: Directory to save figure.
        dpi: Figure resolution.
    """
    point_cb, _, strategy_names = _load_curb_banzhaf(banzhaf_path)

    marginals = point_cb.get('marginals', [])
    if not marginals:
        print('No marginal records in point estimate. Skipping heatmap.')
        return

    # Collect unique CURB sets from marginals
    curb_sets = sorted(
        set(m['curb_set'] for m in marginals),
        key=lambda c: (len(c), sorted(c)),
    )

    # Build matrix: strategy x CURB set
    strat_order = list(strategy_names)
    mat = np.full((len(strat_order), len(curb_sets)), np.nan)

    for rec in marginals:
        s_idx = strat_order.index(rec['strategy'])
        c_idx = curb_sets.index(rec['curb_set'])
        mat[s_idx, c_idx] = rec.get(metric, 0)

    fig, ax = plt.subplots(figsize=(max(12, len(curb_sets) * 0.8), 7))

    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    # Center colormap at 0
    vmax = np.nanmax(np.abs(mat))
    im.set_clim(-vmax, vmax)

    ax.set_yticks(range(len(strat_order)))
    ax.set_yticklabels(
        [SHORT_MAP.get(s, s) for s in strat_order],
        fontsize=10, fontweight='bold',
    )

    col_labels = [_curb_label(c, strategy_names) for c in curb_sets]
    ax.set_xticks(range(len(curb_sets)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=60, ha='right')

    plt.colorbar(im, ax=ax, label=f'Marginal: v(C) - v(C\\{{i}})  [{metric}]')

    ax.set_title(
        f'Per-Coalition Marginal Contributions ({METRIC_LABELS.get(metric, metric)})',
        fontsize=13, fontweight='bold',
    )

    plt.tight_layout()
    out_path = os.path.join(save_dir, filename + '.png') if save_dir else filename + '.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved marginal heatmap: {out_path}')


def create_curb_vs_regular_banzhaf(
    banzhaf_path,
    regular_results_path,
    metric='uw',
    filename='curb_vs_regular_banzhaf',
    save_dir=None,
    dpi=300,
):
    """Scatter: CURB-Banzhaf vs regular Banzhaf per strategy.

    Each point is a strategy. X = regular Banzhaf (all 2^n coalitions),
    Y = CURB-Banzhaf (only CURB coalitions). Diagonal = agreement.

    Args:
        banzhaf_path: Path to curb_banzhaf_results.pkl.
        regular_results_path: Path to iterative_analysis_results.pkl
            (contains regular L3 Banzhaf in bootstrap samples).
        metric: Which metric to compare.
        filename: Output filename stem.
        save_dir: Directory to save figure.
        dpi: Figure resolution.
    """
    point_cb, boot_cb, strategy_names = _load_curb_banzhaf(banzhaf_path)

    # Load regular Banzhaf from iterative analysis
    with open(regular_results_path, 'rb') as f:
        reg_data = pickle.load(f)

    raw_samples = reg_data.get('raw', [])
    banzhaf_key = f'banzhaf_{metric}'

    # Compute regular Banzhaf mean across bootstraps
    reg_banzhaf = {s: [] for s in strategy_names}
    for sample in raw_samples:
        l3 = sample.get('l3', {})
        if l3 and banzhaf_key in l3:
            for s, v in l3[banzhaf_key].items():
                if s in reg_banzhaf:
                    reg_banzhaf[s].append(v)

    if not any(reg_banzhaf.values()):
        print(f'No regular Banzhaf data found for {metric}. Skipping.')
        return

    reg_means = {s: np.mean(v) if v else 0 for s, v in reg_banzhaf.items()}

    # CURB-Banzhaf means
    if boot_cb is not None:
        curb_means = {
            s: boot_cb['banzhaf'][metric].get(s, {}).get('mean', 0)
            for s in strategy_names
        }
    else:
        curb_means = {
            s: point_cb['banzhaf'][metric].get(s, 0) for s in strategy_names
        }

    fig, ax = plt.subplots(figsize=(8, 8))

    rl_set = {'PSRO', 'PPO', 'MAPPO', 'NFSP'}
    for s in strategy_names:
        x = reg_means[s]
        y = curb_means[s]
        sn = SHORT_MAP.get(s, s)
        color = '#666666' if sn in rl_set else '#e6550d'
        ax.scatter(x, y, s=120, c=color, edgecolors='white',
                   linewidths=1.5, zorder=3, alpha=0.9)
        ax.annotate(sn, (x, y), textcoords='offset points',
                    xytext=(6, 6), fontsize=9, fontweight='bold',
                    color=color, zorder=4)

    # Diagonal line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, '--', color='grey', alpha=0.5, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(f'Regular Banzhaf ({METRIC_LABELS.get(metric, metric)})',
                  fontsize=12)
    ax.set_ylabel(f'CURB-Banzhaf ({METRIC_LABELS.get(metric, metric)})',
                  fontsize=12)
    ax.set_title(
        f'CURB-Selected vs Regular Banzhaf\n({METRIC_LABELS.get(metric, metric)})',
        fontsize=14, fontweight='bold',
    )
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = os.path.join(save_dir, filename + '.png') if save_dir else filename + '.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved CURB vs regular Banzhaf scatter: {out_path}')


if __name__ == '__main__':
    from pathlib import Path
    results_path = Path(__file__).parent.parent / 'data' / 'analysis' / 'curb_results.pkl'
    banzhaf_path = Path(__file__).parent.parent / 'data' / 'analysis' / 'curb_banzhaf_results.pkl'
    regular_path = Path(__file__).parent.parent / 'data' / 'analysis' / 'iterative_analysis_results.pkl'
    output_dir = Path(__file__).parent.parent / 'data' / 'analysis'

    create_attractor_landscape(str(results_path), save_dir=str(output_dir))
    create_raincloud_plots(str(results_path), save_dir=str(output_dir))
    create_coalition_network(str(results_path), save_dir=str(output_dir))
    create_absorption_chart(str(results_path), save_dir=str(output_dir))
    create_collapse_diagram(str(results_path), save_dir=str(output_dir))

    # CURB-Banzhaf visualizations (only if results exist)
    if banzhaf_path.exists():
        create_curb_banzhaf_beeswarm(str(banzhaf_path), save_dir=str(output_dir))
        create_marginal_heatmap(str(banzhaf_path), save_dir=str(output_dir))
        if regular_path.exists():
            create_curb_vs_regular_banzhaf(
                str(banzhaf_path), str(regular_path), save_dir=str(output_dir),
            )
