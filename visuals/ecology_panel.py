"""Multi-panel restricted game × bootstrap figure.

Panel layout (shared y-axis):
  [A: Strategy membership] [B: Existence heatmap] [C: Welfare heatmap] [D: Welfare dot+CI]

Reads ecology_matrix.pkl produced by evaluation/curb_ecology_matrix.py.
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list

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

# Strategy display order (RL first, then non-RL, then degenerate)
STRAT_DISPLAY_ORDER = ['PSRO', 'PPO', 'NFSP', 'MAPPO', '5.2:L', '5.2:N', 'EF1', 'Soft', 'Tough', 'Walk']

STRAT_COLORS = {
    'PSRO': '#555555', 'PPO': '#555555', 'NFSP': '#555555', 'MAPPO': '#555555',
    '5.2:L': '#e6550d', '5.2:N': '#e6550d', 'EF1': '#e6550d',
    'Soft': '#bbbbbb', 'Tough': '#bbbbbb', 'Walk': '#bbbbbb',
}


def _restricted_game_label(eco, strategy_names):
    """Short label for a restricted game (CURB set)."""
    names = sorted(
        [SHORT_MAP.get(strategy_names[i], strategy_names[i]) for i in eco]
    )
    if len(names) <= 4:
        return '{' + ', '.join(names) + '}'
    return '{' + ', '.join(names[:3]) + ', +' + str(len(names) - 3) + '}'


def _select_ecologies(ecologies, matrices, minimal_flags, strategy_names,
                       n_show=18, min_freq=0.30):
    """Select a diverse set of ecologies to display.

    Includes minimal singletons (PPO, PSRO), the full game, and
    top-frequency ecologies above min_freq.  Avoids low-frequency
    singletons that add noise.
    """
    mat = matrices['uw']
    exists = ~np.isnan(mat)
    freqs = exists.sum(axis=1) / mat.shape[1]

    selected_idx = []
    eco_to_idx = {eco: i for i, eco in enumerate(ecologies)}

    # Minimal singletons with decent frequency (PPO, PSRO)
    for i, eco in enumerate(ecologies):
        if (len(eco) == 1
                and minimal_flags.get(eco, 0) >= 0.5
                and freqs[i] >= 0.50):
            selected_idx.append(i)

    # Full game
    full = frozenset(range(len(strategy_names)))
    if full in eco_to_idx:
        selected_idx.append(eco_to_idx[full])

    # Fill with top-frequency ecologies above min_freq
    freq_order = np.argsort(-freqs)
    for i in freq_order:
        if len(selected_idx) >= n_show:
            break
        if i not in selected_idx and freqs[i] >= min_freq:
            selected_idx.append(i)

    # Sort: minimal singletons first, then by size asc, freq desc
    def sort_key(i):
        eco = ecologies[i]
        is_min = minimal_flags.get(eco, 0) >= 0.5
        return (0 if is_min else 1, len(eco), -freqs[i])

    selected_idx.sort(key=sort_key)
    return selected_idx


def _sort_bootstraps(exist_matrix):
    """Sort bootstrap columns by hierarchical clustering for visual coherence."""
    # Transpose: cluster on columns (bootstraps)
    # Use a subset of rows for speed if needed
    data = exist_matrix.astype(np.float64).T  # (B, E)
    Z = linkage(data, method='average', metric='hamming')
    return leaves_list(Z)


def create_ecology_panel(
    ecology_matrix_path,
    metric='uw',
    filename='ecology_panel',
    save_dir=None,
    dpi=200,
    n_show=18,
):
    """Create the multi-panel ecology × bootstrap figure."""

    with open(ecology_matrix_path, 'rb') as f:
        data = pickle.load(f)

    ecologies = data['ecologies']
    matrices = data['matrices']
    minimal_flags = data['minimal_flags']
    strategy_names = data['config']['strategy_names']
    n_bootstraps = data['config']['n_bootstraps']

    # Select and order ecologies
    sel_idx = _select_ecologies(
        ecologies, matrices, minimal_flags, strategy_names, n_show=n_show,
    )
    n_eco = len(sel_idx)

    # Build display data
    short_names = [SHORT_MAP.get(s, s) for s in strategy_names]
    strat_order = [short_names.index(s) for s in STRAT_DISPLAY_ORDER
                   if s in short_names]

    # Existence matrix for selected ecologies
    uw_mat = matrices[metric]
    exist_sel = (~np.isnan(uw_mat[sel_idx])).astype(np.float64)  # (n_eco, B)
    welfare_sel = uw_mat[sel_idx]  # (n_eco, B)

    # Sort bootstraps by clustering
    boot_order = _sort_bootstraps(exist_sel)
    exist_sorted = exist_sel[:, boot_order]
    welfare_sorted = welfare_sel[:, boot_order]

    # Labels and metadata
    labels = []
    freqs = []
    for i in sel_idx:
        eco = ecologies[i]
        labels.append(_restricted_game_label(eco, strategy_names))
        freqs.append(exist_sel[sel_idx.index(i)].sum() / n_bootstraps
                     if i in sel_idx else 0)
    freqs = exist_sel.sum(axis=1) / n_bootstraps

    # --- Figure layout ---
    fig = plt.figure(figsize=(22, 0.55 * n_eco + 2.5),
                     constrained_layout=True)
    # Gridspec: [membership | existence | welfare | dot plot]
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1.0, 4, 4, 1.8],
        wspace=0.03,
    )

    y_positions = np.arange(n_eco)

    # === Panel A: Strategy membership ===
    ax_mem = fig.add_subplot(gs[0, 0])

    # Build membership matrix: (n_eco, n_strats) in display order
    mem_matrix = np.zeros((n_eco, len(strat_order)))
    for row, ei in enumerate(sel_idx):
        eco = ecologies[ei]
        for col, si in enumerate(strat_order):
            if si in eco:
                mem_matrix[row, col] = 1.0

    # Color by strategy type
    mem_rgb = np.ones((n_eco, len(strat_order), 4))  # RGBA, default white
    for row in range(n_eco):
        for col, si in enumerate(strat_order):
            if mem_matrix[row, col] > 0:
                sname = short_names[si]
                hex_color = STRAT_COLORS.get(sname, '#333333')
                rgb = mcolors.to_rgba(hex_color)
                mem_rgb[row, col] = rgb
            else:
                mem_rgb[row, col] = (0.95, 0.95, 0.95, 1.0)

    ax_mem.imshow(mem_rgb, aspect='auto', interpolation='nearest')
    ax_mem.set_xticks(range(len(strat_order)))
    ax_mem.set_xticklabels(
        [STRAT_DISPLAY_ORDER[i] for i in range(len(strat_order))],
        fontsize=7, rotation=55, ha='right',
    )
    ax_mem.set_yticks(y_positions)
    # Labels with frequency and minimal marker
    y_labels = []
    for row, ei in enumerate(sel_idx):
        eco = ecologies[ei]
        is_min = minimal_flags.get(eco, 0) >= 0.5
        marker = ' *' if is_min else ''
        y_labels.append(f'{labels[row]}  {freqs[row]*100:.0f}%{marker}')
    ax_mem.set_yticklabels(y_labels, fontsize=8, fontfamily='monospace')
    ax_mem.set_title('Strategy membership', fontsize=9, fontweight='bold')
    ax_mem.tick_params(axis='both', length=0)

    # Grid lines between rows
    for y in np.arange(-0.5, n_eco, 1):
        ax_mem.axhline(y, color='white', linewidth=0.5)
    for x in np.arange(-0.5, len(strat_order), 1):
        ax_mem.axvline(x, color='white', linewidth=0.5)

    # === Panel B: Existence heatmap ===
    ax_exist = fig.add_subplot(gs[0, 1])

    # Custom colormap: light gray for absent, dark blue for present
    exist_cmap = mcolors.ListedColormap(['#f0f0f0', '#2171b5'])
    ax_exist.imshow(
        exist_sorted, aspect='auto', cmap=exist_cmap,
        interpolation='nearest', vmin=0, vmax=1,
    )
    ax_exist.set_yticks([])
    ax_exist.set_xlabel('Bootstrap samples (clustered)', fontsize=8)
    ax_exist.set_title('Existence across bootstraps', fontsize=9,
                       fontweight='bold')
    ax_exist.set_xticks([0, 250, 500, 750, 999])
    ax_exist.set_xticklabels(['0', '250', '500', '750', '1000'], fontsize=7)
    ax_exist.tick_params(axis='y', length=0)

    # === Panel C: Welfare heatmap (per-row z-score) ===
    ax_welf = fig.add_subplot(gs[0, 2])

    # Per-row z-score: highlights within-ecology variation
    welfare_z = np.full_like(welfare_sorted, np.nan)
    for row in range(n_eco):
        vals = welfare_sorted[row]
        present = ~np.isnan(vals)
        if present.sum() > 1:
            mu = np.nanmean(vals)
            sigma = np.nanstd(vals)
            if sigma > 1e-10:
                welfare_z[row, present] = (vals[present] - mu) / sigma
            else:
                welfare_z[row, present] = 0.0

    welfare_z_masked = np.ma.masked_invalid(welfare_z)
    cmap_welf = plt.cm.RdYlBu_r.copy()
    cmap_welf.set_bad('#f0f0f0')

    vabs = 2.5  # clip z-scores for visual contrast
    im = ax_welf.imshow(
        welfare_z_masked, aspect='auto', cmap=cmap_welf,
        interpolation='nearest', vmin=-vabs, vmax=vabs,
    )
    ax_welf.set_yticks([])
    ax_welf.set_xlabel('Bootstrap samples (same order)', fontsize=8)
    ax_welf.set_title(f'Welfare variation ({metric.upper()}, row z-score)',
                      fontsize=9, fontweight='bold')
    ax_welf.set_xticks([0, 250, 500, 750, 999])
    ax_welf.set_xticklabels(['0', '250', '500', '750', '1000'], fontsize=7)
    ax_welf.tick_params(axis='y', length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_welf, fraction=0.03, pad=0.01)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('z-score (within restricted game)', fontsize=7)

    # === Panel D: Welfare dot + CI ===
    ax_dot = fig.add_subplot(gs[0, 3])

    for row, ei in enumerate(sel_idx):
        eco = ecologies[ei]
        vals = uw_mat[ei]
        present = vals[~np.isnan(vals)]
        if len(present) == 0:
            continue

        mean_val = np.mean(present)
        ci_lo = np.percentile(present, 2.5)
        ci_hi = np.percentile(present, 97.5)

        is_min = minimal_flags.get(eco, 0) >= 0.5
        color = '#d62728' if is_min else '#2171b5'
        marker = 'D' if is_min else 'o'

        ax_dot.plot([ci_lo, ci_hi], [row, row], color=color, linewidth=1.5,
                    alpha=0.6, solid_capstyle='round')
        ax_dot.plot(mean_val, row, marker=marker, color=color, markersize=5,
                    markeredgecolor='white', markeredgewidth=0.5, zorder=3)

    ax_dot.set_yticks([])
    ax_dot.set_ylim(n_eco - 0.5, -0.5)
    ax_dot.set_xlabel(f'{metric.upper()}', fontsize=8)
    ax_dot.set_title('Mean + 95% CI', fontsize=9, fontweight='bold')
    ax_dot.tick_params(axis='y', length=0)
    ax_dot.tick_params(axis='x', labelsize=7)
    ax_dot.grid(True, axis='x', alpha=0.3)

    # Legend for dot colors
    from matplotlib.lines import Line2D
    ax_dot.legend(
        handles=[
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#d62728',
                   markersize=6, label='Minimal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2171b5',
                   markersize=6, label='Non-minimal'),
        ],
        loc='lower right', fontsize=6, framealpha=0.8,
    )

    # Suptitle
    fig.suptitle(
        'Restricted Game Landscape: CURB Sets Across Bootstrap Samples',
        fontsize=13, fontweight='bold', y=1.02,
    )

    out_path = Path(save_dir or '.') / (filename + '.png')
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved restricted game panel: {out_path}')
    print(f'  Showing {n_eco} restricted games across {n_bootstraps} bootstraps')


def create_metric_heatmap(
    ecology_matrix_path,
    filename='metric_ecology_heatmap',
    save_dir=None,
    dpi=200,
    n_show=18,
):
    """Vertical two-panel figure: strategy membership + % change from full game.

    Layout: [Strategy membership | Metric heatmap]
    Y-axis: restricted games (shared), X-axis: strategies / metrics.
    """

    METRICS = ['uw', 'nw', 'nw_plus', 'ef1', 'ef1_plus']
    METRIC_LABELS = ['UW', 'NW', 'NW+', 'EF1', 'EF1+']

    with open(ecology_matrix_path, 'rb') as f:
        data = pickle.load(f)

    ecologies = data['ecologies']
    matrices = data['matrices']
    minimal_flags = data['minimal_flags']
    strategy_names = data['config']['strategy_names']

    # Find the full game ecology
    full_eco = frozenset(range(len(strategy_names)))
    eco_to_idx = {eco: i for i, eco in enumerate(ecologies)}
    full_idx = eco_to_idx.get(full_eco)

    # Compute full game baseline for each metric
    full_game_means = {}
    for metric in METRICS:
        mat = matrices[metric]
        if full_idx is not None:
            vals = mat[full_idx]
            present = vals[~np.isnan(vals)]
            full_game_means[metric] = np.nanmean(present) if len(present) > 0 else np.nan
        else:
            full_game_means[metric] = np.nan

    # Select ecologies
    sel_idx = _select_ecologies(
        ecologies, matrices, minimal_flags, strategy_names, n_show=n_show,
    )
    n_eco = len(sel_idx)

    # Sort by frequency (highest first)
    uw_mat = matrices['uw']
    n_bootstraps = uw_mat.shape[1]
    exist = ~np.isnan(uw_mat)
    freqs = exist.sum(axis=1) / n_bootstraps
    sel_idx.sort(key=lambda i: -freqs[i])

    # Build % change from full game: (n_eco, n_metrics) — transposed for vertical
    pct_change = np.full((n_eco, len(METRICS)), np.nan)
    for mi, metric in enumerate(METRICS):
        mat = matrices[metric]
        baseline = full_game_means[metric]
        if np.isnan(baseline) or abs(baseline) < 1e-10:
            continue
        for row, ei in enumerate(sel_idx):
            vals = mat[ei]
            present = vals[~np.isnan(vals)]
            if len(present) > 0:
                pct_change[row, mi] = (np.nanmean(present) - baseline) / abs(baseline) * 100

    # Strategy display info
    short_names = [SHORT_MAP.get(s, s) for s in strategy_names]
    strat_order = [short_names.index(s) for s in STRAT_DISPLAY_ORDER
                   if s in short_names]

    # Y-labels
    y_labels = []
    for row, ei in enumerate(sel_idx):
        eco = ecologies[ei]
        is_min = minimal_flags.get(eco, 0) >= 0.5
        label = _restricted_game_label(eco, strategy_names)
        freq_pct = freqs[ei] * 100
        marker = ' *' if is_min else ''
        y_labels.append(f'{label}  {freq_pct:.0f}%{marker}')

    # --- Figure layout: [membership | heatmap] ---
    fig = plt.figure(figsize=(10, 0.5 * n_eco + 2),
                     constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.03)

    y_positions = np.arange(n_eco)

    # === Panel A: Strategy membership ===
    ax_mem = fig.add_subplot(gs[0, 0])

    mem_matrix = np.zeros((n_eco, len(strat_order)))
    for row, ei in enumerate(sel_idx):
        eco = ecologies[ei]
        for col, si in enumerate(strat_order):
            if si in eco:
                mem_matrix[row, col] = 1.0

    mem_rgb = np.ones((n_eco, len(strat_order), 4))
    for row in range(n_eco):
        for col, si in enumerate(strat_order):
            if mem_matrix[row, col] > 0:
                sname = short_names[si]
                mem_rgb[row, col] = mcolors.to_rgba(
                    STRAT_COLORS.get(sname, '#333333'))
            else:
                mem_rgb[row, col] = (0.95, 0.95, 0.95, 1.0)

    ax_mem.imshow(mem_rgb, aspect='auto', interpolation='nearest')
    ax_mem.set_xticks(range(len(strat_order)))
    ax_mem.set_xticklabels(
        [STRAT_DISPLAY_ORDER[i] for i in range(len(strat_order))],
        fontsize=7, rotation=55, ha='right',
    )
    ax_mem.set_yticks(y_positions)
    ax_mem.set_yticklabels(y_labels, fontsize=8, fontfamily='monospace')
    ax_mem.set_title('Strategy membership', fontsize=9, fontweight='bold')
    ax_mem.tick_params(axis='both', length=0)

    for y in np.arange(-0.5, n_eco, 1):
        ax_mem.axhline(y, color='white', linewidth=0.5)
    for x in np.arange(-0.5, len(strat_order), 1):
        ax_mem.axvline(x, color='white', linewidth=0.5)

    # === Panel B: Metric % change heatmap ===
    ax_heat = fig.add_subplot(gs[0, 1])

    cmap = plt.cm.RdYlBu.copy()
    cmap.set_bad('#f0f0f0')
    pct_masked = np.ma.masked_invalid(pct_change)
    vabs = max(3, np.nanmax(np.abs(pct_change)))
    im = ax_heat.imshow(pct_masked, aspect='auto', cmap=cmap,
                        vmin=-vabs, vmax=vabs, interpolation='nearest')

    # Annotate cells
    for row in range(n_eco):
        for col in range(len(METRICS)):
            val = pct_change[row, col]
            if np.isnan(val):
                continue
            text_color = 'white' if abs(val) > vabs * 0.6 else 'black'
            ax_heat.text(col, row, f'{val:+.1f}%', ha='center', va='center',
                         fontsize=7, color=text_color, fontweight='bold')

    ax_heat.set_xticks(range(len(METRICS)))
    ax_heat.set_xticklabels(METRIC_LABELS, fontsize=9, fontweight='bold')
    ax_heat.set_yticks([])
    ax_heat.set_title('% change from full game', fontsize=9, fontweight='bold')
    ax_heat.tick_params(axis='both', length=0)

    for y in np.arange(-0.5, n_eco, 1):
        ax_heat.axhline(y, color='white', linewidth=1)
    for x in np.arange(-0.5, len(METRICS), 1):
        ax_heat.axvline(x, color='white', linewidth=1)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
    cbar.set_label('% change from full game', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        'Restricted Game Welfare: CURB Sets vs Full Game (* = minimal)',
        fontsize=11, fontweight='bold', y=1.02,
    )

    out_path = Path(save_dir or '.') / (filename + '.png')
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved metric heatmap: {out_path}')
    print(f'  {len(METRICS)} metrics × {n_eco} restricted games')


def _select_showcase_ecologies(curb_metrics, strategy_names):
    """Pick ~6 representative ecologies ordered by size.

    Returns list of (frozenset, sigma_array) tuples.
    Targets: 2 singletons, 1 pair, 1 ~5-set, 1 ~7-set, full game.
    """
    n_strats = len(strategy_names)
    name_to_idx = {s: i for i, s in enumerate(strategy_names)}

    # Fixed targets by strategy composition
    targets = [
        frozenset({name_to_idx['psro']}),
        frozenset({name_to_idx['ppo']}),
        frozenset({name_to_idx['ppo'], name_to_idx['psro']}),
    ]

    # Find a ~5-strategy set containing an LLM strategy
    for eco in curb_metrics:
        if len(eco) == 5 and any(
            strategy_names[i].startswith('openai') or
            strategy_names[i] == 'ef1_bargainer' for i in eco
        ):
            targets.append(eco)
            break

    # Find a ~7-strategy set
    for eco in curb_metrics:
        if len(eco) == 7:
            targets.append(eco)
            break

    # Full game
    targets.append(frozenset(range(n_strats)))

    # Filter to ecologies that actually exist in the data
    showcase = []
    for t in targets:
        if t in curb_metrics:
            showcase.append((t, curb_metrics[t]['sigma']))
    return showcase


def create_payoff_slices(
    curb_results_path,
    filename='restricted_game_payoff_slices',
    save_dir=None,
    dpi=200,
):
    """Row of heatmaps showing the restricted payoff matrix per ecology."""

    with open(curb_results_path, 'rb') as f:
        data = pickle.load(f)

    payoff = data['point_estimate']['payoff_matrix']
    metrics = data['point_estimate']['metrics']
    strategy_names = data['config']['strategy_names']

    showcase = _select_showcase_ecologies(metrics, strategy_names)
    n_panels = len(showcase)

    # Shared colorscale across all subplots
    vmin, vmax = np.inf, -np.inf
    for eco, _ in showcase:
        idx = sorted(eco)
        sub = payoff[np.ix_(idx, idx)]
        vmin = min(vmin, sub.min())
        vmax = max(vmax, sub.max())

    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.5),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    cmap = plt.cm.RdYlBu

    for ax, (eco, sigma) in zip(axes, showcase):
        idx = sorted(eco)
        sub = payoff[np.ix_(idx, idx)]
        names = [SHORT_MAP.get(strategy_names[i], strategy_names[i])
                 for i in idx]

        im = ax.imshow(sub, cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation='nearest')

        # Annotate cell values
        for r in range(len(idx)):
            for c in range(len(idx)):
                val = sub[r, c]
                brightness = (val - vmin) / (vmax - vmin + 1e-10)
                color = 'white' if brightness < 0.3 or brightness > 0.7 else 'black'
                ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                        fontsize=max(5, 8 - len(idx) // 3), color=color)

        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(names, fontsize=7)
        label = _restricted_game_label(eco, strategy_names)
        ax.set_title(f'{label}\n|S|={len(eco)}', fontsize=8, fontweight='bold')

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Expected payoff', fontsize=8)

    fig.suptitle('Restricted Game Payoff Matrices', fontsize=12,
                 fontweight='bold')

    out_path = Path(save_dir or '.') / (filename + '.png')
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved payoff slices: {out_path}  ({n_panels} panels)')


def create_support_panel(
    curb_results_path,
    filename='restricted_game_support',
    save_dir=None,
    dpi=200,
):
    """Row of bar charts showing MENE equilibrium weights per ecology."""

    with open(curb_results_path, 'rb') as f:
        data = pickle.load(f)

    metrics = data['point_estimate']['metrics']
    strategy_names = data['config']['strategy_names']
    short_names = [SHORT_MAP.get(s, s) for s in strategy_names]

    showcase = _select_showcase_ecologies(metrics, strategy_names)
    n_panels = len(showcase)

    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.0),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    x_positions = np.arange(len(STRAT_DISPLAY_ORDER))

    for ax, (eco, sigma) in zip(axes, showcase):
        idx_sorted = sorted(eco)

        # Map sigma to full strategy display order
        weights = np.zeros(len(STRAT_DISPLAY_ORDER))
        in_eco = np.zeros(len(STRAT_DISPLAY_ORDER), dtype=bool)

        for local_i, global_i in enumerate(idx_sorted):
            sname = short_names[global_i]
            if sname in STRAT_DISPLAY_ORDER:
                display_pos = STRAT_DISPLAY_ORDER.index(sname)
                weights[display_pos] = sigma[local_i]
                in_eco[display_pos] = True

        colors = [STRAT_COLORS.get(s, '#333333') if in_eco[j] else '#ffffff'
                  for j, s in enumerate(STRAT_DISPLAY_ORDER)]
        edge_colors = ['#333333' if in_eco[j] else '#cccccc'
                       for j in range(len(STRAT_DISPLAY_ORDER))]

        bars = ax.bar(x_positions, weights, color=colors,
                      edgecolor=edge_colors, linewidth=0.5)

        # Gray out absent strategies
        for j in range(len(STRAT_DISPLAY_ORDER)):
            if not in_eco[j]:
                ax.bar(x_positions[j], 0, color='#f0f0f0',
                       edgecolor='#cccccc', linewidth=0.5)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(STRAT_DISPLAY_ORDER, fontsize=6, rotation=55,
                           ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('NE weight' if ax == axes[0] else '', fontsize=7)
        ax.tick_params(axis='y', labelsize=6)
        label = _restricted_game_label(eco, strategy_names)
        ax.set_title(label, fontsize=8, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    fig.suptitle('MENE Equilibrium Support Across Restricted Games',
                 fontsize=12, fontweight='bold')

    out_path = Path(save_dir or '.') / (filename + '.png')
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved support panel: {out_path}  ({n_panels} panels)')


if __name__ == '__main__':
    data_dir = Path(__file__).parent.parent / 'data' / 'analysis'

    # Existing ecology matrix plots
    pkl_path = str(data_dir / 'ecology_matrix.pkl')
    create_ecology_panel(pkl_path, save_dir=str(data_dir))
    create_metric_heatmap(pkl_path, save_dir=str(data_dir))

    # New: payoff slices + support from curb_results
    curb_path = str(data_dir / 'curb_results.pkl')
    create_payoff_slices(curb_path, save_dir=str(data_dir))
    create_support_panel(curb_path, save_dir=str(data_dir))
