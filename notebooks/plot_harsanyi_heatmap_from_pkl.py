"""
Plot Harsanyi interaction dividend heatmaps from pickle results.
Generates one heatmap per (solver, metric) combination.

Usage:
    python notebooks/plot_harsanyi_heatmap_from_pkl.py [--pkl path/to/results.pkl]
    python notebooks/plot_harsanyi_heatmap_from_pkl.py --min-support 1e-6 --metrics uw nw
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import argparse

DISPLAY = {
    'walk': 'Walk', 'tough': 'Tough', 'soft': 'Soft',
    'mappo': 'MAPPO', 'ppo': 'PPO', 'psro': 'PSRO',
    'nfsp': 'NFSP', 'ef1_bargainer': 'EF1',
    'openai_5.2_low': '5.2-Low', 'openai_5.2_medium': '5.2-Med',
    'openai_5.4_low': '5.4-Low', 'openai_5.2_none': '5.2-None',
    'openai_5.4_medium': '5.4-Med',
}

METRIC_LABELS = {'uw': 'UW', 'nw': 'NW', 'nw_plus': 'NW+', 'ef1': 'EF1', 'ef1_plus': 'EF1+'}


def extract_harsanyi(raw_samples, solver, strategy_names, metric, min_support=0.0):
    """Extract Harsanyi interaction dividend matrix for one metric.

    Args:
        min_support: If > 0, only include bootstrap samples where BOTH agents
            in the pair have equilibrium weight >= min_support.

    Returns:
        (n, n) matrix of mean dividends, (n, n) matrix of p-values (fraction
        of samples where sign differs from mean).
    """
    valid = [s for s in raw_samples if s['level2'].get(solver) is not None]
    if not valid:
        return None, None

    n = len(strategy_names)
    agent_idx = {name: i for i, name in enumerate(strategy_names)}
    sigmas = np.array([s['level1'][solver]['sigma'] for s in valid])

    mean_matrix = np.full((n, n), np.nan)
    sig_matrix = np.full((n, n), np.nan)

    for i, s_a in enumerate(strategy_names):
        for j, s_b in enumerate(strategy_names):
            if i >= j:
                continue

            pair = (s_a, s_b)
            rev_pair = (s_b, s_a)

            dividends = []
            for k, s in enumerate(valid):
                h = s['level2'][solver]['harsanyi']
                if pair in h:
                    val = h[pair][metric]
                elif rev_pair in h:
                    val = h[rev_pair][metric]
                else:
                    continue

                # Support filter: both agents must be active
                if min_support > 0:
                    if (sigmas[k, agent_idx[s_a]] < min_support or
                            sigmas[k, agent_idx[s_b]] < min_support):
                        continue

                dividends.append(val)

            if len(dividends) == 0:
                continue

            dividends = np.array(dividends)
            mean_val = np.mean(dividends)
            mean_matrix[i, j] = mean_val
            mean_matrix[j, i] = mean_val

            # Significance: fraction of samples with opposite sign from mean
            if mean_val > 0:
                p = np.mean(dividends < 0)
            elif mean_val < 0:
                p = np.mean(dividends > 0)
            else:
                p = 1.0
            sig_matrix[i, j] = p
            sig_matrix[j, i] = p

    return mean_matrix, sig_matrix


def plot_harsanyi_heatmap(mean_matrix, sig_matrix, strategy_names, solver_name,
                          metric, min_support=0.0, save_path=None):
    """Plot a single Harsanyi heatmap."""
    n = len(strategy_names)
    display_names = [DISPLAY.get(s, s) for s in strategy_names]

    # Filter to agents that have at least one non-NaN interaction
    has_data = ~np.all(np.isnan(mean_matrix), axis=1)
    idx = np.where(has_data)[0]
    if len(idx) == 0:
        print(f"  No data for {solver_name}/{metric}, skipping.")
        return

    sub_matrix = mean_matrix[np.ix_(idx, idx)]
    sub_sig = sig_matrix[np.ix_(idx, idx)]
    sub_names = [display_names[i] for i in idx]

    # Replace NaN with 0 for display
    plot_matrix = np.nan_to_num(sub_matrix, nan=0.0)

    # Symmetric colormap centered at 0
    vmax = np.nanmax(np.abs(plot_matrix))
    if vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(sub_names)),
                                     max(5, 0.5 * len(sub_names))))
    im = ax.imshow(plot_matrix, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='equal')

    ax.set_xticks(range(len(sub_names)))
    ax.set_xticklabels(sub_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(sub_names)))
    ax.set_yticklabels(sub_names, fontsize=8)

    # Annotate cells with value and significance stars
    for i in range(len(sub_names)):
        for j in range(len(sub_names)):
            if i >= j:
                continue
            val = plot_matrix[i, j]
            p = sub_sig[i, j]

            if np.isnan(p):
                stars = ''
            elif p < 0.01:
                stars = '***'
            elif p < 0.05:
                stars = '**'
            elif p < 0.10:
                stars = '*'
            else:
                stars = ''

            text_color = 'white' if abs(val) > 0.6 * vmax else 'black'
            ax.text(j, i, f'{val:.2f}{stars}', ha='center', va='center',
                    fontsize=7, color=text_color)

    # Mask lower triangle
    for i in range(len(sub_names)):
        for j in range(len(sub_names)):
            if i >= j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=True, color='white', zorder=2))

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Harsanyi Dividend')

    solver_display = {'mene': 'MENE', 'maxent_cce': 'Maxent CCE'}
    support_label = f', support ≥ {min_support}' if min_support > 0 else ''
    ax.set_title(f'Harsanyi Interaction Dividends — {METRIC_LABELS.get(metric, metric)}\n'
                 f'({solver_display.get(solver_name, solver_name)}{support_label})',
                 fontsize=11, fontweight='bold')

    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"  Saved {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str,
                        default='data/analysis/causal_bargaining_100_mecce_mene.pkl')
    parser.add_argument('--outdir', type=str, default='notebooks')
    parser.add_argument('--min-support', type=float, default=0.0,
                        help='Only include samples where both agents have >= this equilibrium weight')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='Which metrics to plot (default: all)')
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        results = pickle.load(f)

    config = results['config']
    strategy_names = config['strategy_names']
    solvers = config['solvers']
    metrics = args.metrics or config['metrics']
    outdir = Path(args.outdir)

    suffix = f'_support{args.min_support}' if args.min_support > 0 else ''

    for solver in solvers:
        for metric in metrics:
            print(f"Processing {solver}/{metric}...")
            mean_matrix, sig_matrix = extract_harsanyi(
                results['raw'], solver, strategy_names, metric,
                min_support=args.min_support)
            if mean_matrix is None:
                print(f"  No valid samples for {solver}, skipping.")
                continue
            save_path = outdir / f'harsanyi_{metric}_{solver}{suffix}.png'
            plot_harsanyi_heatmap(mean_matrix, sig_matrix, strategy_names,
                                 solver, metric, min_support=args.min_support,
                                 save_path=save_path)


if __name__ == '__main__':
    main()
