"""
Plot Level 2 LOO intervals from pickle results.
Generates one figure per solution concept (solver).

Usage:
    python notebooks/plot_l2_loo_from_pkl.py [--pkl path/to/results.pkl]
    python notebooks/plot_l2_loo_from_pkl.py --min-support 1e-6
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
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


def extract_l2_loo(raw_samples, solver, strategy_names, metrics, min_support=0.0):
    """Extract per-agent LOO delta distributions from raw bootstrap samples.

    Args:
        min_support: If > 0, only include bootstrap samples where the agent
            has equilibrium weight >= min_support. This conditions the LOO
            effect on the agent being active in the equilibrium.

    Returns:
        {metric: {agent: (mean, ci_lo, ci_hi, n_active)}}
    """
    valid = [s for s in raw_samples if s['level2'].get(solver) is not None]
    if not valid:
        return None

    # Build per-agent support mask
    agent_idx = {name: i for i, name in enumerate(strategy_names)}
    sigmas = np.array([s['level1'][solver]['sigma'] for s in valid])

    data = {}
    for m in metrics:
        data[m] = {}
        for agent in strategy_names:
            deltas = np.array([s['level2'][solver]['loo'][agent][m] for s in valid])

            # Apply support filter
            if min_support > 0:
                mask = sigmas[:, agent_idx[agent]] >= min_support
                deltas = deltas[mask]

            if len(deltas) == 0:
                continue

            # Skip agents with all-zero effects
            if np.all(deltas == 0):
                continue

            data[m][agent] = (
                np.mean(deltas),
                np.percentile(deltas, 2.5),
                np.percentile(deltas, 97.5),
                len(deltas),
            )
    return data


def plot_l2_loo(data, metrics, solver_name, min_support=0.0, save_path=None):
    """Plot L2 LOO deltas for one solver."""
    all_agents = set()
    for m in metrics:
        all_agents.update(data[m].keys())
    if not all_agents:
        print(f"  No non-zero agents for {solver_name}, skipping.")
        return

    # Sort by mean delta of UW (or first metric)
    sort_metric = 'uw' if 'uw' in metrics else metrics[0]
    agents_sorted = sorted(all_agents,
                           key=lambda s: data[sort_metric].get(s, (0,0,0,0))[0])
    y = np.arange(len(agents_sorted))

    active_metrics = [m for m in metrics if m in data]
    fig, axes = plt.subplots(1, len(active_metrics),
                             figsize=(4 * len(active_metrics), max(5, 0.7 * len(agents_sorted))),
                             sharey=True)
    if len(active_metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, active_metrics):
        for i, agent in enumerate(agents_sorted):
            if agent not in data[m]:
                continue
            mean, ci_lo, ci_hi, n_active = data[m][agent]

            err_lo = max(0.0, mean - ci_lo)
            err_hi = max(0.0, ci_hi - mean)

            color = 'forestgreen' if mean > 0 else 'firebrick'
            ax.errorbar(mean, y[i],
                        xerr=[[err_lo], [err_hi]],
                        fmt='o', color=color, markersize=5, capsize=3,
                        elinewidth=1.5, zorder=3)

        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(METRIC_LABELS.get(m, m), fontweight='bold')
        ax.grid(axis='x', alpha=0.2)
        ax.set_xlabel('Δ Metric')

    # Y-axis labels: include n_active if filtering
    if min_support > 0:
        sort_m = 'uw' if 'uw' in metrics else metrics[0]
        labels = []
        for s in agents_sorted:
            n = data[sort_m].get(s, (0,0,0,0))[3] if s in data.get(sort_m, {}) else 0
            labels.append(f"{DISPLAY.get(s, s)} (n={n})")
        axes[0].set_yticklabels(labels, fontsize=9)
    else:
        axes[0].set_yticklabels([DISPLAY.get(s, s) for s in agents_sorted], fontsize=9)
    axes[0].set_yticks(y)

    legend_elements = [
        Line2D([0], [0], marker='o', color='forestgreen', label='Positive (helpful)',
               markersize=6, linestyle='None'),
        Line2D([0], [0], marker='o', color='firebrick', label='Negative (harmful)',
               markersize=6, linestyle='None'),
    ]
    axes[-1].legend(handles=legend_elements, loc='lower right', fontsize=8)

    solver_display = {'mene': 'MENE', 'maxent_cce': 'Maxent CCE'}
    support_label = f', support ≥ {min_support}' if min_support > 0 else ''
    plt.suptitle(f'Level 2 LOO Deltas ({solver_display.get(solver_name, solver_name)}{support_label})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

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
                        help='Only include samples where agent has >= this equilibrium weight')
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        results = pickle.load(f)

    config = results['config']
    strategy_names = config['strategy_names']
    solvers = config['solvers']
    metrics = config['metrics']
    outdir = Path(args.outdir)

    suffix = f'_support{args.min_support}' if args.min_support > 0 else ''

    for solver in solvers:
        print(f"Processing {solver}...")
        data = extract_l2_loo(results['raw'], solver, strategy_names, metrics,
                              min_support=args.min_support)
        if data is None:
            print(f"  No valid samples for {solver}, skipping.")
            continue
        save_path = outdir / f'l2_loo_{solver}{suffix}.png'
        plot_l2_loo(data, metrics, solver, min_support=args.min_support, save_path=save_path)


if __name__ == '__main__':
    main()
