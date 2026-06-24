"""
Plot CURB-conditional LOO intervals from pickle results.
Generates one figure per solution concept (solver).

Usage:
    python notebooks/plot_curb_intervals_from_pkl.py [--pkl path/to/results.pkl]
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


def extract_curb_intervals(raw_samples, solver, strategy_names, metrics):
    """Extract per-agent min/max delta distributions from raw bootstrap samples.

    Returns:
        {metric: {agent: (min_mean, min_lo, min_hi, max_mean, max_lo, max_hi)}}
    """
    # Filter to samples where this solver produced results
    valid = [s for s in raw_samples if s['level3'].get(solver) is not None]
    if not valid:
        return None

    data = {}
    for m in metrics:
        data[m] = {}
        for agent in strategy_names:
            mins = []
            maxs = []
            for s in valid:
                l3 = s['level3'][solver][agent][m]
                mins.append(l3['min'])
                maxs.append(l3['max'])
            mins = np.array(mins)
            maxs = np.array(maxs)

            # Skip agents with all-zero effects
            if np.all(mins == 0) and np.all(maxs == 0):
                continue

            data[m][agent] = (
                np.mean(mins),
                np.percentile(mins, 2.5),
                np.percentile(mins, 97.5),
                np.mean(maxs),
                np.percentile(maxs, 2.5),
                np.percentile(maxs, 97.5),
            )
    return data


def plot_curb_intervals(data, metrics, solver_name, save_path=None):
    """Plot CURB-conditional LOO intervals for one solver."""
    # Get agents that appear in at least one metric (skip all-zero agents)
    all_agents = set()
    for m in metrics:
        all_agents.update(data[m].keys())
    if not all_agents:
        print(f"  No non-zero agents for {solver_name}, skipping.")
        return

    # Sort by max delta of UW (or first metric if UW not available)
    sort_metric = 'uw' if 'uw' in metrics else metrics[0]
    agents_sorted = sorted(all_agents,
                           key=lambda s: data[sort_metric].get(s, (0,0,0,0,0,0))[3])
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
            min_mean, min_lo, min_hi, max_mean, max_lo, max_hi = data[m][agent]

            # Min delta: red square + CI
            min_err_lo = max(0.0, min_mean - min_lo)
            min_err_hi = max(0.0, min_hi - min_mean)
            ax.errorbar(min_mean, y[i] - 0.15,
                        xerr=[[min_err_lo], [min_err_hi]],
                        fmt='s', color='firebrick', markersize=5, capsize=3,
                        elinewidth=1.5, zorder=3)

            # Max delta: green diamond + CI
            max_err_lo = max(0.0, max_mean - max_lo)
            max_err_hi = max(0.0, max_hi - max_mean)
            ax.errorbar(max_mean, y[i] + 0.15,
                        xerr=[[max_err_lo], [max_err_hi]],
                        fmt='D', color='forestgreen', markersize=5, capsize=3,
                        elinewidth=1.5, zorder=3)

            # Shaded region between means
            ax.fill_betweenx([y[i] - 0.08, y[i] + 0.08], min_mean, max_mean,
                             color='steelblue', alpha=0.15, zorder=1)

        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(METRIC_LABELS.get(m, m), fontweight='bold')
        ax.grid(axis='x', alpha=0.2)
        ax.set_xlabel('Δ Metric')

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([DISPLAY.get(s, s) for s in agents_sorted], fontsize=9)

    legend_elements = [
        Line2D([0], [0], marker='s', color='firebrick', label='Min Δ mean ± 95% CI',
               markersize=6, linestyle='None'),
        Line2D([0], [0], marker='D', color='forestgreen', label='Max Δ mean ± 95% CI',
               markersize=6, linestyle='None'),
    ]
    axes[-1].legend(handles=legend_elements, loc='lower right', fontsize=8)

    solver_display = {'mene': 'MENE', 'maxent_cce': 'Maxent CCE'}
    plt.suptitle(f'CURB-Conditional LOO Intervals ({solver_display.get(solver_name, solver_name)})',
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
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        results = pickle.load(f)

    config = results['config']
    strategy_names = config['strategy_names']
    solvers = config['solvers']
    metrics = config['metrics']
    outdir = Path(args.outdir)

    for solver in solvers:
        print(f"Processing {solver}...")
        data = extract_curb_intervals(results['raw'], solver, strategy_names, metrics)
        if data is None:
            print(f"  No valid samples for {solver}, skipping.")
            continue
        save_path = outdir / f'curb_loo_intervals_{solver}.png'
        plot_curb_intervals(data, metrics, solver, save_path=save_path)


if __name__ == '__main__':
    main()
