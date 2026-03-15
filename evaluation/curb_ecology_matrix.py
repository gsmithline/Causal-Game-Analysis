"""Ecology x Bootstrap matrix analysis for CURB sets.

Loads curb_results.pkl, builds an (E x B) matrix where rows are unique CURB
ecologies and columns are bootstrap samples.  Cell values are equilibrium
welfare (NaN when that ecology was not CURB in a given bootstrap).

Produces per-ecology statistics (frequency, welfare mean/std/CI, minimality)
and co-occurrence / Jaccard similarity across ecologies.

Usage:
    python evaluation/curb_ecology_matrix.py
    python evaluation/curb_ecology_matrix.py --min-freq 0.10
    python evaluation/curb_ecology_matrix.py --input data/analysis/curb_results.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

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

METRICS = ['uw', 'nw', 'nw_plus', 'ef1', 'ef1_plus']

METRIC_LABELS = {
    'uw': 'Utilitarian Welfare',
    'nw': 'Nash Welfare',
    'nw_plus': 'Nash Welfare+',
    'ef1': 'EF1',
    'ef1_plus': 'EF1+',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_curb_results(path):
    """Load curb_results.pkl, return (strategy_names, analyses)."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    strategy_names = data['config']['strategy_names']
    analyses = data['bootstrap']['analyses']
    return strategy_names, analyses


# ---------------------------------------------------------------------------
# Ecology collection & matrix building
# ---------------------------------------------------------------------------

def collect_unique_ecologies(analyses):
    """Gather all unique CURB frozensets across bootstraps, sorted by (size, indices)."""
    ecologies = set()
    for b in analyses:
        ecologies.update(b['metrics'].keys())
    return sorted(ecologies, key=lambda s: (len(s), sorted(s)))


def build_ecology_matrix(analyses, ecologies, metric):
    """Build NaN-padded matrix (E x B) for one metric.

    Returns ndarray of shape (len(ecologies), len(analyses)).
    """
    eco_idx = {e: i for i, e in enumerate(ecologies)}
    E = len(ecologies)
    B = len(analyses)
    matrix = np.full((E, B), np.nan)
    for j, b in enumerate(analyses):
        for cset, vals in b['metrics'].items():
            i = eco_idx[cset]
            matrix[i, j] = vals[metric]
    return matrix


def compute_minimal_flags(analyses):
    """For each ecology, fraction of bootstraps where it was minimal.

    Returns dict: frozenset -> float in [0, 1].
    """
    from collections import Counter
    minimal_counts = Counter()
    total_counts = Counter()
    for b in analyses:
        minimal_sets = set(frozenset(m) for m in b['minimal_curb_sets'])
        for cset in b['metrics']:
            total_counts[cset] += 1
            if cset in minimal_sets:
                minimal_counts[cset] += 1
    return {
        cset: minimal_counts[cset] / total_counts[cset]
        for cset in total_counts
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_ecology_stats(matrix, ecologies, strategy_names, minimal_flags,
                          min_freq, n_bootstraps):
    """Per-ecology stats: frequency, welfare mean/std/CI, size, minimality.

    Returns DataFrame filtered to ecologies with freq >= min_freq.
    """
    rows = []
    for i, eco in enumerate(ecologies):
        vals = matrix[i]
        exists = ~np.isnan(vals)
        freq = exists.sum() / n_bootstraps
        if freq < min_freq:
            continue

        present_vals = vals[exists]
        names = sorted(
            [SHORT_MAP.get(strategy_names[idx], strategy_names[idx])
             for idx in eco]
        )
        if len(names) <= 4:
            label = '{' + ', '.join(names) + '}'
        else:
            label = '{' + ', '.join(names[:3]) + f', +{len(names)-3}' + '}'

        rows.append({
            'ecology': eco,
            'label': label,
            'size': len(eco),
            'freq': freq,
            'mean': np.mean(present_vals),
            'std': np.std(present_vals, ddof=1) if len(present_vals) > 1 else 0.0,
            'ci_lo': np.percentile(present_vals, 2.5),
            'ci_hi': np.percentile(present_vals, 97.5),
            'minimal_rate': minimal_flags.get(eco, 0.0),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('freq', ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Co-occurrence
# ---------------------------------------------------------------------------

def compute_cooccurrence(matrix):
    """Compute co-occurrence counts and Jaccard similarity from existence mask.

    Returns (cooccur, jaccard) — both (E, E) ndarrays.
    """
    exists = (~np.isnan(matrix)).astype(np.float64)  # (E, B)
    cooccur = exists @ exists.T  # (E, E) — count of bootstraps both present

    # Jaccard: |A ∩ B| / |A ∪ B|
    row_sums = exists.sum(axis=1)  # (E,)
    union = row_sums[:, None] + row_sums[None, :] - cooccur
    jaccard = np.divide(cooccur, union, out=np.zeros_like(cooccur),
                        where=union > 0)
    return cooccur, jaccard


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary_table(stats_df, metric_name):
    """Print formatted ecology stats table to stdout."""
    label = METRIC_LABELS.get(metric_name, metric_name)
    print(f'\n=== {metric_name.upper()} ({label}) ===')
    header = (f'  {"Ecology":<40s} {"Size":>4s} {"Freq":>7s} '
              f'{"Mean":>10s} {"Std":>8s} {"[2.5%":>8s} {"97.5%]":>8s} {"Min":>3s}')
    print(header)

    for _, row in stats_df.iterrows():
        is_min = '*' if row['minimal_rate'] >= 0.5 else ''
        print(f'  {row["label"]:<40s} {row["size"]:>4d} {row["freq"]*100:>6.1f}% '
              f'{row["mean"]:>10.2f} {row["std"]:>8.2f} '
              f'{row["ci_lo"]:>8.2f} {row["ci_hi"]:>8.2f} {is_min:>3s}')

    total = len(stats_df)
    print(f'  [showing {total} ecologies at >= specified frequency]')


def print_cooccurrence_table(jaccard, ecologies, strategy_names, min_freq,
                             n_bootstraps, matrix):
    """Print Jaccard similarity matrix for frequent ecologies."""
    # Determine which ecologies are frequent
    exists = ~np.isnan(matrix)
    freqs = exists.sum(axis=1) / n_bootstraps  # (E,)
    frequent_mask = freqs >= min_freq
    frequent_idx = np.where(frequent_mask)[0]

    if len(frequent_idx) == 0:
        print('\n=== Co-occurrence (Jaccard) ===')
        print('  No ecologies above frequency threshold.')
        return

    # Limit to top 15 by frequency for readability
    sorted_idx = sorted(frequent_idx, key=lambda i: -freqs[i])
    show_idx = sorted_idx[:15]

    def short_label(eco):
        names = sorted(
            [SHORT_MAP.get(strategy_names[i], strategy_names[i]) for i in eco]
        )
        if len(names) <= 3:
            return '{' + ','.join(names) + '}'
        return '{' + ','.join(names[:2]) + f',+{len(names)-2}' + '}'

    labels = [short_label(ecologies[i]) for i in show_idx]

    print(f'\n=== Co-occurrence (Jaccard similarity, top {len(show_idx)} ecologies) ===')
    # Header
    col_w = 8
    header = ' ' * 30 + ''.join(f'{l[:col_w]:>{col_w}s}' for l in labels)
    print(header)

    for ri, row_i in enumerate(show_idx):
        row_label = f'  {labels[ri]:<28s}'
        cells = ''
        for col_i in show_idx:
            j = jaccard[row_i, col_i]
            cells += f'{j:>{col_w}.2f}'
        print(row_label + cells)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(path, ecologies, matrices, stats, cooccurrence, jaccard,
                 minimal_flags, strategy_names, n_bootstraps):
    """Save all results to a pkl file."""
    out = {
        'ecologies': ecologies,
        'matrices': matrices,
        'stats': stats,
        'cooccurrence': cooccurrence,
        'jaccard': jaccard,
        'minimal_flags': minimal_flags,
        'config': {
            'n_bootstraps': n_bootstraps,
            'n_ecologies': len(ecologies),
            'strategy_names': strategy_names,
            'metrics': METRICS,
        },
    }
    with open(path, 'wb') as f:
        pickle.dump(out, f)
    print(f'\nSaved results to {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Ecology x Bootstrap matrix analysis for CURB sets')
    parser.add_argument('--input', type=str,
                        default='data/analysis/curb_results.pkl',
                        help='Path to curb_results.pkl')
    parser.add_argument('--output', type=str,
                        default='data/analysis/ecology_matrix.pkl',
                        help='Path to save output pkl')
    parser.add_argument('--min-freq', type=float, default=0.05,
                        help='Minimum frequency to show in tables (default 0.05)')
    args = parser.parse_args()

    print('=' * 68)
    print('ECOLOGY × BOOTSTRAP MATRIX ANALYSIS')
    print('=' * 68)

    strategy_names, analyses = load_curb_results(args.input)
    n_bootstraps = len(analyses)
    print(f'Loaded {n_bootstraps} bootstrap samples, '
          f'{len(strategy_names)} strategies')

    ecologies = collect_unique_ecologies(analyses)
    print(f'Found {len(ecologies)} unique ecologies across all bootstraps')

    minimal_flags = compute_minimal_flags(analyses)

    # Build matrices and compute stats for each metric
    matrices = {}
    stats = {}
    for metric in METRICS:
        mat = build_ecology_matrix(analyses, ecologies, metric)
        matrices[metric] = mat
        df = compute_ecology_stats(
            mat, ecologies, strategy_names, minimal_flags,
            args.min_freq, n_bootstraps,
        )
        stats[metric] = df
        print_summary_table(df, metric)

    # Co-occurrence (metric-independent — use any matrix for existence mask)
    cooccur, jaccard = compute_cooccurrence(matrices['uw'])
    print_cooccurrence_table(
        jaccard, ecologies, strategy_names, args.min_freq,
        n_bootstraps, matrices['uw'],
    )

    # Save
    save_results(
        args.output, ecologies, matrices, stats, cooccur, jaccard,
        minimal_flags, strategy_names, n_bootstraps,
    )


if __name__ == '__main__':
    main()
