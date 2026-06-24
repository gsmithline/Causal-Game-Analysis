"""
Plot CURB coverage ratio (rho) vs Nash clustering cyclicity
for spinning tops games + bargaining domain.

x-axis: rho = |largest minimal CURB set| / |S|
y-axis: Nash clustering cyclicity (fraction of payoff variance within clusters)

Each game is a labeled point. Demonstrates that cyclic games have
high rho (CURB uninformative) while transitive games have low rho.

Usage:
    python scripts/curb_coverage_vs_cyclicity.py [--outdir notebooks]
    python scripts/curb_coverage_vs_cyclicity.py --games-only  # skip CURB (use precomputed)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iterative_game_analysis.nash_clustering import (
    nash_clustering, cyclicity_from_clustering, compute_game_cyclicity
)
from evaluation.curb_analysis import find_minimal_curb_sets_klimm_weibull


# Display names for the figure
DISPLAY = {
    'RPS': 'RPS',
    'Kuhn-poker': 'Kuhn Poker',
    '5,3-Blotto': '5,3-Blotto',
    '5,4-Blotto': '5,4-Blotto',
    '5,5-Blotto': '5,5-Blotto',
    '10,3-Blotto': '10,3-Blotto',
    '10,4-Blotto': '10,4-Blotto',
    'Elo game': 'Elo',
    'Elo game + noise=0.1': 'Elo+0.1',
    'Elo game + noise=0.5': 'Elo+0.5',
    'Elo game + noise=1.0': 'Elo+1.0',
    'Transitive game': 'Transitive',
    'hex(board_size=3)': 'Hex 3x3',
    'tic_tac_toe': 'Tic-Tac-Toe',
    'connect_four': 'Connect Four',
    'AlphaStar': 'AlphaStar',
    'Bargaining': 'Bargaining',
    '3-move parity game 2': 'Parity',
}


def download_bargaining_from_hf(target_dir):
    """Download bargaining crossplay data from HuggingFace."""
    from huggingface_hub import snapshot_download

    print("  Downloading bargaining data from HuggingFace...")
    snapshot_download(
        repo_id="Gsmith43/causal-game-crossplay",
        repo_type="dataset",
        local_dir=str(target_dir),
    )
    print(f"  Downloaded to {target_dir}")


def load_bargaining_matrix(hf_data=False):
    """Load the average bargaining payoff matrix from crossplay data.

    Args:
        hf_data: If True, download from HuggingFace if data/crossplay/ missing.
    """
    from evaluation.original_paper_analysis import (
        load_and_preprocess_data, build_matrices_fast, load_json
    )

    crossplay_dir = Path(__file__).parent.parent / "data" / "crossplay"

    # Download from HuggingFace if needed
    if hf_data and not crossplay_dir.exists():
        download_bargaining_from_hf(crossplay_dir)
    elif not crossplay_dir.exists():
        raise FileNotFoundError(
            f"{crossplay_dir} not found. Use --hf-data to download from HuggingFace."
        )

    matrix_path = crossplay_dir / "metagame_matrix.json"

    if matrix_path.exists():
        strategy_names = load_json(matrix_path)["strategy_names"]
    else:
        strategy_names = [
            "walk", "tough", "nfsp", "mappo", "soft", "ppo", "psro",
            "ef1_bargainer", "openai_5.2_none", "openai_5.2_low",
            "openai_5.4_low", "openai_5.4_medium", "openai_5.2_medium",
        ]

    grouped_data = load_and_preprocess_data(crossplay_dir, strategy_names)
    # Use rng but take all data (no bootstrap resampling — pass None-like seed)
    rng = np.random.default_rng(42)
    matrices = build_matrices_fast(grouped_data, strategy_names, rng, raw_utility=True)
    return matrices["raw_payoff"], strategy_names


def compute_curb_rho(payoff_matrix):
    """Compute CURB coverage ratio: |largest minimal CURB| / n."""
    n = payoff_matrix.shape[0]
    minimals = find_minimal_curb_sets_klimm_weibull(payoff_matrix, n)
    if not minimals:
        return 1.0, 0
    largest = max(len(m) for m in minimals)
    return largest / n, len(minimals)


def _process_game(name, payoff_matrix, precomputed_rho=None, skip_curb=False):
    """Worker function for parallel processing of a single game.

    Returns dict with rho, cyclicity, n, num_clusters, cluster_sizes.
    """
    import warnings
    warnings.filterwarnings("ignore")

    n = payoff_matrix.shape[0]

    # Nash clustering
    cyc_result = compute_game_cyclicity(payoff_matrix)

    # CURB coverage
    if precomputed_rho is not None:
        rho = precomputed_rho
    elif not skip_curb:
        rho, _ = compute_curb_rho(payoff_matrix)
    else:
        return None

    sizes = cyc_result['cluster_sizes']
    non_singleton = [s for s in sizes if s > 1]

    return {
        'rho': rho,
        'cyclicity': cyc_result['cyclicity'],
        'n': n,
        'num_clusters': cyc_result['num_clusters'],
        'cluster_sizes': sizes,
        'max_cluster_size': max(sizes),
        'max_cluster_ratio': max(sizes) / n,
        'mean_cluster_size': np.mean(sizes),
        'num_non_singleton': len(non_singleton),
        'frac_in_non_singleton': sum(non_singleton) / n if non_singleton else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='notebooks')
    parser.add_argument('--spinning-tops-pkl', type=str,
                        default='data/spinning_top_payoffs.pkl')
    parser.add_argument('--max-n-curb', type=int, default=1100,
                        help='Max game size for CURB computation')
    parser.add_argument('--max-n-clustering', type=int, default=1100,
                        help='Max game size for Nash clustering')
    parser.add_argument('--load', type=str, default=None,
                        help='Load precomputed results from pickle (skip computation)')
    parser.add_argument('--hf-data', action='store_true',
                        help='Download bargaining data from HuggingFace if not local')
    args = parser.parse_args()

    outdir = Path(args.outdir)

    # --- If loading from pickle, skip all computation ---
    if args.load:
        print(f"Loading precomputed results from {args.load}...")
        with open(args.load, 'rb') as f:
            results = pickle.load(f)
        # Jump straight to printing and plotting
        _print_and_plot(results, outdir)
        return

    # --- Load spinning tops data ---
    print("Loading spinning tops games...")
    with open(args.spinning_tops_pkl, 'rb') as f:
        st_data = pickle.load(f, encoding='latin1')

    # --- Load bargaining game ---
    print("Loading bargaining game...")
    barg_matrix, barg_names = load_bargaining_matrix(hf_data=args.hf_data)
    print(f"  Bargaining: {len(barg_names)} strategies")

    # --- Precomputed CURB results (from notebook runs) ---
    # These are expensive to compute, so we hardcode known results
    # and compute the rest.
    precomputed_rho = {
        '5,3-Blotto': 18 / 21,
        '5,4-Blotto': 52 / 54,
        '5,5-Blotto': 121 / 126,
        '10,3-Blotto': 63 / 66,
        '10,4-Blotto': 270 / 286,
        'Elo game': 1 / 1000,
        'Elo game + noise=0.1': 18 / 1000,
        'hex(board_size=3)': 270 / 766,
    }

    # --- Select games to include ---
    # Include games that are feasible for both CURB and clustering
    target_games = [
        'RPS', 'Kuhn-poker',
        '5,3-Blotto', '5,4-Blotto', '5,5-Blotto',
        '10,3-Blotto', '10,4-Blotto',
        'Elo game', 'Elo game + noise=0.1',
        'Elo game + noise=0.5', 'Elo game + noise=1.0',
        'Transitive game',
        'hex(board_size=3)',
        'tic_tac_toe',
        '3-move parity game 2',
    ]

    # --- Build job list ---
    jobs = {}  # name -> payoff_matrix

    for name in target_games:
        if name not in st_data:
            print(f"  Skipping {name} (not in pickle)")
            continue
        M = st_data[name].astype(float)
        n = M.shape[0]
        if n > args.max_n_clustering:
            print(f"  Skipping {name} (n={n} > max_n_clustering)")
            continue
        jobs[name] = M

    jobs['Bargaining'] = barg_matrix

    # --- Process all games in parallel ---
    print(f"\nProcessing {len(jobs)} games in parallel...")
    results = {}

    with ProcessPoolExecutor() as executor:
        futures = {}
        for name, M in jobs.items():
            pre_rho = precomputed_rho.get(name)
            skip_curb = M.shape[0] > args.max_n_curb and pre_rho is None
            futures[executor.submit(_process_game, name, M, pre_rho, skip_curb)] = name

        pbar = tqdm(as_completed(futures), total=len(futures), desc="Games")
        for future in pbar:
            name = futures[future]
            pbar.set_postfix_str(name)
            try:
                result = future.result()
                if result is not None:
                    results[name] = result
                    r = result
                    print(f"  {name} (n={r['n']}): rho={r['rho']:.4f}, "
                          f"cyclicity={r['cyclicity']:.4f}, "
                          f"clusters={r['num_clusters']}")
            except Exception as e:
                print(f"  {name}: FAILED - {e}")

    # --- Save results to pickle ---
    save_data = {}
    for name, r in results.items():
        save_data[name] = {
            'rho': r['rho'],
            'cyclicity': r['cyclicity'],
            'n': r['n'],
            'num_clusters': r['num_clusters'],
            'cluster_sizes': r['cluster_sizes'],
        }
    results_pkl = outdir / 'curb_coverage_vs_cyclicity.pkl'
    with open(results_pkl, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved results to {results_pkl}")

    _print_and_plot(results, outdir)


def _print_and_plot(results, outdir):
    """Print summary table and generate the scatter plot."""
    from matplotlib.lines import Line2D

    # --- Print summary table ---
    print("\n" + "=" * 110)
    print(f"{'Game':<25s} {'n':>5s} {'rho':>7s} {'cyc':>7s} {'#clust':>7s} "
          f"{'max_C':>6s} {'max/n':>7s} {'mean_C':>7s} {'#non1':>6s} {'%non1':>7s}")
    print("-" * 110)
    for name, r in sorted(results.items(), key=lambda x: x[1]['rho']):
        print(f"{name:<25s} {r['n']:>5d} {r['rho']:>7.4f} {r['cyclicity']:>7.4f} "
              f"{r['num_clusters']:>7d} {r['max_cluster_size']:>6d} "
              f"{r['max_cluster_ratio']:>7.4f} {r['mean_cluster_size']:>7.1f} "
              f"{r['num_non_singleton']:>6d} {r['frac_in_non_singleton']:>7.4f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Game categories for coloring
    blotto_games = [n for n in results if 'Blotto' in n]
    elo_games = [n for n in results if 'Elo' in n or 'Transitive' in n]
    board_games = [n for n in results if n in (
        'hex(board_size=3)', 'tic_tac_toe', 'connect_four',
        'Kuhn-poker', '3-move parity game 2')]

    colors = {
        'blotto': '#e74c3c',      # red
        'elo': '#3498db',          # blue
        'board': '#2ecc71',        # green
        'special': '#9b59b6',      # purple
        'bargaining': '#e67e22',   # orange
    }

    for name, r in results.items():
        x, y = r['rho'], r['cyclicity']

        if name == 'Bargaining':
            color = colors['bargaining']
            marker = '*'
            size = 200
            zorder = 10
        elif name == 'RPS':
            color = colors['special']
            marker = 'D'
            size = 100
            zorder = 5
        elif name in blotto_games:
            color = colors['blotto']
            marker = 's'
            size = 80
            zorder = 5
        elif name in elo_games:
            color = colors['elo']
            marker = '^'
            size = 80
            zorder = 5
        elif name in board_games:
            color = colors['board']
            marker = 'o'
            size = 80
            zorder = 5
        else:
            color = colors['special']
            marker = 'D'
            size = 80
            zorder = 5

        label = DISPLAY.get(name, name)
        ax.scatter(x, y, c=color, marker=marker, s=size, zorder=zorder,
                   edgecolors='black', linewidths=0.5)

        # Label placement: offset to avoid overlap
        offset_x, offset_y = 0.01, 0.01
        ha = 'left'

        # Adjust specific labels to avoid collisions
        if r['rho'] > 0.9:
            ha = 'right'
            offset_x = -0.01
        if name == 'Elo game':
            offset_y = 0.02
        if name == 'Transitive game':
            offset_y = -0.02

        ax.annotate(label, (x, y), fontsize=7.5,
                    xytext=(offset_x, offset_y), textcoords='offset fontsize',
                    ha=ha, va='bottom')

    # Reference regions
    ax.axvspan(0.0, 0.15, alpha=0.05, color='green', zorder=0)
    ax.axvspan(0.85, 1.0, alpha=0.05, color='red', zorder=0)

    ax.text(0.07, 0.97, 'Level 3\ninformative', transform=ax.transAxes,
            fontsize=8, ha='center', va='top', color='darkgreen', alpha=0.6)
    ax.text(0.93, 0.97, 'Level 3\nuninformative', transform=ax.transAxes,
            fontsize=8, ha='center', va='top', color='darkred', alpha=0.6)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['blotto'],
               markersize=8, markeredgecolor='black', markeredgewidth=0.5,
               label='Blotto (cyclic)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=colors['elo'],
               markersize=8, markeredgecolor='black', markeredgewidth=0.5,
               label='Elo / Transitive'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['board'],
               markersize=8, markeredgecolor='black', markeredgewidth=0.5,
               label='Board games'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=colors['bargaining'],
               markersize=12, markeredgecolor='black', markeredgewidth=0.5,
               label='Bargaining (ours)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=colors['special'],
               markersize=8, markeredgecolor='black', markeredgewidth=0.5,
               label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.9)

    ax.set_xlabel(r'CURB Coverage Ratio $\rho$ = |largest min CURB| / |S|',
                  fontsize=11)
    ax.set_ylabel('Nash Clustering Cyclicity\n(within-cluster variance fraction)',
                  fontsize=11)
    ax.set_title('CURB Coverage vs Game Cyclicity\n'
                 '(Czarnecki et al. 2020 games + Bargaining domain)',
                 fontsize=13, fontweight='bold')

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    save_path = outdir / 'curb_coverage_vs_cyclicity.png'
    fig.savefig(str(save_path), dpi=200, bbox_inches='tight')
    print(f"\nSaved {save_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
