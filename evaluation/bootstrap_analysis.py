"""
Bootstrap analysis for computing average best response graphs.
This computes the bootstrap best response graphs 
Resamples game data with replacement to compute:
- Bootstrap payoff matrices
- Average best response frequencies
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.data_saver import get_matchup_dirname
from visuals.avg_best_response_graph_visual import create_average_best_response_graph



def load_all_games(crossplay_dir: Path, strategy_names: List[str]) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Load all game data from crossplay directory.

    Returns:
        Dict mapping (strategy_p1_name, strategy_p2_name) -> list of games
    """
    crossplay_dir = Path(crossplay_dir)
    all_games = {}

    for strat_i in strategy_names:
        for strat_j in strategy_names:
            matchup_dir = crossplay_dir / get_matchup_dirname(strat_i, strat_j)
            games_path = matchup_dir / "games.json"

            if games_path.exists():
                with open(games_path) as f:
                    data = json.load(f)
                all_games[(strat_i, strat_j)] = data['games']
            else:
                print(f"Warning: No games found for {strat_i} vs {strat_j}")
                all_games[(strat_i, strat_j)] = []

    return all_games


def compute_payoff_matrix_from_games(
    all_games: Dict[Tuple[str, str], List[Dict]],
    strategy_names: List[str],
    sample_indices: Dict[Tuple[str, str], np.ndarray] = None,
) -> np.ndarray:
    """
    Compute payoff matrix from game data.

    Cell [i, j] = Strategy i's average payoff vs Strategy j,
    averaged over both positions (i as P1 vs j, and i as P2 vs j).

    Args:
        all_games: Dict of (p1_strat, p2_strat) -> games list
        strategy_names: List of strategy names
        sample_indices: Optional dict of bootstrap sample indices per matchup.
                       If None, uses all games.

    Returns:
        Payoff matrix of shape (n, n)
    """
    n = len(strategy_names)
    payoff_matrix = np.zeros((n, n))

    for i, strat_i in enumerate(strategy_names):
        for j, strat_j in enumerate(strategy_names):
            # Games where i plays as P1 against j as P2
            games_i_vs_j = all_games.get((strat_i, strat_j), [])
            # Games where j plays as P1 against i as P2 (i is P2 here)
            games_j_vs_i = all_games.get((strat_j, strat_i), [])

            # Get sample indices or use all
            if sample_indices is not None:
                idx_i_vs_j = sample_indices.get((strat_i, strat_j), np.arange(len(games_i_vs_j)))
                idx_j_vs_i = sample_indices.get((strat_j, strat_i), np.arange(len(games_j_vs_i)))
            else:
                idx_i_vs_j = np.arange(len(games_i_vs_j))
                idx_j_vs_i = np.arange(len(games_j_vs_i))

            payoffs_i_as_p1 = [games_i_vs_j[k]['outcome']['payoff_p1'] for k in idx_i_vs_j] if len(games_i_vs_j) > 0 else []

            payoffs_i_as_p2 = [games_j_vs_i[k]['outcome']['payoff_p2'] for k in idx_j_vs_i] if len(games_j_vs_i) > 0 else []

            all_payoffs = payoffs_i_as_p1 + payoffs_i_as_p2
            if all_payoffs:
                payoff_matrix[i, j] = np.mean(all_payoffs)

    return payoff_matrix


def compute_best_response_matrix(payoff_matrix: np.ndarray) -> np.ndarray:
    """
    Compute best response indicator matrix.

    Args:
        payoff_matrix: Shape (n, n) where [i, j] = i's payoff vs j

    Returns:
        BR matrix of shape (n, n) where [j, i] = 1 if j is BR to i, else 0
    """
    n = payoff_matrix.shape[0]
    br_matrix = np.zeros((n, n))

    for i in range(n):  
        payoffs_vs_i = payoff_matrix[:, i]
        best_response = np.argmax(payoffs_vs_i)
        br_matrix[best_response, i] = 1.0

    return br_matrix


def print_single_br_graph(
    br_matrix: np.ndarray,
    strategy_names: List[str],
    bootstrap_num: int = None,
    payoff_matrix: np.ndarray = None,
) -> None:
    """
    Print a single bootstrap's BR graph in text form.

    Shows edges: "BR_strategy -> opponent" for each best response relationship.
    Optionally shows the payoff matrix values.
    """
    n = len(strategy_names)

    if bootstrap_num is not None:
        print(f"\n--- Bootstrap {bootstrap_num} BR Graph ---")

    # Print payoff matrix if provided
    if payoff_matrix is not None:
        print("  Payoff matrix (row i vs col j):")
        header = "    " + "".join(f"{name[:6]:>8}" for name in strategy_names)
        print(header)
        for i, name in enumerate(strategy_names):
            row = f"  {name[:6]:>6}"
            for j in range(n):
                row += f"{payoff_matrix[i, j]:8.4f}"
            print(row)

    # Print BR edges
    print("  BR edges:")
    edges = []
    for i in range(n):  # opponent
        for j in range(n):  # potential BR
            if br_matrix[j, i] == 1.0:
                edges.append(f"    {strategy_names[j]} -> {strategy_names[i]}")

    if edges:
        for edge in edges:
            print(edge)
    else:
        print("    (no BR edges)")


def bootstrap_average_br_matrix(
    all_games: Dict[Tuple[str, str], List[Dict]],
    strategy_names: List[str],
    num_bootstrap: int = 1000,
    seed: int = 42,
    verbose: bool = True,
    print_each_graph: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average best response matrix via bootstrap.

    Args:
        all_games: Dict of (p1_strat, p2_strat) -> games list
        strategy_names: List of strategy names
        num_bootstrap: Number of bootstrap samples
        seed: Random seed
        verbose: Print progress
        print_each_graph: If True, print each bootstrap's BR graph

    Returns:
        (avg_br_matrix, std_br_matrix): Both of shape (n, n)
        avg_br_matrix[j, i] = frequency that strategy j is BR to strategy i
    """
    rng = np.random.default_rng(seed)
    n = len(strategy_names)

    # Collect BR matrices from each bootstrap sample
    br_matrices = []

    for b in range(num_bootstrap):
        if verbose and (b + 1) % 100 == 0:
            print(f"  Bootstrap sample {b + 1}/{num_bootstrap}")

        sample_indices = {}
        for key, games in all_games.items():
            if len(games) > 0:
                sample_indices[key] = rng.choice(len(games), size=len(games), replace=True)
            else:
                sample_indices[key] = np.array([], dtype=int)

        payoff_matrix = compute_payoff_matrix_from_games(all_games, strategy_names, sample_indices)

        br_matrix = compute_best_response_matrix(payoff_matrix)
        br_matrices.append(br_matrix)

        if print_each_graph:
            print_single_br_graph(br_matrix, strategy_names, bootstrap_num=b + 1, payoff_matrix=payoff_matrix)

    # Average over bootstrap samples
    br_matrices = np.array(br_matrices)
    avg_br_matrix = np.mean(br_matrices, axis=0) #TODO: when we avg over samples each cell (i,j) is averaged
    std_br_matrix = np.std(br_matrices, axis=0)

    return avg_br_matrix, std_br_matrix


def print_br_matrix(avg_br_matrix: np.ndarray, strategy_names: List[str]) -> None:
    """Pretty print the average best response matrix."""
    n = len(strategy_names)

    print("\nAverage Best Response Matrix:")
    print("(Row j, Col i) = Frequency that strategy j is BR to strategy i")
    print()

    header = "BR \\ Opp".ljust(12) + "".join(name.ljust(10) for name in strategy_names)
    print(header)
    print("-" * len(header))

    for j, name in enumerate(strategy_names):
        row = name.ljust(12)
        for i in range(n):
            row += f"{avg_br_matrix[j, i]:.3f}".ljust(10)
        print(row)

    print("\nSelf-loop frequencies (strategy is BR to itself):")
    for i, name in enumerate(strategy_names):
        print(f"  {name}: {avg_br_matrix[i, i]:.3f}")



def compute_preference_graph(payoff_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute preference graph over N^2 joint strategy profiles.

    Treats the symmetric meta-game as a two-player game:
      - Player 1's payoff at profile (a, b) = M[a, b]
      - Player 2's payoff at profile (a, b) = M[b, a]

    Edges represent best-response deviations:
      - (a, b) -> (a', b) if a' = argmax_{a''} M[a'', b] and a' != a
      - (a, b) -> (a, b') if b' = argmax_{b''} M[b'', a] and b' != b

    Profiles with no outgoing edges are pure Nash equilibria (sinks).

    Args:
        payoff_matrix: Shape (n, n), role-averaged payoff matrix

    Returns:
        (adj_p1, adj_p2): Two adjacency matrices of shape (n^2, n^2).
        Rows/cols indexed by profile (a, b) flattened as a*n + b.
        adj_p1[src, dst] = 1 if P1 has a BR deviation edge.
        adj_p2[src, dst] = 1 if P2 has a BR deviation edge.
    """
    n = payoff_matrix.shape[0]
    adj_p1 = np.zeros((n * n, n * n))
    adj_p2 = np.zeros((n * n, n * n))

    for a in range(n):
        for b in range(n):
            src = a * n + b

            # Player 1 BR: best response to opponent playing b
            p1_br = np.argmax(payoff_matrix[:, b])
            if p1_br != a:
                dst = p1_br * n + b
                adj_p1[src, dst] = 1.0

            # Player 2 BR: best response to opponent playing a
            p2_br = np.argmax(payoff_matrix[:, a])
            if p2_br != b:
                dst = a * n + p2_br
                adj_p2[src, dst] = 1.0

    return adj_p1, adj_p2

def bootstrap_average_preference_graph(
    all_games: Dict[Tuple[str, str], List[Dict]],
    strategy_names: List[str],
    num_bootstrap: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average preference graph over N^2 profiles via bootstrap.

    For each bootstrap sample, computes the payoff matrix and builds the
    preference graph (BR deviation edges over joint profiles). Averages
    edge frequencies across samples.

    Args:
        all_games: Dict of (p1_strat, p2_strat) -> games list
        strategy_names: List of strategy names
        num_bootstrap: Number of bootstrap samples
        seed: Random seed
        verbose: Print progress

    Returns:
        (avg_adj, std_adj): Both of shape (n^2, n^2)
        avg_adj[src, dst] = frequency that edge src->dst appears across samples
    """
    rng = np.random.default_rng(seed)
    n = len(strategy_names)

    p1_matrices = []
    p2_matrices = []

    for b in range(num_bootstrap):
        if verbose and (b + 1) % 100 == 0:
            print(f"  Bootstrap sample {b + 1}/{num_bootstrap}")

        sample_indices = {}
        for key, games in all_games.items():
            if len(games) > 0:
                sample_indices[key] = rng.choice(len(games), size=len(games), replace=True)
            else:
                sample_indices[key] = np.array([], dtype=int)

        payoff_matrix = compute_payoff_matrix_from_games(all_games, strategy_names, sample_indices)
        adj_p1, adj_p2 = compute_preference_graph(payoff_matrix)
        p1_matrices.append(adj_p1)
        p2_matrices.append(adj_p2)

    p1_matrices = np.array(p1_matrices)
    p2_matrices = np.array(p2_matrices)
    avg_p1 = np.mean(p1_matrices, axis=0)
    avg_p2 = np.mean(p2_matrices, axis=0)

    return avg_p1, avg_p2

def run_bootstrap_analysis(
    crossplay_dir: Path,
    strategy_names: List[str],
    num_bootstrap: int = 1000,
    seed: int = 42,
    output_dir: Path = None,
    save_plot: bool = True,
    print_each_graph: bool = False,
    preference_graph: bool = False,
) -> np.ndarray:
    """
    Run full bootstrap analysis and generate visualization.

    Args:
        crossplay_dir: Path to crossplay data
        strategy_names: List of strategy names
        num_bootstrap: Number of bootstrap samples
        seed: Random seed
        output_dir: Where to save results
        save_plot: Whether to generate the BR graph visualization
        print_each_graph: If True, print each bootstrap's BR graph

    Returns:
        avg_br_matrix
    """
    if preference_graph:
        print(f"Loading game data from {crossplay_dir}...")
        all_games = load_all_games(crossplay_dir, strategy_names)
        total_games = sum(len(g) for g in all_games.values())
        print(f"Loaded {total_games} total games across {len(all_games)} matchups")

        print(f"\nRunning preference graph bootstrap ({num_bootstrap} samples)...")
        avg_p1, avg_p2 = bootstrap_average_preference_graph(
            all_games, strategy_names, num_bootstrap, seed, verbose=True,
        )

        # Identify sinks (pure NE): profiles with no outgoing edges from either player
        n = len(strategy_names)
        print("\nPure Nash Equilibria (sink profiles):")
        for a in range(n):
            for b in range(n):
                src = a * n + b
                if avg_p1[src, :].sum() + avg_p2[src, :].sum() < 0.01:
                    print(f"  ({strategy_names[a]}, {strategy_names[b]})")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            results = {
                "strategy_names": strategy_names,
                "avg_p1": avg_p1.tolist(),
                "avg_p2": avg_p2.tolist(),
                "num_bootstrap": num_bootstrap,
            }
            results_path = output_dir / "bootstrap_preference_graph_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved results to {results_path}")

        if save_plot:
            try:
                from visuals.preference_graph_visual import create_preference_graph
                plot_dir = output_dir if output_dir else Path(".")
                create_preference_graph(
                    avg_p1, avg_p2, strategy_names,
                    filename="average_preference_graph",
                    save_dir=str(plot_dir),
                )
                print(f"Saved preference graph to {plot_dir / 'average_preference_graph.png'}")
            except ImportError as e:
                print(f"Could not generate visualization: {e}")

        return avg_p1, avg_p2
    
    else:
        print(f"Loading game data from {crossplay_dir}...")
        all_games = load_all_games(crossplay_dir, strategy_names)

        total_games = sum(len(g) for g in all_games.values())
        print(f"Loaded {total_games} total games across {len(all_games)} matchups")

        print("\nGames per matchup:")
        for (p1, p2), games in all_games.items():
            print(f"  {p1} vs {p2}: {len(games)} games")

        print(f"\nRunning bootstrap ({num_bootstrap} samples)...")
        avg_br_matrix, std_br_matrix = bootstrap_average_br_matrix(
            all_games, strategy_names, num_bootstrap, seed, verbose=True,
            print_each_graph=print_each_graph,
        )

        print_br_matrix(avg_br_matrix, strategy_names)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            results = {
                "strategy_names": strategy_names,
                "avg_br_matrix": avg_br_matrix.tolist(),
                "std_br_matrix": std_br_matrix.tolist(),
                "num_bootstrap": num_bootstrap,
            }

            results_path = output_dir / "bootstrap_br_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved results to {results_path}")

        if save_plot:
            try:
                plot_dir = output_dir if output_dir else Path(".")
                create_average_best_response_graph(
                    avg_br_matrix,
                    strategy_names,
                    filename="average_best_response_graph",
                    save_dir=str(plot_dir),
                )
                print(f"Saved BR graph to {plot_dir / 'average_best_response_graph.png'}")
            except ImportError as e:
                print(f"Could not generate visualization: {e}")

        return avg_br_matrix


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap analysis for BR graphs")
    parser.add_argument("--print-each", action="store_true", help="Print each bootstrap's BR graph")
    parser.add_argument("--num-bootstrap", type=int, default=1000, help="Number of bootstrap samples")
    parser.add_argument("--preference-graph", action="store_true", help="Compute preference graph over joint profiles")
    args = parser.parse_args()

    crossplay_dir = Path(__file__).parent.parent / "data" / "crossplay"
    output_dir = Path(__file__).parent.parent / "data" / "analysis"

    matrix_path = crossplay_dir / "metagame_matrix.json"
    if matrix_path.exists():
        with open(matrix_path) as f:
            matrix_data = json.load(f)
        strategy_names = matrix_data["strategy_names"]
    else:
        #strategy_names = ["walk", "tough", "soft", "nfsp", "mappo", "ppo", "psro", "openai_5.2_none", "openai_5.2_low", "ef1_bargainer", "openai_5.4_low"]
        strategy_names = ["openai_5.2_low", "openai_5.2_none", "openai_5.4_low"]

    avg_br_matrix = run_bootstrap_analysis(
        crossplay_dir=crossplay_dir,
        strategy_names=strategy_names,
        num_bootstrap=args.num_bootstrap,
        output_dir=output_dir,
        print_each_graph=args.print_each,
        preference_graph=args.preference_graph,
    )
