"""
Bootstrap computation for average best response graph.

For each bootstrap sample:
1. Sample games with replacement from each matchup
2. Construct payoff matrix from sampled games
3. Build adjacency matrix (BR relationships)
4. Average adjacency matrices to get BR frequencies
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_games_from_crossplay(crossplay_dir: Path, strategy_names: List[str]) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Load all game data from crossplay directory.

    Returns:
        Dict mapping (p1_strategy, p2_strategy) -> list of game dicts
    """
    crossplay_dir = Path(crossplay_dir)
    all_games = {}

    for strat_i in strategy_names:
        for strat_j in strategy_names:
            matchup_name = f"{strat_i}_p1_vs_{strat_j}_p2"
            games_path = crossplay_dir / matchup_name / "games.json"

            if games_path.exists():
                with open(games_path) as f:
                    data = json.load(f)
                all_games[(strat_i, strat_j)] = data['games']
            else:
                print(f"Warning: {games_path} not found")
                all_games[(strat_i, strat_j)] = []

    return all_games


def sample_games_and_compute_payoff_matrix(
    all_games: Dict[Tuple[str, str], List[Dict]],
    strategy_names: List[str],
    rng: np.random.Generator,
    sample_size: int = None,
) -> np.ndarray:
    """
    Sample games with replacement and compute payoff matrix.

    Payoff[i, j] = strategy i's average payoff vs strategy j
    (averaged over i as P1 and i as P2)

    Args:
        all_games: Dict of (p1_strat, p2_strat) -> games list
        strategy_names: List of strategy names
        rng: Random number generator
        sample_size: Number of games to sample per matchup direction.
                    If None, samples same size as original (standard bootstrap).
                    Use smaller values (e.g., 200) to get more variance.
    """
    n = len(strategy_names)
    payoff_matrix = np.zeros((n, n))

    for i, strat_i in enumerate(strategy_names):
        for j, strat_j in enumerate(strategy_names):
            # Games where i is P1, j is P2
            games_i_p1 = all_games.get((strat_i, strat_j), [])
            # Games where j is P1, i is P2
            games_j_p1 = all_games.get((strat_j, strat_i), [])

            payoffs = []

            # Sample from games where i is P1
            if len(games_i_p1) > 0:
                n_sample = sample_size if sample_size else len(games_i_p1)
                indices = rng.choice(len(games_i_p1), size=n_sample, replace=True)
                for idx in indices:
                    payoffs.append(games_i_p1[idx]['outcome']['payoff_p1'])

            # Sample from games where i is P2
            if len(games_j_p1) > 0:
                n_sample = sample_size if sample_size else len(games_j_p1)
                indices = rng.choice(len(games_j_p1), size=n_sample, replace=True)
                for idx in indices:
                    payoffs.append(games_j_p1[idx]['outcome']['payoff_p2'])

            if payoffs:
                payoff_matrix[i, j] = np.mean(payoffs)

    return payoff_matrix


def compute_br_adjacency_matrix(payoff_matrix: np.ndarray) -> np.ndarray:
    """
    Compute best response adjacency matrix.

    adj[j, i] = 1 if strategy j is best response to strategy i
    """
    n = payoff_matrix.shape[0]
    adj = np.zeros((n, n))

    for i in range(n):  # For each opponent i
        # Find strategy with highest payoff against i
        best_response = np.argmax(payoff_matrix[:, i])
        adj[best_response, i] = 1.0

    return adj


def bootstrap_average_br_matrix(
    crossplay_dir: Path,
    strategy_names: List[str],
    num_samples: int = 1000,
    seed: int = 42,
    sample_size: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average BR adjacency matrix via bootstrap.

    Args:
        crossplay_dir: Path to crossplay data
        strategy_names: List of strategy names
        num_samples: Number of bootstrap samples
        seed: Random seed

    Returns:
        (avg_br_matrix, payoff_matrix):
            avg_br_matrix[j, i] = frequency that j is BR to i
            payoff_matrix = mean payoff matrix (for reference)
    """
    print(f"Loading games from {crossplay_dir}...")
    all_games = load_games_from_crossplay(crossplay_dir, strategy_names)

    total_games = sum(len(g) for g in all_games.values())
    print(f"Loaded {total_games} games across {len(all_games)} matchups")

    # Diagnostic: Check games per matchup and payoff variance
    print("\nDiagnostics per matchup:")
    for (s1, s2), games in all_games.items():
        if len(games) > 0:
            payoffs_p1 = [g['outcome']['payoff_p1'] for g in games]
            payoffs_p2 = [g['outcome']['payoff_p2'] for g in games]
            print(f"  {s1} vs {s2}: {len(games)} games, "
                  f"P1 payoff std={np.std(payoffs_p1):.4f}, "
                  f"P2 payoff std={np.std(payoffs_p2):.4f}")
        else:
            print(f"  {s1} vs {s2}: 0 games")

    rng = np.random.default_rng(seed)
    n = len(strategy_names)

    # Collect adjacency matrices
    adj_matrices = []

    print(f"\nBootstrap sampling ({num_samples} samples)...")
    for s in range(num_samples):
        if (s + 1) % 200 == 0:
            print(f"  Sample {s + 1}/{num_samples}")

        # Sample games and compute payoff matrix
        payoff_matrix = sample_games_and_compute_payoff_matrix(all_games, strategy_names, rng, sample_size)

        # Compute BR adjacency matrix
        adj = compute_br_adjacency_matrix(payoff_matrix)
        adj_matrices.append(adj)

    # Average adjacency matrices to get frequencies
    adj_matrices = np.array(adj_matrices)
    avg_br_matrix = np.mean(adj_matrices, axis=0)

    # Diagnostic: Count unique BR graphs
    unique_graphs = set()
    for adj in adj_matrices:
        unique_graphs.add(tuple(adj.flatten()))
    print(f"\nUnique BR graphs across {num_samples} samples: {len(unique_graphs)}")
    if len(unique_graphs) == 1:
        print("WARNING: All bootstrap samples produced identical BR graphs!")
        print("This suggests: very few games per matchup, zero payoff variance,")
        print("or payoff gaps too large for sampling noise to affect BR.")

    # Also compute mean payoff matrix for reference
    mean_payoff = sample_games_and_compute_payoff_matrix(
        all_games, strategy_names, np.random.default_rng(seed)
    )

    return avg_br_matrix, mean_payoff


def print_results(avg_br_matrix: np.ndarray, payoff_matrix: np.ndarray, strategy_names: List[str]):
    """Print the results nicely."""
    n = len(strategy_names)

    print("\n" + "=" * 60)
    print("PAYOFF MATRIX (for reference)")
    print("Cell [i,j] = Strategy i's avg payoff vs Strategy j")
    print("=" * 60)

    header = "".ljust(10) + "".join(name.ljust(10) for name in strategy_names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(strategy_names):
        row = name.ljust(10)
        for j in range(n):
            row += f"{payoff_matrix[i, j]:.3f}".ljust(10)
        print(row)

    print("\n" + "=" * 60)
    print("AVERAGE BEST RESPONSE MATRIX (frequencies)")
    print("Cell [j,i] = Frequency that strategy j is BR to strategy i")
    print("=" * 60)

    header = "BR\\Opp".ljust(10) + "".join(name.ljust(10) for name in strategy_names)
    print(header)
    print("-" * len(header))
    for j, name in enumerate(strategy_names):
        row = name.ljust(10)
        for i in range(n):
            row += f"{avg_br_matrix[j, i]:.3f}".ljust(10)
        print(row)

    print("\n" + "=" * 60)
    print("SELF-LOOP FREQUENCIES (strategy is BR to itself)")
    print("=" * 60)
    for i, name in enumerate(strategy_names):
        freq = avg_br_matrix[i, i]
        status = "✓" if freq > 0.5 else "✗"
        print(f"  {name}: {freq:.3f} {status}")


if __name__ == "__main__":
    # Paths
    crossplay_dir = Path(__file__).parent.parent / "data" / "crossplay"
    output_dir = Path(__file__).parent.parent / "data" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load strategy names
    matrix_path = crossplay_dir / "metagame_matrix.json"
    with open(matrix_path) as f:
        strategy_names = json.load(f)["strategy_names"]

    print(f"Strategies: {strategy_names}")

    # Run bootstrap
    avg_br_matrix, payoff_matrix = bootstrap_average_br_matrix(
        crossplay_dir=crossplay_dir,
        strategy_names=strategy_names,
        num_samples=1000,
        seed=42,
    )

    # Print results
    print_results(avg_br_matrix, payoff_matrix, strategy_names)

    # Save results
    results = {
        "strategy_names": strategy_names,
        "avg_br_matrix": avg_br_matrix.tolist(),
        "payoff_matrix": payoff_matrix.tolist(),
    }
    with open(output_dir / "bootstrap_br_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir / 'bootstrap_br_results.json'}")

    # Generate visualization
    from avg_best_response_graph_visual import create_average_best_response_graph

    create_average_best_response_graph(
        avg_br_matrix,
        strategy_names,
        filename="average_best_response_graph",
        save_dir=str(output_dir),
    )
    print(f"Saved graph to {output_dir / 'average_best_response_graph.png'}")
