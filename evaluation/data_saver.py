"""
Data saving utilities for cross-play results.

Saves game data and summaries in organized directory structure.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from evaluation.crossplay import MatchupResult


def _summary_from_games_file(games_path: Path) -> dict:
    """Recompute summary statistics from a (possibly merged) games.json file."""
    with open(games_path, 'r') as f:
        data = json.load(f)

    games = data["games"]
    num_games = len(games)
    payoffs_p1 = [g["outcome"]["payoff_p1"] for g in games]
    payoffs_p2 = [g["outcome"]["payoff_p2"] for g in games]
    results = [g["outcome"]["result"] for g in games]
    accept_count = sum(1 for r in results if r == "accept")

    import numpy as np
    return {
        "strategy_p1": data["strategy_p1"],
        "strategy_p2": data["strategy_p2"],
        "num_games": num_games,
        "avg_payoff_p1": float(np.mean(payoffs_p1)),
        "avg_payoff_p2": float(np.mean(payoffs_p2)),
        "std_payoff_p1": float(np.std(payoffs_p1)),
        "std_payoff_p2": float(np.std(payoffs_p2)),
        "accept_rate": accept_count / num_games if num_games > 0 else 0.0,
        "walk_rate": 1.0 - (accept_count / num_games) if num_games > 0 else 0.0,
    }


def get_matchup_dirname(strategy_p1: str, strategy_p2: str) -> str:
    """Generate directory name for a matchup."""
    return f"{strategy_p1}_p1_vs_{strategy_p2}_p2"


def save_matchup_result(
    result: MatchupResult,
    output_dir: Path,
    save_games: bool = True,
    save_summary: bool = True,
) -> Path:
    """
    Save a matchup result to disk.

    Args:
        result: MatchupResult from cross-play
        output_dir: Base output directory
        save_games: Whether to save full game data
        save_summary: Whether to save summary statistics

    Returns:
        Path to the matchup directory
    """
    output_dir = Path(output_dir)
    matchup_dir = output_dir / get_matchup_dirname(result.strategy_p1, result.strategy_p2)
    matchup_dir.mkdir(parents=True, exist_ok=True)

    if save_games:
        games_path = matchup_dir / "games.json"
        new_data = result.to_dict()

        # Append to existing games if file already exists
        if games_path.exists():
            try:
                with open(games_path, 'r') as f:
                    existing_data = json.load(f)
                # Re-number new game IDs to continue from existing
                offset = len(existing_data.get("games", []))
                for g in new_data["games"]:
                    g["game_id"] += offset
                existing_data["games"].extend(new_data["games"])
                existing_data["num_games"] = len(existing_data["games"])
                new_data = existing_data
            except json.JSONDecodeError:
                print(f"  Warning: corrupt {games_path}, overwriting")
                # Fall through with just new_data

        with open(games_path, 'w') as f:
            json.dump(new_data, f, indent=2)

    if save_summary:
        summary_path = matchup_dir / "summary.json"
        # Recompute summary from merged games file (if it was appended)
        games_path = matchup_dir / "games.json"
        if games_path.exists():
            summary = _summary_from_games_file(games_path)
        else:
            summary = result.compute_summary()
        summary["timestamp"] = datetime.now().isoformat()
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    return matchup_dir


def load_matchup_result(matchup_dir: Path) -> Dict[str, Any]:
    """Load a matchup result from disk."""
    games_path = matchup_dir / "games.json"
    if games_path.exists():
        with open(games_path, 'r') as f:
            return json.load(f)
    return {}


def load_matchup_summary(matchup_dir: Path) -> Dict[str, Any]:
    """Load a matchup summary from disk."""
    summary_path = matchup_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            return json.load(f)
    return {}


def build_metagame_matrix(
    output_dir: Path,
    strategy_names: List[str],
) -> Dict[str, Any]:
    """
    Build metagame payoff matrix from saved matchup summaries.

    Cell [i, j] = Strategy i's average payoff vs Strategy j,
    averaged over both positions (i as P1 vs j as P2, and i as P2 vs j as P1).

    Args:
        output_dir: Directory containing matchup results
        strategy_names: Ordered list of strategy names

    Returns:
        Dict with payoff matrix and metadata
    """
    output_dir = Path(output_dir)
    n = len(strategy_names)

    payoff_matrix = [[0.0] * n for _ in range(n)]
    game_counts = [[0] * n for _ in range(n)]

    # For each pair (i, j), we need:
    # - i as P1 vs j as P2: i gets payoff_p1
    # - j as P1 vs i as P2: i gets payoff_p2
    # Average these for cell [i, j]

    for i, strat_i in enumerate(strategy_names):
        for j, strat_j in enumerate(strategy_names):
            # Matchup: strat_i as P1 vs strat_j as P2
            matchup_i_vs_j = output_dir / get_matchup_dirname(strat_i, strat_j)
            summary_i_vs_j = load_matchup_summary(matchup_i_vs_j)

            # Matchup: strat_j as P1 vs strat_i as P2
            matchup_j_vs_i = output_dir / get_matchup_dirname(strat_j, strat_i)
            summary_j_vs_i = load_matchup_summary(matchup_j_vs_i)

            # Strategy i's payoff when playing as P1 against j
            payoff_i_as_p1 = summary_i_vs_j.get("avg_payoff_p1", 0.0) if summary_i_vs_j else 0.0
            count_i_as_p1 = summary_i_vs_j.get("num_games", 0) if summary_i_vs_j else 0

            # Strategy i's payoff when playing as P2 against j
            payoff_i_as_p2 = summary_j_vs_i.get("avg_payoff_p2", 0.0) if summary_j_vs_i else 0.0
            count_i_as_p2 = summary_j_vs_i.get("num_games", 0) if summary_j_vs_i else 0

            # Average over both positions
            total_games = count_i_as_p1 + count_i_as_p2
            if total_games > 0:
                payoff_matrix[i][j] = (
                    payoff_i_as_p1 * count_i_as_p1 + payoff_i_as_p2 * count_i_as_p2
                ) / total_games
            game_counts[i][j] = total_games

    return {
        "strategy_names": strategy_names,
        "payoff_matrix": payoff_matrix,
        "game_counts": game_counts,
        "timestamp": datetime.now().isoformat(),
    }


def save_metagame_matrix(
    output_dir: Path,
    strategy_names: List[str],
) -> Path:
    """
    Build and save metagame matrix from matchup results.

    Args:
        output_dir: Directory containing matchup results
        strategy_names: Ordered list of strategy names

    Returns:
        Path to saved matrix file
    """
    matrix = build_metagame_matrix(output_dir, strategy_names)

    matrix_path = output_dir / "metagame_matrix.json"
    with open(matrix_path, 'w') as f:
        json.dump(matrix, f, indent=2)

    return matrix_path


def print_metagame_matrix(matrix: Dict[str, Any]) -> None:
    """Pretty print a metagame matrix."""
    names = matrix["strategy_names"]
    payoff = matrix["payoff_matrix"]

    # Header
    header = "i \\ j".ljust(12) + "".join(name.ljust(10) for name in names)
    print(header)
    print("-" * len(header))

    # Rows: each row i shows strategy i's payoff against each opponent j
    for i, name in enumerate(names):
        row = name.ljust(12)
        for j in range(len(names)):
            cell = f"{payoff[i][j]:.3f}"
            row += cell.ljust(10)
        print(row)

    print()
    print("Cell [i,j] = Strategy i's avg payoff vs Strategy j (averaged over P1 & P2)")


if __name__ == "__main__":
    # Test saving and loading
    from evaluation.policy_loader import discover_strategies
    from evaluation.crossplay import run_crossplay
    from pathlib import Path

    strategies_dir = Path(__file__).parent.parent / "strategies"
    strategies = discover_strategies(strategies_dir)

    output_dir = Path(__file__).parent.parent / "data" / "crossplay_test"

    # Run a quick test
    s1, s2 = strategies['ppo'], strategies['nfsp']
    result = run_crossplay(s1, s2, num_games=5, verbose=False)

    # Save
    matchup_dir = save_matchup_result(result, output_dir)
    print(f"Saved to: {matchup_dir}")

    # Load and verify
    summary = load_matchup_summary(matchup_dir)
    print(f"Loaded summary: {summary}")
