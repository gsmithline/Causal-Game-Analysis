#!/usr/bin/env python3
"""
Run crossplay evaluation for psro, ppo, and nfsp.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluation.policy_loader import discover_strategies
from evaluation.crossplay import run_crossplay_batched
from evaluation.data_saver import (
    save_matchup_result,
    save_metagame_matrix,
    print_metagame_matrix,
    build_metagame_matrix,
)


def main():
    strategies_dir = Path(__file__).parent / "strategies"
    output_dir = Path(__file__).parent / "data" / "crossplay"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_games = 1000
    batch_size = 64
    seed = 42

    # Load all strategies
    all_strategies = discover_strategies(strategies_dir)

    # Filter to psro, ppo, nfsp, mappo
    target_names = ["psro", "ppo", "nfsp", "mappo"]
    strategies = {name: all_strategies[name] for name in target_names if name in all_strategies}

    print(f"Loaded strategies: {list(strategies.keys())}")

    strategy_names = list(strategies.keys())
    n = len(strategy_names)
    total_matchups = n * n

    print(f"Running all pairs: {n} strategies = {total_matchups} matchups")
    print(f"Games per matchup: {num_games}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    matchup_count = 0
    for i, name_p1 in enumerate(strategy_names):
        for j, name_p2 in enumerate(strategy_names):
            matchup_count += 1
            strat_p1 = strategies[name_p1]
            strat_p2 = strategies[name_p2]

            print(f"\n[{matchup_count}/{total_matchups}] {name_p1} (P1) vs {name_p2} (P2)")

            # Run cross-play
            result = run_crossplay_batched(
                strat_p1, strat_p2,
                num_games=num_games,
                batch_size=batch_size,
                seed=seed + matchup_count,
                verbose=True,
            )

            # Save results
            save_matchup_result(result, output_dir)

            # Print summary
            summary = result.compute_summary()
            print(f"  Avg payoff: P1={summary['avg_payoff_p1']:.4f}, P2={summary['avg_payoff_p2']:.4f}")
            print(f"  Accept rate: {summary['accept_rate']:.2%}")

    # Build and save metagame matrix
    print("\n" + "=" * 60)
    print("Building metagame matrix...")
    matrix_path = save_metagame_matrix(output_dir, strategy_names)
    print(f"Saved metagame matrix to: {matrix_path}")

    # Print matrix
    print("\nMetagame Payoff Matrix:")
    print("-" * 60)
    matrix = build_metagame_matrix(output_dir, strategy_names)
    print_metagame_matrix(matrix)


if __name__ == "__main__":
    main()
