#!/usr/bin/env python3
"""
Main evaluation script for cross-play between strategies.

Usage:
    # Run all pairs (N x N matrix)
    python run_evaluation.py --all-pairs --num-games 100

    # Run specific pair
    python run_evaluation.py --pair ppo nfsp --num-games 100

    # Run specific pair in both directions
    python run_evaluation.py --pair ppo nfsp --both-directions --num-games 100

    # List available strategies
    python run_evaluation.py --list-strategies
"""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.policy_loader import discover_strategies, load_openai_strategy, Strategy
from evaluation.crossplay import run_crossplay, run_crossplay_batched
from evaluation.data_saver import (
    save_matchup_result,
    save_metagame_matrix,
    print_metagame_matrix,
    build_metagame_matrix,
)



def _run_matchup_worker(
    strat_p1: Strategy,
    strat_p2: Strategy,
    name_p1: str,
    name_p2: str,
    output_dir: Path,
    num_games: int,
    batch_size: int,
    seed: int,
    verbose: bool,
) -> dict:
    """
    Worker function for a single matchup. Thread-safe via deepcopy.

    Creates independent copies of strategies to avoid shared mutable state
    (e.g. _last_trace on LLM policies) across threads.
    """
    s1 = copy.deepcopy(strat_p1)
    s2 = copy.deepcopy(strat_p2)

    if batch_size > 1:
        result = run_crossplay_batched(
            s1, s2,
            num_games=num_games,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose,
        )
    else:
        result = run_crossplay(
            s1, s2,
            num_games=num_games,
            seed=seed,
            verbose=verbose,
        )

    matchup_dir = save_matchup_result(result, output_dir)
    summary = result.compute_summary()

    return {
        "name_p1": name_p1,
        "name_p2": name_p2,
        "summary": summary,
        "matchup_dir": matchup_dir,
    }


def run_all_pairs(
    strategies: dict,
    output_dir: Path,
    num_games: int = 100,
    batch_size: int = 32,
    seed: int = 42,
    verbose: bool = True,
    max_workers: int = 1,
) -> None:
    """
    Run cross-play for all strategy pairs (N x N matrix).

    Each strategy plays as P1 against every strategy as P2.
    Matchups run in parallel when max_workers > 1.
    """
    strategy_names = list(strategies.keys())
    n = len(strategy_names)
    total_matchups = n * n

    print(f"Running all pairs: {n} strategies = {total_matchups} matchups")
    print(f"Games per matchup: {num_games}")
    print(f"Threads: {max_workers}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    if max_workers <= 1:
        # Sequential execution
        pbar = tqdm(total=total_matchups, desc="Matchups", unit="matchup")
        matchup_count = 0
        for i, name_p1 in enumerate(strategy_names):
            for j, name_p2 in enumerate(strategy_names):
                matchup_count += 1
                strat_p1 = strategies[name_p1]
                strat_p2 = strategies[name_p2]

                pbar.set_postfix_str(f"{name_p1} vs {name_p2}")

                if batch_size > 1:
                    result = run_crossplay_batched(
                        strat_p1, strat_p2,
                        num_games=num_games,
                        batch_size=batch_size,
                        seed=seed + matchup_count,
                        verbose=False,
                    )
                else:
                    result = run_crossplay(
                        strat_p1, strat_p2,
                        num_games=num_games,
                        seed=seed + matchup_count,
                        verbose=False,
                    )

                save_matchup_result(result, output_dir)
                pbar.update(1)
        pbar.close()
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            matchup_count = 0
            for i, name_p1 in enumerate(strategy_names):
                for j, name_p2 in enumerate(strategy_names):
                    matchup_count += 1
                    future = executor.submit(
                        _run_matchup_worker,
                        strategies[name_p1], strategies[name_p2],
                        name_p1, name_p2,
                        output_dir, num_games, batch_size,
                        seed + matchup_count, verbose=False,
                    )
                    futures[future] = (name_p1, name_p2)

            pbar = tqdm(total=total_matchups, desc="Matchups", unit="matchup")
            for future in as_completed(futures):
                name_p1, name_p2 = futures[future]
                try:
                    result = future.result()
                    pbar.set_postfix_str(f"{name_p1} vs {name_p2}")
                except Exception as e:
                    pbar.set_postfix_str(f"{name_p1} vs {name_p2} FAILED")
                    tqdm.write(f"  {name_p1} vs {name_p2} failed: {e}")
                pbar.update(1)
            pbar.close()

    # Build and save metagame matrix
    print("\n" + "=" * 60)
    print("Building metagame matrix...")
    matrix_path = save_metagame_matrix(output_dir, strategy_names)
    print(f"Saved metagame matrix to: {matrix_path}")

    # Print matrix
    print("\nMetagame Payoff Matrix (P1 payoff / P2 payoff):")
    print("-" * 60)
    matrix = build_metagame_matrix(output_dir, strategy_names)
    print_metagame_matrix(matrix)


def run_specific_pair(
    strategies: dict,
    name_p1: str,
    name_p2: str,
    output_dir: Path,
    num_games: int = 100,
    both_directions: bool = False,
    batch_size: int = 32,
    seed: int = 42,
    verbose: bool = True,
    max_workers: int = 1,
) -> None:
    """
    Run cross-play for a specific strategy pair.

    Args:
        both_directions: If True, run both A vs B and B vs A
    """
    if name_p1 not in strategies:
        print(f"Error: Strategy '{name_p1}' not found")
        print(f"Available: {list(strategies.keys())}")
        return

    if name_p2 not in strategies:
        print(f"Error: Strategy '{name_p2}' not found")
        print(f"Available: {list(strategies.keys())}")
        return

    pairs_to_run = [(name_p1, name_p2)]
    if both_directions and name_p1 != name_p2:
        pairs_to_run.append((name_p2, name_p1))

    print(f"Running {len(pairs_to_run)} matchup(s)")
    print(f"Games per matchup: {num_games}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    if max_workers > 1 and len(pairs_to_run) > 1:
        # Parallel execution for both-directions
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, (p1_name, p2_name) in enumerate(pairs_to_run):
                future = executor.submit(
                    _run_matchup_worker,
                    strategies[p1_name], strategies[p2_name],
                    p1_name, p2_name,
                    output_dir, num_games, batch_size,
                    seed + idx, verbose=False,
                )
                futures[future] = (p1_name, p2_name)

            pbar = tqdm(total=len(pairs_to_run), desc="Matchups", unit="matchup")
            for future in as_completed(futures):
                p1_name, p2_name = futures[future]
                try:
                    result = future.result()
                    s = result["summary"]
                    pbar.set_postfix_str(f"{p1_name} vs {p2_name}")
                    tqdm.write(
                        f"  {p1_name} vs {p2_name}: "
                        f"P1={s['avg_payoff_p1']:.4f}, P2={s['avg_payoff_p2']:.4f}, "
                        f"Accept={s['accept_rate']:.2%}"
                    )
                except Exception as e:
                    tqdm.write(f"  {p1_name} vs {p2_name} failed: {e}")
                pbar.update(1)
            pbar.close()
    else:
        # Sequential execution
        pbar = tqdm(total=len(pairs_to_run), desc="Matchups", unit="matchup")
        for p1_name, p2_name in pairs_to_run:
            strat_p1 = strategies[p1_name]
            strat_p2 = strategies[p2_name]

            pbar.set_postfix_str(f"{p1_name} vs {p2_name}")

            if batch_size > 1:
                result = run_crossplay_batched(
                    strat_p1, strat_p2,
                    num_games=num_games,
                    batch_size=batch_size,
                    seed=seed,
                    verbose=False,
                )
            else:
                result = run_crossplay(
                    strat_p1, strat_p2,
                    num_games=num_games,
                    seed=seed,
                    verbose=False,
                )

            matchup_dir = save_matchup_result(result, output_dir)
            summary = result.compute_summary()
            tqdm.write(
                f"  {p1_name} vs {p2_name}: "
                f"P1={summary['avg_payoff_p1']:.4f}, P2={summary['avg_payoff_p2']:.4f}, "
                f"Accept={summary['accept_rate']:.2%}"
            )
            pbar.update(1)
        pbar.close()


def list_strategies(strategies_dir: Path) -> None:
    """List all available strategies."""
    strategies = discover_strategies(strategies_dir)
    print(f"\nAvailable strategies ({len(strategies)}):")
    print("-" * 40)
    for name, strat in strategies.items():
        print(f"  {name:12} - {strat.algorithm} ({type(strat.p1_policy).__name__})")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-play evaluation between trained strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all pairs with 100 games each
  python run_evaluation.py --all-pairs --num-games 100

  # Run PPO vs NFSP (PPO as P1)
  python run_evaluation.py --pair ppo nfsp --num-games 100

  # Run PPO vs NFSP in both directions
  python run_evaluation.py --pair ppo nfsp --both-directions --num-games 100

  # Run one strategy vs all others (both directions)
  python run_evaluation.py --vs-all openai_5.2_none -n 10 -t 4

  # List available strategies
  python run_evaluation.py --list-strategies
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--all-pairs",
        action="store_true",
        help="Run cross-play for all strategy pairs (N x N)"
    )
    mode_group.add_argument(
        "--pair",
        nargs=2,
        metavar=("P1", "P2"),
        help="Run cross-play for a specific pair (P1 strategy, P2 strategy)"
    )
    mode_group.add_argument(
        "--vs-all",
        metavar="STRATEGY",
        help="Run a strategy against all others in both directions"
    )
    mode_group.add_argument(
        "--list-strategies",
        action="store_true",
        help="List available strategies and exit"
    )

    # Options
    parser.add_argument(
        "--num-games", "-n",
        type=int,
        default=100,
        help="Number of games per matchup (default: 100)"
    )
    parser.add_argument(
        "--both-directions",
        action="store_true",
        help="For --pair mode: run both A vs B and B vs A"
    )
    parser.add_argument(
        "--strategies-dir",
        type=Path,
        default=Path(__file__).parent.parent / "strategies",
        help="Directory containing trained strategies"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "crossplay",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for parallel game evaluation (default: 32)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=1,
        help="Number of threads for parallel matchup execution (default: 1, sequential)"
    )
    parser.add_argument(
        "--openai-models",
        nargs="+",
        metavar="SPEC",
        help="OpenAI model specs to load (e.g. 5 5.2:none 5.2:low 5.1:medium)"
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        metavar="STRATEGY",
        help="Exclude these strategies from the run (e.g. --exclude walk tough)"
    )

    args = parser.parse_args()

    # Handle list mode
    if args.list_strategies:
        list_strategies(args.strategies_dir)
        return

    # Load strategies
    print(f"Loading strategies from: {args.strategies_dir}")
    strategies = discover_strategies(args.strategies_dir)

    if not strategies:
        print("Error: No strategies found!")
        return

    # Load additional OpenAI models from --openai-models specs
    if args.openai_models:
        for spec in args.openai_models:
            if ":" in spec:
                model, effort = spec.split(":", 1)
            else:
                model, effort = spec, "none"
            try:
                strat = load_openai_strategy(name="openai", path=Path(), model=model, reasoning_effort=effort)
                strategies[strat.name] = strat
                print(f"Loaded OpenAI strategy: {strat.name}")
            except Exception as e:
                print(f"Error loading OpenAI model {spec}: {e}")

    # Remove excluded strategies
    if args.exclude:
        for name in args.exclude:
            if name in strategies:
                del strategies[name]
                print(f"Excluded strategy: {name}")
            else:
                print(f"Warning: --exclude '{name}' not found in loaded strategies")

    print(f"Loaded {len(strategies)} strategies: {list(strategies.keys())}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    if args.vs_all:
        name = args.vs_all
        if name not in strategies:
            print(f"Error: Strategy '{name}' not found")
            print(f"Available: {list(strategies.keys())}")
            return

        # Build pairs: name vs all others (both directions)
        pairs = []
        for other in strategies:
            pairs.append((name, other))
            if other != name:
                pairs.append((other, name))

        print(f"Running {name} vs all: {len(pairs)} matchups")
        print(f"Games per matchup: {args.num_games}")
        print(f"Threads: {args.threads}")
        print(f"Output directory: {args.output_dir}")
        print("=" * 60)

        if args.threads > 1:
            with ThreadPoolExecutor(max_workers=args.threads) as executor:
                futures = {}
                for idx, (p1, p2) in enumerate(pairs):
                    future = executor.submit(
                        _run_matchup_worker,
                        strategies[p1], strategies[p2],
                        p1, p2,
                        args.output_dir, args.num_games, args.batch_size,
                        args.seed + idx, verbose=False,
                    )
                    futures[future] = (p1, p2)

                pbar = tqdm(total=len(pairs), desc="Matchups", unit="matchup")
                for future in as_completed(futures):
                    p1, p2 = futures[future]
                    try:
                        result = future.result()
                        s = result["summary"]
                        pbar.set_postfix_str(f"{p1} vs {p2}")
                        tqdm.write(
                            f"  {p1} vs {p2}: "
                            f"P1={s['avg_payoff_p1']:.4f}, P2={s['avg_payoff_p2']:.4f}, "
                            f"Accept={s['accept_rate']:.2%}"
                        )
                    except Exception as e:
                        tqdm.write(f"  {p1} vs {p2} failed: {e}")
                    pbar.update(1)
                pbar.close()
        else:
            pbar = tqdm(total=len(pairs), desc="Matchups", unit="matchup")
            for idx, (p1, p2) in enumerate(pairs):
                pbar.set_postfix_str(f"{p1} vs {p2}")
                strat_p1 = strategies[p1]
                strat_p2 = strategies[p2]

                if args.batch_size > 1:
                    result = run_crossplay_batched(
                        strat_p1, strat_p2,
                        num_games=args.num_games,
                        batch_size=args.batch_size,
                        seed=args.seed + idx,
                        verbose=False,
                    )
                else:
                    result = run_crossplay(
                        strat_p1, strat_p2,
                        num_games=args.num_games,
                        seed=args.seed + idx,
                        verbose=False,
                    )

                save_matchup_result(result, args.output_dir)
                summary = result.compute_summary()
                tqdm.write(
                    f"  {p1} vs {p2}: "
                    f"P1={summary['avg_payoff_p1']:.4f}, P2={summary['avg_payoff_p2']:.4f}, "
                    f"Accept={summary['accept_rate']:.2%}"
                )
                pbar.update(1)
            pbar.close()
        return

    if args.all_pairs:
        run_all_pairs(
            strategies=strategies,
            output_dir=args.output_dir,
            num_games=args.num_games,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose=not args.quiet,
            max_workers=args.threads,
        )
    elif args.pair:
        run_specific_pair(
            strategies=strategies,
            name_p1=args.pair[0],
            name_p2=args.pair[1],
            output_dir=args.output_dir,
            num_games=args.num_games,
            both_directions=args.both_directions,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose=not args.quiet,
            max_workers=args.threads,
        )


if __name__ == "__main__":
    main()
