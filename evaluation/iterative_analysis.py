"""CLI wrapper for iterative meta-game analysis.

Thin script that calls into src/iterative_game_analysis/full_analysis.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iterative_game_analysis.full_analysis import (
    load_crossplay_to_dataframe,
    run_full_pipeline,
    save_results,
    print_results,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Iterative meta-game analysis with L1/L2/L3 bootstrap"
    )
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver", type=str, default="mene")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--raw-utility",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use raw utility space (default). --no-raw-utility for normalized.",
    )
    parser.add_argument(
        "--include-l3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Level 3 (Shapley/Banzhaf) analysis. --no-include-l3 to skip.",
    )
    parser.add_argument(
        "--l3-method",
        type=str,
        default="both",
        choices=["shapley", "banzhaf", "both"],
    )
    args = parser.parse_args()

    crossplay_dir = Path(__file__).parent.parent / "data" / "crossplay"

    # Load strategy names
    matrix_path = crossplay_dir / "metagame_matrix.json"
    if matrix_path.exists():
        with open(matrix_path) as f:
            strategy_names = json.load(f)["strategy_names"]
    else:
        strategy_names = [
            "walk", "tough", "nfsp", "mappo", "soft", "ppo", "psro",
            "openai_5.2_none",
        ]

    # Load data
    print("Loading crossplay data...")
    df = load_crossplay_to_dataframe(
        crossplay_dir, strategy_names, raw_utility=args.raw_utility,
    )

    # Run analysis
    results = run_full_pipeline(
        df=df,
        strategy_names=strategy_names,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
        solver=args.solver,
        include_l3=args.include_l3,
        l3_method=args.l3_method,
    )

    # Print
    print_results(results["aggregated"], raw_utility=args.raw_utility)

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path(__file__).parent.parent
            / "data" / "analysis" / "iterative_analysis_results"
        )

    print(f"\nSaving results...")
    save_results(results, output_path)
    print("Done.")