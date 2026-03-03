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
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers for L3 coalition solves (default 1).",
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=None,
        help="If set, run as one chunk of a parallel job. "
             "Offsets seed by chunk-id and saves raw results only.",
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
            "openai_5.2_none", "openai_5.2_low", "ef1_bargainer"
        ]

    # Load data
    print("Loading crossplay data...")
    df = load_crossplay_to_dataframe(
        crossplay_dir, strategy_names, raw_utility=args.raw_utility,
    )

    # Offset seed for chunk mode
    seed = args.seed if args.chunk_id is None else args.seed + args.chunk_id

    # Run analysis
    results = run_full_pipeline(
        df=df,
        strategy_names=strategy_names,
        num_bootstrap=args.num_bootstrap,
        seed=seed,
        solver=args.solver,
        include_l3=args.include_l3,
        l3_method=args.l3_method,
        n_workers=args.n_workers,
    )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path(__file__).parent.parent
            / "data" / "analysis" / "iterative_analysis_results"
        )

    if args.chunk_id is not None:
        # Chunk mode: save only raw results, skip aggregation/printing
        import pickle
        chunk_path = output_path.parent / f"{output_path.name}_chunk{args.chunk_id}.pkl"
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunk_path, "wb") as f:
            pickle.dump(results["raw"], f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Chunk {args.chunk_id}: saved {len(results['raw'])} samples to {chunk_path}")
    else:
        # Normal mode: print and save everything
        print_results(results["aggregated"], raw_utility=args.raw_utility)
        print(f"\nSaving results...")
        save_results(results, output_path)
        print("Done.")