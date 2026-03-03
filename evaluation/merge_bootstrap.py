"""Merge chunked bootstrap results into a single analysis output."""

import argparse
import glob
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iterative_game_analysis.full_analysis import (
    aggregate_results,
    save_results,
    print_results,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge bootstrap chunk files")
    parser.add_argument(
        "--chunk-dir",
        type=str,
        required=True,
        help="Directory containing *_chunk*.pkl files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output base path (default: chunk-dir/iterative_analysis_results)",
    )
    parser.add_argument(
        "--raw-utility",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    chunk_files = sorted(glob.glob(str(chunk_dir / "*_chunk*.pkl")))

    if not chunk_files:
        print(f"No chunk files found in {chunk_dir}")
        sys.exit(1)

    print(f"Found {len(chunk_files)} chunk files:")
    all_raw = []
    for cf in chunk_files:
        with open(cf, "rb") as f:
            raw = pickle.load(f)
        print(f"  {Path(cf).name}: {len(raw)} samples")
        all_raw.extend(raw)

    print(f"\nTotal bootstrap samples: {len(all_raw)}")

    # Infer strategy names from the first sample
    strategy_names = list(all_raw[0]["l1"].keys())
    print(f"Strategies: {strategy_names}")

    aggregated = aggregate_results(all_raw, strategy_names)
    print_results(aggregated, raw_utility=args.raw_utility)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = chunk_dir / "iterative_analysis_results"

    results = {
        "raw": all_raw,
        "aggregated": aggregated,
        "config": {
            "strategy_names": strategy_names,
            "num_bootstrap": len(all_raw),
            "merged_from": [str(cf) for cf in chunk_files],
        },
    }

    print(f"\nSaving merged results...")
    save_results(results, output_path)
    print("Done.")
