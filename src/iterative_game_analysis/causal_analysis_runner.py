from __future__ import annotations

import pickle
import json
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .metagame import MetaGame
from .utils import compute_regret
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.iterative_game_analysis.causal_analysis import analyze_one_bootstrap, _aggregate_results, save_results, print_results




def run_causal_pipeline(
    build_bootstrap_fn: Callable,
    strategy_names: List[str],
    solvers: List[str] = None,
    metrics: List[str] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    non_ablatable: Optional[set] = None,
    do_harsanyi: bool = True,
    do_curb: bool = True,
    mc_curb_budget: int = 50,
    max_workers: Optional[int] = None,
    verbose: bool = True,
):
    """Run the unified causal metagame analysis pipeline.

    Args:
        build_bootstrap_fn: Callable that takes (rng,) and returns a dict with:
            - "payoff": (n, n) payoff matrix for equilibrium solving
            - metric_name: (n, n) matrix for each metric
            e.g., {"payoff": M, "uw": M_uw, "nw": M_nw, "coop": M_coop}
        strategy_names: List of strategy names.
        solvers: List of solver names. Default: ["mene"].
        metrics: List of metric names (must be keys returned by build_bootstrap_fn,
            excluding "payoff"). Default: all keys except "payoff".
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.
        non_ablatable: Set of strategy names that cannot be removed.
        do_harsanyi: Whether to compute second-order LOO effects (Level 2 pairwise interaction terms; formerly labeled "Harsanyi dividends").
        do_curb: Whether to compute CURB analysis (Level 3).
        mc_curb_budget: MC samples for non-minimal CURB sets.
        verbose: Whether to print progress.

    Returns:
        Dict with:
            - "raw": List of per-bootstrap results
            - "config": Configuration parameters
            - "aggregated": Aggregated statistics
    """
    if solvers is None:
        solvers = ["mene"] 
    non_ablatable = non_ablatable or set()

    # Infer metrics from first bootstrap if not specified
    rng_probe = np.random.default_rng(seed)
    if metrics is None:
        first_boot = build_bootstrap_fn(rng_probe)
        metrics = [k for k in first_boot.keys() if k != "payoff"]

    # Pre-generate all bootstrap matrices (needed for parallel or sequential)
    if verbose:
        print(f"Generating {n_bootstrap} bootstrap samples...")
    rng = np.random.default_rng(seed)
    all_boots = []
    for _ in tqdm(range(n_bootstrap), desc="Building matrices", disable=not verbose):
        all_boots.append(build_bootstrap_fn(rng))

    # Worker function for parallel execution
    def _process_one(boot):
        payoff_matrix = boot["payoff"]
        metric_matrices = {m: boot[m] for m in metrics}
        return analyze_one_bootstrap(
            payoff_matrix=payoff_matrix,
            metric_matrices=metric_matrices,
            strategy_names=strategy_names,
            solvers=solvers,
            metrics=metrics,
            non_ablatable=non_ablatable,
            do_harsanyi=do_harsanyi,
            do_curb=do_curb,
            mc_curb_budget=mc_curb_budget,
        )

    if max_workers is not None and max_workers > 1:
        if verbose:
            print(f"Running {n_bootstrap} bootstraps with {max_workers} workers...")
        raw_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_one, boot): i
                       for i, boot in enumerate(all_boots)}
            pbar = tqdm(total=n_bootstrap, desc="Bootstrap samples", disable=not verbose)
            for future in as_completed(futures):
                raw_results.append((futures[future], future.result()))
                pbar.update(1)
            pbar.close()
        # Sort by original index to maintain reproducibility
        raw_results.sort(key=lambda x: x[0])
        raw_results = [r for _, r in raw_results]
    else:
        raw_results = []
        for boot in tqdm(all_boots, desc="Bootstrap samples", disable=not verbose):
            raw_results.append(_process_one(boot))

    #aggregate results
    aggregated = _aggregate_results(raw_results, strategy_names, solvers, metrics)

    return {
        "raw": raw_results,
        "config": {
            "strategy_names": strategy_names,
            "solvers": solvers,
            "metrics": metrics,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "do_harsanyi": do_harsanyi,
            "do_curb": do_curb,
        },
        "aggregated": aggregated,
    }



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    parser = argparse.ArgumentParser(description="Causal metagame analysis")
    parser.add_argument("--domain", choices=["bargaining", "pd"], default="bargaining")
    parser.add_argument("--n-bootstrap", type=int, default=10)
    parser.add_argument("--solvers", nargs="+", default=["mene"])
    parser.add_argument("--no-harsanyi", action="store_true")
    parser.add_argument("--no-curb", action="store_true")
    parser.add_argument("--mc-curb-budget", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Number of parallel workers. Default: sequential.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.domain == "bargaining":
        from evaluation.original_paper_analysis import (
            load_and_preprocess_data, build_matrices_fast,
        )
        crossplay_dir = Path(__file__).parent.parent.parent / "data" / "crossplay"
        strategy_names = [
            "walk", "tough", "soft", "openai_5.2_none", "openai_5.2_low",
            "ef1_bargainer", "nfsp", "ppo", "psro", "mappo",
            "openai_5.4_low", "openai_5.2_medium", "openai_5.4_medium",
        ]
        print(f"Loading bargaining data from {crossplay_dir}...")
        grouped_data = load_and_preprocess_data(crossplay_dir, strategy_names)

        def build_bootstrap(rng):
            matrices = build_matrices_fast(
                grouped_data, strategy_names, rng, raw_utility=True
            )
            return {
                "payoff": matrices["raw_payoff"],
                "uw": matrices["uw"],
                "nw": matrices["nw"],
                "nw_plus": matrices["nw_plus"],
                "ef1": np.nan_to_num(matrices["ef1"], nan=0.0),
                "ef1_plus": np.nan_to_num(matrices["ef1_plus"], nan=0.0),
            }

        metrics = ["uw", "nw", "nw_plus", "ef1", "ef1_plus"]
        non_ablatable = {"walk"}

    elif args.domain == "pd":
        pd_path = Path(__file__).parent.parent.parent / "data" / "pd_tournament.pkl"
        if not pd_path.exists():
            pd_path = Path(__file__).parent.parent.parent / "notebooks" / "pd_data" / "pd_tournament.pkl"
        print(f"Loading PD data from {pd_path}...")
        with open(pd_path, "rb") as f:
            pd_data = pickle.load(f)

        strategy_names = pd_data["strategy_names"]
        payoff_reps = pd_data["payoff_reps"]
        coop_matrix = pd_data["coop_matrix"]
        n = len(strategy_names)
        n_reps = payoff_reps.shape[0]

        def build_bootstrap(rng):
            boot = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    idx = rng.choice(n_reps, size=n_reps, replace=True)
                    boot[i, j] = payoff_reps[idx, i, j].mean()
            return {"payoff": boot, "coop": coop_matrix}

        metrics = ["payoff", "coop"]
        non_ablatable = set()

    print(f"\nRunning causal analysis:")
    print(f"Domain: {args.domain}")
    print(f"Strategies: {len(strategy_names)}")
    print(f"Solvers: {args.solvers}")
    print(f"Bootstraps: {args.n_bootstrap}")
    print(f"Second-order LOO (pairwise): {not args.no_harsanyi}")
    print(f"CURB: {not args.no_curb}")
    print()

    results = run_causal_pipeline(
        build_bootstrap_fn=build_bootstrap,
        strategy_names=strategy_names,
        solvers=args.solvers,
        metrics=metrics,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        non_ablatable=non_ablatable,
        do_harsanyi=not args.no_harsanyi,
        do_curb=not args.no_curb,
        mc_curb_budget=args.mc_curb_budget,
        max_workers=args.max_workers,
    )

    print_results(results)

    if args.output:
        save_results(results, args.output)
    else:
        default_path = Path(__file__).parent.parent.parent / "data" / "analysis" / f"causal_{args.domain}"
        save_results(results, default_path)