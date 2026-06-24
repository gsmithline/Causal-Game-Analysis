"""
Level 1 MENE analysis: Bootstrap means ± SE with paired significance tests.

For each metric, identifies the top-performing agent and tests whether it is
significantly better than the next-best agent using a paired bootstrap test.

Output format matches the paper's Figure 1 tables.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import argparse
import sys
import hashlib
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.original_paper_analysis import (
    load_single_matchup,
    build_matrices_fast,
    load_json,
    MAX_UW,
    MAX_NW,
    MAX_NW_PLUS,
)
from src.iterative_game_analysis.metagame import MetaGame
from src.iterative_game_analysis.utils import compute_regret


def _matchup_cache_key(crossplay_dir: Path, si: str, sj: str) -> str:
    """Hash per-matchup based on games.json mtime+size. Busts on data change."""
    path = crossplay_dir / f"{si}_p1_vs_{sj}_p2" / "games.json"
    if not path.exists():
        return None
    st = path.stat()
    h = hashlib.sha256(f"{st.st_mtime_ns}:{st.st_size}".encode())
    return h.hexdigest()[:12]


def load_grouped_cached(crossplay_dir: Path, strategy_names: List[str]) -> Dict:
    """Load matchup data with per-matchup pickle cache. Resumes on interrupt."""
    cache_dir = crossplay_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)

    grouped_data = {}
    for si in strategy_names:
        for sj in strategy_names:
            key = _matchup_cache_key(crossplay_dir, si, sj)
            if key is None:
                print(f"Warning: No games found for {si} vs {sj}")
                continue
            cache_path = cache_dir / f"{si}_p1_vs_{sj}_p2_{key}.pkl"

            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    grouped_data[(si, sj)] = pickle.load(f)
                continue

            result = load_single_matchup(crossplay_dir, si, sj)
            if result is None:
                continue
            grouped_data[(si, sj)] = result
            with open(cache_path, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    return grouped_data


def paired_bootstrap_pvalue(top_samples: np.ndarray, next_samples: np.ndarray, higher_is_better: bool) -> float:
    """Compute p-value for paired bootstrap test.

    For each bootstrap sample, we already have the metric for the top and
    next-best agent computed on the same resampled game. The test statistic
    is the paired difference. The p-value is the fraction of samples where
    the difference goes the wrong way.
    """
    diff = top_samples - next_samples
    if higher_is_better:
        # Top should be >= next, so wrong way is diff < 0
        p = np.mean(diff < 0)
    else:
        # Top should be <= next (e.g. regret), so wrong way is diff > 0
        p = np.mean(diff > 0)
    return p


def significance_stars(p: float) -> str:
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


def run_level1(
    crossplay_dir: Path,
    strategy_names: List[str],
    num_bootstrap: int = 1000,
    seed: int = 42,
    raw_utility: bool = True,
) -> Dict:
    """Run Level 1 MENE bootstrap and return per-sample metric arrays."""

    print(f"Loading games from {crossplay_dir}...")
    grouped_data = load_grouped_cached(crossplay_dir, strategy_names)

    total_games = sum(d['n_games'] for d in grouped_data.values())
    print(f"Loaded {total_games:,} total games")

    rng = np.random.default_rng(seed)
    n = len(strategy_names)

    # Per-sample arrays: [n_bootstrap, n_agents]
    uw_all = []
    nw_all = []
    nw_plus_all = []
    ef1_all = []
    ef1_plus_all = []
    regret_all = []

    print(f"\nRunning {num_bootstrap} bootstrap samples (MENE)...")
    for _ in tqdm(range(num_bootstrap), desc="Bootstrap"):
        matrices = build_matrices_fast(grouped_data, strategy_names, rng, raw_utility=raw_utility)
        eq_matrix = matrices["raw_payoff"] if raw_utility else matrices["payoff"]

        metagame = MetaGame(policies=strategy_names, payoff_matrix=eq_matrix)
        sigma = metagame.solve("mene")

        # Per-agent metrics at equilibrium: M @ sigma
        uw_all.append(matrices["uw"] @ sigma)
        nw_all.append(matrices["nw"] @ sigma)
        nw_plus_all.append(matrices["nw_plus"] @ sigma)

        ef1_clean = np.nan_to_num(matrices["ef1"], nan=0.0)
        ef1_all.append(ef1_clean @ sigma)

        ef1_plus_clean = np.nan_to_num(matrices["ef1_plus"], nan=0.0)
        ef1_plus_all.append(ef1_plus_clean @ sigma)

        regret_matrix = matrices["raw_payoff"] if raw_utility else matrices["payoff"]
        regret_vec, _, _ = compute_regret(sigma, regret_matrix)
        regret_all.append(regret_vec)

    return {
        "strategy_names": strategy_names,
        "num_bootstrap": num_bootstrap,
        "raw_utility": raw_utility,
        "uw": np.array(uw_all),
        "nw": np.array(nw_all),
        "nw_plus": np.array(nw_plus_all),
        "ef1": np.array(ef1_all),
        "ef1_plus": np.array(ef1_plus_all),
        "regret": np.array(regret_all),
    }


def print_table(results: Dict) -> None:
    """Print table matching the paper's Figure 1 format."""
    strategy_names = results["strategy_names"]
    n = len(strategy_names)
    raw = results["raw_utility"]

    # Metric configs: (key, divisor, multiplier, higher_is_better, label, fmt)
    if raw:
        uw_div, nw_div, nw_plus_div = MAX_UW, MAX_NW, MAX_NW_PLUS
    else:
        uw_div, nw_div, nw_plus_div = 2.0, 1.0, 1.0

    metrics = [
        ("regret", 1.0, 1.0, False, "Reg↓"),
        ("nw", nw_div, 100.0, True, "NW↑"),
        ("nw_plus", nw_plus_div, 100.0, True, "NW+↑"),
        ("uw", uw_div, 100.0, True, "UW↑"),
        ("ef1", 1.0, 100.0, True, "EF1↑"),
        ("ef1_plus", 1.0, 100.0, True, "EF1+↑"),
    ]

    # Precompute: for each metric, find top and next-best agent by mean
    metric_stats = {}
    for key, div, mult, higher, label in metrics:
        samples = results[key]  # [B, n]
        scaled = samples / div * mult if div != 1.0 or mult != 1.0 else samples
        means = np.mean(scaled, axis=0)
        ses = np.std(scaled, axis=0) / np.sqrt(scaled.shape[0])   


        if higher:
            sorted_idx = np.argsort(-means)
        else:
            sorted_idx = np.argsort(means)

        top_idx = sorted_idx[0]
        next_idx = sorted_idx[1]

        p = paired_bootstrap_pvalue(scaled[:, top_idx], scaled[:, next_idx], higher)
        stars = significance_stars(p)

        metric_stats[key] = {
            "means": means,
            "ses": ses,
            "top_idx": top_idx,
            "next_idx": next_idx,
            "p": p,
            "stars": stars,
            "scaled": scaled,
        }

    # Print header
    header_labels = [m[4] for m in metrics]
    print(f"\n{'Agent':<20s}", end="")
    for label in header_labels:
        print(f"  {label:>16s}", end="")
    print()
    print("-" * (20 + 18 * len(metrics)))

    # Print rows
    for i, name in enumerate(strategy_names):
        print(f"{name:<20s}", end="")
        for key, div, mult, higher, label in metrics:
            stats = metric_stats[key]
            mean = stats["means"][i]
            se = stats["ses"][i]

            # Format value
            if key == "regret":
                cell = f"{mean:.2f}±{se:.2f}"
            else:
                cell = f"{mean:.2f}±{se:.2f}"

            # Add stars if this is the top agent
            if i == stats["top_idx"]:
                cell = cell + stats["stars"]

            # Bold marker for best value
            if i == stats["top_idx"]:
                cell = f"[{cell}]"

            print(f"  {cell:>16s}", end="")
        print()

    # Print significance legend
    print()
    print("Brackets [] mark best value per metric.")
    print("Stars compare best to next-best agent (paired bootstrap):")
    print("  * p < 0.10, ** p < 0.05, *** p < 0.01")

    # Print p-values
    print("\nPaired significance (top vs next-best):")
    for key, div, mult, higher, label in metrics:
        stats = metric_stats[key]
        top_name = strategy_names[stats["top_idx"]]
        next_name = strategy_names[stats["next_idx"]]
        print(f"  {label}: {top_name} vs {next_name}, p={stats['p']:.4f} {stats['stars']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 1 MENE table with paired bootstrap significance")
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-utility", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    crossplay_dir = Path(__file__).parent.parent / "data" / "crossplay"

    matrix_path = crossplay_dir / "metagame_matrix.json"
    if matrix_path.exists():
        strategy_names = load_json(matrix_path)["strategy_names"]
    else:
        strategy_names = [
            "walk", "tough", "nfsp", "mappo", "soft", "ppo", "psro",
            "ef1_bargainer", "aspiration", "openai_5.2_none", "openai_5.2_low",
            "openai_5.4_low", "openai_5.4_medium", "openai_5.2_medium",
        ]

    results = run_level1(
        crossplay_dir=crossplay_dir,
        strategy_names=strategy_names,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
        raw_utility=args.raw_utility,
    )

    print_table(results)
