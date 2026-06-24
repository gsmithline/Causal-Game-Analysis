"""
Original paper analysis: Bootstrap metrics for UW, NW, NW+, Regret, and EF1.
Computes equilibrium-weighted welfare metrics with confidence intervals.

Optimized for large datasets with stratified bootstrap and vectorized operations.
Welfare metrics (UW, NW, NW+) are computed in raw utility space and normalized
by the expected maximum welfare constants from the game distribution.
"""

import numpy as np

# Item quantities for the bargaining game
ITEM_QUANTITIES = np.array([7, 4, 1])

# Normalization constants: expected maximum welfare over the game distribution
# (computed by generating millions of random game settings and averaging the
# maximum achievable welfare for each setting over all possible allocations)
MAX_UW = 805.9   # Expected max utilitarian welfare (sum of raw payoffs)
MAX_NW = 378.7   # Expected max Nash welfare (sqrt of product, incl. BATNA outcome)
MAX_NW_PLUS = 81.7  # Expected max NW on advantages
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import argparse
from tqdm import tqdm
import orjson
import json
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.iterative_game_analysis.metagame import MetaGame
from src.iterative_game_analysis.utils import compute_regret
import polarix as plx
import jax.numpy as jnp
HAS_POLARIX = True
import numpy as np
from itertools import product
import math
import jax
from functools import partial



def _solve_maxent_cce(eq_matrix, subset, max_iterations=1_000_000, gap_threshold=1e-4):
    """Solve maxent CCE using Polarix. Returns sigma or None if not converged."""
    game = plx.Game(
        payoffs=jnp.stack([jnp.array(eq_matrix), jnp.array(eq_matrix.T)]),
        actions=(np.array(subset), np.array(subset)),
        players=('row', 'column'),
        symmetry_groups=(0, 0),
    )
    ce = plx.solve(game, plx.ce_maxent, max_num_iterations=max_iterations)
    ce_gap = float(ce.extra['ce_gap'])
    if ce_gap > gap_threshold:
        return None
    sigma = np.array(plx.marginals_from_joint(ce.joint)[0])
    return sigma

def _solve_maxent_affinity(eq_matrix, subset, max_iterations=1_000_000, gap_threshold=1e-4, kernel_variance=100):
    game = plx.Game(
        payoffs=jnp.stack([jnp.array(eq_matrix), jnp.array(eq_matrix.T)]),
        actions=(np.array(subset), np.array(subset)),
        players=('row', 'column'),
        symmetry_groups=(0, 0),
    )

    key = jax.random.PRNGKey(42)
    kernel_fn = plx.affinity_kernel(key, kernel_variance=kernel_variance) #affinity kernel
    solver = partial(plx.max_affinity_entropy_marginals, kernel_fn=kernel_fn, )
    result = plx.solve(game, solver, max_num_iterations=max_iterations)
    sigma = np.array(result.marginals[0])
    return sigma



def load_json(path):
        with open(path, 'rb') as f:
            return orjson.loads(f.read())
def dump_json(obj, path):
        with open(path, 'wb') as f:
            f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))

# import json
#     def load_json(path):
#         with open(path, 'r') as f:
#             return json.load(f)
#     def dump_json(obj, path):
#         import json
#         with open(path, 'w') as f:
#             json.dump(obj, f, indent=2)


def is_ef1_vectorized(
    utilities_p1: np.ndarray,
    utilities_p2: np.ndarray,
    allocation_p1: np.ndarray,
    allocation_p2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized EF1 check for multiple games.

    Args:
        utilities_p1: [n_games, 3] P1's values per item type
        utilities_p2: [n_games, 3] P2's values per item type
        allocation_p1: [n_games, 3] P1's allocation
        allocation_p2: [n_games, 3] P2's allocation

    Returns:
        [n_games] boolean array, True if allocation is EF1
    """
    # P1's EF1 check toward P2
    p1_own_value = np.sum(utilities_p1 * allocation_p1, axis=1)
    p1_other_value = np.sum(utilities_p1 * allocation_p2, axis=1)
    # Max value of single item type in P2's bundle (where P2 has > 0)
    p1_max_item = np.max(np.where(allocation_p2 > 0, utilities_p1, 0), axis=1)
    p1_ef1 = p1_own_value >= p1_other_value - p1_max_item

    # P2's EF1 check toward P1
    p2_own_value = np.sum(utilities_p2 * allocation_p2, axis=1)
    p2_other_value = np.sum(utilities_p2 * allocation_p1, axis=1)
    # Max value of single item type in P1's bundle (where P1 has > 0)
    p2_max_item = np.max(np.where(allocation_p1 > 0, utilities_p2, 0), axis=1)
    p2_ef1 = p2_own_value >= p2_other_value - p2_max_item

    return p1_ef1 & p2_ef1


def exists_ef1_beating_batnas(
    utilities_p1: np.ndarray,
    utilities_p2: np.ndarray,
    raw_batna_p1: np.ndarray,
    raw_batna_p2: np.ndarray,
) -> np.ndarray:
    """
    For each game, check if there exists at least one allocation that is
    both EF1 and strictly beats both players' BATNAs.

    Enumerates all 80 possible allocations (items = [7, 4, 1]).

    Args:
        utilities_p1: (n_games, 3) per-item values for P1
        utilities_p2: (n_games, 3) per-item values for P2
        raw_batna_p1: (n_games,) P1's BATNA in raw utility space
        raw_batna_p2: (n_games,) P2's BATNA in raw utility space

    Returns:
        (n_games,) boolean — True if a valid allocation exists
    """
    items = ITEM_QUANTITIES  # [7, 4, 1]

    # All possible P1 allocations: (80, 3)
    all_allocs = np.array([
        (a, b, c)
        for a in range(items[0] + 1)
        for b in range(items[1] + 1)
        for c in range(items[2] + 1)
    ])
    p2_allocs = items - all_allocs 


    p1_payoffs = utilities_p1 @ all_allocs.T
    p2_payoffs = utilities_p2 @ p2_allocs.T


    _EPS = 1e-4
    beats_batna = (
        (p1_payoffs > raw_batna_p1[:, None] - _EPS)
        & (p2_payoffs > raw_batna_p2[:, None] - _EPS)
    )


    p1_other_vals = utilities_p1 @ p2_allocs.T  
    p2_has_items = p2_allocs > 0 
    p1_vals_masked = np.where(
        p2_has_items[None, :, :], utilities_p1[:, None, :], 0
    ) 
    p1_max_remove = np.max(p1_vals_masked, axis=2)  
    p1_ef1 = p1_payoffs >= p1_other_vals - p1_max_remove


    p2_other_vals = utilities_p2 @ all_allocs.T  
    p1_has_items = all_allocs > 0  
    p2_vals_masked = np.where(
        p1_has_items[None, :, :], utilities_p2[:, None, :], 0
    )  
    p2_max_remove = np.max(p2_vals_masked, axis=2)  
    p2_ef1 = p2_payoffs >= p2_other_vals - p2_max_remove

   
    valid = p1_ef1 & p2_ef1 & beats_batna  
    return np.any(valid, axis=1)  



    
    
def load_single_matchup(
    crossplay_dir: Path,
    strat_i: str,
    strat_j: str,
) -> Dict[str, np.ndarray]:
    """Load + preprocess a single matchup. Returns None if missing."""
    matchup_dir = crossplay_dir / f"{strat_i}_p1_vs_{strat_j}_p2"
    games_path = matchup_dir / "games.json"

    if not games_path.exists():
        print(f"Warning: No games found for {strat_i} vs {strat_j}")
        return None

    print(f"  Loading {strat_i} vs {strat_j}...")
    data = load_json(games_path)

    games = data["games"]

    # Cap aspiration vs openai_5.2_none (both directions) at 2000 games
    # to keep that matchup balanced — it has more games than the rest.
    pair = {strat_i, strat_j}
    if pair == {"aspiration", "openai_5.2_none"} and len(games) > 2000:
        games = games[:2000]

    return _preprocess_games(games)


def load_and_preprocess_data(
    crossplay_dir: Path,
    strategy_names: List[str],
) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    """Load all games and pre-process into numpy arrays grouped by strategy pair."""
    grouped_data = {}
    for strat_i in strategy_names:
        for strat_j in strategy_names:
            result = load_single_matchup(crossplay_dir, strat_i, strat_j)
            if result is not None:
                grouped_data[(strat_i, strat_j)] = result
    return grouped_data


def _preprocess_games(games) -> Dict[str, np.ndarray]:
    """Extract numpy arrays from a list of game dicts."""
    n_games = len(games)

    payoff_i = np.zeros(n_games)
    payoff_j = np.zeros(n_games)
    batna_i = np.zeros(n_games)
    batna_j = np.zeros(n_games)
    is_accept = np.zeros(n_games, dtype=bool)
    ef1 = np.zeros(n_games, dtype=bool)
    utilities_p1_all = np.zeros((n_games, 3))
    utilities_p2_all = np.zeros((n_games, 3))

    has_allocation = False
    utilities_p1_list = []
    utilities_p2_list = []
    allocation_p1_list = []
    allocation_p2_list = []
    accept_indices = []

    for idx, game in enumerate(games):
        outcome = game["outcome"]
        payoff_i[idx] = outcome["payoff_p1"]
        payoff_j[idx] = outcome["payoff_p2"]
        batna_i[idx] = outcome["batna_p1"]
        batna_j[idx] = outcome["batna_p2"]

        if outcome.get("utilities_p1"):
            utilities_p1_all[idx] = outcome["utilities_p1"]
            utilities_p2_all[idx] = outcome["utilities_p2"]

        if outcome["result"] == "accept":
            is_accept[idx] = True
            if outcome.get("utilities_p1") and outcome.get("allocation_p1"):
                has_allocation = True
                accept_indices.append(idx)
                utilities_p1_list.append(outcome["utilities_p1"])
                utilities_p2_list.append(outcome["utilities_p2"])
                allocation_p1_list.append(outcome["allocation_p1"])
                allocation_p2_list.append(outcome["allocation_p2"])

    if has_allocation and len(accept_indices) > 0:
        utilities_p1_arr = np.array(utilities_p1_list)
        utilities_p2_arr = np.array(utilities_p2_list)
        allocation_p1_arr = np.array(allocation_p1_list)
        allocation_p2_arr = np.array(allocation_p2_list)

        ef1_results = is_ef1_vectorized(
            utilities_p1_arr, utilities_p2_arr,
            allocation_p1_arr, allocation_p2_arr
        )
        ef1[accept_indices] = ef1_results

    max_possible_i = np.sum(utilities_p1_all * ITEM_QUANTITIES, axis=1)
    max_possible_j = np.sum(utilities_p2_all * ITEM_QUANTITIES, axis=1)
    raw_payoff_i = payoff_i * max_possible_i
    raw_payoff_j = payoff_j * max_possible_j
    raw_batna_i = batna_i * max_possible_i
    raw_batna_j = batna_j * max_possible_j

    _EPS = 1e-4
    outcome_beats_batnas = (raw_payoff_i > raw_batna_i - _EPS) & (raw_payoff_j > raw_batna_j - _EPS)

    ef1_rational_exists = exists_ef1_beating_batnas(
        utilities_p1_all, utilities_p2_all, raw_batna_i, raw_batna_j
    )

    return {
        'payoff_i': payoff_i,
        'payoff_j': payoff_j,
        'batna_i': batna_i,
        'batna_j': batna_j,
        'raw_payoff_i': raw_payoff_i,
        'raw_payoff_j': raw_payoff_j,
        'raw_batna_i': raw_batna_i,
        'raw_batna_j': raw_batna_j,
        'is_accept': is_accept,
        'ef1': ef1,
        'outcome_beats_batnas': outcome_beats_batnas,
        'ef1_rational_exists': ef1_rational_exists,
        'n_games': n_games,
    }


def build_matrices_fast(
    grouped_data: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    strategy_names: List[str],
    rng: np.random.Generator,
    raw_utility: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Build payoff matrices using stratified bootstrap (resample within each pair).

    Vectorized operations for speed.
    """
    n = len(strategy_names)
    policy_to_idx = {p: i for i, p in enumerate(strategy_names)}

    # Initialize matrices
    # payoff_p1[i,j] = i's payoff as P1 vs j
    # payoff_p2[i,j] = j's payoff as P2 vs i (in the i_p1_vs_j matchup)
    payoff_p1_matrix = np.zeros((n, n))
    payoff_p2_matrix = np.zeros((n, n))
    raw_payoff_p1_matrix = np.zeros((n, n))
    raw_payoff_p2_matrix = np.zeros((n, n))
    nw_matrix = np.zeros((n, n))
    uw_matrix = np.zeros((n, n))
    nw_plus_matrix = np.zeros((n, n))
    ef1_matrix = np.full((n, n), np.nan)
    ef1_matrix_plus = np.full((n, n), np.nan)

    for (pi, pj), data in grouped_data.items():
        i, j = policy_to_idx[pi], policy_to_idx[pj]
        n_games = data['n_games']

        if n_games == 0:
            continue

        # Stratified bootstrap: resample within this pair
        indices = rng.choice(n_games, size=n_games, replace=True)

        # Resample arrays (normalized, for payoff matrix / equilibrium)
        payoffs_i = data['payoff_i'][indices]
        payoffs_j = data['payoff_j'][indices]
        is_accept = data['is_accept'][indices]
        ef1_vals = data['ef1'][indices]

        # Payoff matrix stays in normalized space (for equilibrium computation)
        payoff_p1_matrix[i, j] = np.mean(payoffs_i)
        payoff_p2_matrix[i, j] = np.mean(payoffs_j)

        # Raw utility payoff matrix (for regret in raw space)
        raw_payoff_p1_matrix[i, j] = np.mean(data['raw_payoff_i'][indices])
        raw_payoff_p2_matrix[i, j] = np.mean(data['raw_payoff_j'][indices])

        # Choose welfare source based on flag
        if raw_utility:
            w_i = data['raw_payoff_i'][indices]
            w_j = data['raw_payoff_j'][indices]
            b_i = data['raw_batna_i'][indices]
            b_j = data['raw_batna_j'][indices]
        else:
            w_i = payoffs_i
            w_j = payoffs_j
            b_i = data['batna_i'][indices]
            b_j = data['batna_j'][indices]

        # NW: sqrt(p1 * p2)
        nw_vals = np.sqrt(np.maximum(w_i, 0) * np.maximum(w_j, 0))
        nw_matrix[i, j] = np.mean(nw_vals)

        # UW: p1 + p2
        uw_matrix[i, j] = np.mean(w_i + w_j)

        # NW+: sqrt(advantage_i * advantage_j)
        adv_i = np.maximum(w_i - b_i, 0)
        adv_j = np.maximum(w_j - b_j, 0)
        nw_plus_vals = np.sqrt(adv_i * adv_j)
        nw_plus_matrix[i, j] = np.mean(nw_plus_vals)

        # EF1 frequency: fraction of accepts that are EF1
        n_accept = np.sum(is_accept)
        if n_accept > 0:
            ef1_matrix[i, j] = np.sum(ef1_vals & is_accept) / n_accept
        
        # EF1+: only counted over games where bargaining is rational
        # numerator: accept AND ef1 AND outcome beats both BATNAs
        # denominator: accept AND exists an allocation that is EF1 and beats both BATNAs
        ef1_rational = data['ef1_rational_exists'][indices]
        outcome_beats = data['outcome_beats_batnas'][indices]
        denom = np.sum(is_accept & ef1_rational)
        if denom > 0:
            numer = np.sum(ef1_vals & is_accept & outcome_beats & ef1_rational)
            ef1_matrix_plus[i, j] = numer / denom




        

    # Symmetrize for symmetric game analysis.
    # payoff[i,j] should be i's expected payoff against j, averaged over roles:
    #   = 0.5 * (i's payoff as P1 from i_vs_j) + 0.5 * (i's payoff as P2 from j_vs_i)
    # payoff_p2_matrix[j,i] = i's payoff as P2 in the j_p1_vs_i matchup
    payoff_matrix = 0.5 * (payoff_p1_matrix + payoff_p2_matrix.T)
    raw_payoff_matrix = 0.5 * (raw_payoff_p1_matrix + raw_payoff_p2_matrix.T)

    # UW, NW, NW+, EF1 are game-level metrics (symmetric by nature), but
    # values differ across role assignments due to P1/P2 asymmetry.
    # Average both directions to remove role bias.
    nw_matrix = 0.5 * (nw_matrix + nw_matrix.T)
    uw_matrix = 0.5 * (uw_matrix + uw_matrix.T)
    nw_plus_matrix = 0.5 * (nw_plus_matrix + nw_plus_matrix.T)
    ef1_matrix = 0.5 * (ef1_matrix + ef1_matrix.T)
    ef1_matrix_plus = 0.5 * (ef1_matrix_plus + ef1_matrix_plus.T)

    return {
        "payoff": payoff_matrix,
        "raw_payoff": raw_payoff_matrix,
        "nw": nw_matrix,
        "uw": uw_matrix,
        "nw_plus": nw_plus_matrix,
        "ef1": ef1_matrix,
        "ef1_plus": ef1_matrix_plus,
    }


def summarize_per_agent(samples, strategy_names):
    """Compute summary statistics per agent from vector bootstrap samples."""
    arr = np.array(samples)  # shape: [n_bootstrap, n_agents]
    result = {}
    for i, name in enumerate(strategy_names):
        agent_samples = arr[:, i]
        result[name] = {
            "mean": float(np.mean(agent_samples)),
            "std": float(np.std(agent_samples)),
            "ci_lower": float(np.percentile(agent_samples, 2.5)),
            "ci_upper": float(np.percentile(agent_samples, 97.5))
        }
    return result


def compute_metrics_at_equilibrium(
    sigma: np.ndarray,
    matrices: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute per-agent UW, NW, NW+, EF1 at equilibrium."""
    uw = matrices["uw"] @ sigma
    nw = matrices["nw"] @ sigma
    nw_plus = matrices["nw_plus"] @ sigma

    # EF1: handle NaN by treating as 0
    ef1_clean = np.nan_to_num(matrices["ef1"], nan=0.0)
    ef1 = ef1_clean @ sigma

    ef1_plus_clean = np.nan_to_num(matrices["ef1_plus"], nan=0.0)
    ef1_plus = ef1_plus_clean @ sigma

    return {"uw": uw, "nw": nw, "nw_plus": nw_plus, "ef1": ef1, "ef1_plus": ef1_plus}


def run_bootstrap_analysis(
    crossplay_dir: Path,
    strategy_names: List[str],
    num_bootstrap: int = 1000,
    seed: int = 42,
    solver: str = "mene",
    raw_utility: bool = True,
) -> Dict:
    """Run bootstrap analysis for original paper metrics."""

    print(f"Loading and preprocessing games from {crossplay_dir}...")
    grouped_data = load_and_preprocess_data(crossplay_dir, strategy_names)

    total_games = sum(d['n_games'] for d in grouped_data.values())
    total_accepts = sum(np.sum(d['is_accept']) for d in grouped_data.values())
    print(f"Loaded {total_games:,} total games ({total_accepts:,} accepts)")

    rng = np.random.default_rng(seed)

    # Storage for bootstrap results
    uw_samples = []
    nw_samples = []
    nw_plus_samples = []
    ef1_samples = []
    ef1_plus_samples = []
    regret_samples = []
    sigma_samples = []
    support_counts = np.zeros(len(strategy_names), dtype=int)  # Track support frequency

    # Marginal decomposition: contrib[i,j] = σⱼ · M[i,j] averaged over bootstraps
    n = len(strategy_names)
    contrib_metrics = ["uw", "nw", "nw_plus", "ef1", "ef1_plus"]
    contrib_accum = {key: np.zeros((n, n)) for key in contrib_metrics}

    print(f"\nRunning {num_bootstrap} bootstrap samples...")
    n_skipped = 0

    for _ in tqdm(range(num_bootstrap), desc="Bootstrap"):
        # Build matrices with stratified bootstrap
        matrices = build_matrices_fast(grouped_data, strategy_names, rng, raw_utility=raw_utility)

        # Create metagame and solve
        eq_matrix = matrices["raw_payoff"] if raw_utility else matrices["payoff"]
        if solver == "maxent_cce":
            eq_matrix = (eq_matrix + eq_matrix.T) / 2  # symmetric game
            sigma = _solve_maxent_cce(eq_matrix, strategy_names)
            if sigma is None:
                n_skipped += 1
                continue
        elif solver == "maxaffent_ne":
            eq_matrix = (eq_matrix + eq_matrix.T) / 2  # symmetric game
            sigma = _solve_maxent_affinity(eq_matrix=eq_matrix)
            if sigma is None:
                n_skipped += 1
                continue
        else:
            metagame = MetaGame(
                policies=strategy_names,
                payoff_matrix=eq_matrix,
            )
            sigma = metagame.solve(solver)

        # Track which strategies are in support (probability > 1e-6)
        support_counts += (sigma > 1e-2).astype(int)

        # Accumulate marginal decomposition: contrib[i,j] = σⱼ · M[i,j]
        for mk in contrib_metrics:
            M = matrices[mk]
            M_clean = np.nan_to_num(M, nan=0.0)
            contrib_accum[mk] += M_clean * sigma[np.newaxis, :]

        # Compute per-agent metrics at equilibrium
        metrics = compute_metrics_at_equilibrium(sigma, matrices)
        uw_samples.append(metrics["uw"])
        nw_samples.append(metrics["nw"])
        nw_plus_samples.append(metrics["nw_plus"])
        ef1_samples.append(metrics["ef1"])
        ef1_plus_samples.append(metrics["ef1_plus"])

        # Compute per-agent regret (in raw utility space if flag is set)
        regret_matrix = matrices["raw_payoff"] if raw_utility else matrices["payoff"]
        regret_vec, _, _ = compute_regret(sigma, regret_matrix)
        regret_samples.append(regret_vec)

        sigma_samples.append(sigma.tolist())

    n_converged = num_bootstrap - n_skipped
    if n_skipped > 0:
        print(f"\n  {n_skipped}/{num_bootstrap} samples skipped (CCE did not converge)")
        print(f"  Using {n_converged} converged samples for analysis")

    # Average marginal contributions over converged samples
    marginal_contributions = {key: (acc / max(n_converged, 1)).tolist()
                              for key, acc in contrib_accum.items()}

    # Compute support frequency as percentage
    support_freq = {
        name: float(support_counts[i] / max(n_converged, 1) * 100)
        for i, name in enumerate(strategy_names)
    }

    results = {
        "strategy_names": strategy_names,
        "num_bootstrap": n_converged,
        "raw_utility": raw_utility,
        "support_frequency": support_freq,
        "uw": summarize_per_agent(uw_samples, strategy_names),
        "nw": summarize_per_agent(nw_samples, strategy_names),
        "nw_plus": summarize_per_agent(nw_plus_samples, strategy_names),
        "ef1": summarize_per_agent(ef1_samples, strategy_names),
        "ef1_plus": summarize_per_agent(ef1_plus_samples, strategy_names),
        "regret": summarize_per_agent(regret_samples, strategy_names),
        "equilibrium": {
            "mean": np.mean(sigma_samples, axis=0).tolist(),
            "std": np.std(sigma_samples, axis=0).tolist(),
        },
        "marginal_contributions": marginal_contributions,
    }

    return results


def print_results(results: Dict) -> None:
    """Pretty print the results (welfare as percentages, matching paper style)."""
    print("\n" + "=" * 60)
    print("ORIGINAL PAPER ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nStrategies: {results['strategy_names']}")
    print(f"Bootstrap samples: {results['num_bootstrap']}")

    strategy_names = results['strategy_names']

    print("\n--- Equilibrium Distribution ---")
    eq = results["equilibrium"]
    for i, name in enumerate(strategy_names):
        print(f"  {name:8s}: {eq['mean'][i]:.4f} +/- {eq['std'][i]:.4f}")

    print("\n--- Support Frequency % (in how many bootstraps) ---")
    for name in strategy_names:
        freq = results["support_frequency"][name]
        print(f"  {name:8s}: {freq:.1f}%")

    raw = results.get("raw_utility", True)
    if raw:
        print("\n--- Per-Agent Regret (95% CI) [raw utility] ---")
        for name in strategy_names:
            r = results["regret"][name]
            print(f"  {name:8s}: {r['mean']:.2f} +/- {r['std']:.2f}  "
                  f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]")
    else:
        print("\n--- Per-Agent Regret (95% CI) ---")
        for name in strategy_names:
            r = results["regret"][name]
            print(f"  {name:8s}: {r['mean']*100:.2f} +/- {r['std']*100:.2f}  "
                  f"[{r['ci_lower']*100:.2f}, {r['ci_upper']*100:.2f}]")

    if raw:
        uw_div, nw_div, nw_plus_div = MAX_UW, MAX_NW, MAX_NW_PLUS
        label = "raw utility"
    else:
        uw_div, nw_div, nw_plus_div = 2.0, 1.0, 1.0
        label = "normalized utility"

    print(f"\n--- Per-Agent NW % (95% CI) [{label}, div={nw_div}] ---")
    for name in strategy_names:
        m = results["nw"][name]
        print(f"  {name:8s}: {m['mean']/nw_div*100:.2f} +/- {m['std']/nw_div*100:.2f}  "
              f"[{m['ci_lower']/nw_div*100:.2f}, {m['ci_upper']/nw_div*100:.2f}]")

    print(f"\n--- Per-Agent NW+ % (95% CI) [{label}, div={nw_plus_div}] ---")
    for name in strategy_names:
        m = results["nw_plus"][name]
        print(f"  {name:8s}: {m['mean']/nw_plus_div*100:.2f} +/- {m['std']/nw_plus_div*100:.2f}  "
              f"[{m['ci_lower']/nw_plus_div*100:.2f}, {m['ci_upper']/nw_plus_div*100:.2f}]")

    print(f"\n--- Per-Agent UW % (95% CI) [{label}, div={uw_div}] ---")
    for name in strategy_names:
        m = results["uw"][name]
        print(f"  {name:8s}: {m['mean']/uw_div*100:.2f} +/- {m['std']/uw_div*100:.2f}  "
              f"[{m['ci_lower']/uw_div*100:.2f}, {m['ci_upper']/uw_div*100:.2f}]")

    print("\n--- Per-Agent EF1 Frequency % (95% CI) ---")
    for name in strategy_names:
        m = results["ef1"][name]
        print(f"  {name:8s}: {m['mean']*100:.2f} +/- {m['std']*100:.2f}  "
              f"[{m['ci_lower']*100:.2f}, {m['ci_upper']*100:.2f}]")

    print("\n--- Per-Agent EF1+ Frequency % (95% CI) [rational games only] ---")
    for name in strategy_names:
        m = results["ef1_plus"][name]
        print(f"  {name:8s}: {m['mean']*100:.2f} +/- {m['std']*100:.2f}  "
              f"[{m['ci_lower']*100:.2f}, {m['ci_upper']*100:.2f}]")


def plot_marginal_decomposition(results: Dict) -> None:
    """Generate stacked bar charts showing marginal decomposition of equilibrium values.

    For each strategy i, V_i = Σⱼ σⱼ · M[i,j]. Each colored segment shows
    opponent j's contribution to i's rating.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visuals.visualize_analysis import (
        STRATEGY_COLORS, DISPLAY_NAMES, STRATEGY_ORDER,
    )

    strategy_names = results["strategy_names"]
    contrib_data = results["marginal_contributions"]
    n = len(strategy_names)

    # Metric -> (normalization divisor, display label)
    metric_cfg = {
        "uw":       (MAX_UW,       "Utilitarian Welfare"),
        "nw":       (MAX_NW,       "Nash Welfare"),
        "nw_plus":  (MAX_NW_PLUS,  "NW+"),
        "ef1":      (1.0,          "EF1 Frequency"),
        "ef1_plus": (1.0,          "EF1+ Frequency"),
    }

    fig_dir = Path(__file__).parent.parent / "data" / "analysis" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, (divisor, label) in metric_cfg.items():
        contrib = np.array(contrib_data[metric_key])  # (n, n)
        # Normalize to percentage
        contrib_pct = contrib / divisor * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(n)
        bottom = np.zeros(n)

        for j in range(n):
            segment = contrib_pct[:, j]  # opponent j's contribution to each strategy i
            color = STRATEGY_COLORS.get(strategy_names[j], f"C{j}")
            display = DISPLAY_NAMES.get(strategy_names[j], strategy_names[j])
            ax.bar(x, segment, bottom=bottom, color=color, edgecolor="white",
                   linewidth=0.3, label=display)
            bottom += segment

        ax.set_xticks(x)
        ax.set_xticklabels(
            [DISPLAY_NAMES.get(s, s) for s in strategy_names],
            rotation=30, ha="right",
        )
        unit = "pp" if metric_key in ("ef1", "ef1_plus") else "% of max"
        ax.set_ylabel(f"Equilibrium Value ({unit})")
        ax.set_title(f"Marginal Decomposition – {label}")
        ax.legend(title="Opponent", bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=8)
        fig.tight_layout()
        fname = fig_dir / f"marginal_decomp_{metric_key}.png"
        fig.savefig(str(fname), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Original paper bootstrap analysis")
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver", type=str, default="mene")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--raw-utility", action=argparse.BooleanOptionalAction, default=True,
                        help="Use raw utility space (default). --no-raw-utility for normalized /2 mode.")
    parser.add_argument("--plot", action="store_true",
                        help="Generate marginal decomposition figures")
    args = parser.parse_args()

    crossplay_dir = Path(__file__).parent.parent / "data" / "crossplay"

    # Load strategy names from metagame matrix
    matrix_path = crossplay_dir / "metagame_matrix.json"
    if matrix_path.exists():
        strategy_names = load_json(matrix_path)["strategy_names"]
    else:
        strategy_names = ["walk", "tough", "nfsp", "mappo", "soft", "ppo", "psro", "ef1_bargainer", 
                          "openai_5.2_none", "openai_5.2_low", "openai_5.4_low",
                          "openai_5.4_medium", "openai_5.2_medium" ]

    results = run_bootstrap_analysis(
        crossplay_dir=crossplay_dir,
        strategy_names=strategy_names,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
        solver=args.solver,
        raw_utility=args.raw_utility,
    )

    print_results(results)

    if args.plot:
        print("\nGenerating marginal decomposition figures...")
        plot_marginal_decomposition(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent / "data" / "analysis" / "original_paper_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(results, output_path)
    print(f"\nSaved results to {output_path}")
