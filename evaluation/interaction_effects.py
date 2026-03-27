"""
Interaction effects via paired bootstraps.

Architecture
============
The key idea: ONE bootstrap resample of the raw data induces ALL subgames.

    resample raw data → build full N×N matrices → slice rows/cols for each subgame

Because the full game and every restricted game come from the SAME resample,
comparisons between them are PAIRED across bootstrap index b:

    d_b = W(full, b) - W(LOO_i, b)

This is a paired bootstrap test (like a paired t-test, but nonparametric).
The paired differences have much lower variance than comparing independently
bootstrapped subgames, because the shared resampled payoff matrix cancels out
the dominant source of noise.

Analysis per bootstrap sample b
-------------------------------
For each sample b, we resample the raw crossplay data (stratified by pair),
build all metric matrices (UW, NW, NW+, EF1, EF1+), then for EVERY subgame
(full, LOO, LTO) we:
  - Slice the matrices to the subgame's strategy subset
  - Solve MENE on the payoff matrix
  - Record per-agent values (M[i,:] @ σ), aggregate welfare (σᵀMσ), regret, σ

All results are indexed by bootstrap sample b, enabling paired comparisons.

Comparisons (all paired)
------------------------
  1. Singleton marginal effects: d_b = W(full, b) - W(LOO_i, b)
  2. Harsanyi dividends: d_b = W(full,b) - W(LOO_A,b) - W(LOO_B,b) + W(LTO_AB,b)
  3. Per-agent spillovers: d_b = V_j(full, b) - V_j(LOO_i, b)

Each uses element-wise paired differences across the same bootstrap index.
95% CIs on the paired differences determine statistical significance.

Parallelization
---------------
We parallelize across bootstrap samples (not across subgames), since each
sample must process all subgames from the same resample. Each worker handles
a batch of bootstrap samples.
"""

import sys
import json
import itertools
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iterative_game_analysis.metagame import MetaGame

# Optional: Polarix for maxent CCE solver
try:
    import polarix as plx
    import jax.numpy as jnp
    HAS_POLARIX = True
except ImportError:
    HAS_POLARIX = False
from src.iterative_game_analysis.utils import compute_regret
from evaluation.original_paper_analysis import (
    load_and_preprocess_data,
    build_matrices_fast,
    compute_metrics_at_equilibrium,
    summarize_per_agent,
    load_json,
    dump_json,
    MAX_UW, MAX_NW, MAX_NW_PLUS,
)

METRIC_NAMES = ["uw", "nw", "nw_plus", "ef1", "ef1_plus"]


def _solve_maxent_cce(eq_matrix, subset, max_iterations=1_000_000, gap_threshold=1e-2):
    """Solve maxent CCE using Polarix. Returns sigma or None if not converged.

    Returns None when ce_gap > gap_threshold, allowing callers to skip
    non-converged bootstrap samples.
    """
    if not HAS_POLARIX:
        raise ImportError("Polarix is required for maxent_cce solver. pip install polarix")
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

METRIC_MAX = {
    "uw": MAX_UW,
    "nw": MAX_NW,
    "nw_plus": MAX_NW_PLUS,
    "ef1": 1.0,
    "ef1_plus": 1.0,
}


# ── sub-game enumeration ─────────────────────────────────────────────


def enumerate_subgames(strategy_names, non_ablatable=None):
    """Generate all sub-games: full, leave-one-out, leave-two-out.

    Args:
        strategy_names: Full list of strategy names.
        non_ablatable: Set of strategies that cannot be removed.

    Returns:
        List of (label, subset) tuples.
    """
    non_ablatable = set(non_ablatable or [])
    ablatable = [s for s in strategy_names if s not in non_ablatable]

    subgames = [("full", list(strategy_names))]

    for s in ablatable:
        subset = [q for q in strategy_names if q != s]
        subgames.append((f"loo_{s}", subset))

    for a, b in itertools.combinations(ablatable, 2):
        subset = [q for q in strategy_names if q != a and q != b]
        subgames.append((f"lto_{a}_{b}", subset))

    return subgames


# ── fixed-eq: solve once on mean matrix, bootstrap only evaluations ──


def solve_mean_equilibria(grouped_data, strategy_names, subgames, solver, raw_utility):
    """Solve MENE once on the mean (non-bootstrapped) matrix for each subgame.

    Returns a dict mapping label -> sigma (ndarray).
    """
    # Build mean matrices using identity resample (no randomness)
    n = len(strategy_names)
    policy_to_idx = {p: i for i, p in enumerate(strategy_names)}

    # We need to build matrices without resampling. Use a dummy rng trick:
    # build_matrices_fast resamples with rng.choice(n, n, replace=True).
    # Instead, we manually build mean matrices from grouped_data.
    from evaluation.original_paper_analysis import (
        MAX_UW, MAX_NW, MAX_NW_PLUS,
    )

    payoff_p1 = np.zeros((n, n))
    payoff_p2 = np.zeros((n, n))
    raw_payoff_p1 = np.zeros((n, n))
    raw_payoff_p2 = np.zeros((n, n))
    nw_mat = np.zeros((n, n))
    uw_mat = np.zeros((n, n))
    nw_plus_mat = np.zeros((n, n))
    ef1_mat = np.full((n, n), np.nan)
    ef1_plus_mat = np.full((n, n), np.nan)

    for (pi, pj), data in grouped_data.items():
        i, j = policy_to_idx[pi], policy_to_idx[pj]
        if data['n_games'] == 0:
            continue

        payoff_p1[i, j] = np.mean(data['payoff_i'])
        payoff_p2[i, j] = np.mean(data['payoff_j'])
        raw_payoff_p1[i, j] = np.mean(data['raw_payoff_i'])
        raw_payoff_p2[i, j] = np.mean(data['raw_payoff_j'])

        if raw_utility:
            w_i, w_j = data['raw_payoff_i'], data['raw_payoff_j']
            b_i, b_j = data['raw_batna_i'], data['raw_batna_j']
        else:
            w_i, w_j = data['payoff_i'], data['payoff_j']
            b_i, b_j = data['batna_i'], data['batna_j']

        nw_mat[i, j] = np.mean(np.sqrt(np.maximum(w_i, 0) * np.maximum(w_j, 0)))
        uw_mat[i, j] = np.mean(w_i + w_j)
        adv_i = np.maximum(w_i - b_i, 0)
        adv_j = np.maximum(w_j - b_j, 0)
        nw_plus_mat[i, j] = np.mean(np.sqrt(adv_i * adv_j))

        is_accept = data['is_accept']
        ef1_vals = data['ef1']
        n_accept = np.sum(is_accept)
        if n_accept > 0:
            ef1_mat[i, j] = np.sum(ef1_vals & is_accept) / n_accept
            rational = is_accept & (data['payoff_i'] >= data['batna_i']) & (data['payoff_j'] >= data['batna_j'])
            n_rational = np.sum(rational)
            if n_rational > 0:
                ef1_plus_mat[i, j] = np.sum(ef1_vals & rational) / n_rational

    # Symmetrize
    payoff_sym = (payoff_p1 + payoff_p2.T) / 2
    raw_payoff_sym = (raw_payoff_p1 + raw_payoff_p2.T) / 2

    mean_matrices = {
        "payoff": payoff_sym,
        "raw_payoff": raw_payoff_sym,
        "uw": (uw_mat + uw_mat.T) / 2,
        "nw": (nw_mat + nw_mat.T) / 2,
        "nw_plus": (nw_plus_mat + nw_plus_mat.T) / 2,
        "ef1": np.nanmean(np.stack([ef1_mat, ef1_mat.T]), axis=0),
        "ef1_plus": np.nanmean(np.stack([ef1_plus_mat, ef1_plus_mat.T]), axis=0),
    }

    # Solve equilibrium for each subgame on the mean matrix
    fixed_sigmas = {}
    for label, subset in subgames:
        idx = [policy_to_idx[s] for s in subset]
        sub_matrices = {k: mat[np.ix_(idx, idx)] for k, mat in mean_matrices.items()}
        eq_matrix = sub_matrices["raw_payoff"] if raw_utility else sub_matrices["payoff"]
        if solver == "maxent_cce":
            sigma = _solve_maxent_cce(eq_matrix, subset)
            if sigma is None:
                raise RuntimeError(f"CCE did not converge for mean-matrix subgame '{label}'. "
                                   "Try increasing max_iterations.")
        else:
            mg = MetaGame(policies=subset, payoff_matrix=eq_matrix)
            sigma = mg.solve(solver)
        fixed_sigmas[label] = sigma
        print(f"  {label}: sigma = [{', '.join(f'{s}:{v:.3f}' for s, v in zip(subset, sigma) if v > 0.005)}]")

    return fixed_sigmas


def _eval_at_fixed_sigma(matrices, subset, strategy_names, sigma, raw_utility):
    """Evaluate metrics at a pre-computed sigma (no solving).

    Same as _analyze_one_subgame but skips the equilibrium solver.
    """
    policy_to_idx = {p: i for i, p in enumerate(strategy_names)}
    idx = [policy_to_idx[s] for s in subset]

    sub_matrices = {
        key: mat[np.ix_(idx, idx)] for key, mat in matrices.items()
    }

    per_agent = {}
    for m in METRIC_NAMES:
        M = np.nan_to_num(sub_matrices[m], nan=0.0)
        per_agent[m] = M @ sigma

    aggregate = {}
    for m in METRIC_NAMES:
        M = np.nan_to_num(sub_matrices[m], nan=0.0)
        aggregate[m] = float(sigma @ M @ sigma)

    regret_matrix = sub_matrices["raw_payoff"] if raw_utility else sub_matrices["payoff"]
    regret_vec, _, _ = compute_regret(sigma, regret_matrix)

    return {
        "per_agent": per_agent,
        "aggregate": aggregate,
        "regret": regret_vec,
        "sigma": sigma,
    }


def bootstrap_one_sample_fixed_eq(
    grouped_data, strategy_names, subgames, fixed_sigmas, rng, raw_utility=True,
):
    """Run ALL subgame evaluations on ONE bootstrap resample at FIXED equilibria.

    Like bootstrap_one_sample, but uses pre-computed sigmas instead of solving.
    Only payoff matrix noise is bootstrapped — equilibrium is held constant.
    """
    matrices = build_matrices_fast(grouped_data, strategy_names, rng,
                                   raw_utility=raw_utility)

    results = {}
    for label, subset in subgames:
        results[label] = _eval_at_fixed_sigma(
            matrices, subset, strategy_names, fixed_sigmas[label], raw_utility,
        )

    return results


# ── single bootstrap sample: all subgames at once ────────────────────


def _analyze_one_subgame(matrices, subset, strategy_names, solver, raw_utility):
    """Analyze one subgame from pre-built full-game matrices.

    Slices the full matrices down to the subset, solves equilibrium,
    and returns per-agent values, aggregate welfare, regret, and sigma.

    Args:
        matrices: Full-game metric matrices from build_matrices_fast.
        subset: Strategy names for this subgame.
        strategy_names: Full list of strategy names (for index lookup).
        solver: Equilibrium solver name.
        raw_utility: Whether to use raw utility for equilibrium.

    Returns:
        Dict with per-agent values, aggregate welfare, regret, sigma.
        Returns None if maxent_cce solver did not converge.
    """
    policy_to_idx = {p: i for i, p in enumerate(strategy_names)}
    idx = [policy_to_idx[s] for s in subset]
    n_sub = len(subset)

    # Slice all matrices to this subset
    sub_matrices = {
        key: mat[np.ix_(idx, idx)] for key, mat in matrices.items()
    }

    # Solve equilibrium on the payoff matrix
    eq_matrix = sub_matrices["raw_payoff"] if raw_utility else sub_matrices["payoff"]
    if solver == "maxent_cce":
        eq_matrix = (eq_matrix + eq_matrix.T) / 2  # symmetric game
        sigma = _solve_maxent_cce(eq_matrix, subset)
        if sigma is None:
            return None
    else:
        mg = MetaGame(policies=subset, payoff_matrix=eq_matrix)
        sigma = mg.solve(solver)

    # Per-agent metrics: M[i,:] @ sigma for each metric
    per_agent = {}
    for m in METRIC_NAMES:
        M = np.nan_to_num(sub_matrices[m], nan=0.0)
        per_agent[m] = M @ sigma  # shape (n_sub,)

    # Aggregate equilibrium welfare: σᵀMσ for each metric
    aggregate = {}
    for m in METRIC_NAMES:
        M = np.nan_to_num(sub_matrices[m], nan=0.0)
        aggregate[m] = float(sigma @ M @ sigma)

    # Regret
    regret_matrix = sub_matrices["raw_payoff"] if raw_utility else sub_matrices["payoff"]
    regret_vec, _, _ = compute_regret(sigma, regret_matrix)

    return {
        "per_agent": per_agent,       # {metric: ndarray of shape (n_sub,)}
        "aggregate": aggregate,        # {metric: float}
        "regret": regret_vec,          # ndarray of shape (n_sub,)
        "sigma": sigma,                # ndarray of shape (n_sub,)
    }


def bootstrap_one_sample(
    grouped_data,
    strategy_names,
    subgames,
    rng,
    solver="mene",
    raw_utility=True,
):
    """Run ALL subgame analyses on ONE bootstrap resample.

    This is the core of the paired bootstrap: one resample of the raw data
    produces the full matrix, and every subgame is a slice of that same matrix.

    Args:
        grouped_data: Output of load_and_preprocess_data (full dataset).
        strategy_names: Full list of strategy names.
        subgames: List of (label, subset) from enumerate_subgames.
        rng: numpy random generator for this sample.
        solver: Equilibrium solver name.
        raw_utility: Whether to use raw utility for equilibrium.

    Returns:
        Dict mapping label -> subgame result dict, all from the SAME resample.
    """
    # 1. Resample once → build full matrices
    matrices = build_matrices_fast(grouped_data, strategy_names, rng,
                                   raw_utility=raw_utility)

    # 2. For each subgame, slice and solve
    results = {}
    for label, subset in subgames:
        res = _analyze_one_subgame(
            matrices, subset, strategy_names, solver, raw_utility,
        )
        if res is None:
            return None  # CCE didn't converge — skip this bootstrap sample
        results[label] = res

    return results


# ── batch worker for parallelization ────────────────────────────────

# Module-level shared data for worker processes
_SHARED = None


def _init_worker(grouped_data, strategy_names, subgames, solver, raw_utility,
                  fixed_sigmas=None):
    """Initialize worker process with shared data."""
    global _SHARED
    _SHARED = {
        "grouped_data": grouped_data,
        "strategy_names": strategy_names,
        "subgames": subgames,
        "solver": solver,
        "raw_utility": raw_utility,
        "fixed_sigmas": fixed_sigmas,
    }


def _worker_batch(seed_and_count):
    """Process a batch of bootstrap samples in one worker.

    Args:
        seed_and_count: Tuple of (seed, n_samples) for this batch.

    Returns:
        List of per-sample result dicts.
    """
    seed, n_samples = seed_and_count
    rng = np.random.default_rng(seed)
    batch_results = []
    for _ in range(n_samples):
        if _SHARED["fixed_sigmas"] is not None:
            result = bootstrap_one_sample_fixed_eq(
                _SHARED["grouped_data"],
                _SHARED["strategy_names"],
                _SHARED["subgames"],
                _SHARED["fixed_sigmas"],
                rng,
                _SHARED["raw_utility"],
            )
        else:
            result = bootstrap_one_sample(
                _SHARED["grouped_data"],
                _SHARED["strategy_names"],
                _SHARED["subgames"],
                rng,
                _SHARED["solver"],
                _SHARED["raw_utility"],
            )
        if result is not None:  # None = CCE didn't converge, skip sample
            batch_results.append(result)
    return batch_results


# ── main runner ──────────────────────────────────────────────────────


def run_paired_bootstraps(
    grouped_data,
    strategy_names,
    subgames,
    num_bootstrap=1000,
    seed=42,
    solver="mene",
    raw_utility=True,
    max_workers=4,
    fixed_sigmas=None,
):
    """Run paired bootstrap: each sample produces all subgames from one resample.

    Parallelizes across bootstrap samples (not subgames). Each worker processes
    a batch of samples, where each sample resamples data once and analyzes all
    subgames from that single resample.

    Args:
        grouped_data: Output of load_and_preprocess_data.
        strategy_names: Full list of strategy names.
        subgames: List of (label, subset) from enumerate_subgames.
        num_bootstrap: Number of bootstrap samples.
        seed: Base random seed.
        solver: Equilibrium solver name.
        raw_utility: Whether to use raw utility.
        max_workers: Number of parallel workers.
        fixed_sigmas: If provided, use these pre-computed equilibria instead of
            solving per sample. Dict mapping label -> sigma ndarray.

    Returns:
        List of dicts, one per bootstrap sample. Each dict maps
        label -> subgame result (per_agent, aggregate, regret, sigma).
    """
    # Split samples into batches, one per worker
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(max_workers)

    # Divide samples across workers
    base = num_bootstrap // max_workers
    remainder = num_bootstrap % max_workers
    batches = []
    for i in range(max_workers):
        n = base + (1 if i < remainder else 0)
        if n > 0:
            ent = child_seeds[i].entropy
            s = int(ent[0]) if hasattr(ent, '__getitem__') else int(ent)
            batches.append((s, n))

    n_subgames = len(subgames)
    mode = "fixed-eq" if fixed_sigmas else "re-solve"
    print(f"\nRunning {num_bootstrap} paired bootstrap samples "
          f"({n_subgames} subgames each, {len(batches)} workers, {mode})...")

    all_samples = []

    with ProcessPoolExecutor(
        max_workers=len(batches),
        initializer=_init_worker,
        initargs=(grouped_data, strategy_names, subgames, solver, raw_utility,
                  fixed_sigmas),
    ) as executor:
        futures = {
            executor.submit(_worker_batch, batch): batch
            for batch in batches
        }

        pbar = tqdm(total=num_bootstrap, desc="Bootstrap samples", unit="sample")
        for future in as_completed(futures):
            batch_results = future.result()
            all_samples.extend(batch_results)
            pbar.update(len(batch_results))
        pbar.close()

    n_converged = len(all_samples)
    if n_converged < num_bootstrap:
        n_skipped = num_bootstrap - n_converged
        print(f"\n  {n_skipped}/{num_bootstrap} samples skipped (CCE did not converge)")
        print(f"  Using {n_converged} converged samples for analysis")

    return all_samples


# ── aggregate results ────────────────────────────────────────────────


def aggregate_samples(all_samples, subgames, strategy_names):
    """Aggregate per-sample results into summary statistics per subgame.

    Produces the same output format as the old independent-bootstrap version,
    so existing print/display functions still work.

    Args:
        all_samples: List of per-sample dicts from run_paired_bootstraps.
        subgames: List of (label, subset) tuples.
        strategy_names: Full list of strategy names.

    Returns:
        Dict mapping label -> aggregated result dict (same format as before).
    """
    num_bootstrap = len(all_samples)
    subgame_map = {label: subset for label, subset in subgames}

    all_results = {}

    for label, subset in subgames:
        n = len(subset)

        # Collect arrays across bootstrap samples for this subgame
        per_agent_samples = {m: [] for m in METRIC_NAMES}
        aggregate_welfare = {m: [] for m in METRIC_NAMES}
        regret_samples = []
        sigma_samples = []
        support_counts = np.zeros(n, dtype=int)

        for sample in all_samples:
            r = sample[label]

            sigma = r["sigma"]
            sigma_samples.append(sigma.tolist())
            support_counts += (sigma > 1e-2).astype(int)

            for m in METRIC_NAMES:
                per_agent_samples[m].append(r["per_agent"][m])
                aggregate_welfare[m].append(r["aggregate"][m])

            regret_samples.append(r["regret"])

        # Summarize
        support_freq = {
            name: float(support_counts[i] / num_bootstrap * 100)
            for i, name in enumerate(subset)
        }

        # Per-agent raw arrays (for paired comparisons)
        per_agent_raw = {}
        for m in METRIC_NAMES:
            arr = np.array(per_agent_samples[m])  # (num_bootstrap, n)
            per_agent_raw[m] = {
                name: arr[:, i].tolist() for i, name in enumerate(subset)
            }

        regret_arr = np.array(regret_samples)  # (num_bootstrap, n)
        regret_raw = {
            name: regret_arr[:, i].tolist() for i, name in enumerate(subset)
        }

        all_results[label] = {
            "label": label,
            "subset": subset,
            "num_bootstrap": num_bootstrap,
            "support_frequency": support_freq,
            # Per-agent summaries
            "uw": summarize_per_agent(per_agent_samples["uw"], subset),
            "nw": summarize_per_agent(per_agent_samples["nw"], subset),
            "nw_plus": summarize_per_agent(per_agent_samples["nw_plus"], subset),
            "ef1": summarize_per_agent(per_agent_samples["ef1"], subset),
            "ef1_plus": summarize_per_agent(per_agent_samples["ef1_plus"], subset),
            "regret": summarize_per_agent(regret_samples, subset),
            # Raw per-agent sample arrays (for paired comparisons)
            "per_agent_raw": per_agent_raw,
            "regret_raw": regret_raw,
            # Aggregate equilibrium welfare per metric
            "aggregate_welfare": {m: vals for m, vals in aggregate_welfare.items()},
            # Equilibrium distribution
            "equilibrium": {
                "mean": np.mean(sigma_samples, axis=0).tolist(),
                "std": np.std(sigma_samples, axis=0).tolist(),
            },
            "sigma_samples": sigma_samples,
        }

    return all_results


# ── paired comparisons ───────────────────────────────────────────────


def compute_paired_comparisons(all_results, strategy_names, non_ablatable=None):
    """Compare restricted-game distributions to the full game using PAIRED diffs.

    Because all subgames in a single bootstrap sample share the same resampled
    data, the differences are paired across bootstrap index b:

        d_b = metric(full, b) - metric(LOO_i, b)

    This controls for sampling noise in the payoff matrix and yields much
    tighter CIs than independent bootstraps.

    Computes:
      1. Singleton marginal effects: W(full, b) - W(LOO_i, b)
      2. Harsanyi dividends: W(full,b) - W(LOO_A,b) - W(LOO_B,b) + W(LTO_AB,b)
      3. Per-agent spillovers: V_j(full, b) - V_j(LOO_i, b)

    Returns:
        Dict with comparison summaries.
    """
    non_ablatable = set(non_ablatable or [])
    ablatable = [s for s in strategy_names if s not in non_ablatable]
    full_agg = all_results["full"]["aggregate_welfare"]

    def _summarize_diff(arr):
        """Summarize a paired bootstrap difference distribution at multiple alpha levels."""
        ci_90_lo, ci_90_hi = np.percentile(arr, 5), np.percentile(arr, 95)
        ci_95_lo, ci_95_hi = np.percentile(arr, 2.5), np.percentile(arr, 97.5)
        ci_99_lo, ci_99_hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)

        # IQR-trimmed: keep only inner 50% of samples, recompute CIs
        q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr_mask = (arr >= q25) & (arr <= q75)
        iqr_arr = arr[iqr_mask]
        iqr_ci_90 = [float(np.percentile(iqr_arr, 5)), float(np.percentile(iqr_arr, 95))]
        iqr_ci_95 = [float(np.percentile(iqr_arr, 2.5)), float(np.percentile(iqr_arr, 97.5))]
        iqr_ci_99 = [float(np.percentile(iqr_arr, 0.5)), float(np.percentile(iqr_arr, 99.5))]

        return {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "iqr_mean": float(np.mean(iqr_arr)),
            "std": float(np.std(arr)),
            "ci_90": [float(ci_90_lo), float(ci_90_hi)],
            "ci_95": [float(ci_95_lo), float(ci_95_hi)],
            "ci_99": [float(ci_99_lo), float(ci_99_hi)],
            "sig_10": bool(ci_90_lo > 0 or ci_90_hi < 0),
            "sig_05": bool(ci_95_lo > 0 or ci_95_hi < 0),
            "sig_01": bool(ci_99_lo > 0 or ci_99_hi < 0),
            # IQR-trimmed significance (CIs on inner 50% of samples)
            "iqr_ci_90": iqr_ci_90,
            "iqr_ci_95": iqr_ci_95,
            "iqr_ci_99": iqr_ci_99,
            "iqr_sig_10": bool(iqr_ci_90[0] > 0 or iqr_ci_90[1] < 0),
            "iqr_sig_05": bool(iqr_ci_95[0] > 0 or iqr_ci_95[1] < 0),
            "iqr_sig_01": bool(iqr_ci_99[0] > 0 or iqr_ci_99[1] < 0),
        }

    # 1. Singleton marginal effects (paired): W(full, b) - W(LOO_i, b)
    marginal_effects = {m: {} for m in METRIC_NAMES}
    for s in ablatable:
        label = f"loo_{s}"
        if label not in all_results:
            continue
        loo_agg = all_results[label]["aggregate_welfare"]
        for m in METRIC_NAMES:
            # Element-wise paired diff: same bootstrap index b
            diff = np.array(full_agg[m]) - np.array(loo_agg[m])
            marginal_effects[m][s] = _summarize_diff(diff)

    # 2. Harsanyi dividends (paired)
    harsanyi = {m: {} for m in METRIC_NAMES}
    for a, b in itertools.combinations(ablatable, 2):
        loo_a = all_results.get(f"loo_{a}", {}).get("aggregate_welfare")
        loo_b = all_results.get(f"loo_{b}", {}).get("aggregate_welfare")
        lto_ab = all_results.get(f"lto_{a}_{b}", {}).get("aggregate_welfare")
        if not (loo_a and loo_b and lto_ab):
            continue
        for m in METRIC_NAMES:
            # All four terms from the SAME bootstrap sample b
            H = (np.array(full_agg[m])
                 - np.array(loo_a[m])
                 - np.array(loo_b[m])
                 + np.array(lto_ab[m]))
            harsanyi[m][f"{a} x {b}"] = _summarize_diff(H)

    # 3. Per-agent spillovers (paired): V_j(full, b) - V_j(LOO_i, b)
    spillovers = {}
    for s in ablatable:
        label = f"loo_{s}"
        if label not in all_results:
            continue
        loo_result = all_results[label]
        spillovers[s] = {m: {} for m in METRIC_NAMES}
        for m in METRIC_NAMES:
            full_raw = all_results["full"]["per_agent_raw"][m]
            loo_raw = loo_result["per_agent_raw"][m]
            for agent in loo_result["subset"]:
                if agent in full_raw and agent in loo_raw:
                    # Paired: same bootstrap index b
                    diff = np.array(full_raw[agent]) - np.array(loo_raw[agent])
                    spillovers[s][m][agent] = _summarize_diff(diff)

    return {
        "marginal_effects": marginal_effects,
        "harsanyi_dividends": harsanyi,
        "per_agent_spillovers": spillovers,
    }


# ── display ──────────────────────────────────────────────────────────


def print_subgame_results(all_results):
    """Print full results for each sub-game."""

    print("\n" + "=" * 90)
    print("SUB-GAME BOOTSTRAP RESULTS (PAIRED)")
    print("All subgames derived from the same bootstrap resample.")
    print("=" * 90)

    # Print full game first, then LOOs, then LTOs
    order = ["full"]
    order += sorted([k for k in all_results if k.startswith("loo_")])
    order += sorted([k for k in all_results if k.startswith("lto_")])

    for label in order:
        if label not in all_results:
            continue
        r = all_results[label]
        subset = r["subset"]
        n = len(subset)
        B = r["num_bootstrap"]

        print(f"\n{'=' * 70}")
        print(f"  {label.upper()}  ({n} strategies)")
        print(f"  Strategies: {subset}")
        print(f"  Bootstrap samples: {B}")
        print(f"{'=' * 70}")

        # Equilibrium distribution
        eq = r["equilibrium"]
        print(f"\n  --- Equilibrium Distribution ---")
        for i, name in enumerate(subset):
            print(f"    {name:<20}: {eq['mean'][i]:.4f} +/- {eq['std'][i]:.4f}")

        # Support frequency
        print(f"\n  --- Support Frequency % ---")
        for name in subset:
            freq = r["support_frequency"].get(name, 0)
            print(f"    {name:<20}: {freq:.1f}%")

        # Aggregate welfare (sigma^T M sigma)
        print(f"\n  --- Aggregate Equilibrium Welfare (sigma^T M sigma) ---")
        for m in METRIC_NAMES:
            vals = r["aggregate_welfare"][m]
            arr = np.array(vals)
            div = METRIC_MAX[m]
            unit = "pp" if m in ("ef1", "ef1_plus") else "% of max"
            mean_pct = np.mean(arr) / div * 100
            std_pct = np.std(arr) / div * 100
            lo_pct = np.percentile(arr, 2.5) / div * 100
            hi_pct = np.percentile(arr, 97.5) / div * 100
            print(f"    {m:>10}: {mean_pct:>8.2f} +/- {std_pct:>5.2f}  "
                  f"[{lo_pct:>8.2f}, {hi_pct:>8.2f}] {unit}")

        # Per-agent regret
        print(f"\n  --- Per-Agent Regret (95% CI) [raw utility] ---")
        for name in subset:
            info = r["regret"].get(name, {})
            if info:
                print(f"    {name:<20}: {info['mean']:>8.2f} +/- {info['std']:>5.2f}  "
                      f"[{info['ci_lower']:>8.2f}, {info['ci_upper']:>8.2f}]")

        # Per-agent UW
        print(f"\n  --- Per-Agent UW % (95% CI) [raw utility, div={MAX_UW}] ---")
        for name in subset:
            info = r["uw"].get(name, {})
            if info:
                print(f"    {name:<20}: {info['mean']/MAX_UW*100:>8.2f} +/- {info['std']/MAX_UW*100:>5.2f}  "
                      f"[{info['ci_lower']/MAX_UW*100:>8.2f}, {info['ci_upper']/MAX_UW*100:>8.2f}]")

        # Per-agent NW
        print(f"\n  --- Per-Agent NW % (95% CI) [raw utility, div={MAX_NW}] ---")
        for name in subset:
            info = r["nw"].get(name, {})
            if info:
                print(f"    {name:<20}: {info['mean']/MAX_NW*100:>8.2f} +/- {info['std']/MAX_NW*100:>5.2f}  "
                      f"[{info['ci_lower']/MAX_NW*100:>8.2f}, {info['ci_upper']/MAX_NW*100:>8.2f}]")

        # Per-agent NW+
        print(f"\n  --- Per-Agent NW+ % (95% CI) [raw utility, div={MAX_NW_PLUS}] ---")
        for name in subset:
            info = r["nw_plus"].get(name, {})
            if info:
                print(f"    {name:<20}: {info['mean']/MAX_NW_PLUS*100:>8.2f} +/- {info['std']/MAX_NW_PLUS*100:>5.2f}  "
                      f"[{info['ci_lower']/MAX_NW_PLUS*100:>8.2f}, {info['ci_upper']/MAX_NW_PLUS*100:>8.2f}]")

        # Per-agent EF1
        print(f"\n  --- Per-Agent EF1 Frequency % (95% CI) ---")
        for name in subset:
            info = r["ef1"].get(name, {})
            if info:
                print(f"    {name:<20}: {info['mean']*100:>8.2f} +/- {info['std']*100:>5.2f}  "
                      f"[{info['ci_lower']*100:>8.2f}, {info['ci_upper']*100:>8.2f}]")

        # Per-agent EF1+
        print(f"\n  --- Per-Agent EF1+ Frequency % (95% CI) [rational games only] ---")
        for name in subset:
            info = r["ef1_plus"].get(name, {})
            if info:
                print(f"    {name:<20}: {info['mean']*100:>8.2f} +/- {info['std']*100:>5.2f}  "
                      f"[{info['ci_lower']*100:>8.2f}, {info['ci_upper']*100:>8.2f}]")


def _sig_stars(info):
    """Return significance stars: *** (p<.01), ** (p<.05), * (p<.1)."""
    if info.get("sig_01"):
        return "***"
    if info.get("sig_05"):
        return "**"
    if info.get("sig_10"):
        return "*"
    return ""


def _count_sig(effects):
    """Count significant effects at each level."""
    n01 = sum(1 for v in effects.values() if v.get("sig_01"))
    n05 = sum(1 for v in effects.values() if v.get("sig_05"))
    n10 = sum(1 for v in effects.values() if v.get("sig_10"))
    return n01, n05, n10


def print_paired_comparisons(comparisons, strategy_names):
    """Print paired comparison tables."""

    sig_legend = "Significance: * p<.10, ** p<.05, *** p<.01"

    # ── Singleton marginal effects ──
    print("\n" + "=" * 85)
    print("SINGLETON MARGINAL EFFECTS: W(S) - W(S\\{i})")
    print("Positive = removing strategy hurts welfare.")
    print("PAIRED bootstrap (same resample for full and LOO).")
    print(sig_legend)
    print("=" * 85)

    for m in METRIC_NAMES:
        effects = comparisons["marginal_effects"][m]
        if not effects:
            continue
        div = METRIC_MAX[m]
        unit = "pp" if m in ("ef1", "ef1_plus") else "% of max"
        print(f"\n  {m.upper()} ({unit}):")
        print(f"  {'Strategy':<20} {'Mean':>10} {'95% CI':>24} {'Sig':>5}")
        print("  " + "-" * 62)
        for s, info in sorted(effects.items(), key=lambda x: -abs(x[1]["mean"])):
            stars = _sig_stars(info)
            mean_pct = info["mean"] / div * 100
            lo_pct = info["ci_95"][0] / div * 100
            hi_pct = info["ci_95"][1] / div * 100
            print(f"  {s:<20} {mean_pct:>10.4f}  [{lo_pct:>10.4f}, {hi_pct:>10.4f}] {stars:>5}")
        n01, n05, n10 = _count_sig(effects)
        print(f"  Significant: {n01} at .01, {n05} at .05, {n10} at .10 (of {len(effects)})")

    # ── Harsanyi dividends ──
    print("\n" + "=" * 85)
    print("PAIRWISE HARSANYI DIVIDENDS: W(S) - W(S\\{A}) - W(S\\{B}) + W(S\\{A,B})")
    print("+ve = complementary (together they create more welfare)")
    print("-ve = substitutes (they crowd each other out)")
    print("PAIRED bootstrap.")
    print(sig_legend)
    print("=" * 85)

    for m in METRIC_NAMES:
        dividends = comparisons["harsanyi_dividends"][m]
        if not dividends:
            continue
        div = METRIC_MAX[m]
        unit = "pp" if m in ("ef1", "ef1_plus") else "% of max"
        print(f"\n  {m.upper()} ({unit}):")
        print(f"  {'Pair':<35} {'Mean':>10} {'95% CI':>24} {'Sig':>5}")
        print("  " + "-" * 77)
        for pair, info in sorted(dividends.items(), key=lambda x: -abs(x[1]["mean"])):
            stars = _sig_stars(info)
            mean_pct = info["mean"] / div * 100
            lo_pct = info["ci_95"][0] / div * 100
            hi_pct = info["ci_95"][1] / div * 100
            print(f"  {pair:<35} {mean_pct:>10.4f}  [{lo_pct:>10.4f}, {hi_pct:>10.4f}] {stars:>5}")
        n01, n05, n10 = _count_sig(dividends)
        print(f"  Significant: {n01} at .01, {n05} at .05, {n10} at .10 (of {len(dividends)})")

    # ── Per-agent spillovers ──
    print("\n" + "=" * 85)
    print("PER-AGENT SPILLOVERS: How does removing strategy i affect agent j's welfare?")
    print("Diff = V_j(full, b) - V_j(LOO_i, b). Positive = agent j benefits from i.")
    print("PAIRED bootstrap.")
    print(sig_legend)
    print("=" * 85)

    spill = comparisons.get("per_agent_spillovers", {})
    for removed in sorted(spill.keys()):
        print(f"\n  Removing: {removed}")
        print(f"  {'Agent':<20}", end="")
        for m in METRIC_NAMES:
            print(f" {'':>2}{m:>8}", end="")
        print()
        print("  " + "-" * (20 + 10 * len(METRIC_NAMES)))

        # Collect agents, sort by abs UW diff
        agents = list(spill[removed].get("uw", {}).keys())
        agents.sort(key=lambda a: -abs(spill[removed]["uw"][a]["mean"]))

        for agent in agents:
            print(f"  {agent:<20}", end="")
            for m in METRIC_NAMES:
                info = spill[removed].get(m, {}).get(agent, {})
                if info:
                    stars = _sig_stars(info)
                    div = METRIC_MAX[m]
                    val = info["mean"] / div * 100
                    s_pad = stars.ljust(3)
                    print(f" {val:>6.3f}{s_pad}", end="")
                else:
                    print(f" {'n/a':>9}", end="")
            print()

        # Count significant spillovers per metric
        for m in METRIC_NAMES:
            n01, n05, n10 = _count_sig(spill[removed].get(m, {}))
            if n10 > 0:
                print(f"    {m}: {n01}***, {n05}**, {n10}* (of {len(agents)})")


# ── main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Interaction effects via paired bootstraps"
    )
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--solver", type=str, default="mene",
        help="Equilibrium solver: mene, maxent_cce, etc. "
             "Note: maxent_cce uses Polarix/JAX which may not work with "
             "multiprocessing; use --max-workers 1 for maxent_cce.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--non-ablatable", nargs="*", default=["walk"],
        help="Strategies that cannot be removed",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--raw-utility", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--fixed-eq", action="store_true", default=False,
        help="Solve MENE once on mean matrix, bootstrap only metric evaluations",
    )
    args = parser.parse_args()

    crossplay_dir = Path(__file__).parent.parent / "data" / "crossplay"

    # Load strategy names
    matrix_path = crossplay_dir / "metagame_matrix.json"
    if matrix_path.exists():
        strategy_names = load_json(matrix_path)["strategy_names"]
    else:
        strategy_names = [
            "walk", "tough", "nfsp", "mappo", "soft", "ppo", "psro",
            "ef1_bargainer", "openai_5.2_none", "openai_5.2_low", "openai_5.4_low",
            "openai_5.4_medium", "openai_5.2_medium"
        ]

    print(f"Strategies ({len(strategy_names)}): {strategy_names}")
    print(f"Non-ablatable: {args.non_ablatable}")

    # Load data once
    print(f"\nLoading games from {crossplay_dir}...")
    grouped_data = load_and_preprocess_data(crossplay_dir, strategy_names)
    total = sum(d["n_games"] for d in grouped_data.values())
    print(f"Loaded {total:,} games across {len(grouped_data)} matchups")

    # Enumerate sub-games
    subgames = enumerate_subgames(strategy_names, set(args.non_ablatable))
    n_full = 1
    n_loo = sum(1 for l, _ in subgames if l.startswith("loo_"))
    n_lto = sum(1 for l, _ in subgames if l.startswith("lto_"))
    print(f"\nSub-games: {n_full} full + {n_loo} LOO + {n_lto} LTO = {len(subgames)} total")

    # Optionally solve equilibria once on mean matrix
    fixed_sigmas = None
    if args.fixed_eq:
        print("\nSolving equilibria on mean (non-bootstrapped) matrices...")
        fixed_sigmas = solve_mean_equilibria(
            grouped_data, strategy_names, subgames, args.solver, args.raw_utility,
        )

    # Run paired bootstraps
    all_samples = run_paired_bootstraps(
        grouped_data=grouped_data,
        strategy_names=strategy_names,
        subgames=subgames,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
        solver=args.solver,
        raw_utility=args.raw_utility,
        max_workers=args.max_workers,
        fixed_sigmas=fixed_sigmas,
    )

    # Aggregate into per-subgame summaries
    all_results = aggregate_samples(all_samples, subgames, strategy_names)

    # Print each sub-game's full results
    print_subgame_results(all_results)

    # Compute and print paired comparisons
    comparisons = compute_paired_comparisons(
        all_results, strategy_names, set(args.non_ablatable)
    )
    print_paired_comparisons(comparisons, strategy_names)

    # Save everything
    out_path = Path(args.output) if args.output else (
        Path(__file__).parent.parent / "data" / "analysis" / f"interaction_effects_mecce_1000.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "strategy_names": strategy_names,
        "non_ablatable": args.non_ablatable,
        "num_bootstrap": args.num_bootstrap,
        "solver": args.solver,
        "raw_utility": args.raw_utility,
        "paired": True,
        "fixed_eq": args.fixed_eq,
        # All sub-game results (full distributions)
        "subgame_results": all_results,
        # Comparison summaries
        "comparisons": comparisons,
    }
    dump_json(save_data, out_path)
    print(f"\nSaved results to {out_path}")


# TODO: Compositional effects analysis (iterative version)
# Aggregate equilibrium welfare (σᵀMσ) is stable across LOO games, but
# individual agent values (M[i,:] @ σ) shift significantly. Explore:
#
# 1. Spillover heatmap: rows = removed strategy, cols = remaining agents,
#    cells = % change in agent value relative to full game baseline.
#    cell(remove_i, agent_j, metric_m) =
#        (E[V_j | LOO_i] - E[V_j | full]) / E[V_j | full] * 100
#    One heatmap per metric (UW, NW, NW+, EF1, EF1+), or stacked panel.
#    Overlay significance stars from bootstrap CIs.
#    Shows "friends" vs "enemies" in the ecosystem.
#
# 2. Equilibrium sensitivity decomposition: for each LOO game, decompose
#    agent value change into:
#    - Direct effect: same σ, different matrix (lost matchup)
#    - Indirect effect: different σ, same agents (equilibrium shift)
#    Compute by evaluating old σ on restricted game vs new σ on restricted game.


if __name__ == "__main__":
    main()
