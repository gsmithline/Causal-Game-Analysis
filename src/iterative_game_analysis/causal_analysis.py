"""Unified causal metagame analysis pipeline.

Implements the three-level counterfactual framework:
  Level 1: Equilibrium analysis (per-agent metrics at equilibrium)
  Level 2: LOO + Harsanyi dividends + solver comparison
  Level 3: CURB-conditional LOO (LOO within strategically coherent restricted games)

All levels operate on the same bootstrapped empirical game per sample,
maintaining paired structure for tight confidence intervals.

Supports multiple equilibrium solvers: MENE, maxent CCE (Polarix), LLE (Polarix).

Usage:
    from iterative_game_analysis.causal_analysis import run_causal_pipeline

    results = run_causal_pipeline(
        grouped_data=grouped_data,
        strategy_names=strategy_names,
        metric_matrices_fn=build_matrices,
        solvers=["mene", "maxent_cce", "lle"],
        metrics=["uw", "nw", "coop"],
        n_bootstrap=1000,
    )
"""

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

# Optional Polarix imports
try:
    import polarix as plx
    import jax.numpy as jnp
    import jax
    HAS_POLARIX = True
except ImportError:
    HAS_POLARIX = False


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def _solve_mene(payoff_matrix, strategy_names):
    """Solve MENE. Returns sigma or None."""
    try:
        mg = MetaGame(policies=strategy_names, payoff_matrix=payoff_matrix)
        return mg.solve("mene")
    except Exception:
        return None


def _solve_maxent_cce(payoff_matrix, strategy_names, max_iterations=200_000,
                       gap_threshold=1e-3):
    """Solve maxent CCE via Polarix. Returns sigma or None if not converged."""
    if not HAS_POLARIX:
        raise ImportError("Polarix required for maxent_cce. pip install polarix")
    #eq_matrix = (payoff_matrix + payoff_matrix.T) / 2  # symmetrize for Polarix
    eq_matrix = payoff_matrix
    game = plx.Game(
        payoffs=jnp.stack([jnp.array(eq_matrix), jnp.array(eq_matrix.T)]),
        actions=(np.array(strategy_names), np.array(strategy_names)),
        players=('row', 'column'),
        symmetry_groups=(0, 0),
    )
    ce = plx.solve(game, plx.ce_maxent, max_num_iterations=max_iterations)
    ce_gap = float(ce.extra['ce_gap'])
    if ce_gap > gap_threshold:
        return None
    sigma = np.array(plx.marginals_from_joint(ce.joint)[0])
    return sigma


def _solve_lle(payoff_matrix, strategy_names, max_iterations=200_000):
    """Solve LLE via Polarix. Returns sigma or None."""
    if not HAS_POLARIX:
        raise ImportError("Polarix required for LLE. pip install polarix")
    #eq_matrix = (payoff_matrix + payoff_matrix.T) / 2
    eq_matrix = payoff_matrix
    game = plx.Game(
        payoffs=jnp.stack([jnp.array(eq_matrix), jnp.array(eq_matrix.T)]),
        actions=(np.array(strategy_names), np.array(strategy_names)),
        players=('row', 'column'),
        symmetry_groups=(0, 0),
    )
    try:
        result = plx.solve(game, plx.lle, max_num_iterations=max_iterations)
        sigma = np.array(plx.marginals_from_joint(result.joint)[0])
        return sigma
    except Exception:
        return None


SOLVER_FNS = {
    "mene": _solve_mene,
    "maxent_cce": _solve_maxent_cce,
    "lle": _solve_lle,
}


def _solve(solver_name, payoff_matrix, strategy_names):
    """Dispatch to the appropriate solver."""
    fn = SOLVER_FNS[solver_name]
    return fn(payoff_matrix, strategy_names)


# ---------------------------------------------------------------------------
# CURB set computation (import from evaluation if available, else minimal)
# ---------------------------------------------------------------------------

try:
    from evaluation.curb_analysis import (
        find_minimal_curb_sets_klimm_weibull,
        curb_closure,
    )
    HAS_CURB = True
except ImportError:
    HAS_CURB = False


# ---------------------------------------------------------------------------
# Core per-bootstrap analysis
# ---------------------------------------------------------------------------

def _compute_regret(sigma, payoff_matrix):
    """Compute per-agent regret using the same convention as original_paper_analysis.

    regret[i] = nash_value - expected_utils[i] >= 0 at equilibrium.
    Strategies in support have regret ≈ 0; out-of-support have regret > 0.
    """
    regret, _, _ = compute_regret(sigma, payoff_matrix)
    return regret


def _eval_welfare(sigma, metric_matrix):
    """Evaluate welfare: sigma^T M sigma."""
    return float(sigma @ metric_matrix @ sigma)


def _eval_per_agent(sigma, metric_matrix):
    """Evaluate per-agent values: M @ sigma."""
    return metric_matrix @ sigma


def analyze_one_bootstrap(
    payoff_matrix,
    metric_matrices,
    strategy_names,
    solvers,
    metrics,
    non_ablatable=None,
    do_harsanyi=True,
    do_curb=True,
    mc_curb_budget=50,
):
    """Run all three levels of analysis on one bootstrapped game.

    Args:
        payoff_matrix: (n, n) payoff matrix for equilibrium solving.
        metric_matrices: Dict[str, ndarray] mapping metric name to (n, n) matrix.
        strategy_names: List of strategy names.
        solvers: List of solver names (e.g., ["mene", "maxent_cce", "lle"]).
        metrics: List of metric names (keys into metric_matrices).
        non_ablatable: Set of strategy names that cannot be removed.
        do_harsanyi: Whether to compute pairwise Harsanyi dividends.
        do_curb: Whether to compute Level 3 CURB analysis.
        mc_curb_budget: Number of MC samples for non-minimal CURB sets.

    Returns:
        Dict with keys: level1, level2, level3, curb_info.
    """
    n = len(strategy_names)
    non_ablatable = non_ablatable or set()
    result = {"level1": {}, "level2": {}, "level3": {}, "curb_info": {}}

    # Level 1: Equilibrium analysis — equivalent to original_paper_analysis.py.
    # Computes the same quantities:
    #   sigma:         equilibrium mixture (same solver, same payoff matrix)
    #   per_agent:     (M_k @ sigma) per metric — per-agent value at equilibrium
    #   regret:        compute_regret(sigma, payoff_matrix) — same as original
    #   welfare:       sigma^T M_k sigma — total ecosystem welfare
    #   welfare_share: sigma_i * (M_k @ sigma)_i — agent i's share of welfare
    # Note: values are in raw payoff units (not 0-100 normalized as in
    # original_paper_analysis). Rankings are identical.
    for solver in solvers:
        sigma = _solve(solver, payoff_matrix, strategy_names)
        if sigma is None:
            result["level1"][solver] = None
            continue

        l1 = {
            "sigma": sigma,
            "regret": _compute_regret(sigma, payoff_matrix),
            "welfare": {},
            "per_agent": {},
            "welfare_share": {},
        }
        for m in metrics:
            M_k = metric_matrices[m]
            l1["welfare"][m] = _eval_welfare(sigma, M_k)
            per_agent = _eval_per_agent(sigma, M_k)
            l1["per_agent"][m] = per_agent
            l1["welfare_share"][m] = sigma * per_agent  # σᵢ(Mσ)ᵢ

        result["level1"][solver] = l1

    for solver in solvers:
        if result["level1"].get(solver) is None:
            result["level2"][solver] = None
            continue

        l2 = {"loo": {}, "harsanyi": {}}

        # LOO for each agent
        for i, s in enumerate(strategy_names):
            if s in non_ablatable:
                l2["loo"][s] = {m: 0.0 for m in metrics}
                continue

            loo_idx = [j for j in range(n) if j != i]
            loo_names = [strategy_names[j] for j in loo_idx]
            loo_payoff = payoff_matrix[np.ix_(loo_idx, loo_idx)]

            sigma_loo = _solve(solver, loo_payoff, loo_names)
            if sigma_loo is None:
                l2["loo"][s] = {m: 0.0 for m in metrics}
                continue

            deltas = {}
            for m in metrics:
                M_k = metric_matrices[m]
                w_full = result["level1"][solver]["welfare"][m]
                M_loo = M_k[np.ix_(loo_idx, loo_idx)]
                w_loo = _eval_welfare(sigma_loo, M_loo)
                deltas[m] = w_full - w_loo
            l2["loo"][s] = deltas

        # Harsanyi dividends for each pair
        if do_harsanyi:
            ablatable = [s for s in strategy_names if s not in non_ablatable]
            for s_a, s_b in combinations(ablatable, 2):
                i_a = strategy_names.index(s_a)
                i_b = strategy_names.index(s_b)
                lto_idx = [j for j in range(n) if j != i_a and j != i_b]
                lto_names = [strategy_names[j] for j in lto_idx]
                lto_payoff = payoff_matrix[np.ix_(lto_idx, lto_idx)]

                sigma_lto = _solve(solver, lto_payoff, lto_names)
                if sigma_lto is None:
                    l2["harsanyi"][(s_a, s_b)] = {m: 0.0 for m in metrics}
                    continue

                dividends = {}
                for m in metrics:
                    M_k = metric_matrices[m]
                    w_full = result["level1"][solver]["welfare"][m]
                    w_loo_a = w_full - l2["loo"][s_a][m]  # W(LOO_a) = W(full) - delta_a
                    w_loo_b = w_full - l2["loo"][s_b][m]
                    M_lto = M_k[np.ix_(lto_idx, lto_idx)]
                    w_lto = _eval_welfare(sigma_lto, M_lto)
                    #dividends[m] = w_full - (w_full - l2["loo"][s_a][m]) - (w_full - l2["loo"][s_b][m]) + w_lto
                    dividends[m] = w_full - w_loo_a - w_loo_b + w_lto #TODO check this fix
                l2["harsanyi"][(s_a, s_b)] = dividends

        result["level2"][solver] = l2

    # ── Level 3: CURB-conditional LOO ──
    if do_curb and HAS_CURB:
        # Find CURB sets
        minimals = find_minimal_curb_sets_klimm_weibull(payoff_matrix, n)
        all_curbs = set(frozenset(m) for m in minimals)
        # MC sampling for non-minimals
        rng = np.random.default_rng()
        for _ in range(mc_curb_budget):
            seed_size = rng.integers(2, n + 1)
            seed = set(rng.choice(n, size=seed_size, replace=False).tolist())
            c = curb_closure(payoff_matrix, seed)
            all_curbs.add(c)

        result["curb_info"]["minimal_curb_sets"] = [list(m) for m in minimals]
        result["curb_info"]["all_curb_sets"] = [list(c) for c in all_curbs]
        result["curb_info"]["n_curb_sets"] = len(all_curbs)

        # Solve NE within each CURB set, keyed by solver
        result["level3"] = {}
        for solver in solvers:
            curb_results = []
            for c in all_curbs:
                c_idx = sorted(c)
                c_names = [strategy_names[j] for j in c_idx]
                c_payoff = payoff_matrix[np.ix_(c_idx, c_idx)]

                sigma_c = _solve(solver, c_payoff, c_names)
                if sigma_c is None:
                    continue

                welfare = {}
                for m in metrics:
                    M_k = metric_matrices[m]
                    M_c = M_k[np.ix_(c_idx, c_idx)]
                    welfare[m] = _eval_welfare(sigma_c, M_c)

                curb_results.append({
                    "curb_set": frozenset(c),
                    "idx": c_idx,
                    "sigma": sigma_c,
                    **welfare,
                })

            # LOO within each CURB set
            l3 = {}
            for i, s in enumerate(strategy_names):
                if s in non_ablatable:
                    l3[s] = {m: {"min": 0.0, "max": 0.0} for m in metrics}
                    continue

                loo_deltas = {m: [] for m in metrics}

                for cr in curb_results:
                    if i not in cr["curb_set"]:
                        continue
                    if len(cr["curb_set"]) < 2:
                        continue

                    c_minus_i = [j for j in cr["idx"] if j != i]
                    if len(c_minus_i) < 1:
                        continue

                    c_minus_names = [strategy_names[j] for j in c_minus_i]
                    c_minus_payoff = payoff_matrix[np.ix_(c_minus_i, c_minus_i)]

                    sigma_loo_c = _solve(solver, c_minus_payoff, c_minus_names)
                    if sigma_loo_c is None:
                        continue

                    for m in metrics:
                        M_k = metric_matrices[m]
                        M_loo = M_k[np.ix_(c_minus_i, c_minus_i)]
                        w_loo = _eval_welfare(sigma_loo_c, M_loo)
                        loo_deltas[m].append(cr[m] - w_loo)

                agent_l3 = {}
                for m in metrics:
                    if loo_deltas[m]:
                        agent_l3[m] = {
                            "min": min(loo_deltas[m]),
                            "max": max(loo_deltas[m]),
                            "all_deltas": loo_deltas[m],
                        }
                    else:
                        agent_l3[m] = {"min": 0.0, "max": 0.0, "all_deltas": []}
                l3[s] = agent_l3

            result["level3"][solver] = l3

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

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
        do_harsanyi: Whether to compute Harsanyi dividends (Level 2).
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
        from concurrent.futures import ThreadPoolExecutor, as_completed
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

    # Aggregate
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
# Aggregation
# ---------------------------------------------------------------------------

def _summarize(values):
    """Compute mean, std, 95% CI."""
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
    }


def _aggregate_results(raw_results, strategy_names, solvers, metrics):
    """Aggregate raw bootstrap results into summary statistics."""
    n_bootstrap = len(raw_results)
    agg = {
        "level1": {},
        "level2": {},
        "level3": {},
        "curb_survival": {},
    }

    # ── Level 1 aggregation ──
    for solver in solvers:
        solver_agg = {"welfare": {}, "per_agent": {}, "welfare_share": {}, "regret": {}, "support_frequency": {}}

        # Sigma samples
        sigma_samples = []
        for r in raw_results:
            l1 = r["level1"].get(solver)
            if l1 is not None:
                sigma_samples.append(l1["sigma"])

        if not sigma_samples:
            agg["level1"][solver] = None
            continue

        sigmas = np.array(sigma_samples)
        n_converged = len(sigma_samples)

        # Support frequency
        support_counts = np.sum(sigmas > 1e-2, axis=0)
        for i, name in enumerate(strategy_names):
            solver_agg["support_frequency"][name] = float(support_counts[i] / n_converged * 100)

        # Equilibrium distribution
        solver_agg["equilibrium"] = {
            "mean": np.mean(sigmas, axis=0).tolist(),
            "std": np.std(sigmas, axis=0).tolist(),
        }

        # Per-metric welfare and per-agent values
        for m in metrics:
            welfare_vals = [r["level1"][solver]["welfare"][m]
                           for r in raw_results if r["level1"].get(solver)]
            solver_agg["welfare"][m] = _summarize(welfare_vals)

            solver_agg["per_agent"][m] = {}
            for i, name in enumerate(strategy_names):
                vals = [r["level1"][solver]["per_agent"][m][i]
                        for r in raw_results if r["level1"].get(solver)]
                solver_agg["per_agent"][m][name] = _summarize(vals)

        # Welfare share: σᵢ(Mσ)ᵢ per metric
        for m in metrics:
            solver_agg["welfare_share"][m] = {}
            for i, name in enumerate(strategy_names):
                vals = [r["level1"][solver]["welfare_share"][m][i]
                        for r in raw_results if r["level1"].get(solver)]
                solver_agg["welfare_share"][m][name] = _summarize(vals)

        # Regret
        for i, name in enumerate(strategy_names):
            vals = [r["level1"][solver]["regret"][i]
                    for r in raw_results if r["level1"].get(solver)]
            solver_agg["regret"][name] = _summarize(vals)

        solver_agg["n_converged"] = n_converged
        agg["level1"][solver] = solver_agg

    # ── Level 2 aggregation ──
    for solver in solvers:
        if agg["level1"].get(solver) is None:
            agg["level2"][solver] = None
            continue

        solver_l2 = {"loo": {}, "harsanyi": {}}

        for name in strategy_names:
            solver_l2["loo"][name] = {}
            for m in metrics:
                vals = [r["level2"][solver]["loo"][name][m]
                        for r in raw_results
                        if r["level2"].get(solver) and name in r["level2"][solver]["loo"]]
                solver_l2["loo"][name][m] = _summarize(vals) if vals else _summarize([0.0])

        # Harsanyi
        if raw_results[0]["level2"].get(solver) and raw_results[0]["level2"][solver].get("harsanyi"):
            for pair in raw_results[0]["level2"][solver]["harsanyi"]:
                solver_l2["harsanyi"][pair] = {}
                for m in metrics:
                    vals = [r["level2"][solver]["harsanyi"][pair][m]
                            for r in raw_results
                            if r["level2"].get(solver) and pair in r["level2"][solver].get("harsanyi", {})]
                    solver_l2["harsanyi"][pair][m] = _summarize(vals) if vals else _summarize([0.0])

        agg["level2"][solver] = solver_l2

    # ── Level 3 aggregation (keyed by solver) ──
    for solver in solvers:
        agg["level3"][solver] = {}
        for name in strategy_names:
            agg["level3"][solver][name] = {}
            for m in metrics:
                min_vals = [r["level3"][solver][name][m]["min"]
                            for r in raw_results
                            if solver in r.get("level3", {})
                            and name in r["level3"][solver]]
                max_vals = [r["level3"][solver][name][m]["max"]
                            for r in raw_results
                            if solver in r.get("level3", {})
                            and name in r["level3"][solver]]

                agg["level3"][solver][name][m] = {
                    "min": _summarize(min_vals) if min_vals else _summarize([0.0]),
                    "max": _summarize(max_vals) if max_vals else _summarize([0.0]),
                }

    # CURB survival
    curb_counts = {}
    for r in raw_results:
        for c in r.get("curb_info", {}).get("all_curb_sets", []):
            key = frozenset(c)
            curb_counts[key] = curb_counts.get(key, 0) + 1

    # Top 20 CURB sets by frequency
    top_curbs = sorted(curb_counts.items(), key=lambda x: -x[1])[:20]
    agg["curb_survival"] = {
        str(sorted(c)): count / n_bootstrap * 100
        for c, count in top_curbs
    }

    # Minimal CURB survival
    minimal_counts = {}
    for r in raw_results:
        for c in r.get("curb_info", {}).get("minimal_curb_sets", []):
            key = frozenset(c)
            minimal_counts[key] = minimal_counts.get(key, 0) + 1
    top_minimals = sorted(minimal_counts.items(), key=lambda x: -x[1])[:20]
    agg["minimal_curb_survival"] = {
        str(sorted(c)): count / n_bootstrap * 100
        for c, count in top_minimals
    }

    return agg


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_results(results):
    """Pretty-print causal analysis results."""
    agg = results["aggregated"]
    config = results["config"]
    strategy_names = config["strategy_names"]
    solvers = config["solvers"]
    metrics = config["metrics"]

    print("=" * 70)
    print("CAUSAL METAGAME ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Strategies: {strategy_names}")
    print(f"Solvers: {solvers}")
    print(f"Metrics: {metrics}")
    print(f"Bootstraps: {config['n_bootstrap']}")

    for solver in solvers:
        l1 = agg["level1"].get(solver)
        if l1 is None:
            print(f"\n--- Solver {solver}: did not converge ---")
            continue

        print(f"\n{'='*70}")
        print(f"SOLVER: {solver} ({l1['n_converged']}/{config['n_bootstrap']} converged)")
        print(f"{'='*70}")

        # Equilibrium
        print("\n--- Equilibrium Distribution ---")
        eq = l1["equilibrium"]
        for i, name in enumerate(strategy_names):
            if eq["mean"][i] > 0.001 or eq["std"][i] > 0.001:
                print(f"  {name:<30}: {eq['mean'][i]:.4f} +/- {eq['std'][i]:.4f}")

        # Support
        print("\n--- Support Frequency % ---")
        for name in strategy_names:
            freq = l1["support_frequency"][name]
            if freq > 0:
                print(f"  {name:<30}: {freq:.1f}%")

        # Regret
        print("\n--- Per-Agent Regret (95% CI) ---")
        for name in strategy_names:
            r = l1["regret"][name]
            print(f"  {name:<30}: {r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")

        # Per-metric welfare
        for m in metrics:
            w = l1["welfare"][m]
            print(f"\n--- Aggregate {m.upper()} (95% CI) ---")
            print(f"  {w['mean']:.4f} [{w['ci_lower']:.4f}, {w['ci_upper']:.4f}]")

        # Level 2 LOO
        l2 = agg["level2"].get(solver)
        if l2:
            print(f"\n--- Level 2: LOO Effects ---")
            for m in metrics:
                print(f"\n  Metric: {m}")
                for name in strategy_names:
                    d = l2["loo"][name][m]
                    sig = "**" if d["ci_lower"] > 0 or d["ci_upper"] < 0 else ""
                    print(f"    {name:<30}: {d['mean']:+.4f} [{d['ci_lower']:.4f}, {d['ci_upper']:.4f}] {sig}")

    # Level 3
    if config["do_curb"]:
        print(f"\n{'='*70}")
        print("LEVEL 3: CURB-CONDITIONAL LOO")
        print(f"{'='*70}")

        print("\n--- Minimal CURB Survival (top 10) ---")
        for key, freq in list(agg["minimal_curb_survival"].items())[:10]:
            print(f"  {freq:5.1f}% | {key}")

        for solver in solvers:
            if solver not in agg["level3"]:
                continue
            print(f"\n--- CURB-Conditional LOO Intervals (solver: {solver}) ---")
            for m in metrics:
                print(f"\n  Metric: {m}")
                print(f"  {'Strategy':<30} {'Min Delta':>12} {'Min 95% CI':>24} {'Max Delta':>12} {'Max 95% CI':>24}")
                print(f"  {'-'*105}")
                for name in strategy_names:
                    l3 = agg["level3"][solver][name][m]
                    mn = l3["min"]
                    mx = l3["max"]
                    if abs(mn["mean"]) < 0.001 and abs(mx["mean"]) < 0.001:
                        continue
                    print(f"  {name:<30} {mn['mean']:>+.4f} [{mn['ci_lower']:>+.4f}, {mn['ci_upper']:>+.4f}]   "
                          f"{mx['mean']:>+.4f} [{mx['ci_lower']:>+.4f}, {mx['ci_upper']:>+.4f}]")


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_results(results, output_path):
    """Save results to pickle and JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pkl_path = output_path.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to {pkl_path}")


def load_results(output_path):
    """Load results from pickle."""
    pkl_path = Path(output_path).with_suffix(".pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


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
    print(f"  Domain: {args.domain}")
    print(f"  Strategies: {len(strategy_names)}")
    print(f"  Solvers: {args.solvers}")
    print(f"  Bootstraps: {args.n_bootstrap}")
    print(f"  Harsanyi: {not args.no_harsanyi}")
    print(f"  CURB: {not args.no_curb}")
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
