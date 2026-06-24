"""Unified causal metagame analysis pipeline.

Implements the three-level counterfactual framework:
  Level 1: Equilibrium analysis (per-agent metrics at equilibrium)
  Level 2: LOO + second-order LOO effects (Möbius coefficients at the
           grand coalition, a.k.a. Harsanyi dividends in cooperative game
           theory) + solver comparison
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

import polarix as plx
import jax.numpy as jnp
import jax
from evaluation.curb_analysis import ( find_minimal_curb_sets_klimm_weibull, curb_closure)

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




def _compute_level1(payoff_matrix, metric_matrices, strategy_names, solvers, metrics):
    """Level 1: Equilibrium analysis. Returns dict keyed by solver.

    Equivalent to original_paper_analysis.py. Computes:
      sigma:         equilibrium mixture (same solver, same payoff matrix)
      per_agent:     (M_k @ sigma) per metric -- per-agent value at equilibrium
      regret:        compute_regret(sigma, payoff_matrix)
      welfare:       sigma^T M_k sigma -- total ecosystem welfare
      welfare_share: sigma_i * (M_k @ sigma)_i -- agent i's share of welfare
    Values are in raw payoff units. Rankings are identical to original_paper_analysis.
    """
    level1 = {}
    for solver in solvers:
        sigma = _solve(solver, payoff_matrix, strategy_names)
        if sigma is None:
            level1[solver] = None
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
            l1["welfare_share"][m] = sigma * per_agent
        level1[solver] = l1
    return level1

  


def _compute_level2(payoff_matrix, metric_matrices, strategy_names, solvers, metrics, level1, non_ablatable, do_harsanyi):
    """Level 2: full-game LOO + second-order LOO effects. Returns dict keyed by solver."""
    n = len(strategy_names)
    level2 = {}

    for solver in solvers:
        if level1.get(solver) is None:
            level2[solver] = None
            continue

        l2 = {"loo": {}, "harsanyi": {}}

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
                w_full = level1[solver]["welfare"][m]
                M_loo = M_k[np.ix_(loo_idx, loo_idx)]
                w_loo = _eval_welfare(sigma_loo, M_loo)
                deltas[m] = w_full - w_loo
            l2["loo"][s] = deltas
            l2["loo"][s]["sigma"] = sigma_loo

        # Second-order LOO effects (pairwise interactions; stored under
        # the "harsanyi" key for back-compat with existing pickles)
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
                    w_full = level1[solver]["welfare"][m]
                    w_loo_a = w_full - l2["loo"][s_a][m]
                    w_loo_b = w_full - l2["loo"][s_b][m]
                    M_lto = M_k[np.ix_(lto_idx, lto_idx)]
                    w_lto = _eval_welfare(sigma_lto, M_lto)
                    dividends[m] = w_full - w_loo_a - w_loo_b + w_lto
                l2["harsanyi"][(s_a, s_b)] = dividends

        level2[solver] = l2
    return level2
  

def _compute_level3(payoff_matrix, metric_matrices, strategy_names, solvers, metrics, non_ablatable, mc_curb_budget):
    """Level 3: CURB-conditional LOO. Returns (level3_dict, curb_info_dict)."""
    n = len(strategy_names)

    # Find CURB sets
    minimals = find_minimal_curb_sets_klimm_weibull(payoff_matrix, n)
    all_curbs = set(frozenset(m) for m in minimals)
    # TODO: Replace or supplement MC with lattice enumeration from minimals:
    # 1. Enumerate all subsets of minimal CURB sets, compute closure of each union.
    #    With k minimals this is 2^k - 1 closures (exact, deterministic).
    # 2. Then bias remaining MC budget toward seeds that include strategies
    #    outside the already-found CURB sets, to discover novel non-minimals
    #    that aren't just unions of known minimals.
    # MC sampling for non-minimals
    rng = np.random.default_rng()
    for _ in range(mc_curb_budget):
        seed_size = rng.integers(2, n + 1)
        seed = set(rng.choice(n, size=seed_size, replace=False).tolist())
        c = curb_closure(payoff_matrix, seed)
        all_curbs.add(c)

    curb_info = {
        "minimal_curb_sets": [list(m) for m in minimals],
        "all_curb_sets": [list(c) for c in all_curbs],
        "n_curb_sets": len(all_curbs),
    }

    # Solve NE within each CURB set, keyed by solver
    level3 = {}
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
            loo_sigmas = []

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

                loo_sigmas.append({
                    "curb_set": cr["curb_set"],
                    "sigma_curb": cr["sigma"],
                    "sigma_loo": sigma_loo_c,
                    "curb_idx": cr["idx"],
                    "loo_idx": c_minus_i,
                })

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
            agent_l3["sigmas"] = loo_sigmas
            l3[s] = agent_l3

        level3[solver] = l3
    return level3, curb_info
    



def analyze_one_bootstrap(payoff_matrix, metric_matrices, strategy_names, solvers, metrics,
                          non_ablatable=None, do_harsanyi=True, do_curb=True, mc_curb_budget=50):
    """Run all three levels on one bootstrapped game."""
    non_ablatable = non_ablatable or set()
    result = {}
    result["level1"] = _compute_level1(payoff_matrix, metric_matrices, strategy_names, solvers, metrics)
    result["level2"] = _compute_level2(payoff_matrix, metric_matrices, strategy_names, solvers, metrics,
                                        result["level1"], non_ablatable, do_harsanyi)
    if do_curb:
        result["level3"], result["curb_info"] = _compute_level3(payoff_matrix, metric_matrices, strategy_names,
                                                                 solvers, metrics, non_ablatable, mc_curb_budget)
    else:
        result["level3"] = {}
        result["curb_info"] = {}
    return result


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

        # Second-order LOO effects (dict key kept as "harsanyi" for pickle back-compat)
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


