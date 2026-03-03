"""Per-Strategy Iterative Expansion (PSIE) analysis.

PSIE starts from each strategy as a seed, iteratively adds best responses
until no outside strategy can profitably deviate (epsilon-Nash), and records
the full expansion trace. This reveals which strategies are "essential" to
the ecosystem and whether distinct CURB-like sets exist.

Usage:
    python evaluation/psie_analysis.py                     # full run
    python evaluation/psie_analysis.py --no-bootstrap      # point estimate only
    python evaluation/psie_analysis.py --max-bootstrap 10  # quick sanity check
"""

import argparse
import multiprocessing
import pickle
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iterative_game_analysis.metagame import MetaGame
from src.iterative_game_analysis.utils import compute_regret


def _summarize(samples):
    """Compute mean, std, 95% CI from bootstrap samples."""
    arr = np.array(samples)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
    }


def _compute_step_metrics(matrices, support, sigma):
    """Compute welfare/fairness metrics for a restricted equilibrium.

    Args:
        matrices: Full NxN matrices dict with keys payoff, nw, nw_plus, ef1, ef1_plus.
        support: List of strategy indices in the current support.
        sigma: Equilibrium mixture over the support.

    Returns:
        (metrics_dict, per_agent_dict) where metrics_dict has scalar welfare values
        and per_agent_dict has per-strategy vectors (matrix_sub @ sigma).
    """
    metric_keys = ["payoff", "nw", "nw_plus", "ef1", "ef1_plus"]
    metrics = {}
    per_agent = {}
    for key in metric_keys:
        sub = matrices[key][np.ix_(support, support)]
        if key in ("ef1", "ef1_plus"):
            sub = np.nan_to_num(sub, nan=0.0)
        label = "uw" if key == "payoff" else key
        metrics[label] = float(sigma @ sub @ sigma)
        per_agent[label] = (sub @ sigma).tolist()
    return metrics, per_agent


def _compute_in_out(seed_idx, matrices, support, sigma, strategy_names, solver, eq_cache):
    """Compute in/out counterfactual for the seed strategy at one expansion step.

    Compares the current equilibrium (seed IN) to the equilibrium on
    support \\ {seed} (seed OUT), giving the marginal contribution of the
    seed to the restricted ecosystem.

    Args:
        seed_idx: Global index of the seed strategy.
        matrices: Full NxN matrices dict.
        support: Current support indices (seed is in here).
        sigma: Equilibrium mixture over the support.
        strategy_names: Full list of strategy names.
        solver: Equilibrium solver name.
        eq_cache: Shared cache keyed by frozenset(support) -> (sigma, eq_value).

    Returns:
        Dict with in_metrics, out_metrics, delta, entry_mass, equilibrium_shift,
        or None if support has only the seed (can't remove it).
    """
    if len(support) < 2:
        return None

    payoff_matrix = matrices["payoff"]
    seed_pos = support.index(seed_idx)

    # --- "In" metrics (already computed, but grab entry_mass) ---
    in_metrics, _ = _compute_step_metrics(matrices, support, sigma)
    entry_mass = float(sigma[seed_pos])

    # --- "Out" game: support without the seed ---
    support_out = [s for s in support if s != seed_idx]
    out_key = frozenset(support_out)

    if out_key in eq_cache:
        sigma_out, eq_value_out = eq_cache[out_key]
    else:
        if len(support_out) == 1:
            sigma_out = np.array([1.0])
            eq_value_out = float(payoff_matrix[support_out[0], support_out[0]])
        else:
            subset_names = [strategy_names[i] for i in support_out]
            sub_payoff = payoff_matrix[np.ix_(support_out, support_out)]
            sub_game = MetaGame(subset_names, sub_payoff)
            sigma_out = sub_game.solve(solver)
            eq_value_out = float(sigma_out @ sub_payoff @ sigma_out)
        eq_cache[out_key] = (sigma_out, eq_value_out)

    out_metrics, _ = _compute_step_metrics(matrices, support_out, sigma_out)

    # --- Delta: in - out ---
    delta = {k: in_metrics[k] - out_metrics[k] for k in in_metrics}

    # --- Equilibrium shift: L1 distance of incumbent play ---
    # Compare sigma_in (restricted to incumbents) vs sigma_out
    # sigma_in on incumbents = sigma with seed_pos removed, renormalized
    sigma_incumbents_in = np.delete(sigma, seed_pos)
    incumbent_mass = float(np.sum(sigma_incumbents_in))
    if incumbent_mass > 0:
        sigma_incumbents_in = sigma_incumbents_in / incumbent_mass
    eq_shift = float(np.sum(np.abs(sigma_incumbents_in - sigma_out)))

    return {
        "in_metrics": in_metrics,
        "out_metrics": out_metrics,
        "delta": delta,
        "entry_mass": entry_mass,
        "equilibrium_shift": eq_shift,
    }


def run_psie_single_seed(seed_idx, matrices, strategy_names, solver, epsilon, eq_cache):
    """Run PSIE algorithm from a single seed strategy.

    Starting from one strategy, iteratively adds the outside strategy with
    the highest profitable deviation until no outside strategy can gain more
    than epsilon.

    Args:
        seed_idx: Index of the seed strategy.
        matrices: Dict with keys 'payoff', 'nw', 'nw_plus', 'ef1', 'ef1_plus'
            (each full NxN matrix).
        strategy_names: List of strategy names.
        solver: Equilibrium solver name.
        epsilon: Deviation threshold for stopping.
        eq_cache: Shared dict keyed by frozenset(support_indices) -> (sigma, eq_value).

    Returns:
        Dict with 'seed', 'trace' (list of step dicts), 'terminal_support',
        'terminal_sigma', 'terminal_eq_value'.
    """
    payoff_matrix = matrices["payoff"]
    n = len(strategy_names)
    support = [seed_idx]
    trace = []

    while True:
        support_key = frozenset(support)

        # Check cache for this support set
        if support_key in eq_cache:
            sigma, eq_value = eq_cache[support_key]
        else:
            if len(support) == 1:
                # Short-circuit 1x1 game
                sigma = np.array([1.0])
                eq_value = float(payoff_matrix[support[0], support[0]])
            else:
                subset_names = [strategy_names[i] for i in support]
                sub_payoff = payoff_matrix[np.ix_(support, support)]
                sub_game = MetaGame(subset_names, sub_payoff)
                sigma = sub_game.solve(solver)
                eq_value = float(sigma @ sub_payoff @ sigma)
            eq_cache[support_key] = (sigma, eq_value)

        # Compute deviation gains for all outside strategies
        outside = [i for i in range(n) if i not in support]
        deviation_gains = {}
        for s_idx in outside:
            # s' expected payoff against the restricted equilibrium
            s_payoff = float(payoff_matrix[s_idx, support] @ sigma)
            deviation_gains[s_idx] = s_payoff - eq_value

        # Find best responder
        if deviation_gains:
            best_idx = max(deviation_gains, key=deviation_gains.get)
            best_gain = deviation_gains[best_idx]
        else:
            best_idx = None
            best_gain = 0.0

        # Compute welfare/fairness metrics at this step
        step_metrics, step_per_agent = _compute_step_metrics(matrices, support, sigma)

        # Compute in/out counterfactual for the seed strategy
        in_out = _compute_in_out(
            seed_idx, matrices, support, sigma, strategy_names, solver, eq_cache,
        )

        # Record step
        step = {
            "support": list(support),
            "support_names": [strategy_names[i] for i in support],
            "sigma": sigma.copy(),
            "eq_value": eq_value,
            "deviation_gains": {
                strategy_names[k]: v for k, v in deviation_gains.items()
            },
            "max_deviation_gain": best_gain,
            "best_responder": strategy_names[best_idx] if best_idx is not None else None,
            "metrics": step_metrics,
            "per_agent": step_per_agent,
            "in_out": in_out,
        }
        trace.append(step)

        # Check stopping condition
        if best_gain <= epsilon or best_idx is None:
            break

        # Add best responder to support
        support = sorted(support + [best_idx])

    return {
        "seed": strategy_names[seed_idx],
        "seed_idx": seed_idx,
        "trace": trace,
        "terminal_support": set(step["support_names"]),
        "terminal_sigma": sigma.copy(),
        "terminal_eq_value": eq_value,
        "n_steps": len(trace),
    }


def run_psie_all_seeds(matrices, strategy_names, solver="mene", epsilon=0.0):
    """Run PSIE from every strategy as seed.

    Args:
        matrices: Dict with keys 'payoff', 'nw', 'nw_plus', 'ef1', 'ef1_plus'
            (each full NxN matrix).
        strategy_names: List of strategy names.
        solver: Equilibrium solver name.
        epsilon: Deviation threshold.

    Returns:
        List of trace dicts (one per seed).
    """
    eq_cache = {}
    traces = []
    for seed_idx in range(len(strategy_names)):
        trace = run_psie_single_seed(
            seed_idx, matrices, strategy_names, solver, epsilon, eq_cache,
        )
        traces.append(trace)
    return traces


def analyze_traces(traces, strategy_names, matrices):
    """Extract summary statistics from PSIE traces.

    Args:
        traces: List of trace dicts from run_psie_all_seeds.
        strategy_names: List of strategy names.
        matrices: Dict with matrix keys (uses 'payoff' for terminal regret).

    Returns:
        Dict with unique_supports, all_converge_same, path_lengths,
        entry_order, never_entered, always_entered, terminal_regrets.
    """
    payoff_matrix = matrices["payoff"]
    n = len(strategy_names)

    # Terminal support sets
    terminal_supports = [frozenset(t["terminal_support"]) for t in traces]
    unique_supports = list({tuple(sorted(s)) for s in terminal_supports})

    all_converge_same = len(set(terminal_supports)) == 1

    # Path lengths
    path_lengths = [t["n_steps"] for t in traces]

    # Entry order: for each strategy, which step it entered across seeds
    entry_order = {name: [] for name in strategy_names}
    for t in traces:
        entered = set()
        for step_idx, step in enumerate(t["trace"]):
            for name in step["support_names"]:
                if name not in entered:
                    entered.add(name)
                    entry_order[name].append(step_idx)
        # Strategies that never entered this trace
        for name in strategy_names:
            if name not in entered:
                entry_order[name].append(None)

    # Never entered / always entered across all seeds
    never_entered = []
    always_entered = []
    for name in strategy_names:
        in_all = all(name in t["terminal_support"] for t in traces)
        in_none = all(name not in t["terminal_support"] for t in traces)
        if in_all:
            always_entered.append(name)
        if in_none:
            never_entered.append(name)

    # Terminal regrets: max regret of each terminal equilibrium in the full game
    terminal_regrets = []
    for t in traces:
        # Build full-game sigma from terminal
        terminal_indices = t["trace"][-1]["support"]
        terminal_sigma = t["terminal_sigma"]
        sigma_full = np.zeros(n)
        for i, idx in enumerate(terminal_indices):
            sigma_full[idx] = terminal_sigma[i]
        regret_vec, _, _ = compute_regret(sigma_full, payoff_matrix)
        terminal_regrets.append(float(np.max(regret_vec)))

    # Per-seed terminal support (tuple of sorted names)
    terminal_support_per_seed = [
        tuple(sorted(t["terminal_support"])) for t in traces
    ]

    # Trajectory crossings: support sets visited by >1 seed
    from collections import defaultdict
    support_visitors = defaultdict(list)
    for t in traces:
        for step_idx, step in enumerate(t["trace"]):
            key = frozenset(step["support_names"])
            support_visitors[key].append((t["seed"], step_idx))
    crossings = []
    for support, visitors in support_visitors.items():
        if len(visitors) > 1:
            crossings.append({
                "support": tuple(sorted(support)),
                "visitors": visitors,  # list of (seed_name, step_idx)
            })

    return {
        "unique_supports": unique_supports,
        "all_converge_same": all_converge_same,
        "path_lengths": path_lengths,
        "entry_order": entry_order,
        "never_entered": never_entered,
        "always_entered": always_entered,
        "terminal_regrets": terminal_regrets,
        "terminal_support_per_seed": terminal_support_per_seed,
        "crossings": crossings,
    }


def _bootstrap_worker(args):
    """Worker for parallel bootstrap PSIE.

    Top-level function so it's picklable by multiprocessing.
    """
    sample_matrices, strategy_names, solver, epsilon = args
    sample_matrices = {k: np.array(v, dtype=float) for k, v in sample_matrices.items()}
    traces = run_psie_all_seeds(sample_matrices, strategy_names, solver, epsilon)
    analysis = analyze_traces(traces, strategy_names, sample_matrices)
    analysis["traces"] = traces
    return analysis


def aggregate_bootstrap_analysis(all_analyses, strategy_names):
    """Aggregate PSIE analyses across bootstrap samples.

    Args:
        all_analyses: List of analysis dicts from analyze_traces.
        strategy_names: List of strategy names.

    Returns:
        Dict with aggregated statistics (mean, std, CI) for all metrics.
    """
    n_samples = len(all_analyses)

    # Convergence rate
    convergence_rate = sum(
        1 for a in all_analyses if a["all_converge_same"]
    ) / n_samples

    # Path lengths per seed
    n_seeds = len(strategy_names)
    path_lengths_per_seed = {}
    for seed_idx in range(n_seeds):
        samples = [a["path_lengths"][seed_idx] for a in all_analyses]
        path_lengths_per_seed[strategy_names[seed_idx]] = _summarize(samples)

    # Entry step per strategy (excluding None entries = never entered)
    entry_step = {}
    for name in strategy_names:
        # Collect the mean entry step per bootstrap (across seeds, ignoring None)
        samples = []
        for a in all_analyses:
            steps = [s for s in a["entry_order"][name] if s is not None]
            if steps:
                samples.append(float(np.mean(steps)))
        if samples:
            entry_step[name] = _summarize(samples)
        else:
            entry_step[name] = {"mean": None, "std": None, "ci_lower": None, "ci_upper": None}

    # Never-entered / always-entered frequencies
    never_entered_freq = {}
    always_entered_freq = {}
    for name in strategy_names:
        never_entered_freq[name] = sum(
            1 for a in all_analyses if name in a["never_entered"]
        ) / n_samples
        always_entered_freq[name] = sum(
            1 for a in all_analyses if name in a["always_entered"]
        ) / n_samples

    # Most common terminal support sets
    support_counter = Counter()
    for a in all_analyses:
        for s in a["unique_supports"]:
            support_counter[s] += 1
    most_common_supports = support_counter.most_common(10)

    # Terminal regrets
    terminal_regrets_per_seed = {}
    for seed_idx in range(n_seeds):
        samples = [a["terminal_regrets"][seed_idx] for a in all_analyses]
        terminal_regrets_per_seed[strategy_names[seed_idx]] = _summarize(samples)

    # Most common terminal support per seed
    terminal_per_seed = {}
    for seed_idx in range(n_seeds):
        counter = Counter()
        for a in all_analyses:
            counter[a["terminal_support_per_seed"][seed_idx]] += 1
        terminal_per_seed[strategy_names[seed_idx]] = counter.most_common(3)

    # Trajectory crossings across bootstrap samples
    crossing_rate = sum(
        1 for a in all_analyses if len(a["crossings"]) > 0
    ) / n_samples
    # Most commonly shared support sets
    crossing_counter = Counter()
    for a in all_analyses:
        for c in a["crossings"]:
            crossing_counter[c["support"]] += 1
    most_common_crossings = crossing_counter.most_common(10)

    return {
        "convergence_rate": convergence_rate,
        "path_lengths_per_seed": path_lengths_per_seed,
        "entry_step": entry_step,
        "never_entered_freq": never_entered_freq,
        "always_entered_freq": always_entered_freq,
        "most_common_supports": most_common_supports,
        "terminal_regrets_per_seed": terminal_regrets_per_seed,
        "terminal_per_seed": terminal_per_seed,
        "crossing_rate": crossing_rate,
        "most_common_crossings": most_common_crossings,
        "n_samples": n_samples,
    }


def print_psie_results(point_analysis, bootstrap_agg, strategy_names, point_traces):
    """Print formatted PSIE analysis results.

    Args:
        point_analysis: Analysis dict from analyze_traces (point estimate).
        bootstrap_agg: Aggregated bootstrap dict (or None if no bootstrap).
        strategy_names: List of strategy names.
        point_traces: List of trace dicts from run_psie_all_seeds.
    """
    print("\n" + "=" * 70)
    print("PSIE (Per-Strategy Iterative Expansion) ANALYSIS")
    print("=" * 70)

    # Section 1: Point estimate traces
    print("\n--- Section 1: Expansion Traces (Point Estimate) ---")
    for t in point_traces:
        print(f"\n  Seed: {t['seed']}")
        for step_idx, step in enumerate(t["trace"]):
            support_str = ", ".join(step["support_names"])
            print(f"    Step {step_idx}: support=[{support_str}]")
            sigma_str = ", ".join(
                f"{step['support_names'][i]}={step['sigma'][i]:.4f}"
                for i in range(len(step["sigma"]))
            )
            print(f"      sigma: {sigma_str}")
            print(f"      eq_value: {step['eq_value']:.4f}")
            if step["best_responder"]:
                print(f"      best_responder: {step['best_responder']} "
                      f"(gain={step['max_deviation_gain']:.4f})")
            else:
                print(f"      max_deviation_gain: {step['max_deviation_gain']:.4f} (converged)")
            # Show welfare/fairness metrics
            if "metrics" in step:
                m = step["metrics"]
                print(f"      metrics: UW={m['uw']:.4f}  NW={m['nw']:.4f}  "
                      f"NW+={m['nw_plus']:.4f}  EF1={m['ef1']:.4f}  "
                      f"EF1+={m['ef1_plus']:.4f}")
            # Show per-agent values
            if "per_agent" in step:
                pa = step["per_agent"]
                for j, name in enumerate(step["support_names"]):
                    vals = "  ".join(
                        f"{k}={pa[k][j]:.4f}"
                        for k in ["uw", "nw", "nw_plus", "ef1", "ef1_plus"]
                    )
                    print(f"        {name}: {vals}")
            # Show in/out counterfactual for the seed
            io = step.get("in_out")
            if io is not None:
                d = io["delta"]
                print(f"      in/out ({t['seed']}): entry_mass={io['entry_mass']:.4f}  "
                      f"eq_shift={io['equilibrium_shift']:.4f}")
                print(f"        delta: UW={d['uw']:+.4f}  NW={d['nw']:+.4f}  "
                      f"NW+={d['nw_plus']:+.4f}  EF1={d['ef1']:+.4f}  "
                      f"EF1+={d['ef1_plus']:+.4f}")
            # Show top 3 deviation gains
            if step["deviation_gains"]:
                sorted_gains = sorted(
                    step["deviation_gains"].items(), key=lambda x: x[1], reverse=True,
                )
                top = sorted_gains[:3]
                gains_str = ", ".join(f"{name}={gain:.4f}" for name, gain in top)
                print(f"      top deviations: {gains_str}")

    # Section 2: Terminal support sets
    print("\n--- Section 2: Terminal Support Sets ---")
    print(f"  All seeds converge to same support: {point_analysis['all_converge_same']}")
    print(f"  Unique terminal supports ({len(point_analysis['unique_supports'])}):")
    for i, support in enumerate(point_analysis["unique_supports"]):
        print(f"    {i+1}. {{{', '.join(support)}}}")

    seeds_per_support = {}
    for t in point_traces:
        key = tuple(sorted(t["terminal_support"]))
        seeds_per_support.setdefault(key, []).append(t["seed"])
    for support, seeds in seeds_per_support.items():
        print(f"    Seeds -> {{{', '.join(support)}}}: {seeds}")

    # Section 3: Entry order analysis
    print("\n--- Section 3: Entry Order Analysis ---")
    print("  Average entry step per strategy (across seeds):")
    entry_data = []
    for name in strategy_names:
        steps = [s for s in point_analysis["entry_order"][name] if s is not None]
        if steps:
            avg = np.mean(steps)
            entry_data.append((name, avg, len(steps)))
        else:
            entry_data.append((name, float("inf"), 0))
    entry_data.sort(key=lambda x: x[1])
    for name, avg_step, n_entries in entry_data:
        if n_entries > 0:
            print(f"    {name:20s}: step {avg_step:.1f} (entered in {n_entries}/{len(point_traces)} seeds)")
        else:
            print(f"    {name:20s}: never entered")

    print(f"\n  Always in terminal support: {point_analysis['always_entered']}")
    print(f"  Never in terminal support:  {point_analysis['never_entered']}")

    # Section 3b: Trajectory crossings
    crossings = point_analysis.get("crossings", [])
    print(f"\n--- Section 3b: Trajectory Crossings ---")
    if crossings:
        print(f"  {len(crossings)} shared support set(s):")
        for c in crossings:
            names = ", ".join(c["support"])
            print(f"    {{{names}}}")
            for seed, step in c["visitors"]:
                print(f"      {seed:20s} @ step {step}")
    else:
        print("  No crossings — every seed takes a unique path.")

    # Section 4: Terminal equilibrium quality
    print("\n--- Section 4: Terminal Equilibrium Quality ---")
    print("  Max regret of terminal equilibrium in full game:")
    for i, t in enumerate(point_traces):
        regret = point_analysis["terminal_regrets"][i]
        print(f"    Seed {t['seed']:20s}: max_regret = {regret:.4f}")

    # Section 4b: Terminal welfare metrics and in/out
    print("\n--- Section 4b: Terminal Welfare & In/Out Summary ---")
    for t in point_traces:
        terminal = t["trace"][-1]
        m = terminal.get("metrics", {})
        support_str = ", ".join(terminal["support_names"])
        print(f"\n  Seed: {t['seed']}  ->  terminal=[{support_str}]")
        print(f"    Welfare:  UW={m.get('uw',0):.2f}  NW={m.get('nw',0):.2f}  "
              f"NW+={m.get('nw_plus',0):.2f}  EF1={m.get('ef1',0):.4f}  "
              f"EF1+={m.get('ef1_plus',0):.4f}")
        # Per-agent at terminal
        pa = terminal.get("per_agent", {})
        if pa:
            for j, name in enumerate(terminal["support_names"]):
                vals = "  ".join(
                    f"{k}={pa[k][j]:.2f}" for k in ["uw", "nw", "nw_plus"]
                )
                fair = "  ".join(
                    f"{k}={pa[k][j]:.4f}" for k in ["ef1", "ef1_plus"]
                )
                print(f"      {name:20s}: {vals}  {fair}")
        # In/out at terminal
        io = terminal.get("in_out")
        if io is not None:
            d = io["delta"]
            print(f"    In/Out ({t['seed']}): entry_mass={io['entry_mass']:.4f}  "
                  f"eq_shift={io['equilibrium_shift']:.4f}")
            print(f"      delta: UW={d['uw']:+.2f}  NW={d['nw']:+.2f}  "
                  f"NW+={d['nw_plus']:+.2f}  EF1={d['ef1']:+.4f}  "
                  f"EF1+={d['ef1_plus']:+.4f}")
        else:
            print(f"    In/Out: N/A (singleton terminal)")

    # Section 5: Bootstrap uncertainty
    if bootstrap_agg is not None:
        print("\n--- Section 5: Bootstrap Uncertainty ---")
        print(f"  Bootstrap samples: {bootstrap_agg['n_samples']}")
        print(f"  Convergence rate (all seeds -> same support): "
              f"{bootstrap_agg['convergence_rate']:.1%}")

        print("\n  Path lengths per seed (mean +/- std [95% CI]):")
        for name in strategy_names:
            m = bootstrap_agg["path_lengths_per_seed"][name]
            print(f"    {name:20s}: {m['mean']:.1f} +/- {m['std']:.1f}  "
                  f"[{m['ci_lower']:.0f}, {m['ci_upper']:.0f}]")

        print("\n  Entry step per strategy (mean +/- std [95% CI]):")
        entry_items = []
        for name in strategy_names:
            m = bootstrap_agg["entry_step"][name]
            if m["mean"] is not None:
                entry_items.append((name, m))
            else:
                entry_items.append((name, None))
        entry_items.sort(key=lambda x: x[1]["mean"] if x[1] else float("inf"))
        for name, m in entry_items:
            if m is not None:
                print(f"    {name:20s}: {m['mean']:.2f} +/- {m['std']:.2f}  "
                      f"[{m['ci_lower']:.2f}, {m['ci_upper']:.2f}]")
            else:
                print(f"    {name:20s}: never entered (in any bootstrap)")

        print("\n  Always-entered frequency (in all terminal supports):")
        for name in strategy_names:
            freq = bootstrap_agg["always_entered_freq"][name]
            print(f"    {name:20s}: {freq:.1%}")

        print("\n  Never-entered frequency (in no terminal support):")
        for name in strategy_names:
            freq = bootstrap_agg["never_entered_freq"][name]
            if freq > 0:
                print(f"    {name:20s}: {freq:.1%}")

        print("\n  Most common terminal support sets:")
        for support, count in bootstrap_agg["most_common_supports"][:5]:
            print(f"    {{{', '.join(support)}}}: {count}/{bootstrap_agg['n_samples']} "
                  f"({count/bootstrap_agg['n_samples']:.1%})")

        print("\n  Most common terminal per seed:")
        n_bs = bootstrap_agg["n_samples"]
        for name in strategy_names:
            tops = bootstrap_agg["terminal_per_seed"][name]
            top_support, top_count = tops[0]
            print(f"    {name:20s}: {{{', '.join(top_support)}}} "
                  f"({top_count}/{n_bs} = {top_count/n_bs:.0%})")
            for support, count in tops[1:]:
                print(f"    {'':20s}  {{{', '.join(support)}}} "
                      f"({count}/{n_bs} = {count/n_bs:.0%})")

        print(f"\n  Trajectory crossings: {bootstrap_agg['crossing_rate']:.0%} "
              f"of bootstrap samples have at least one crossing")
        if bootstrap_agg["most_common_crossings"]:
            print("  Most commonly shared support sets:")
            for support, count in bootstrap_agg["most_common_crossings"][:5]:
                print(f"    {{{', '.join(support)}}}: {count}/{n_bs} "
                      f"({count/n_bs:.0%})")

        print("\n  Terminal max regret per seed (mean +/- std [95% CI]):")
        for name in strategy_names:
            m = bootstrap_agg["terminal_regrets_per_seed"][name]
            print(f"    {name:20s}: {m['mean']:.4f} +/- {m['std']:.4f}  "
                  f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description="PSIE (Per-Strategy Iterative Expansion) analysis",
    )
    parser.add_argument(
        "--input", type=str,
        default=None,
        help="Path to iterative_analysis_results.pkl",
    )
    parser.add_argument(
        "--output", type=str,
        default=None,
        help="Path to save PSIE results pkl",
    )
    parser.add_argument("--solver", type=str, default="mene")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Deviation threshold for epsilon-Nash (default 0.0)")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap, only run point estimate")
    parser.add_argument("--max-bootstrap", type=int, default=None,
                        help="Cap number of bootstrap samples to use")
    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent.parent
    input_path = Path(args.input) if args.input else (
        base_dir / "data" / "analysis" / "iterative_analysis_results.pkl"
    )
    output_path = Path(args.output) if args.output else (
        base_dir / "data" / "analysis" / "psie_results.pkl"
    )

    # Load data
    print(f"Loading data from {input_path}...")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    strategy_names = data["config"]["strategy_names"]
    raw_samples = data["raw"]
    print(f"  Strategies: {strategy_names}")
    print(f"  Bootstrap samples available: {len(raw_samples)}")

    # Build average matrices (point estimate)
    matrix_keys = ["payoff", "nw", "nw_plus", "ef1", "ef1_plus"]
    print("Building average matrices...")
    avg_matrices = {}
    for key in matrix_keys:
        stacked = np.array([s["matrices"][key] for s in raw_samples])
        avg_matrices[key] = np.nanmean(stacked, axis=0)
    print(f"  Shape: {avg_matrices['payoff'].shape}")

    # Run PSIE point estimate
    print(f"\nRunning PSIE point estimate (solver={args.solver}, epsilon={args.epsilon})...")
    point_traces = run_psie_all_seeds(
        avg_matrices, strategy_names, solver=args.solver, epsilon=args.epsilon,
    )
    point_analysis = analyze_traces(point_traces, strategy_names, avg_matrices)

    # Run bootstrap
    bootstrap_result = None
    if not args.no_bootstrap:
        n_bootstrap = len(raw_samples)
        if args.max_bootstrap is not None:
            n_bootstrap = min(n_bootstrap, args.max_bootstrap)

        n_workers = min(n_bootstrap, multiprocessing.cpu_count())
        print(f"\nRunning PSIE on {n_bootstrap} bootstrap samples "
              f"({n_workers} workers)...")

        worker_args = [
            ({k: raw_samples[b_idx]["matrices"][k] for k in matrix_keys},
             strategy_names, args.solver, args.epsilon)
            for b_idx in range(n_bootstrap)
        ]

        with multiprocessing.Pool(n_workers) as pool:
            all_analyses = list(tqdm(
                pool.imap_unordered(_bootstrap_worker, worker_args),
                total=n_bootstrap, desc="Bootstrap PSIE",
            ))

        print("Aggregating bootstrap results...")
        bootstrap_agg = aggregate_bootstrap_analysis(all_analyses, strategy_names)
        bootstrap_result = {
            "analyses": all_analyses,
            "aggregated": bootstrap_agg,
        }
    else:
        bootstrap_agg = None

    # Print results
    print_psie_results(point_analysis, bootstrap_agg, strategy_names, point_traces)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "solver": args.solver,
            "epsilon": args.epsilon,
            "strategy_names": strategy_names,
            "input_path": str(input_path),
        },
        "point_estimate": {
            "matrices": avg_matrices,
            "traces": point_traces,
            "analysis": point_analysis,
        },
        "bootstrap": bootstrap_result,
    }
    with open(output_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
