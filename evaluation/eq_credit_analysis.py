"""Equilibrium-consistent credit analysis with fresh MENE recomputation.

For each of the 1000 bootstrap samples, re-solves all 2^10 − 1 = 1023
coalition equilibria from the stored raw payoff matrices using MENE, then
extracts three families of quantities:

1. **First-order credits** C_i(S) — marginal contribution of strategy i
   to the full set S of all 10 strategies.
2. **Second-order influences** Infl_{j→i}(S) = C_i(S) − C_i(S\\{j}) —
   how much strategy j's presence changes strategy i's credit.
3. **Greedy add / remove paths** — sequences that greedily maximise
   (or minimise loss of) each welfare metric.

All quantities are bootstrap-aggregated (mean, std, 95% CI).

Usage:
    python evaluation/eq_credit_analysis.py --workers 4
"""

import argparse
import itertools
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from iterative_game_analysis.metagame import MetaGame
from visuals.visualize_analysis import STRATEGY_ORDER

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

METRICS = ["uw", "nw", "nw_plus", "ef1", "ef1_plus"]

DATA_DIR = Path(__file__).parent.parent / "data" / "analysis"
INPUT_PKL = DATA_DIR / "iterative_analysis_results.pkl"
OUTPUT_PKL = DATA_DIR / "eq_credit_results.pkl"


def _summarize(samples):
    """Compute mean, std, 95% CI from bootstrap samples."""
    arr = np.array(samples)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
    }


# ---------------------------------------------------------------------------
# Per-sample extraction
# ---------------------------------------------------------------------------


def _solve_coalition(args):
    """Worker: solve equilibrium for one coalition. Top-level for pickling."""
    coalition_key, policy_subset, sub_payoff = args
    sub_game = MetaGame(policy_subset, sub_payoff)
    sigma = sub_game.solve("mene")
    return coalition_key, sigma


def _recompute_equilibria(payoff_matrix, policies, n_workers=1):
    """Solve MENE for every non-empty subset of policies.

    Args:
        payoff_matrix: Full n×n payoff matrix (rows/cols match policies order).
        policies: Ordered list of policy names.
        n_workers: Parallel workers (1 = sequential).

    Returns:
        Dict mapping frozenset(coalition) → equilibrium sigma.
    """
    policy_to_idx = {p: i for i, p in enumerate(policies)}

    tasks = []
    for r in range(1, len(policies) + 1):
        for subset in itertools.combinations(policies, r):
            policy_subset = sorted(subset)
            key = frozenset(policy_subset)
            indices = [policy_to_idx[p] for p in policy_subset]
            sub_payoff = payoff_matrix[np.ix_(indices, indices)]
            tasks.append((key, policy_subset, sub_payoff))

    cache = {}
    if n_workers <= 1:
        for task in tasks:
            key, sigma = _solve_coalition(task)
            cache[key] = sigma
    else:
        from multiprocessing import Pool
        with Pool(n_workers) as pool:
            for key, sigma in pool.imap_unordered(_solve_coalition, tasks):
                cache[key] = sigma

    return cache


def _build_value_table(eq_cache, matrices, policies):
    """Evaluate each welfare metric at each coalition's equilibrium.

    Args:
        eq_cache: frozenset(coalition) → sigma from _recompute_equilibria.
        matrices: Dict with 'payoff', 'nw', 'nw_plus', 'ef1', 'ef1_plus' matrices.
        policies: Ordered list of policy names.

    Returns:
        Dict[metric_name, Dict[frozenset, float]].
    """
    policy_to_idx = {p: i for i, p in enumerate(policies)}
    wf_names = ["uw", "nw", "nw_plus", "ef1", "ef1_plus"]
    wf_matrices = {
        "uw": matrices["payoff"],
        "nw": matrices["nw"],
        "nw_plus": matrices["nw_plus"],
        "ef1": matrices["ef1"],
        "ef1_plus": matrices["ef1_plus"],
    }

    value_table = {wf: {} for wf in wf_names}
    for key, sigma in eq_cache.items():
        policy_subset = sorted(key)
        indices = [policy_to_idx[p] for p in policy_subset]
        for wf in wf_names:
            sub = wf_matrices[wf][np.ix_(indices, indices)]
            sub_clean = np.nan_to_num(sub, nan=0.0)
            value_table[wf][key] = float(sigma @ sub_clean @ sigma)

    return value_table


def _build_index_from_values(value_table, policies):
    """Build the (frozenset(coalition_with), policy) → marginals index.

    For each policy i and each subset S of the other n-1 policies,
    computes marginals[m] = value_table[m][S∪{i}] - value_table[m][S]
    (0.0 when S is empty).

    Returns the same index structure that _extract_credits / _greedy_add etc. expect.
    """
    index = {}
    others_map = {p: [q for q in policies if q != p] for p in policies}

    for policy in policies:
        others = others_map[policy]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                S = frozenset(subset)
                S_with = S | {policy}
                v_without = {m: value_table[m].get(S, 0.0) for m in METRICS}
                v_with = {m: value_table[m][S_with] for m in METRICS}
                marginals = {m: v_with[m] - v_without[m] for m in METRICS}
                index[(S_with, policy)] = marginals

    return index


def _extract_credits(index, policies):
    """First-order credits C_i(S) for each strategy i."""
    S = frozenset(policies)
    credits = {}
    for p in policies:
        marginals = index.get((S, p))
        if marginals is None:
            raise KeyError(f"Missing marginals for policy={p}, coalition_with={S}")
        credits[p] = dict(marginals)  # copy
    return credits


def _extract_influences(index, policies):
    """Second-order influences Infl_{j→i}(S) for all pairs (i, j)."""
    S = frozenset(policies)
    influences = {}  # (i, j) → {metric: value}
    for i in policies:
        c_i_S = index[(S, i)]
        for j in policies:
            if j == i:
                continue
            S_without_j = S - {j}
            c_i_S_minus_j = index[(S_without_j, i)]
            influences[(i, j)] = {
                m: c_i_S[m] - c_i_S_minus_j[m] for m in METRICS
            }
    return influences


def _coalition_value(index, coalition, policies_in_coalition):
    """Reconstruct Y(X) by summing marginals along an arbitrary insertion order."""
    if not policies_in_coalition:
        return {m: 0.0 for m in METRICS}
    # Build up from ∅
    value = {m: 0.0 for m in METRICS}
    current = set()
    for p in sorted(policies_in_coalition):  # deterministic order
        current.add(p)
        key = (frozenset(current), p)
        marg = index[key]
        for m in METRICS:
            value[m] += marg[m]
    return value


def _greedy_add(index, policies, metric):
    """Greedy-add path: start ∅, add argmax Y(X∪{s}) − Y(X) each step."""
    remaining = set(policies)
    current = set()
    path = []  # list of (strategy_added, Y_value_after)
    y_val = 0.0

    for _ in range(len(policies)):
        best_s, best_gain = None, -np.inf
        for s in remaining:
            candidate = frozenset(current | {s})
            marg = index[(candidate, s)]
            if marg[metric] > best_gain:
                best_gain = marg[metric]
                best_s = s
        current.add(best_s)
        remaining.remove(best_s)
        y_val += best_gain
        path.append({"strategy": best_s, "value": y_val})

    return path


def _greedy_remove(index, policies, metric):
    """Greedy-remove path: start S, remove argmin Y(X) − Y(X\\{r}) each step."""
    current = set(policies)
    y_val_dict = _coalition_value(index, current, current)
    y_val = y_val_dict[metric]
    path = [{"strategy": None, "value": y_val}]  # starting point

    for _ in range(len(policies)):
        best_r, best_loss = None, np.inf
        for r in current:
            # loss = Y(X) - Y(X\{r}) = marginal of r in X
            marg = index[(frozenset(current), r)]
            loss = marg[metric]
            if loss < best_loss:
                best_loss = loss
                best_r = r
        current.remove(best_r)
        y_val -= best_loss
        path.append({"strategy": best_r, "value": y_val})

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Eq-credit analysis with fresh MENE solves")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for MENE solves (default: 1)")
    args = parser.parse_args()
    n_workers = args.workers

    print(f"Loading {INPUT_PKL} ...")
    with open(INPUT_PKL, "rb") as f:
        raw = pickle.load(f)

    samples = raw["raw"]
    n_samples = len(samples)
    print(f"  {n_samples} bootstrap samples loaded.")
    print(f"  Workers: {n_workers}")

    policies = STRATEGY_ORDER  # canonical order

    # Accumulators
    credit_acc = {p: {m: [] for m in METRICS} for p in policies}
    influence_acc = {
        (i, j): {m: [] for m in METRICS}
        for i in policies for j in policies if i != j
    }
    greedy_add_acc = {m: [] for m in METRICS}    # list of paths
    greedy_remove_acc = {m: [] for m in METRICS}

    for sample in tqdm(samples, desc="Processing samples"):
        # Recompute all 1023 coalition equilibria from raw matrices
        payoff = sample["matrices"]["payoff"]
        eq_cache = _recompute_equilibria(payoff, policies, n_workers)
        value_table = _build_value_table(eq_cache, sample["matrices"], policies)
        index = _build_index_from_values(value_table, policies)

        # 1. First-order credits
        credits = _extract_credits(index, policies)
        for p in policies:
            for m in METRICS:
                credit_acc[p][m].append(credits[p][m])

        # 2. Second-order influences
        influences = _extract_influences(index, policies)
        for (i, j), vals in influences.items():
            for m in METRICS:
                influence_acc[(i, j)][m].append(vals[m])

        # 3. Greedy paths
        for m in METRICS:
            greedy_add_acc[m].append(_greedy_add(index, policies, m))
            greedy_remove_acc[m].append(_greedy_remove(index, policies, m))

    # --- Aggregate ---
    print("Aggregating ...")

    # Credits
    credit_summary = {}
    for p in policies:
        credit_summary[p] = {m: _summarize(credit_acc[p][m]) for m in METRICS}

    # Influences
    influence_summary = {}
    for (i, j) in influence_acc:
        influence_summary[(i, j)] = {
            m: _summarize(influence_acc[(i, j)][m]) for m in METRICS
        }

    # Greedy paths — aggregate per step
    def _aggregate_paths(path_acc):
        """Aggregate list-of-paths into per-step summaries."""
        n_steps = len(path_acc[0])
        result = []
        for step in range(n_steps):
            values = [path_acc[s][step]["value"] for s in range(len(path_acc))]
            # Most common strategy at this step
            strategies = [path_acc[s][step]["strategy"] for s in range(len(path_acc))]
            strategy_counts = defaultdict(int)
            for s in strategies:
                strategy_counts[s] += 1
            most_common = max(strategy_counts, key=strategy_counts.get)
            result.append({
                "most_common_strategy": most_common,
                "strategy_counts": dict(strategy_counts),
                "value": _summarize(values),
            })
        return result

    greedy_add_summary = {
        m: _aggregate_paths(greedy_add_acc[m]) for m in METRICS
    }
    greedy_remove_summary = {
        m: _aggregate_paths(greedy_remove_acc[m]) for m in METRICS
    }

    results = {
        "credits": credit_summary,
        "influences": influence_summary,
        "greedy_add": greedy_add_summary,
        "greedy_remove": greedy_remove_summary,
        "n_samples": n_samples,
        "policies": policies,
        "metrics": METRICS,
    }

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved to {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
