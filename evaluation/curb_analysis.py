"""CURB set analysis across bootstrap samples.

Enumerates all CURB sets (Closed Under Rational Behavior, Basu & Weibull 1991)
in the symmetric 2-player empirical meta-game. A subset S is CURB if CBR(S) ⊆ S,
where CBR(S) is the set of all strategies that are a best response to *some*
mixture over S.

With n strategies there are 2^n - 1 non-empty subsets to check. For n=10 this
is 1023, very tractable.

Follows the algorithmic approach of Benisch, Davis & Sandholm (JAIR 2010).

Usage:
    python evaluation/curb_analysis.py                     # full run
    python evaluation/curb_analysis.py --no-bootstrap      # point estimate only
    python evaluation/curb_analysis.py --max-bootstrap 20  # quick sanity check
    python evaluation/curb_analysis.py --method closure    # closure-based algorithm
    python evaluation/curb_analysis.py --method compare --no-bootstrap  # compare both
"""

import argparse
import multiprocessing
import pickle
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iterative_game_analysis.metagame import MetaGame
from visuals.visualize_analysis import DISPLAY_NAMES, STRATEGY_ORDER

# Metric keys for welfare/fairness matrices (excluding payoff which is used
# to solve the equilibrium and to derive UW).
METRIC_KEYS = ("nw", "nw_plus", "ef1", "ef1_plus")


# ---------------------------------------------------------------------------
# Welfare/fairness metric helpers
# ---------------------------------------------------------------------------


def _summarize(samples):
    """Compute mean, std, 95% CI from bootstrap samples."""
    arr = np.array(samples)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
    }


def solve_restricted_equilibrium(payoff_sub, policy_names, solver="mene"):
    """Solve for equilibrium on a restricted subgame.

    Args:
        payoff_sub: (k, k) payoff matrix of the restricted game.
        policy_names: List of k policy name strings.
        solver: Solver name (default "mene").

    Returns:
        1-D numpy array of equilibrium mixture (length k).
    """
    if payoff_sub.shape[0] == 1:
        return np.array([1.0])
    mg = MetaGame(policies=policy_names, payoff_matrix=payoff_sub)
    return mg.solve(solver)


def compute_curb_metrics(curb_indices, matrices, strategy_names, solver="mene"):
    """Solve restricted equilibrium and compute welfare/fairness metrics.

    Args:
        curb_indices: Sorted list/tuple of strategy indices in the CURB set.
        matrices: Dict with keys "payoff", "nw", "nw_plus", "ef1", "ef1_plus".
        strategy_names: Full list of strategy names.
        solver: Solver name for equilibrium computation.

    Returns:
        Dict with keys: sigma, eq_value, uw, nw, nw_plus, ef1, ef1_plus.
        Returns None if equilibrium solving fails.
    """
    idx = sorted(curb_indices)
    ix = np.ix_(idx, idx)

    payoff_sub = matrices["payoff"][ix]
    policy_names = [strategy_names[i] for i in idx]

    try:
        sigma = solve_restricted_equilibrium(payoff_sub, policy_names, solver)
    except Exception:
        return None

    # eq_value = expected payoff per player; uw = 2x that (symmetric game)
    eq_value = float(sigma @ payoff_sub @ sigma)
    result = {
        "sigma": sigma,
        "eq_value": eq_value,
        "uw": 2.0 * eq_value,
    }

    for key in METRIC_KEYS:
        m_sub = matrices[key][ix]
        # EF1 matrices may have NaN for walk matchups — treat as 0
        m_sub = np.nan_to_num(m_sub, nan=0.0)
        result[key] = float(sigma @ m_sub @ sigma)

    return result


# ---------------------------------------------------------------------------
# CURB-selected Banzhaf attribution
# ---------------------------------------------------------------------------
#
# Uses CURB sets as the coalition pool for Banzhaf, instead of all 2^n subsets.
# For each strategy i:
#   β_CURB(i) = (1/|{C : i ∈ C}|) * Σ_{C : i ∈ C} [v(C) - v(C \ {i})]
#
# Only solves equilibria for CURB sets and their "minus one" variants,
# making it much cheaper than full Banzhaf (which needs all 2^n subsets).


# Metrics to compute CURB-Banzhaf for.
CURB_BANZHAF_METRICS = ("uw", "nw", "nw_plus", "ef1", "ef1_plus")


def _solve_and_evaluate(subset_indices, matrices, strategy_names):
    """Solve equilibrium on a subset and evaluate all metrics.

    Args:
        subset_indices: Sorted list of strategy indices.
        matrices: Dict with all metric matrices.
        strategy_names: Full list of strategy names.

    Returns:
        Dict: {metric -> float} or None if solve fails.
    """
    idx = sorted(subset_indices)
    ix = np.ix_(idx, idx)
    payoff_sub = matrices["payoff"][ix]
    policy_names = [strategy_names[i] for i in idx]

    try:
        sigma = solve_restricted_equilibrium(payoff_sub, policy_names)
    except Exception:
        return None

    metric_map = {
        "uw": matrices["payoff"],
        "nw": matrices["nw"],
        "nw_plus": matrices["nw_plus"],
        "ef1": matrices["ef1"],
        "ef1_plus": matrices["ef1_plus"],
    }
    result = {}
    for m, M in metric_map.items():
        sub = np.nan_to_num(M[ix], nan=0.0)
        result[m] = float(sigma @ sub @ sigma)
    return result


def compute_curb_banzhaf(matrices, strategy_names, curb_sets,
                         include_full_game=True):
    """Compute CURB-selected Banzhaf values.

    Uses CURB sets as the coalition pool for Banzhaf. For each strategy i,
    averages marginal contribution v(C) - v(C \\ {i}) across all CURB sets
    C that contain i.

    Only solves equilibria for CURB sets and their "minus one" variants.

    Args:
        matrices: Dict with keys "payoff", "nw", "nw_plus", "ef1", "ef1_plus".
        strategy_names: Full list of strategy names.
        curb_sets: List of frozensets (each a set of strategy indices).
        include_full_game: If True, include the full game as a coalition.

    Returns:
        Dict with:
          "banzhaf": {metric: {strategy_name: float}}
          "counts": {strategy_name: int} — how many CURB sets contain each
          "marginals": list of per-(strategy, CURB set) marginal records
    """
    n = len(strategy_names)

    # Build coalition list (optionally add full game)
    coalitions = list(curb_sets)
    full_game = frozenset(range(n))
    if include_full_game and full_game not in coalitions:
        coalitions.append(full_game)

    # Collect all subsets we need: each coalition C, and each C \ {i}
    subsets_needed = set()
    for C in coalitions:
        subsets_needed.add(C)
        for i in C:
            minus_i = C - {i}
            if minus_i:  # skip empty set
                subsets_needed.add(minus_i)

    # Solve all needed subsets (with caching to avoid duplicates)
    value_cache = {}  # frozenset -> {metric -> float}
    for S in subsets_needed:
        vals = _solve_and_evaluate(S, matrices, strategy_names)
        if vals is not None:
            value_cache[S] = vals

    # Compute CURB-Banzhaf: accumulate marginals per strategy
    banzhaf_sums = {m: defaultdict(float) for m in CURB_BANZHAF_METRICS}
    counts = defaultdict(int)
    marginals = []

    for C in coalitions:
        if C not in value_cache:
            continue
        v_C = value_cache[C]

        for i in C:
            s_name = strategy_names[i]
            counts[s_name] += 1
            C_minus_i = C - {i}

            # v(C \ {i}): 0 if empty, otherwise look up
            if not C_minus_i:
                v_without = {m: 0.0 for m in CURB_BANZHAF_METRICS}
            elif C_minus_i in value_cache:
                v_without = value_cache[C_minus_i]
            else:
                continue  # solve failed for C \ {i}, skip this marginal

            marginal_rec = {"strategy": s_name, "curb_set": C}
            for m in CURB_BANZHAF_METRICS:
                mc = v_C[m] - v_without[m]
                banzhaf_sums[m][s_name] += mc
                marginal_rec[m] = mc
            marginals.append(marginal_rec)

    # Average by number of coalitions containing each strategy
    banzhaf = {m: {} for m in CURB_BANZHAF_METRICS}
    for m in CURB_BANZHAF_METRICS:
        for i in range(n):
            s_name = strategy_names[i]
            c = counts[s_name]
            banzhaf[m][s_name] = banzhaf_sums[m][s_name] / c if c > 0 else 0.0

    return {
        "banzhaf": banzhaf,
        "counts": dict(counts),
        "marginals": marginals,
    }


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def compute_cbr(payoff_matrix, S_indices):
    """Compute the Conditional Best Response set CBR(S).

    For each strategy i in {0, ..., n-1}, check whether i is a best response
    to *some* mixture sigma over S. This is an LP feasibility problem:

        Find sigma_j >= 0 for j in S, sum(sigma) = 1
        s.t.  sum_j sigma_j * U[i,j] >= sum_j sigma_j * U[k,j]  for all k

    If feasible, then i in CBR(S).

    Args:
        payoff_matrix: (n, n) symmetric game payoff matrix.
        S_indices: Iterable of strategy indices defining subset S.

    Returns:
        frozenset of strategy indices in CBR(S).
    """
    n = payoff_matrix.shape[0]
    S = list(S_indices)
    m = len(S)
    cbr = set()

    for i in range(n):
        # We want to find sigma in Delta(S) such that strategy i is a BR.
        # Variables: sigma_j for j in S (length m)
        # Objective: dummy (feasibility only) -> minimize 0
        c = np.zeros(m)

        # Inequality constraints: A_ub @ sigma <= b_ub
        # For each competitor k != i:
        #   sum_j sigma_j * U[k,j] - sum_j sigma_j * U[i,j] <= 0
        #   i.e. sum_j sigma_j * (U[k,j] - U[i,j]) <= 0
        A_ub_rows = []
        b_ub_rows = []
        for k in range(n):
            if k == i:
                continue
            row = np.array([payoff_matrix[k, S[j]] - payoff_matrix[i, S[j]]
                            for j in range(m)])
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)

        A_ub = np.array(A_ub_rows) if A_ub_rows else None
        b_ub = np.array(b_ub_rows) if b_ub_rows else None

        # Equality constraint: sum(sigma) = 1
        A_eq = np.ones((1, m))
        b_eq = np.array([1.0])

        # Bounds: sigma_j >= 0
        bounds = [(0.0, None)] * m

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")

        if result.success:
            cbr.add(i)

    return frozenset(cbr)


def is_curb(payoff_matrix, S_indices):
    """Check if subset S is CURB: CBR(S) ⊆ S.

    Args:
        payoff_matrix: (n, n) payoff matrix.
        S_indices: Iterable of strategy indices.

    Returns:
        True if S is CURB.
    """
    S = frozenset(S_indices)
    cbr = compute_cbr(payoff_matrix, S)
    return cbr.issubset(S)


def find_all_curb_sets(payoff_matrix, n_strategies):
    """Enumerate all non-empty subsets and return those that are CURB.

    Args:
        payoff_matrix: (n, n) payoff matrix.
        n_strategies: Number of strategies.

    Returns:
        List of frozensets, each a CURB set.
    """
    curb_sets = []
    for size in range(1, n_strategies + 1):
        for subset in combinations(range(n_strategies), size):
            S = frozenset(subset)
            if is_curb(payoff_matrix, S):
                curb_sets.append(S)
    return curb_sets


def find_minimal_curb_sets(all_curb_sets):
    """Filter to minimal CURB sets (no proper CURB subset).

    Args:
        all_curb_sets: List of frozensets that are CURB.

    Returns:
        List of frozensets that are minimal CURB.
    """
    curb_set = set(all_curb_sets)
    minimal = []
    for S in all_curb_sets:
        is_minimal = True
        for T in curb_set:
            if T < S:  # T is a strict subset of S and also CURB
                is_minimal = False
                break
        if is_minimal:
            minimal.append(S)
    return minimal


def curb_closure(payoff_matrix, S_indices):
    """Compute the CURB closure of S: smallest CURB set containing S.

    Iteratively: S_{t+1} = S_t union CBR(S_t) until stable.

    Args:
        payoff_matrix: (n, n) payoff matrix.
        S_indices: Iterable of strategy indices.

    Returns:
        frozenset: the CURB closure.
    """
    S = frozenset(S_indices)
    while True:
        cbr = compute_cbr(payoff_matrix, S)
        S_new = S | cbr
        if S_new == S:
            return S
        S = S_new


def find_minimal_curb_sets_via_closure(payoff_matrix, n_strategies):
    """Find all minimal CURB sets using the closure algorithm.

    For each strategy i, curb_closure({i}) yields the smallest CURB set
    containing i. Every minimal CURB set is the closure of some singleton.
    We collect unique closures, then filter to those with no proper subset
    among the closures (Benisch, Davis & Sandholm, JAIR 2010).

    Complexity: n closure calls, each O(n) iterations of compute_cbr.

    Args:
        payoff_matrix: (n, n) payoff matrix.
        n_strategies: Number of strategies.

    Returns:
        List of frozensets, each a minimal CURB set.
    """
    seen = set()
    closures = []
    for i in range(n_strategies):
        closure = curb_closure(payoff_matrix, [i])
        if closure not in seen:
            seen.add(closure)
            closures.append(closure)
    # Filter to globally minimal: no proper subset among closures
    minimals = []
    for S in closures:
        if not any(T < S for T in seen):
            minimals.append(S)
    return minimals


def _lp_check_one(args):
    """Check if strategy h is a BR to some mixture over S. Top-level for parallelism."""
    h, payoff_matrix, S = args
    n = payoff_matrix.shape[0]
    m = len(S)
    c_lp = np.zeros(m)
    all_k = [k for k in range(n) if k != h]
    A_ub = payoff_matrix[all_k][:, S] - payoff_matrix[h, S][np.newaxis, :]
    b_ub = np.zeros(len(all_k))
    A_eq = np.ones((1, m))
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * m
    result = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")
    return (h, result.success)


def _expand_one_wcurb(args):
    """Expand a single wCURB candidate to its closure. Picklable top-level function."""
    start_T, pure_br, n = args

    def reachable_from_profile(i, j):
        visited = set()
        stack = [(i, j)]
        while stack:
            a, b = stack.pop()
            if (a, b) in visited:
                continue
            visited.add((a, b))
            br_b = pure_br[b]
            if (br_b, b) not in visited:
                stack.append((br_b, b))
            br_a = pure_br[a]
            if (a, br_a) not in visited:
                stack.append((a, br_a))
        return visited

    C_s = set(start_T)
    converged = False
    while not converged:
        converged = True
        for t_i in range(n):
            for t_j in range(n):
                if t_i in C_s and t_j in C_s:
                    reachable = reachable_from_profile(t_i, t_j)
                    T1 = set(a for a, b in reachable)
                    T2 = set(b for a, b in reachable)
                    T_t = T1 | T2
                    if not T_t.issubset(C_s):
                        C_s = C_s | T_t
                        converged = False
    return frozenset(C_s)


def find_minimal_curb_sets_klimm_weibull(payoff_matrix, n_strategies, parallel=False,
                                          max_workers=None):
    """Find all minimal CURB sets using Klimm & Weibull (2009) algorithm.

    Faithful implementation of Algorithms 1 and 2 from:
      Klimm & Weibull, "Finding all minimal CURB sets", 2009.

    Algorithm 1: Find all minimal wCURB configurations via the pure
      best-reply graph on strategy profiles.
      - For each profile s, compute P(s) = reachable profiles via BR graph
      - Compute T(s) = minimal product set containing P(s)
      - Iteratively expand: for each t in C_s, if T(t) not in C_s, expand
      - Collect minimal results into family C

    Algorithm 2: Promote wCURB candidates to sCURB (= CURB for 2-player)
      via LP feasibility checks on the strong stability sets.
      - Maintains family T of candidates (sub-complete w.r.t. sCURB)
      - Picks size-minimal T, checks all h outside T via LP
      - If violator found: adds h to T AND updates all T' in family
      - If no violator: T is confirmed sCURB, removed from family

    For two-player games, CURB = sCURB (Remark 1 in paper).

    Args:
        payoff_matrix: (n, n) symmetric game payoff matrix.
        n_strategies: Number of strategies.
        parallel: If True, parallelize expansion and LP checks.
        max_workers: Number of workers for parallel mode.

    Returns:
        List of frozensets, each a minimal CURB set.
    """
    n = n_strategies
    if max_workers is None:
        import multiprocessing as _mp
        max_workers = _mp.cpu_count()

    # ── Algorithm 1: Find all minimal wCURB configurations ──
    # Step 1a: Build best-reply graph and compute P(s) for all profiles
    pure_br = np.argmax(payoff_matrix, axis=0)  # pure_br[j] = BR to j

    def reachable_from_profile(i, j):
        """Compute P(s) = profiles reachable from (i,j) via BR graph."""
        visited = set()
        stack = [(i, j)]
        while stack:
            a, b = stack.pop()
            if (a, b) in visited:
                continue
            visited.add((a, b))
            # Player 1 deviates: BR to b
            br_b = pure_br[b]
            if (br_b, b) not in visited:
                stack.append((br_b, b))
            # Player 2 deviates: BR to a
            br_a = pure_br[a]
            if (a, br_a) not in visited:
                stack.append((a, br_a))
        return visited

    def compute_T(reachable):
        """Compute T(s) = minimal product set containing P(s).
        For symmetric 2-player: T = T1 union T2 (projections)."""
        T1 = frozenset(a for a, b in reachable)
        T2 = frozenset(b for a, b in reachable)
        return T1 | T2

    # Precompute T(s) for all profiles s = (i, j)
    # Paper: "foreach s in S do T(s) := x_{i in N} union_{t in P(s)} supp(t)"
    T_of_profile = {}
    profile_bar = tqdm(total=n * n, desc="Alg 1: Profile reachability", unit="profile")
    for s_i in range(n):
        for s_j in range(n):
            reachable = reachable_from_profile(s_i, s_j)
            T_of_profile[(s_i, s_j)] = frozenset(compute_T(reachable))
            profile_bar.update(1)
    profile_bar.close()

    unique_Ts = set(T_of_profile.values())
    print(f"  Unique T(s) values: {len(unique_Ts)}")

    # Paper Algorithm 1 lines 4-22: foreach s in S, expand C_s and collect minimals
    # The paper iterates over all profiles s in S (= all (i,j) pairs)
    # C_s starts as T(s), then expands by checking T(t) for all t in C_s
    family_C = []  # the output family of minimal wCURB configurations
    found_set = set()

    all_profiles = [(i, j) for i in range(n) for j in range(n)]

    if parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Deduplicate: only expand unique T(s) values
        sorted_Ts = sorted(unique_Ts, key=len)

        expanded_cache = {}
        minimal_wcurbs = []
        found = set()

        pbar = tqdm(total=len(sorted_Ts),
                    desc="Alg 1: Expand wCURB candidates (parallel)")
        chunk_size = max(max_workers * 4, 1)
        idx = 0
        while idx < len(sorted_Ts):
            chunk = []
            while idx < len(sorted_Ts) and len(chunk) < chunk_size:
                c = sorted_Ts[idx]
                idx += 1
                # Early pruning: skip supersets of known minimals
                if any(m.issubset(c) for m in found):
                    pbar.update(1)
                    continue
                chunk.append(c)

            if not chunk:
                continue

            args_list = [(c, pure_br, n) for c in chunk]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_expand_one_wcurb, a): a_idx
                           for a_idx, a in enumerate(args_list)}
                for future in as_completed(futures):
                    C_frozen = future.result()
                    pbar.update(1)
                    if C_frozen in found:
                        continue

                    # Minimality check (Algorithm 1 lines 16-21)
                    is_minimal = True
                    to_remove = []
                    for existing in minimal_wcurbs:
                        if existing < C_frozen:
                            is_minimal = False
                            break
                        if C_frozen < existing:
                            to_remove.append(existing)
                    if is_minimal:
                        for r in to_remove:
                            minimal_wcurbs.remove(r)
                            found.discard(r)
                        minimal_wcurbs.append(C_frozen)
                        found.add(C_frozen)
                        print(f"\n    Found minimal wCURB, size={len(C_frozen)}")
        pbar.close()
    else:
        # Sequential: follows paper Algorithm 1 directly
        minimal_wcurbs = []
        found = set()

        sorted_Ts = sorted(unique_Ts, key=len)
        for start_T in tqdm(sorted_Ts, desc="Alg 1: Expand wCURB candidates"):
            # Early pruning
            if any(m.issubset(start_T) for m in found):
                continue

            # Lines 5-15: expand C_s until converged
            C_s = set(start_T)
            converged = False
            while not converged:
                converged = True
                # "foreach t in C_s": iterate over profiles in C_s x C_s
                for t_i in range(n):
                    for t_j in range(n):
                        if t_i in C_s and t_j in C_s:
                            T_t = T_of_profile.get((t_i, t_j), frozenset())
                            # Line 10: if T(t) not subset of C_s, expand
                            if not T_t.issubset(C_s):
                                C_s = C_s | T_t
                                converged = False

            C_frozen = frozenset(C_s)

            # Lines 16-21: minimality check and family update
            is_minimal = True
            to_remove = []
            for existing in minimal_wcurbs:
                if existing < C_frozen:
                    is_minimal = False
                    break
                if C_frozen < existing:
                    to_remove.append(existing)
            if is_minimal:
                for r in to_remove:
                    minimal_wcurbs.remove(r)
                    found.discard(r)
                if C_frozen not in found:
                    minimal_wcurbs.append(C_frozen)
                    found.add(C_frozen)

    print(f"  Minimal wCURB sets: {len(minimal_wcurbs)}")
    for i, m in enumerate(minimal_wcurbs):
        print(f"    wCURB {i}: size={len(m)}")

    # ── Algorithm 2: Promote wCURB to sCURB via LP (= CURB for 2-player) ──
    # Paper: maintains family T sub-complete w.r.t. sCURB sets.
    # Picks size-minimal T in T, checks LP for all h outside T.
    # If violator found: add h to T, update all T' in T, restart.
    # If no violator: T confirmed, remove from T, add to C.
    print(f"  Alg 2: LP verification{'  (parallel)' if parallel else ''}...")

    # Initialize family T with minimal wCURBs
    family_T = [set(m) for m in minimal_wcurbs]
    confirmed_curb = []
    lp_checks = 0
    round_num = 0

    while family_T:
        # Line 2: Choose size-minimal T in T
        family_T.sort(key=len)
        T_set = family_T[0]
        round_num += 1

        # Lines 3-9: Check all h outside T via LP
        outside = [h for h in range(n) if h not in T_set]
        violator = None

        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            args_list = [(h, payoff_matrix, sorted(T_set)) for h in outside]

            pbar = tqdm(total=len(outside),
                        desc=f"  Alg 2 round {round_num} (|T|={len(T_set)}, "
                             f"checking {len(outside)}, family={len(family_T)})")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_lp_check_one, a): a[0] for a in args_list}
                for future in as_completed(futures):
                    h, feasible = future.result()
                    lp_checks += 1
                    pbar.update(1)
                    if feasible and violator is None:
                        violator = h
                        for f in futures:
                            f.cancel()
                        break
            pbar.close()
        else:
            for h in tqdm(outside,
                          desc=f"  Alg 2 round {round_num} (|T|={len(T_set)}, "
                               f"checking {len(outside)}, family={len(family_T)})"):
                S = sorted(T_set)
                m = len(S)
                c_lp = np.zeros(m)
                all_k = [k for k in range(n) if k != h]
                A_ub = payoff_matrix[all_k][:, S] - payoff_matrix[h, S][np.newaxis, :]
                b_ub = np.zeros(len(all_k))
                A_eq = np.ones((1, m))
                b_eq = np.array([1.0])
                bounds = [(0.0, None)] * m
                result = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                                 bounds=bounds, method="highs")
                lp_checks += 1
                if result.success:
                    violator = h
                    break

        if violator is not None:
            # Lines 6-7: Violator found. Update T and all T' in family.
            # Paper: "Update all T in T" = add violator to all candidates
            # that need it (i.e., don't already contain it)
            print(f"    ✗ Violator: strategy {violator}, updating family")
            for T_prime in family_T:
                T_prime.add(violator)
            # Goto line 2 (restart with updated family)
        else:
            # Lines 11-13: T is confirmed sCURB.
            T_frozen = frozenset(T_set)
            confirmed_curb.append(T_frozen)
            family_T.pop(0)  # Remove T from family
            print(f"    ✓ Confirmed CURB set, size={len(T_frozen)}")

            # Line 13: "Update all T in T" = remove confirmed and
            # any T' that is a superset of confirmed
            family_T = [T_prime for T_prime in family_T
                        if not T_frozen.issubset(T_prime)]

    # Filter to minimal among confirmed
    curb_sets = []
    for C in confirmed_curb:
        if not any(C2 < C for C2 in confirmed_curb):
            curb_sets.append(C)

    print(f"  LP checks: {lp_checks}, confirmed minimal CURB sets: {len(curb_sets)}")
    return curb_sets


def find_all_curb_sets_via_closure(payoff_matrix, n_strategies):
    """Find all CURB sets using closure-derived minimals to prune search.

    Every CURB set must be a superset of at least one minimal CURB set.
    We first find minimals via closure, then only check subsets that contain
    at least one minimal — skipping subsets that provably cannot be CURB.

    Args:
        payoff_matrix: (n, n) payoff matrix.
        n_strategies: Number of strategies.

    Returns:
        List of frozensets, each a CURB set.
    """
    minimals = find_minimal_curb_sets_via_closure(payoff_matrix, n_strategies)
    curb_sets = []
    for size in range(1, n_strategies + 1):
        for subset in combinations(range(n_strategies), size):
            S = frozenset(subset)
            # Prune: S must contain at least one minimal CURB set
            if not any(m.issubset(S) for m in minimals):
                continue
            if is_curb(payoff_matrix, S):
                curb_sets.append(S)
    return curb_sets


# ---------------------------------------------------------------------------
# Bootstrap parallelization
# ---------------------------------------------------------------------------


def _compute_metrics_for_curb_sets(curb_sets, matrices, strategy_names, solver="mene"):
    """Compute welfare/fairness metrics for a list of CURB sets.

    Returns dict mapping frozenset -> metrics dict (or None on failure).
    """
    metrics = {}
    for S in curb_sets:
        m = compute_curb_metrics(S, matrices, strategy_names, solver)
        if m is not None:
            metrics[S] = m
    return metrics


def _curb_worker(args):
    """Top-level picklable worker for bootstrap parallelization.

    Runs find_all_curb_sets on one bootstrap payoff matrix (brute force).
    """
    matrices, n_strategies, strategy_names = args
    matrices = {k: np.array(v, dtype=float) for k, v in matrices.items()}
    sample_payoff = matrices["payoff"]
    all_curb = find_all_curb_sets(sample_payoff, n_strategies)
    minimal_curb = find_minimal_curb_sets(all_curb)
    metrics = _compute_metrics_for_curb_sets(all_curb, matrices, strategy_names)
    return {
        "all_curb_sets": all_curb,
        "minimal_curb_sets": minimal_curb,
        "metrics": metrics,
    }


def _curb_worker_closure(args):
    """Top-level picklable worker using closure-based algorithm.

    Finds minimals via closure, then all CURB sets via pruned enumeration.
    """
    matrices, n_strategies, strategy_names = args
    matrices = {k: np.array(v, dtype=float) for k, v in matrices.items()}
    sample_payoff = matrices["payoff"]
    all_curb = find_all_curb_sets_via_closure(sample_payoff, n_strategies)
    minimal_curb = find_minimal_curb_sets_via_closure(sample_payoff, n_strategies)
    metrics = _compute_metrics_for_curb_sets(all_curb, matrices, strategy_names)
    return {
        "all_curb_sets": all_curb,
        "minimal_curb_sets": minimal_curb,
        "metrics": metrics,
    }


def _curb_banzhaf_worker(args):
    """Top-level picklable worker for CURB-selected Banzhaf computation.

    For one bootstrap sample: finds CURB sets, then computes CURB-Banzhaf
    (one value per strategy, using CURB sets as the coalition pool).
    """
    matrices, n_strategies, strategy_names = args
    matrices = {k: np.array(v, dtype=float) for k, v in matrices.items()}

    # Find all CURB sets for this bootstrap sample
    sample_payoff = matrices["payoff"]
    all_curb = find_all_curb_sets(sample_payoff, n_strategies)

    # Compute CURB-Banzhaf (includes full game as a coalition)
    result = compute_curb_banzhaf(
        matrices, strategy_names, all_curb, include_full_game=True,
    )

    return {
        "all_curb_sets": all_curb,
        "banzhaf": result["banzhaf"],
        "counts": result["counts"],
    }


def aggregate_curb_banzhaf_results(all_results, strategy_names):
    """Aggregate CURB-Banzhaf results across bootstrap samples.

    Collects per-strategy Banzhaf values from each bootstrap and computes
    mean, std, 95% CI.

    Args:
        all_results: List of dicts from _curb_banzhaf_worker.
        strategy_names: List of strategy names.

    Returns:
        Dict with aggregated statistics per strategy per metric.
    """
    n_samples = len(all_results)

    # Collect: metric -> strategy -> [values across bootstraps]
    collector = {m: defaultdict(list) for m in CURB_BANZHAF_METRICS}
    count_collector = defaultdict(list)  # strategy -> [count per bootstrap]

    for res in all_results:
        for m in CURB_BANZHAF_METRICS:
            for strat, val in res["banzhaf"][m].items():
                collector[m][strat].append(val)
        for strat, c in res["counts"].items():
            count_collector[strat].append(c)

    # Summarize
    banzhaf_summary = {}
    for m in CURB_BANZHAF_METRICS:
        banzhaf_summary[m] = {}
        for strat in strategy_names:
            vals = collector[m].get(strat, [])
            if vals:
                banzhaf_summary[m][strat] = _summarize(vals)
            else:
                banzhaf_summary[m][strat] = _summarize([0.0])

    count_summary = {}
    for strat in strategy_names:
        vals = count_collector.get(strat, [])
        if vals:
            count_summary[strat] = _summarize(vals)

    # Also store raw per-bootstrap values for beeswarm plotting
    raw_values = {m: {s: list(collector[m].get(s, []))
                      for s in strategy_names}
                  for m in CURB_BANZHAF_METRICS}

    return {
        "banzhaf": banzhaf_summary,
        "raw_values": raw_values,
        "counts": count_summary,
        "n_samples": n_samples,
        "strategy_names": strategy_names,
    }


def aggregate_curb_results(all_results, strategy_names):
    """Aggregate CURB analysis results across bootstrap samples.

    Args:
        all_results: List of dicts from _curb_worker.
        strategy_names: List of strategy names.

    Returns:
        Dict with aggregated statistics.
    """
    n_samples = len(all_results)
    n = len(strategy_names)

    # Frequency of each CURB set
    curb_counter = Counter()
    minimal_counter = Counter()
    n_curb_per_sample = []
    n_minimal_per_sample = []

    # Strategy membership
    any_curb_membership = Counter()  # how many samples each strategy appears in any CURB
    minimal_membership = Counter()

    # Size distribution
    size_counts = defaultdict(list)  # size -> [count per sample]

    for res in all_results:
        all_curb = res["all_curb_sets"]
        minimal = res["minimal_curb_sets"]

        n_curb_per_sample.append(len(all_curb))
        n_minimal_per_sample.append(len(minimal))

        for S in all_curb:
            curb_counter[S] += 1
            for idx in S:
                any_curb_membership[idx] += 1

        for S in minimal:
            minimal_counter[S] += 1
            for idx in S:
                minimal_membership[idx] += 1

        # Size distribution for this sample
        sample_sizes = Counter(len(S) for S in all_curb)
        for size in range(1, n + 1):
            size_counts[size].append(sample_sizes.get(size, 0))

    # Convert to frequencies
    curb_freq = {S: count / n_samples for S, count in curb_counter.most_common()}
    minimal_freq = {S: count / n_samples for S, count in minimal_counter.most_common()}

    # Strategy membership frequency (in any CURB set, across samples)
    # Normalize: for each strategy, what fraction of samples it appears in at least one CURB set
    strat_in_any_curb = {}
    strat_in_minimal = {}
    for idx in range(n):
        # Count samples where this strategy appears in at least one CURB set
        count_any = 0
        count_min = 0
        for res in all_results:
            if any(idx in S for S in res["all_curb_sets"]):
                count_any += 1
            if any(idx in S for S in res["minimal_curb_sets"]):
                count_min += 1
        strat_in_any_curb[strategy_names[idx]] = count_any / n_samples
        strat_in_minimal[strategy_names[idx]] = count_min / n_samples

    # Size distribution stats
    size_distribution = {}
    for size in range(1, n + 1):
        vals = size_counts[size]
        if vals:
            size_distribution[size] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    # --- Aggregate welfare/fairness metrics per CURB set ---
    metric_names = ("eq_value", "uw") + METRIC_KEYS
    curb_welfare = {}
    for S, count in curb_counter.most_common():
        freq = count / n_samples
        # Collect metric values across samples that contain this CURB set
        metric_samples = {m: [] for m in metric_names}
        for res in all_results:
            if S in res.get("metrics", {}):
                mdict = res["metrics"][S]
                for m in metric_names:
                    metric_samples[m].append(mdict[m])
        # Only summarise if we have enough observations
        if len(metric_samples["eq_value"]) >= 2:
            entry = {"freq": freq}
            for m in metric_names:
                entry[m] = _summarize(metric_samples[m])
            curb_welfare[S] = entry

    return {
        "curb_set_frequencies": curb_freq,
        "minimal_curb_frequencies": minimal_freq,
        "strategy_in_any_curb": strat_in_any_curb,
        "strategy_in_minimal_curb": strat_in_minimal,
        "n_curb_sets_per_sample": {
            "mean": float(np.mean(n_curb_per_sample)),
            "std": float(np.std(n_curb_per_sample)),
        },
        "n_minimal_per_sample": {
            "mean": float(np.mean(n_minimal_per_sample)),
            "std": float(np.std(n_minimal_per_sample)),
        },
        "size_distribution": size_distribution,
        "curb_welfare": curb_welfare,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _dn(strategy):
    """Return display name for a strategy."""
    return DISPLAY_NAMES.get(strategy, strategy)


def _set_str(S_indices, strategy_names):
    """Format a frozenset of indices as a comma-separated display-name string."""
    names = sorted([strategy_names[i] for i in S_indices],
                   key=lambda s: STRATEGY_ORDER.index(s) if s in STRATEGY_ORDER else 999)
    return "{" + ", ".join(_dn(s) for s in names) + "}"


def print_curb_results(point_result, bootstrap_agg, strategy_names):
    """Print formatted CURB analysis results."""
    print("\n" + "=" * 70)
    print("CURB SET ANALYSIS")
    print("=" * 70)

    # --- Point estimate ---
    all_curb = point_result["all_curb_sets"]
    minimal = point_result["minimal_curb_sets"]
    n = len(strategy_names)

    print(f"\n--- Point Estimate ---")
    print(f"  Total CURB sets found: {len(all_curb)} (out of {2**n - 1} non-empty subsets)")

    # Group by size
    by_size = defaultdict(list)
    for S in all_curb:
        by_size[len(S)].append(S)

    minimal_set = set(minimal)

    for size in sorted(by_size.keys()):
        sets = by_size[size]
        print(f"\n  Size {size}: {len(sets)} CURB set(s)")
        for S in sets:
            tag = " [MINIMAL]" if S in minimal_set else ""
            print(f"    {_set_str(S, strategy_names)}{tag}")

    print(f"\n  Minimal CURB sets ({len(minimal)}):")
    for S in minimal:
        print(f"    {_set_str(S, strategy_names)}")

    # Point estimate welfare/fairness metrics
    point_metrics = point_result.get("metrics")
    if point_metrics:
        print(f"\n  Welfare/fairness metrics at restricted equilibrium:")
        print(f"    {'CURB set':<45s} {'EqVal':>7s} {'UW':>7s} {'NW':>7s} "
              f"{'NW+':>7s} {'EF1':>7s} {'EF1+':>7s}")
        print(f"    {'-'*45} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for S in minimal:
            if S in point_metrics:
                m = point_metrics[S]
                print(f"    {_set_str(S, strategy_names):<45s} "
                      f"{m['eq_value']:7.4f} {m['uw']:7.4f} {m['nw']:7.4f} "
                      f"{m['nw_plus']:7.4f} {m['ef1']:7.4f} {m['ef1_plus']:7.4f}")

    # --- Bootstrap ---
    if bootstrap_agg is not None:
        print(f"\n--- Bootstrap Uncertainty ({bootstrap_agg['n_samples']} samples) ---")

        stats = bootstrap_agg["n_curb_sets_per_sample"]
        print(f"\n  CURB sets per sample: {stats['mean']:.1f} +/- {stats['std']:.1f}")
        stats_min = bootstrap_agg["n_minimal_per_sample"]
        print(f"  Minimal CURB sets per sample: {stats_min['mean']:.1f} +/- {stats_min['std']:.1f}")

        print(f"\n  Most frequent CURB sets (top 15):")
        for S, freq in list(bootstrap_agg["curb_set_frequencies"].items())[:15]:
            tag = ""
            if S in bootstrap_agg["minimal_curb_frequencies"]:
                tag = f"  [minimal in {bootstrap_agg['minimal_curb_frequencies'][S]:.0%} of samples]"
            print(f"    {_set_str(S, strategy_names)}: {freq:.0%}{tag}")

        print(f"\n  Most frequent minimal CURB sets:")
        for S, freq in list(bootstrap_agg["minimal_curb_frequencies"].items())[:10]:
            print(f"    {_set_str(S, strategy_names)}: {freq:.0%}")

        print(f"\n  Strategy membership (in any CURB set):")
        for name in strategy_names:
            freq = bootstrap_agg["strategy_in_any_curb"].get(name, 0)
            print(f"    {_dn(name):20s}: {freq:.0%}")

        print(f"\n  Strategy membership (in minimal CURB sets):")
        for name in strategy_names:
            freq = bootstrap_agg["strategy_in_minimal_curb"].get(name, 0)
            print(f"    {_dn(name):20s}: {freq:.0%}")

        print(f"\n  Size distribution (mean count per sample):")
        for size in sorted(bootstrap_agg["size_distribution"].keys()):
            d = bootstrap_agg["size_distribution"][size]
            if d["mean"] > 0:
                print(f"    Size {size:2d}: {d['mean']:.1f} +/- {d['std']:.1f}")

        # Welfare/fairness metrics (bootstrap)
        curb_welfare = bootstrap_agg.get("curb_welfare", {})
        if curb_welfare:
            # Show metrics for minimal CURB sets that appear frequently
            minimal_freq = bootstrap_agg["minimal_curb_frequencies"]
            frequent_minimals = [S for S in minimal_freq if S in curb_welfare]

            if frequent_minimals:
                print(f"\n  Welfare/fairness at restricted equilibrium (minimal CURB sets):")
                print(f"    {'CURB set':<40s} {'Freq':>5s}  {'EqVal':>14s} "
                      f"{'UW':>14s} {'NW':>14s} {'NW+':>14s} "
                      f"{'EF1':>14s} {'EF1+':>14s}")
                print(f"    {'-'*40} {'-'*5}  {'-'*14} {'-'*14} {'-'*14} "
                      f"{'-'*14} {'-'*14} {'-'*14}")
                for S in frequent_minimals:
                    w = curb_welfare[S]
                    freq = w["freq"]
                    def _fmt(m):
                        return f"{m['mean']:.4f}±{m['std']:.4f}"
                    print(f"    {_set_str(S, strategy_names):<40s} {freq:5.0%}  "
                          f"{_fmt(w['eq_value']):>14s} {_fmt(w['uw']):>14s} "
                          f"{_fmt(w['nw']):>14s} {_fmt(w['nw_plus']):>14s} "
                          f"{_fmt(w['ef1']):>14s} {_fmt(w['ef1_plus']):>14s}")


def cross_reference_psie(curb_sets, strategy_names, psie_path):
    """Check if PSIE terminal supports are CURB sets.

    Args:
        curb_sets: List of frozensets (index-based) that are CURB.
        strategy_names: List of strategy names.
        psie_path: Path to psie_results.pkl.
    """
    if not psie_path.exists():
        print(f"\n  [PSIE cross-reference skipped: {psie_path} not found]")
        return

    with open(psie_path, "rb") as f:
        psie = pickle.load(f)

    point = psie.get("point_estimate", {})
    analysis = point.get("analysis", {})
    unique_supports = analysis.get("unique_supports", [])

    curb_set_of_sets = set(curb_sets)

    print(f"\n--- PSIE Cross-Reference ---")
    for support_tuple in unique_supports:
        # Convert name-based support to index-based frozenset
        indices = frozenset(
            strategy_names.index(name) for name in support_tuple
            if name in strategy_names
        )
        is_curb = indices in curb_set_of_sets
        support_str = "{" + ", ".join(_dn(s) for s in support_tuple) + "}"
        status = "YES" if is_curb else "NO"
        print(f"  PSIE terminal {support_str}: CURB? {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_brute_force(avg_payoff, n):
    """Run brute-force CURB enumeration. Returns (all_curb, minimal_curb)."""
    all_curb = find_all_curb_sets(avg_payoff, n)
    minimal_curb = find_minimal_curb_sets(all_curb)
    return all_curb, minimal_curb


def _run_closure(avg_payoff, n):
    """Run closure-based CURB algorithm. Returns (all_curb, minimal_curb)."""
    minimal_curb = find_minimal_curb_sets_via_closure(avg_payoff, n)
    all_curb = find_all_curb_sets_via_closure(avg_payoff, n)
    return all_curb, minimal_curb


def _run_curb_banzhaf(args, base_dir, input_path, curb_results_path):
    """Standalone CURB-Banzhaf mode.

    Loads existing curb_results.pkl for CURB sets and
    iterative_analysis_results.pkl for raw matrices.
    Only computes the Banzhaf marginals (v(C) - v(C\\{i})).
    """
    banzhaf_output = base_dir / "data" / "analysis" / "curb_banzhaf_results.pkl"
    matrix_keys = ("payoff",) + METRIC_KEYS

    # Load raw bootstrap data (for matrices)
    print(f"Loading raw data from {input_path}...")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    strategy_names = data["config"]["strategy_names"]
    raw_samples = data["raw"]
    n = len(strategy_names)
    print(f"  Strategies ({n}): {[_dn(s) for s in strategy_names]}")
    print(f"  Bootstrap samples available: {len(raw_samples)}")

    # Load existing CURB results (for point estimate CURB sets)
    print(f"Loading CURB sets from {curb_results_path}...")
    with open(curb_results_path, "rb") as f:
        curb_data = pickle.load(f)

    all_curb = curb_data["point_estimate"]["all_curb_sets"]
    print(f"  Point estimate CURB sets: {len(all_curb)}")

    # Build average matrices for point estimate
    avg_matrices = {}
    for key in matrix_keys:
        stacked = np.array([s["matrices"][key] for s in raw_samples])
        avg_matrices[key] = np.nanmean(stacked, axis=0)

    print(f"\n{'='*70}")
    print("CURB-SELECTED BANZHAF ATTRIBUTION")
    print(f"{'='*70}")

    # Point estimate
    print("\nPoint estimate: computing CURB-Banzhaf...")
    point_cb = compute_curb_banzhaf(
        avg_matrices, strategy_names, all_curb, include_full_game=True,
    )

    print(f"\n  Coalitions per strategy (how many CURB sets contain each):")
    for s_name in strategy_names:
        c = point_cb["counts"].get(s_name, 0)
        print(f"    {_dn(s_name):20s}: {c} CURB sets")

    print(f"\n  CURB-Banzhaf (point estimate):")
    print(f"    {'Strategy':<20s} {'UW':>10s} {'NW':>10s} {'NW+':>10s} "
          f"{'EF1':>10s} {'EF1+':>10s}")
    print(f"    {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for s_name in strategy_names:
        vals = [f"{point_cb['banzhaf'][m].get(s_name, 0):10.4f}"
                for m in CURB_BANZHAF_METRICS]
        print(f"    {_dn(s_name):<20s} {' '.join(vals)}")

    # Bootstrap (re-finds CURB sets per sample — fast — then solves marginals)
    banzhaf_bootstrap_agg = None
    if not args.no_bootstrap:
        n_bootstrap = len(raw_samples)
        if args.max_bootstrap is not None:
            n_bootstrap = min(n_bootstrap, args.max_bootstrap)

        n_workers = min(n_bootstrap, multiprocessing.cpu_count())

        print(f"\nRunning CURB-Banzhaf on {n_bootstrap} "
              f"bootstrap samples ({n_workers} workers)...")
        print("  (Re-finds CURB sets per sample, then solves only marginals)")

        banzhaf_args = [
            (
                {k: raw_samples[b_idx]["matrices"][k] for k in matrix_keys},
                n,
                strategy_names,
            )
            for b_idx in range(n_bootstrap)
        ]

        with multiprocessing.Pool(n_workers) as pool:
            banzhaf_results_list = list(tqdm(
                pool.imap_unordered(_curb_banzhaf_worker, banzhaf_args),
                total=n_bootstrap, desc="Bootstrap CURB-Banzhaf",
            ))

        print("Aggregating bootstrap CURB-Banzhaf results...")
        banzhaf_bootstrap_agg = aggregate_curb_banzhaf_results(
            banzhaf_results_list, strategy_names,
        )

        # Print bootstrap summary
        print(f"\n  CURB-Banzhaf (bootstrap mean +/- std):")
        print(f"    {'Strategy':<20s} {'UW':>18s} {'NW':>18s} "
              f"{'EF1':>18s} {'#CURB sets':>12s}")
        print(f"    {'-'*20} {'-'*18} {'-'*18} {'-'*18} {'-'*12}")
        bs = banzhaf_bootstrap_agg["banzhaf"]
        cs = banzhaf_bootstrap_agg["counts"]
        for s_name in strategy_names:
            parts = []
            for m in ("uw", "nw", "ef1"):
                s = bs[m].get(s_name, {})
                parts.append(f"{s.get('mean',0):.4f}+/-{s.get('std',0):.4f}")
            cnt = cs.get(s_name, {})
            cnt_str = f"{cnt.get('mean',0):.1f}+/-{cnt.get('std',0):.1f}"
            print(f"    {_dn(s_name):<20s} {'  '.join(parts)}  {cnt_str:>12s}")

    # Save results
    banzhaf_output.parent.mkdir(parents=True, exist_ok=True)
    banzhaf_data = {
        "config": {
            "strategy_names": strategy_names,
            "n_strategies": n,
            "input_path": str(input_path),
            "n_bootstrap": banzhaf_bootstrap_agg["n_samples"]
                if banzhaf_bootstrap_agg else 0,
        },
        "point_estimate": point_cb,
        "bootstrap": banzhaf_bootstrap_agg,
    }
    with open(banzhaf_output, "wb") as f:
        pickle.dump(banzhaf_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved CURB-Banzhaf results to {banzhaf_output}")


def main():
    parser = argparse.ArgumentParser(
        description="CURB set analysis across bootstrap samples",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to iterative_analysis_results.pkl",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save CURB results pkl",
    )
    parser.add_argument("--solver", type=str, default="mene",
                        help="Solver name (unused, for CLI consistency)")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap, only run point estimate")
    parser.add_argument("--max-bootstrap", type=int, default=None,
                        help="Cap number of bootstrap samples to use")
    parser.add_argument(
        "--method", type=str, default="brute_force",
        choices=["brute_force", "closure", "compare"],
        help="Algorithm: brute_force (enumerate all 2^n-1), "
             "closure (Benisch et al. closure, default), "
             "compare (run both and verify)",
    )
    parser.add_argument(
        "--banzhaf", action="store_true",
        help="Compute CURB-selected Banzhaf attribution "
             "(separate pass over bootstrap samples)",
    )
    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent.parent
    input_path = Path(args.input) if args.input else (
        base_dir / "data" / "analysis" / "iterative_analysis_results.pkl"
    )
    output_path = Path(args.output) if args.output else (
        base_dir / "data" / "analysis" / "curb_results.pkl"
    )
    psie_path = base_dir / "data" / "analysis" / "psie_results.pkl"

    # --- Standalone CURB-Banzhaf mode ---
    if args.banzhaf:
        _run_curb_banzhaf(args, base_dir, input_path, output_path)
        return

    # Load data
    print(f"Loading data from {input_path}...")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    strategy_names = data["config"]["strategy_names"]
    raw_samples = data["raw"]
    n = len(strategy_names)
    print(f"  Strategies ({n}): {[_dn(s) for s in strategy_names]}")
    print(f"  Bootstrap samples available: {len(raw_samples)}")
    print(f"  Non-empty subsets to check: {2**n - 1}")
    print(f"  Method: {args.method}")

    # Build average matrices (point estimate) for all metric keys
    matrix_keys = ("payoff",) + METRIC_KEYS
    print(f"\nBuilding average matrices ({', '.join(matrix_keys)})...")
    avg_matrices = {}
    for key in matrix_keys:
        stacked = np.array([s["matrices"][key] for s in raw_samples])
        avg_matrices[key] = np.nanmean(stacked, axis=0)
    avg_payoff = avg_matrices["payoff"]
    print(f"  Shape: {avg_payoff.shape}")

    # --- Point estimate ---
    if args.method == "compare":
        print(f"\n--- Method Comparison (point estimate) ---")

        t0 = time.perf_counter()
        all_curb_bf, minimal_curb_bf = _run_brute_force(avg_payoff, n)
        t_bf = time.perf_counter() - t0

        t0 = time.perf_counter()
        all_curb_cl, minimal_curb_cl = _run_closure(avg_payoff, n)
        t_cl = time.perf_counter() - t0

        # Verify identical results
        bf_minimals = set(minimal_curb_bf)
        cl_minimals = set(minimal_curb_cl)
        bf_all = set(all_curb_bf)
        cl_all = set(all_curb_cl)

        minimals_match = bf_minimals == cl_minimals
        all_match = bf_all == cl_all

        print(f"\n  {'Method':<15s} {'Time (s)':>10s} {'Minimals':>10s} {'Total':>10s}")
        print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'brute_force':<15s} {t_bf:>10.3f} {len(minimal_curb_bf):>10d} {len(all_curb_bf):>10d}")
        print(f"  {'closure':<15s} {t_cl:>10.3f} {len(minimal_curb_cl):>10d} {len(all_curb_cl):>10d}")
        print(f"\n  Minimals match: {minimals_match}")
        print(f"  All CURB sets match: {all_match}")

        if not minimals_match:
            print(f"  WARNING: Minimal CURB sets differ!")
            print(f"    brute_force only: {bf_minimals - cl_minimals}")
            print(f"    closure only:     {cl_minimals - bf_minimals}")
        if not all_match:
            print(f"  WARNING: All CURB sets differ!")
            print(f"    brute_force only: {bf_all - cl_all}")
            print(f"    closure only:     {cl_all - bf_all}")

        # Use brute_force results as canonical
        all_curb, minimal_curb = all_curb_bf, minimal_curb_bf

    elif args.method == "closure":
        print(f"\nRunning closure-based CURB algorithm on point estimate...")
        all_curb, minimal_curb = _run_closure(avg_payoff, n)

    else:  # brute_force
        print(f"\nRunning CURB enumeration on point estimate ({2**n - 1} subsets)...")
        all_curb, minimal_curb = _run_brute_force(avg_payoff, n)

    # Compute welfare/fairness metrics at restricted equilibrium (point estimate)
    print("Computing welfare/fairness metrics for point estimate CURB sets...")
    point_metrics = _compute_metrics_for_curb_sets(
        all_curb, avg_matrices, strategy_names,
    )
    print(f"  Solved {len(point_metrics)}/{len(all_curb)} CURB sets")

    point_result = {
        "all_curb_sets": all_curb,
        "minimal_curb_sets": minimal_curb,
        "metrics": point_metrics,
        "curb_by_size": defaultdict(list),
    }
    for S in all_curb:
        point_result["curb_by_size"][len(S)].append(S)
    point_result["curb_by_size"] = dict(point_result["curb_by_size"])

    # --- Bootstrap ---
    bootstrap_agg = None
    if not args.no_bootstrap:
        n_bootstrap = len(raw_samples)
        if args.max_bootstrap is not None:
            n_bootstrap = min(n_bootstrap, args.max_bootstrap)

        n_workers = min(n_bootstrap, multiprocessing.cpu_count())

        # Select worker based on method
        if args.method in ("closure", "compare"):
            worker_fn = _curb_worker_closure
            method_label = "closure"
        else:
            worker_fn = _curb_worker
            method_label = "brute_force"

        print(f"\nRunning CURB on {n_bootstrap} bootstrap samples "
              f"({n_workers} workers, method={method_label})...")

        worker_args = [
            (
                {k: raw_samples[b_idx]["matrices"][k] for k in matrix_keys},
                n,
                strategy_names,
            )
            for b_idx in range(n_bootstrap)
        ]

        with multiprocessing.Pool(n_workers) as pool:
            all_results = list(tqdm(
                pool.imap_unordered(worker_fn, worker_args),
                total=n_bootstrap, desc="Bootstrap CURB",
            ))

        print("Aggregating bootstrap results...")
        bootstrap_agg = aggregate_curb_results(all_results, strategy_names)
        bootstrap_result = {
            "analyses": all_results,
            "aggregated": bootstrap_agg,
        }
    else:
        bootstrap_result = None

    # --- Print results ---
    bootstrap_agg = bootstrap_result["aggregated"] if bootstrap_result else None
    print_curb_results(point_result, bootstrap_agg, strategy_names)

    # --- PSIE cross-reference ---
    cross_reference_psie(all_curb, strategy_names, psie_path)

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "strategy_names": strategy_names,
            "n_strategies": n,
            "input_path": str(input_path),
            "n_bootstrap": bootstrap_agg["n_samples"] if bootstrap_agg else 0,
            "method": args.method,
        },
        "point_estimate": {
            "payoff_matrix": avg_payoff,
            "all_curb_sets": all_curb,
            "minimal_curb_sets": minimal_curb,
            "curb_by_size": point_result["curb_by_size"],
            "metrics": point_metrics,
        },
        "bootstrap": bootstrap_result,
    }
    with open(output_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved results to {output_path}")



if __name__ == "__main__":
    main()
