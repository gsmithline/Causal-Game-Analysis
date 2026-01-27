"""Analysis functions for the three levels of causal meta-game analysis."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from iterative_game_analysis.metagame import MetaGame
from iterative_game_analysis.utils import l1_norm

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Level 1: Interaction-Level Analysis (No Re-Equilibration)
# =============================================================================


def baseline_value(
    metagame: MetaGame, policy: str, sigma: NDArray[np.floating]
) -> float:
    """Compute baseline equilibrium interaction value for a policy.

    U_B(π_i) = Σ_π σ_B(π) · μ(π_i, π)

    This is the expected outcome for policy π_i when facing a "typical"
    partner drawn from the baseline equilibrium mixture σ_B.

    Args:
        metagame: The meta-game (baseline library B).
        policy: The incumbent policy π_i.
        sigma: Equilibrium mixture σ_B over baseline policies.

    Returns:
        Expected baseline value U_B(π_i).
    """
    return metagame.expected_value(policy, sigma)


def partner_lift(
    metagame: MetaGame,
    incumbent: str,
    candidate: str,
    sigma: NDArray[np.floating],
) -> float:
    """Compute Partner Lift for an incumbent facing a candidate.

    PL_1(π_i; π_j | B) = μ(π_i, π_j) - U_B(π_i)

    Measures: if incumbent π_i faces candidate π_j instead of a typical
    equilibrium partner, what is the expected change in outcome?

    Args:
        metagame: Meta-game containing both incumbent and candidate.
        incumbent: The incumbent policy π_i (in baseline B).
        candidate: The candidate policy π_j (may be outside B).
        sigma: Equilibrium mixture σ_B over baseline (for U_B computation).

    Returns:
        Partner lift value.
    """
    # μ(π_i, π_j) - direct pairwise outcome
    pairwise = metagame.pairwise_payoff(incumbent, candidate)

    # U_B(π_i) - baseline value (only over baseline policies in sigma)
    baseline = baseline_value(metagame, incumbent, sigma)

    return pairwise - baseline


def level1_analysis(
    metagame: MetaGame,
    baseline_policies: list[str],
    candidate: str,
    solver: str = "mene",
) -> dict:
    """Compute all Level 1 metrics for a candidate against a baseline.

    Level 1 measures the direct interaction effect of candidate π_j without
    letting the ecosystem adapt (no re-equilibration).

    Args:
        metagame: Full meta-game containing baseline and candidate.
        baseline_policies: List of policies in baseline library B.
        candidate: The candidate policy π_j to evaluate.
        solver: Solver for computing baseline equilibrium.

    Returns:
        Dict containing:
            - sigma_B: Baseline equilibrium mixture
            - per_incumbent: Dict of partner lift per incumbent {π_i: PL_1}
            - uniform_avg: Uniform average of partner lifts
            - equilibrium_avg: Equilibrium-weighted average
            - min: Worst-case partner lift
            - max: Best-case partner lift
    """
    # Get baseline sub-game and solve for equilibrium
    baseline_game = metagame.subset(baseline_policies)
    sigma_B = baseline_game.solve(solver)

    # Create mapping from baseline policies to their equilibrium weights
    sigma_dict = {p: sigma_B[i] for i, p in enumerate(baseline_policies)}

    # Compute partner lift for each incumbent
    per_incumbent = {}
    for incumbent in baseline_policies:
        # Need to expand sigma to full metagame indices for baseline_value
        sigma_full = np.zeros(metagame.n_policies)
        for p, weight in sigma_dict.items():
            sigma_full[metagame.policy_index(p)] = weight

        pl = partner_lift(metagame, incumbent, candidate, sigma_full)
        per_incumbent[incumbent] = pl

    lifts = list(per_incumbent.values())

    # Aggregations
    uniform_avg = float(np.mean(lifts))
    equilibrium_avg = sum(sigma_dict[p] * per_incumbent[p] for p in baseline_policies)

    return {
        "sigma_B": sigma_B,
        "per_incumbent": per_incumbent,
        "uniform_avg": uniform_avg,
        "equilibrium_avg": equilibrium_avg,
        "min": float(np.min(lifts)),
        "max": float(np.max(lifts)),
    }


# =============================================================================
# Level 2: Ecosystem-Level Analysis (Re-Equilibration)
# =============================================================================


def ecosystem_value(
    metagame: MetaGame,
    sigma: NDArray[np.floating],
    welfare_fn: str = "utilitarian",
) -> float:
    """Compute ecosystem value W(S) for a library with equilibrium mixture.

    W(S) = Y_eco(σ_S, M_S)

    Args:
        metagame: Meta-game for library S.
        sigma: Equilibrium mixture σ_S.
        welfare_fn: Welfare function ("utilitarian", "nash", "egalitarian").

    Returns:
        Ecosystem welfare value.
    """
    return metagame.welfare(sigma, welfare_fn)


def ecosystem_lift(
    metagame: MetaGame,
    baseline_policies: list[str],
    candidate: str,
    solver: str = "mene",
    welfare_fn: str = "utilitarian",
) -> dict:
    """Compute ecosystem lift from adding a candidate to baseline.

    Δ_eco(π_j | B) = W(B⁺) - W(B)

    where B⁺ = B ∪ {π_j}

    Level 2 measures the ecosystem effect of making π_j available to the
    system and allowing strategic adaptation (re-equilibration).

    Args:
        metagame: Full meta-game containing baseline and candidate.
        baseline_policies: List of policies in baseline library B.
        candidate: The candidate policy π_j.
        solver: Solver for computing equilibria.
        welfare_fn: Welfare function for ecosystem value.

    Returns:
        Dict containing:
            - delta_eco: Ecosystem lift Δ_eco
            - W_B: Baseline ecosystem value W(B)
            - W_B_plus: Extended ecosystem value W(B⁺)
            - sigma_B: Baseline equilibrium
            - sigma_B_plus: Extended equilibrium (full, including candidate)
            - entry_mass: σ_{B⁺}(π_j) - equilibrium weight of candidate
            - equilibrium_shift: ||σ_{B⁺} - σ_B||_1 (restricted to B)
            - incumbent_shifts: Per-incumbent value shift Δ_inc(π_i; π_j | B)
    """
    # Baseline game B
    baseline_game = metagame.subset(baseline_policies)
    sigma_B = baseline_game.solve(solver)
    W_B = ecosystem_value(baseline_game, sigma_B, welfare_fn)

    # Extended game B⁺ = B ∪ {π_j}
    extended_policies = baseline_policies + [candidate]
    extended_game = metagame.subset(extended_policies)
    sigma_B_plus = extended_game.solve(solver)
    W_B_plus = ecosystem_value(extended_game, sigma_B_plus, welfare_fn)

    # Entry mass: equilibrium weight of candidate in B⁺
    candidate_idx = extended_game.policy_index(candidate)
    entry_mass = float(sigma_B_plus[candidate_idx])

    # Equilibrium shift: L1 distance restricted to baseline policies
    sigma_B_plus_restricted = sigma_B_plus[:-1]  # All except candidate
    equilibrium_shift = l1_norm(sigma_B_plus_restricted, sigma_B)

    # Incumbent value shifts: Δ_inc(π_i; π_j | B) = V_{B⁺}(π_i) - V_B(π_i)
    incumbent_shifts = {}
    for i, policy in enumerate(baseline_policies):
        V_B = baseline_game.expected_value(policy, sigma_B)
        # For V_{B⁺}, need to use extended game
        V_B_plus = extended_game.expected_value(policy, sigma_B_plus)
        incumbent_shifts[policy] = V_B_plus - V_B

    return {
        "delta_eco": W_B_plus - W_B,
        "W_B": W_B,
        "W_B_plus": W_B_plus,
        "sigma_B": sigma_B,
        "sigma_B_plus": sigma_B_plus,
        "entry_mass": entry_mass,
        "equilibrium_shift": equilibrium_shift,
        "incumbent_shifts": incumbent_shifts,
    }


# =============================================================================
# Level 3: Ecosystem Attribution (Shapley/Banzhaf)
# =============================================================================


def shapley_value(
    policies: list[str],
    value_fn: Callable[[list[str]], float],
    n_samples: int | None = None,
) -> dict[str, float]:
    """Compute Shapley values for ecosystem attribution.

    φ(π) = E_{uniform ≺}[v(Pred_≺(π) ∪ {π}) - v(Pred_≺(π))]

    Assigns synergy-aware credit for ecosystem outcomes.

    Args:
        policies: List of all policies.
        value_fn: Function that takes a list of policies and returns
            ecosystem value W(S) for that sub-library.
        n_samples: Number of permutation samples for Monte Carlo approximation.
            If None, computes exact Shapley values (exponential in |policies|).

    Returns:
        Dict mapping each policy to its Shapley value.
    """
    n = len(policies)

    if n_samples is None and n <= 10:
        # Exact computation for small games
        return _shapley_exact(policies, value_fn)
    else:
        # Monte Carlo approximation
        n_samples = n_samples or 1000
        return _shapley_monte_carlo(policies, value_fn, n_samples)


def _shapley_exact(
    policies: list[str], value_fn: Callable[[list[str]], float]
) -> dict[str, float]:
    """Exact Shapley value computation."""
    n = len(policies)
    shapley = {p: 0.0 for p in policies}

    # Iterate over all permutations
    for perm in itertools.permutations(policies):
        current_coalition: list[str] = []
        prev_value = value_fn([])  # Empty coalition value

        for policy in perm:
            current_coalition.append(policy)
            curr_value = value_fn(current_coalition)
            shapley[policy] += curr_value - prev_value
            prev_value = curr_value

    # Average over all permutations
    n_perms = math.factorial(n)
    return {p: v / n_perms for p, v in shapley.items()}


def _shapley_monte_carlo(
    policies: list[str],
    value_fn: Callable[[list[str]], float],
    n_samples: int,
) -> dict[str, float]:
    """Monte Carlo approximation of Shapley values."""
    rng = np.random.default_rng()
    shapley = {p: 0.0 for p in policies}

    for _ in range(n_samples):
        # Random permutation
        perm = list(rng.permutation(policies))
        current_coalition: list[str] = []
        prev_value = value_fn([])

        for policy in perm:
            current_coalition.append(policy)
            curr_value = value_fn(current_coalition)
            shapley[policy] += curr_value - prev_value
            prev_value = curr_value

    return {p: v / n_samples for p, v in shapley.items()}


def make_ecosystem_value_fn(
    metagame: MetaGame,
    solver: str = "mene",
    welfare_fn: str = "utilitarian",
) -> Callable[[list[str]], float]:
    """Create a value function for Shapley/Banzhaf computation.

    This ensures the equilibrium σ_S is computed from M_S and applied to M_S,
    as required for correct Level 3 analysis.

    Args:
        metagame: The full meta-game.
        solver: Solver for computing equilibria.
        welfare_fn: Welfare function for ecosystem value.

    Returns:
        A function v(S) that computes σ_S^T M_S σ_S for any coalition S.
    """

    def value_fn(policies: list[str]) -> float:
        if len(policies) == 0:
            return 0.0
        sub_game = metagame.subset(policies)
        sigma_S = sub_game.solve(solver)
        return sub_game.welfare(sigma_S, welfare_fn)

    return value_fn


def banzhaf_value(
    policies: list[str],
    value_fn: Callable[[list[str]], float],
    n_samples: int | None = None,
) -> dict[str, float]:
    """Compute Banzhaf values for ecosystem attribution.

    β(π) = E_{S ⊆ Π\\{π}}[v(S ∪ {π}) - v(S)]

    Unlike Shapley, Banzhaf considers all coalitions with equal weight.

    Args:
        policies: List of all policies.
        value_fn: Function that returns ecosystem value for a sub-library.
        n_samples: Number of samples for Monte Carlo. If None, exact computation.

    Returns:
        Dict mapping each policy to its Banzhaf value.
    """
    n = len(policies)

    if n_samples is None and n <= 15:
        return _banzhaf_exact(policies, value_fn)
    else:
        n_samples = n_samples or 1000
        return _banzhaf_monte_carlo(policies, value_fn, n_samples)


def _banzhaf_exact(
    policies: list[str], value_fn: Callable[[list[str]], float]
) -> dict[str, float]:
    """Exact Banzhaf value computation."""
    n = len(policies)
    banzhaf = {p: 0.0 for p in policies}

    for policy in policies:
        others = [p for p in policies if p != policy]
        # Iterate over all subsets of others
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                S = list(subset)
                S_with_policy = S + [policy]
                marginal = value_fn(S_with_policy) - value_fn(S)
                banzhaf[policy] += marginal

        # Average over 2^(n-1) coalitions
        banzhaf[policy] /= 2 ** (n - 1)

    return banzhaf


def _banzhaf_monte_carlo(
    policies: list[str],
    value_fn: Callable[[list[str]], float],
    n_samples: int,
) -> dict[str, float]:
    """Monte Carlo approximation of Banzhaf values."""
    rng = np.random.default_rng()
    banzhaf = {p: 0.0 for p in policies}

    for policy in policies:
        others = [p for p in policies if p != policy]
        for _ in range(n_samples):
            # Random subset of others
            mask = rng.random(len(others)) < 0.5
            S = [others[i] for i in range(len(others)) if mask[i]]
            S_with_policy = S + [policy]
            marginal = value_fn(S_with_policy) - value_fn(S)
            banzhaf[policy] += marginal

        banzhaf[policy] /= n_samples

    return banzhaf


def level3_analysis(
    metagame: MetaGame,
    policies: list[str] | None = None,
    solver: str = "mene",
    welfare_fn: str = "utilitarian",
    method: str = "shapley",
    n_samples: int | None = None,
) -> dict:
    """Compute Level 3 ecosystem attribution for all policies.

    Level 3 assigns credit for ecosystem outcomes using cooperative game theory,
    accounting for synergies between policies.

    Args:
        metagame: The meta-game to analyze.
        policies: Policies to include (defaults to all in metagame).
        solver: Solver for computing equilibria.
        welfare_fn: Welfare function for ecosystem value.
        method: Attribution method ("shapley" or "banzhaf").
        n_samples: Number of samples for Monte Carlo (None for exact).

    Returns:
        Dict containing:
            - attributions: Dict mapping each policy to its attribution value
            - total_value: W(Π) - ecosystem value of full library
            - method: Attribution method used
            - efficiency_gap: Difference between sum of attributions and total value
                (should be ~0 for Shapley, may differ for Banzhaf)
    """
    if policies is None:
        policies = metagame.policies

    # Create properly-structured value function
    value_fn = make_ecosystem_value_fn(metagame, solver, welfare_fn)

    # Compute total ecosystem value
    total_value = value_fn(policies)

    # Compute attributions
    if method == "shapley":
        attributions = shapley_value(policies, value_fn, n_samples)
    elif method == "banzhaf":
        attributions = banzhaf_value(policies, value_fn, n_samples)
    else:
        raise ValueError(f"Unknown attribution method: {method}")

    # Check efficiency (Shapley should sum to total value)
    attribution_sum = sum(attributions.values())
    efficiency_gap = total_value - attribution_sum

    return {
        "attributions": attributions,
        "total_value": total_value,
        "method": method,
        "efficiency_gap": efficiency_gap,
    }


# =============================================================================
# EF1 Frequency (Fairness Metric for Bargaining)
# =============================================================================


def ef1_frequency(
    df: pd.DataFrame,
    policy_i_col: str = "policy_i",
    policy_j_col: str = "policy_j",
    ef1_col: str = "ef1",
) -> pd.DataFrame:
    """Compute EF1 frequency matrix between policy groups.

    EF1 (Envy-Free up to 1 item) measures fairness in allocation/bargaining.
    The frequency is computed as the fraction of interactions where
    the allocation satisfied EF1.

    Args:
        df: DataFrame with columns for policies and EF1 indicator.
        policy_i_col: Column for row policy.
        policy_j_col: Column for column policy.
        ef1_col: Column with binary EF1 indicator (1 if EF1 satisfied).

    Returns:
        DataFrame with EF1 frequency for each (policy_i, policy_j) pair.
    """
    grouped = df.groupby([policy_i_col, policy_j_col])[ef1_col].agg(["mean", "count"])
    grouped.columns = ["ef1_frequency", "n_samples"]
    return grouped.reset_index()


def ef1_frequency_matrix(
    df: pd.DataFrame,
    policies: list[str] | None = None,
    policy_i_col: str = "policy_i",
    policy_j_col: str = "policy_j",
    ef1_col: str = "ef1",
) -> tuple[NDArray[np.floating], list[str]]:
    """Compute EF1 frequency as a matrix.

    Args:
        df: DataFrame with EF1 indicators.
        policies: List of policies (inferred if None).
        policy_i_col: Column for row policy.
        policy_j_col: Column for column policy.
        ef1_col: Column with binary EF1 indicator.

    Returns:
        Tuple of (EF1 frequency matrix, list of policies).
    """
    if policies is None:
        all_policies = set(df[policy_i_col].unique()) | set(df[policy_j_col].unique())
        policies = sorted(all_policies)

    n = len(policies)
    policy_to_idx = {p: i for i, p in enumerate(policies)}

    freq_matrix = np.full((n, n), np.nan)
    grouped = df.groupby([policy_i_col, policy_j_col])[ef1_col].mean()

    for (pi, pj), freq in grouped.items():
        if pi in policy_to_idx and pj in policy_to_idx:
            i, j = policy_to_idx[pi], policy_to_idx[pj]
            freq_matrix[i, j] = freq

    return freq_matrix, policies


def aggregate_ef1_between_groups(
    df: pd.DataFrame,
    group_a: list[str],
    group_b: list[str],
    policy_i_col: str = "policy_i",
    policy_j_col: str = "policy_j",
    ef1_col: str = "ef1",
) -> dict[str, float]:
    """Compute aggregate EF1 frequency between two policy groups.

    Useful for comparing fairness between different LLM families or
    agent types in bargaining experiments.

    Args:
        df: DataFrame with EF1 indicators.
        group_a: First group of policies.
        group_b: Second group of policies.
        policy_i_col: Column for row policy.
        policy_j_col: Column for column policy.
        ef1_col: Column with binary EF1 indicator.

    Returns:
        Dict with:
            - a_vs_b: EF1 frequency when group A policies are row
            - b_vs_a: EF1 frequency when group B policies are row
            - overall: Overall EF1 frequency between groups
            - within_a: EF1 frequency within group A
            - within_b: EF1 frequency within group B
    """
    results = {}

    # A vs B (A is row policy)
    mask_a_vs_b = df[policy_i_col].isin(group_a) & df[policy_j_col].isin(group_b)
    if mask_a_vs_b.any():
        results["a_vs_b"] = float(df.loc[mask_a_vs_b, ef1_col].mean())
    else:
        results["a_vs_b"] = np.nan

    # B vs A (B is row policy)
    mask_b_vs_a = df[policy_i_col].isin(group_b) & df[policy_j_col].isin(group_a)
    if mask_b_vs_a.any():
        results["b_vs_a"] = float(df.loc[mask_b_vs_a, ef1_col].mean())
    else:
        results["b_vs_a"] = np.nan

    # Overall between groups
    mask_between = mask_a_vs_b | mask_b_vs_a
    if mask_between.any():
        results["overall"] = float(df.loc[mask_between, ef1_col].mean())
    else:
        results["overall"] = np.nan

    # Within group A
    mask_within_a = df[policy_i_col].isin(group_a) & df[policy_j_col].isin(group_a)
    if mask_within_a.any():
        results["within_a"] = float(df.loc[mask_within_a, ef1_col].mean())
    else:
        results["within_a"] = np.nan

    # Within group B
    mask_within_b = df[policy_i_col].isin(group_b) & df[policy_j_col].isin(group_b)
    if mask_within_b.any():
        results["within_b"] = float(df.loc[mask_within_b, ef1_col].mean())
    else:
        results["within_b"] = np.nan

    return results
