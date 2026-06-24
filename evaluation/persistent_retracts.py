"""Persistent retract (σ-CURB) computation for symmetric 2-player games.

Persistent retracts (Kalai & Samet 1984) coincide with minimal CURB sets defined
under the *refined* best-reply correspondence σ, rather than the standard β
(Balkenborg, Hofbauer & Kuzmics 2013, *Theoretical Economics*).

For two-player games in the class 𝒢* (no own-payoff-equivalent pure strategies
that are not payoff-equivalent), BHK 2013 show:

    σ-BR(G)  =  β-BR(G')

where G' is G with the payoffs of *weakly inferior* pure strategies reduced by
a small ε. A pure strategy is weakly inferior if it is

  (i) weakly dominated by some mixture over the other strategies, or
  (ii) own-payoff-equivalent to a proper mixture over the other strategies.

This module implements (i) and (ii) as LP feasibility tests. The perturbation
and downstream β-CURB call are in a follow-up step.

References
----------
Kalai, E. and Samet, D. (1984). Persistent equilibria in strategic games.
    IJGT 13(3), 129-144.
Balkenborg, D., Hofbauer, J. and Kuzmics, C. (2013). Refined best reply
    correspondence and dynamics. Theoretical Economics 8, 165-192.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import linprog


def _other_indices(i: int, n: int) -> list[int]:
    return [j for j in range(n) if j != i]


# ---------------------------------------------------------------------------
# Self-contained β-CURB enumeration (small games only).
#
# For tests and any small-n use case where the heavy import chain in
# ``evaluation.curb_analysis`` (which transitively loads MetaGame and the
# visualization layer) is unwanted. Algorithmic content mirrors that file
# but is duplicated here so this module is stand-alone.
# ---------------------------------------------------------------------------


def _compute_cbr(payoff_matrix: np.ndarray, S_indices) -> frozenset[int]:
    """LP-based conditional best-response set CBR(S).

    Strategy i ∈ {0..n-1} is in CBR(S) iff there exists σ ∈ Δ(S) such that
    σᵀ M[i, :] ≥ σᵀ M[k, :] for all k. Equivalent (up to sign) to the
    constraint Σ_j σ_j · (M[k, j] − M[i, j]) ≤ 0 for all k ≠ i.
    """
    n = payoff_matrix.shape[0]
    S = list(S_indices)
    m = len(S)
    cbr: set[int] = set()
    for i in range(n):
        c = np.zeros(m)
        A_ub_rows = []
        for k in range(n):
            if k == i:
                continue
            row = np.array(
                [payoff_matrix[k, S[j]] - payoff_matrix[i, S[j]] for j in range(m)]
            )
            A_ub_rows.append(row)
        A_ub = np.array(A_ub_rows) if A_ub_rows else None
        b_ub = np.zeros(len(A_ub_rows)) if A_ub_rows else None
        A_eq = np.ones((1, m))
        b_eq = np.array([1.0])
        bounds = [(0.0, None)] * m
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res.success:
            cbr.add(i)
    return frozenset(cbr)


def _brute_force_minimal_beta_curb(
    payoff_matrix: np.ndarray,
    n_strategies: int,
) -> list[frozenset[int]]:
    """Enumerate all 2^n − 1 non-empty subsets, return the minimal β-CURB sets.

    Intended for small games (n ≲ 12). For larger games use
    ``evaluation.curb_analysis.find_minimal_curb_sets_klimm_weibull``.
    """
    curb_sets = []
    for size in range(1, n_strategies + 1):
        for subset in combinations(range(n_strategies), size):
            S = frozenset(subset)
            if _compute_cbr(payoff_matrix, S).issubset(S):
                curb_sets.append(S)
    curb_set_lookup = set(curb_sets)
    minimal = []
    for S in curb_sets:
        if not any((T < S) for T in curb_set_lookup):
            minimal.append(S)
    return minimal


def is_weakly_dominated(
    payoff_matrix: np.ndarray,
    i: int,
    tol: float = 1e-9,
) -> bool:
    """Test whether pure strategy i is weakly dominated by a mixture over j ≠ i.

    Strategy i is weakly dominated if there exists σ ∈ Δ(S \\ {i}) such that
    σᵀ M[:, k] ≥ M[i, k] for all opponent pure strategies k, with strict
    inequality for at least one k.

    Tested via the LP

        maximize  Σ_j σ_j · R_j   where R_j = Σ_k M[j, k]
        subject to  Σ_j σ_j · M[j, k] ≥ M[i, k]   ∀ k
                    Σ_j σ_j = 1, σ_j ≥ 0

    over j ∈ S \\ {i}. If the optimum is > Σ_k M[i, k] + tol, then there is a
    σ achieving strict improvement somewhere while remaining ≥ on every k,
    i.e., i is weakly dominated.

    Parameters
    ----------
    payoff_matrix : (n, n) array
        Row player's payoffs in a symmetric 2-player game.
    i : int
        Index of the candidate strategy.
    tol : float
        Numerical tolerance for the strict-improvement test.

    Returns
    -------
    bool : True if strategy i is weakly dominated.
    """
    n = payoff_matrix.shape[0]
    if n < 2:
        return False
    others = _other_indices(i, n)
    m = len(others)

    # Objective: maximize Σ_j σ_j · R_j, so minimize -Σ_j σ_j · R_j.
    row_sums = payoff_matrix[others, :].sum(axis=1)
    c = -row_sums

    # Inequality constraints: σᵀ M[:, k] ≥ M[i, k] for all k.
    # linprog uses A_ub @ x ≤ b_ub, so flip sign:
    #     -Σ_j σ_j · M[j, k] ≤ -M[i, k]
    A_ub = -payoff_matrix[others, :].T  # shape (n, m)
    b_ub = -payoff_matrix[i, :]         # shape (n,)

    A_eq = np.ones((1, m))
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * m

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    if not result.success:
        return False

    achieved_sum = -result.fun  # since we minimized -Σ
    own_sum = payoff_matrix[i, :].sum()
    return achieved_sum > own_sum + tol


def is_own_payoff_equivalent_to_mixture(
    payoff_matrix: np.ndarray,
    i: int,
    tol: float = 1e-9,
    support_eps: float = 1e-6,
    require_proper: bool = True,
) -> bool:
    """Test whether pure strategy i is own-payoff-equivalent to a mixture.

    Strategy i is own-payoff-equivalent to σ ∈ Δ(S \\ {i}) for the player
    using it if σᵀ M[:, k] = M[i, k] for every opponent pure strategy k. In
    other words, the row player's payoff from playing i equals the row
    player's payoff from playing σ, against every column strategy.

    BHK 2013 defines weak inferiority in terms of *proper* mixtures
    (support ≥ 2). The default ``require_proper=True`` enforces this by
    constraining σ_j ≤ 1 − support_eps for every j: combined with Σσ = 1,
    this forces at least two components to be positive. With
    ``require_proper=False`` (legacy behaviour) the LP also accepts
    degenerate σ = e_j, which flags pairs of own-payoff-equivalent *pure*
    strategies as "weakly inferior" — strictly outside BHK 2013's class 𝒢*.

    Tested as an LP feasibility problem with equality constraints

        Σ_j σ_j · M[j, k] = M[i, k]   ∀ k
        Σ_j σ_j = 1
        0 ≤ σ_j ≤ 1 − support_eps   (when require_proper=True)

    over j ∈ S \\ {i}.

    Parameters
    ----------
    payoff_matrix : (n, n) array
        Row player's payoffs in a symmetric 2-player game.
    i : int
        Index of the candidate strategy.
    tol : float
        Numerical tolerance — currently unused; scipy's HiGHS handles
        residuals. Kept for signature parity with the dominance test.
    support_eps : float
        Minimum mass that must reside outside any single component, when
        ``require_proper=True``. Equivalent to forcing every σ_j ≤
        1 − support_eps; with Σσ = 1 this guarantees support size ≥ 2.
    require_proper : bool
        If True (default), require a proper mixture per BHK 2013. If
        False, allow degenerate single-component mixtures.

    Returns
    -------
    bool : True if strategy i is own-payoff-equivalent to a (proper) mixture
        over the other strategies.
    """
    n = payoff_matrix.shape[0]
    if n < 2:
        return False
    others = _other_indices(i, n)
    m = len(others)

    # Pure feasibility: dummy zero objective.
    c = np.zeros(m)

    # Equality constraints:
    #   Σ_j σ_j · M[j, k] = M[i, k]   for each k
    #   Σ_j σ_j = 1
    A_eq_match = payoff_matrix[others, :].T   # shape (n, m)
    b_eq_match = payoff_matrix[i, :]           # shape (n,)
    A_eq = np.vstack([A_eq_match, np.ones((1, m))])
    b_eq = np.concatenate([b_eq_match, np.array([1.0])])

    if require_proper:
        # No single σ_j may carry the full mass; with Σσ = 1 this implies
        # support size ≥ 2.
        upper = 1.0 - support_eps
        bounds = [(0.0, upper)] * m
    else:
        bounds = [(0.0, None)] * m

    _ = tol
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return bool(result.success)


def perturb_for_sigma_curb(
    payoff_matrix: np.ndarray,
    weakly_inferior: frozenset[int],
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Construct the perturbed game G' whose β-CURB sets are G's σ-CURB sets.

    For a symmetric 2-player game G with row-player payoff matrix M, reducing
    the row-player's payoff for playing strategy i means M[i, :] -= ε. By the
    role symmetry of G (col-player payoff matrix is Mᵀ), the same operation
    also reduces the col-player's payoff for playing i: (Mᵀ)[:, i] = M[i, :].
    So a single subtraction M[i, :] -= ε for each weakly inferior i
    implements both the row and column perturbations of BHK 2013.

    The result M' is not coordinate-symmetric (M'[i, k] ≠ M'[k, i] in
    general), but the game G' = (M', M'ᵀ) remains role-symmetric, which is
    what the downstream β-CURB algorithm assumes.

    Parameters
    ----------
    payoff_matrix : (n, n) array
        Row player's payoffs in the original symmetric game G.
    weakly_inferior : iterable of int
        Indices of weakly inferior pure strategies in G (see
        :func:`find_weakly_inferior`).
    epsilon : float
        Perturbation magnitude. Must be smaller than any payoff gap that
        matters for the CURB structure — defaults to 1e-6, which is safe
        for payoffs of order 1–100.

    Returns
    -------
    (n, n) array : the perturbed row-player matrix M'.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive; got {epsilon}")
    M_prime = payoff_matrix.astype(float, copy=True)
    for i in weakly_inferior:
        M_prime[i, :] -= epsilon
    return M_prime


def find_minimal_persistent_retracts(
    payoff_matrix: np.ndarray,
    n_strategies: int | None = None,
    epsilon: float = 1e-6,
    tol: float = 1e-9,
    curb_finder=None,
    support_eps: float = 1e-6,
    require_proper: bool = True,
) -> list[frozenset[int]]:
    """Compute the minimal persistent retracts (= minimal σ-CURB sets) of G.

    Implements the 2-player perturbation route of Balkenborg, Hofbauer &
    Kuzmics (2013): identify the weakly inferior pure strategies, reduce
    their payoffs by a small ε, and run the standard β-CURB algorithm on
    the perturbed game. The resulting minimal β-CURB sets of G' coincide
    with the minimal σ-CURB sets of G, which are exactly the persistent
    retracts (Kalai & Samet 1984).

    Parameters
    ----------
    payoff_matrix : (n, n) array
        Row player's payoffs in the symmetric 2-player game.
    n_strategies : int, optional
        Number of strategies. Defaults to ``payoff_matrix.shape[0]``.
    epsilon : float
        Perturbation magnitude. See :func:`perturb_for_sigma_curb`.
    tol : float
        Numerical tolerance for the weakly-inferior identification.
    curb_finder : callable, optional
        β-CURB function with signature ``(payoff_matrix, n_strategies) ->
        list[frozenset[int]]``. Defaults to
        ``find_minimal_curb_sets_klimm_weibull``.

    Returns
    -------
    list[frozenset[int]] : minimal persistent retracts of G.

    Notes
    -----
    BHK 2013's result requires G to lie in the class 𝒢*, i.e., have no
    own-payoff-equivalent pure strategies that are *not* themselves payoff-
    equivalent (the appendix argues this is essentially WLOG). Games with
    near-equivalent strategies (e.g., MAPPO ≈ PPO in the bargaining domain)
    sit at the edge of 𝒢*; the LP-based identification in
    :func:`is_own_payoff_equivalent_to_mixture` flags them, after which the
    perturbation breaks the equivalence numerically.
    """
    if n_strategies is None:
        n_strategies = payoff_matrix.shape[0]
    if curb_finder is None:
        # Local import to avoid a circular dependency at module load time
        # (curb_analysis pulls in MetaGame and the visualization layer).
        from evaluation.curb_analysis import find_minimal_curb_sets_klimm_weibull
        curb_finder = find_minimal_curb_sets_klimm_weibull

    weakly_inferior = find_weakly_inferior(
        payoff_matrix, tol=tol,
        support_eps=support_eps, require_proper=require_proper,
    )
    M_prime = perturb_for_sigma_curb(payoff_matrix, weakly_inferior, epsilon=epsilon)
    return curb_finder(M_prime, n_strategies)


def find_weakly_inferior(
    payoff_matrix: np.ndarray,
    tol: float = 1e-9,
    support_eps: float = 1e-6,
    require_proper: bool = True,
) -> frozenset[int]:
    """Return the set of pure strategies that are weakly inferior.

    A pure strategy is *weakly inferior* (BHK 2013) if it is either weakly
    dominated by a mixture over the other strategies, or own-payoff-
    equivalent to a *proper* mixture (support ≥ 2) of them. These are the
    strategies whose payoffs we will perturb to construct the modified
    game G' whose β-CURB sets equal the σ-CURB sets (persistent retracts)
    of the original game G.

    Parameters
    ----------
    payoff_matrix : (n, n) array
        Row player's payoffs in a symmetric 2-player game.
    tol : float
        Numerical tolerance for the dominance test.
    support_eps : float
        Minimum mass off any single component for the own-payoff
        equivalence test (passed to
        :func:`is_own_payoff_equivalent_to_mixture`).
    require_proper : bool
        If True (default), the own-payoff equivalence test requires a
        proper mixture. See
        :func:`is_own_payoff_equivalent_to_mixture`.

    Returns
    -------
    frozenset of int : indices of weakly inferior pure strategies.
    """
    n = payoff_matrix.shape[0]
    weakly_inferior = set()
    for i in range(n):
        if is_weakly_dominated(payoff_matrix, i, tol=tol):
            weakly_inferior.add(i)
            continue
        if is_own_payoff_equivalent_to_mixture(
            payoff_matrix, i, tol=tol,
            support_eps=support_eps, require_proper=require_proper,
        ):
            weakly_inferior.add(i)
    return frozenset(weakly_inferior)
