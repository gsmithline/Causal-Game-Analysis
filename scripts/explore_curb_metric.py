"""Explore the CURB coverage ratio ρ across domains.

For each domain (bargaining, iterated PD, Czarnecki spinning-top) compute:

  * |min CURB| / |S| under the standard β best-reply correspondence (ρ_β)
  * |min CURB| / |S| under the refined σ correspondence (ρ_σ, persistent
    retract), via the BHK 2013 perturbation route
  * the number of weakly inferior pure strategies (LP-identified)
  * a cheap cyclicity proxy ‖M − Mᵀ‖_F / ‖M‖_F for side-by-side
    comparison with Bertrand et al. 2023's disc decomposition

The cyclicity proxy is the Frobenius norm of the antisymmetric component
relative to the matrix; Bertrand's λ_disc / λ_transitive ratio is a more
faithful invariant but requires fitting their disc model. We compute the
proxy here and leave the disc-fit for a follow-up.

Usage
-----
    uv run python scripts/explore_curb_metric.py
    uv run python scripts/explore_curb_metric.py --max-n 30  # cap game size
    uv run python scripts/explore_curb_metric.py --domain bargaining
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.curb_analysis import find_minimal_curb_sets_klimm_weibull
from evaluation.persistent_retracts import (
    find_minimal_persistent_retracts,
    find_weakly_inferior,
)


def cyclicity_frobenius(M: np.ndarray) -> float:
    """‖M − Mᵀ‖_F / ‖M‖_F. Zero for symmetric M, larger for antisymmetric."""
    num = np.linalg.norm(M - M.T, ord="fro")
    den = np.linalg.norm(M, ord="fro")
    return float(num / den) if den > 0 else 0.0


def bertrand_normal_decomposition(M: np.ndarray, tol_rel: float = 1e-6) -> dict:
    """Bertrand 2023 Theorem 1 (normal decomposition) magnitudes via SVD.

    For a skew-symmetric matrix A, Theorem 1 gives A = Σ_l λ_l (u_l v_lᵀ −
    v_l u_lᵀ) with orthogonal pairs and λ_1 ≥ … ≥ λ_⌊n/2⌋. The pair
    magnitudes λ_l are exactly the SVD singular values of A, which come in
    duplicate pairs (σ_{2j−1} = σ_{2j} = λ_j) for skew-symmetric inputs.

    We apply this to the antisymmetric component A = (M − Mᵀ) / 2, so the
    measure is defined on any matrix (including non-zero-sum payoffs, where
    A captures the cyclic part). For zero-sum games where M is already
    skew-symmetric, A = M.

    Returns
    -------
    dict with keys:
        lambda_1: largest pair magnitude (dominant — Bertrand's "main"
            disc component).
        lambda_2: second pair magnitude.
        lambda_ratio: λ_2 / λ_1 (close to 0 ⇒ single-disc / near-pure
            transitive-or-cyclic; close to 1 ⇒ mixed cyclic structure
            beyond the dominant pair).
        top_pair_fraction: λ_1² / Σ_l λ_l² (variance explained by
            dominant pair).
        n_significant_pairs: count of pairs with λ_l > tol_rel * λ_1.
    """
    A = (M - M.T) / 2.0
    s = np.linalg.svd(A, compute_uv=False)
    # Pair magnitudes: every other singular value of the skew-symmetric A.
    pairs = s[::2]
    if pairs.size == 0 or pairs[0] == 0:
        return {
            "lambda_1": 0.0,
            "lambda_2": 0.0,
            "lambda_ratio": 0.0,
            "top_pair_fraction": 1.0,
            "n_significant_pairs": 0,
        }
    lam1 = float(pairs[0])
    lam2 = float(pairs[1]) if pairs.size >= 2 else 0.0
    n_sig = int(np.sum(pairs > tol_rel * lam1))
    sq_sum = float(np.sum(pairs ** 2))
    return {
        "lambda_1": lam1,
        "lambda_2": lam2,
        "lambda_ratio": (lam2 / lam1) if lam1 > 0 else 0.0,
        "top_pair_fraction": (lam1 ** 2 / sq_sum) if sq_sum > 0 else 1.0,
        "n_significant_pairs": n_sig,
    }


def compute_rho_pair(
    M: np.ndarray,
    name: str,
    epsilon: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """Compute ρ_β, ρ_σ, |W|, cyclicity for one game matrix."""
    n = M.shape[0]
    t0 = time.perf_counter()
    beta_min = find_minimal_curb_sets_klimm_weibull(M, n_strategies=n)
    t_beta = time.perf_counter() - t0

    t1 = time.perf_counter()
    W = find_weakly_inferior(M)
    sigma_min = find_minimal_persistent_retracts(M, n_strategies=n, epsilon=epsilon)
    t_sigma = time.perf_counter() - t1

    rho_beta = max((len(C) for C in beta_min), default=0) / n
    rho_sigma = max((len(C) for C in sigma_min), default=0) / n
    cyc = cyclicity_frobenius(M)
    nd = bertrand_normal_decomposition(M)

    result = {
        "name": name,
        "n": n,
        "rho_beta": rho_beta,
        "rho_sigma": rho_sigma,
        "n_weakly_inferior": len(W),
        "frobenius_cyclicity": cyc,
        "lambda_1": nd["lambda_1"],
        "lambda_2": nd["lambda_2"],
        "lambda_ratio": nd["lambda_ratio"],
        "top_pair_fraction": nd["top_pair_fraction"],
        "n_significant_pairs": nd["n_significant_pairs"],
        "n_min_curb_beta": len(beta_min),
        "n_min_curb_sigma": len(sigma_min),
        "t_beta_seconds": t_beta,
        "t_sigma_seconds": t_sigma,
    }
    if verbose:
        print(
            f"  {name:35s}  n={n:4d}  ρ_β={rho_beta:.3f}  ρ_σ={rho_sigma:.3f}  "
            f"|W|={len(W):3d}  cyc={cyc:.3f}  "
            f"λ₂/λ₁={nd['lambda_ratio']:.3f}  top%={nd['top_pair_fraction']:.3f}  "
            f"t_β={t_beta:.1f}s  t_σ={t_sigma:.1f}s"
        )
    return result


# ---------------------------------------------------------------------------
# Domain-specific loaders
# ---------------------------------------------------------------------------


def load_bargaining() -> list[tuple[str, np.ndarray]]:
    """Return [(label, payoff_matrix)] for the bargaining domain.

    The point-estimate payoff matrix used by the existing CURB pipeline
    lives in ``data/analysis/curb_results.pkl`` under
    ``point_estimate.payoff_matrix`` (10×10). The standalone
    ``data/bargaining_matrices.pkl`` is an empty stub.
    """
    path = ROOT / "data" / "analysis" / "curb_results.pkl"
    with open(path, "rb") as f:
        cr = pickle.load(f)
    M = np.asarray(cr["point_estimate"]["payoff_matrix"], dtype=float)
    return [("bargaining/avg_payoff", M)]


def load_pd() -> list[tuple[str, np.ndarray]]:
    """Return [(label, payoff_matrix)] for the iterated PD domain.

    Uses the tournament pickle's `payoff_matrix` (30 strategies).
    """
    path = ROOT / "notebooks" / "pd_data" / "pd_tournament.pkl"
    with open(path, "rb") as f:
        pdt = pickle.load(f)
    M = np.asarray(pdt["payoff_matrix"], dtype=float)
    return [("pd/tournament_payoff", M)]


def load_spinning_top(max_n: int | None = 50) -> list[tuple[str, np.ndarray]]:
    """Return [(label, payoff_matrix)] for spinning-top games up to max_n.

    The Czarnecki et al. 2020 pickle stores *payoff* matrices already in
    skew-symmetric form (P + Pᵀ = 0, range [−1, 1], zero diagonal). No
    transformation is needed; the matrices feed directly into CURB
    analysis as row-player payoffs.
    """
    path = ROOT / "data" / "spinning_top_payoffs.pkl"
    with open(path, "rb") as f:
        st = pickle.load(f)
    games = []
    for game, M in st.items():
        M = np.asarray(M, dtype=float)
        if max_n is not None and M.shape[0] > max_n:
            continue
        games.append((f"spinning_top/{game}", M))
    games.sort(key=lambda kv: kv[1].shape[0])
    return games


DOMAINS = {
    "bargaining": load_bargaining,
    "pd": load_pd,
    "spinning_top": load_spinning_top,
}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=list(DOMAINS.keys()) + ["all"],
        default="all",
        help="Which domain(s) to evaluate.",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=50,
        help="Skip spinning-top games larger than this (Klimm-Weibull "
        "scales poorly past ~50 strategies).",
    )
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    selected = [args.domain] if args.domain != "all" else list(DOMAINS.keys())

    all_results: list[dict] = []
    for domain in selected:
        print(f"\n=== {domain} ===")
        if domain == "spinning_top":
            games = DOMAINS[domain](max_n=args.max_n)
        else:
            games = DOMAINS[domain]()
        for label, M in games:
            try:
                res = compute_rho_pair(M, label, epsilon=args.epsilon)
                all_results.append(res)
            except Exception as exc:
                print(f"  {label}: FAILED ({exc.__class__.__name__}: {exc})")

    # Summary table
    print("\n\n=== Summary ===")
    print(
        f"{'game':35s}  {'n':>4s}  {'ρ_β':>5s}  {'ρ_σ':>5s}  "
        f"{'Δρ':>5s}  {'|W|':>4s}  {'cyc':>5s}"
    )
    for r in all_results:
        delta = r["rho_beta"] - r["rho_sigma"]
        print(
            f"{r['name']:35s}  {r['n']:>4d}  {r['rho_beta']:>5.3f}  "
            f"{r['rho_sigma']:>5.3f}  {delta:>5.3f}  "
            f"{r['n_weakly_inferior']:>4d}  {r['frobenius_cyclicity']:>5.3f}"
        )

    # Save for downstream plotting
    out_path = ROOT / "data" / "analysis" / "curb_metric_exploration.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved {len(all_results)} game results to {out_path}")


if __name__ == "__main__":
    main()
