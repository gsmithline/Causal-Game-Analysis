"""Build an iterated Prisoner's Dilemma payoff + cooperation dataset.

Runs an Axelrod (https://axelrod.readthedocs.io) tournament with a
configurable strategy pool, and outputs per-rep payoff and cooperation
matrices suitable for the causal-metagame pipeline (β-CURB, σ-CURB,
ρ diagnostic, Bertrand-style spectral measure).

Output pickle structure matches the legacy ``notebooks/pd_data/pd_tournament.pkl``:

    strategy_names : list[str]               # length n
    n_strategies   : int                     # n
    turns          : int
    repetitions    : int
    noise          : float
    payoff_matrix  : ndarray (n, n)          # row-player mean per-turn payoff
    coop_matrix    : ndarray (n, n)          # mean co-cooperation rate
    nw_matrix      : ndarray (n, n)          # social welfare = u_i + u_j
    payoff_reps    : ndarray (R, n, n)       # per-repetition payoffs (for bootstrap)
    coop_reps      : ndarray (R, n, n) | None  # only if --per-rep-coop is set
    pool           : str                     # "short" / "all" / explicit list

Usage
-----
    uv run python scripts/build_pd_dataset.py
    uv run python scripts/build_pd_dataset.py --max-n 30 --reps 20 --out pd_small.pkl
    uv run python scripts/build_pd_dataset.py --pool all --turns 200 --reps 50
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import axelrod as axl
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def select_players(pool: str, max_n: int | None = None):
    if pool == "short":
        strategies = axl.short_run_time_strategies
    elif pool == "all":
        strategies = axl.strategies
    else:
        names = {s.strip() for s in pool.split(",")}
        strategies = [s for s in axl.strategies if s.name in names]
    players = [s() for s in strategies]
    if max_n is not None:
        players = players[:max_n]
    return players


def run_tournament(
    players,
    turns: int,
    repetitions: int,
    noise: float,
    seed: int,
):
    t = axl.Tournament(
        players,
        turns=turns,
        repetitions=repetitions,
        noise=noise,
        seed=seed,
    )
    result = t.play(progress_bar=False)

    payoff_matrix = np.asarray(result.payoff_matrix, dtype=float)        # (n, n)
    coop_matrix = np.asarray(result.normalised_cooperation, dtype=float)  # (n, n)
    payoffs_rep = np.asarray(result.payoffs, dtype=float)                 # (n, n, R)

    # Move repetition axis to front for consistency with legacy pickle.
    payoff_reps = np.transpose(payoffs_rep, (2, 0, 1))                    # (R, n, n)

    nw_matrix = payoff_matrix + payoff_matrix.T                           # (n, n)

    return {
        "payoff_matrix": payoff_matrix,
        "coop_matrix": coop_matrix,
        "nw_matrix": nw_matrix,
        "payoff_reps": payoff_reps,
    }


def maybe_compute_curb_diagnostics(payoff_matrix: np.ndarray):
    """Print ρ_β, ρ_σ, |W| and Bertrand λ for a quick sanity check.

    Imported lazily so the script remains useful even if the σ-CURB / CURB
    modules are missing.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from evaluation.curb_analysis import find_minimal_curb_sets_klimm_weibull
        from evaluation.persistent_retracts import (
            find_minimal_persistent_retracts,
            find_weakly_inferior,
        )
    except Exception as exc:
        print(f"[diagnostics skipped: {exc}]")
        return

    n = payoff_matrix.shape[0]
    print(f"\n  Diagnostics on payoff_matrix ({n}×{n}):")
    t0 = time.perf_counter()
    beta_min = find_minimal_curb_sets_klimm_weibull(payoff_matrix, n_strategies=n)
    t_beta = time.perf_counter() - t0
    t1 = time.perf_counter()
    W = find_weakly_inferior(payoff_matrix)
    sigma_min = find_minimal_persistent_retracts(payoff_matrix, n_strategies=n)
    t_sigma = time.perf_counter() - t1

    rho_b = max((len(C) for C in beta_min), default=0) / n
    rho_s = max((len(C) for C in sigma_min), default=0) / n

    A = (payoff_matrix - payoff_matrix.T) / 2
    pairs = np.linalg.svd(A, compute_uv=False)[::2]
    top_pct = (pairs[0] ** 2 / np.sum(pairs ** 2)) if pairs[0] > 0 else 1.0
    lam_ratio = (pairs[1] / pairs[0]) if pairs.size >= 2 and pairs[0] > 0 else 0.0

    print(f"    ρ_β = {rho_b:.3f}  ({t_beta:.1f}s)")
    print(f"    ρ_σ = {rho_s:.3f}  ({t_sigma:.1f}s)")
    print(f"    |W| = {len(W)}")
    print(f"    Bertrand λ_2/λ_1 = {lam_ratio:.3f}  top% = {top_pct:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool", default="short",
        help='"short" (default), "all", or a comma-separated list of strategy names.',
    )
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--turns", type=int, default=200)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", default=None,
        help="Output pickle path. Defaults to notebooks/pd_data/pd_tournament_axelrod.pkl.",
    )
    parser.add_argument(
        "--no-diagnostics", action="store_true",
        help="Skip CURB / ρ / Bertrand diagnostics on the resulting payoff matrix.",
    )
    args = parser.parse_args()

    players = select_players(args.pool, args.max_n)
    n = len(players)
    print(
        f"PD tournament: pool={args.pool!r}, n={n}, turns={args.turns}, "
        f"reps={args.reps}, noise={args.noise}, seed={args.seed}"
    )

    t0 = time.perf_counter()
    res = run_tournament(
        players, turns=args.turns, repetitions=args.reps,
        noise=args.noise, seed=args.seed,
    )
    print(f"  tournament: {time.perf_counter() - t0:.1f}s")

    out_path = (
        Path(args.out) if args.out is not None
        else ROOT / "notebooks" / "pd_data" / "pd_tournament_axelrod.pkl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "strategy_names": [p.name for p in players],
        "n_strategies": n,
        "turns": args.turns,
        "repetitions": args.reps,
        "noise": args.noise,
        "pool": args.pool,
        **res,
    }
    with open(out_path, "wb") as f:
        pickle.dump(record, f)
    print(f"  saved → {out_path}")
    print(f"  payoff_matrix  : {res['payoff_matrix'].shape}  range [{res['payoff_matrix'].min():.2f}, {res['payoff_matrix'].max():.2f}]")
    print(f"  coop_matrix    : {res['coop_matrix'].shape}  range [{res['coop_matrix'].min():.2f}, {res['coop_matrix'].max():.2f}]")
    print(f"  payoff_reps    : {res['payoff_reps'].shape}")

    if not args.no_diagnostics:
        maybe_compute_curb_diagnostics(res["payoff_matrix"])


if __name__ == "__main__":
    main()
