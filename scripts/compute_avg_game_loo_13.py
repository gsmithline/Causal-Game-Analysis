"""Compute LOO ΔUW on the 13-strategy bargaining average game (MENE solver).

Difference vs compute_avg_game_loo.py: loads raw crossplay data and builds
the average payoff/UW matrices over ALL rollouts (not a single bootstrap).
Matches the strategy set used in the README worked example.

Output: scripts/avg_game_loo_13.json
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import types
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Bypass package __init__ (heavy import chain).
pkg = types.ModuleType("iga_pkg")
pkg.__path__ = [str(ROOT / "src" / "iterative_game_analysis")]
sys.modules["iga_pkg"] = pkg

_utils_spec = importlib.util.spec_from_file_location(
    "iga_pkg.utils", ROOT / "src" / "iterative_game_analysis" / "utils.py",
)
_utils = importlib.util.module_from_spec(_utils_spec)
sys.modules["iga_pkg.utils"] = _utils
_utils_spec.loader.exec_module(_utils)

solvers_pkg = types.ModuleType("iga_pkg.solvers")
solvers_pkg.__path__ = [str(ROOT / "src" / "iterative_game_analysis" / "solvers")]
sys.modules["iga_pkg.solvers"] = solvers_pkg

_base_spec = importlib.util.spec_from_file_location(
    "iga_pkg.solvers.base", ROOT / "src" / "iterative_game_analysis" / "solvers" / "base.py",
)
_base = importlib.util.module_from_spec(_base_spec)
sys.modules["iga_pkg.solvers.base"] = _base
_base_spec.loader.exec_module(_base)

_mene_src = (ROOT / "src/iterative_game_analysis/solvers/mene.py").read_text() \
    .replace("from .base import register_solver", "from iga_pkg.solvers.base import register_solver") \
    .replace("from ..utils import simplex_projection", "from iga_pkg.utils import simplex_projection")
mene_mod = types.ModuleType("iga_pkg.solvers.mene")
exec(compile(_mene_src, str(ROOT / "src/iterative_game_analysis/solvers/mene.py"), "exec"),
     mene_mod.__dict__)
MENESolver = mene_mod.MENESolver

# Stub polarix + jax so the data-loading module imports cleanly.
sys.modules["polarix"] = types.ModuleType("polarix")
_jax = types.ModuleType("jax")
_jax_np = types.ModuleType("jax.numpy")
_jax.numpy = _jax_np
sys.modules["jax"] = _jax
sys.modules["jax.numpy"] = _jax_np
# Also stub the metagame + utils imports the data module pulls.
_metagame = types.ModuleType("src.iterative_game_analysis.metagame")
class _MetaGameStub:  # noqa: D401
    pass
_metagame.MetaGame = _MetaGameStub
sys.modules["src.iterative_game_analysis.metagame"] = _metagame
_src_pkg = types.ModuleType("src")
_src_pkg.__path__ = [str(ROOT / "src")]
sys.modules["src"] = _src_pkg
_iga_pkg = types.ModuleType("src.iterative_game_analysis")
_iga_pkg.__path__ = [str(ROOT / "src" / "iterative_game_analysis")]
sys.modules["src.iterative_game_analysis"] = _iga_pkg
_iga_utils = types.ModuleType("src.iterative_game_analysis.utils")
_iga_utils.compute_regret = lambda *a, **kw: 0.0
sys.modules["src.iterative_game_analysis.utils"] = _iga_utils

# Use the project's data loader.
from evaluation.original_paper_analysis import load_and_preprocess_data  # noqa: E402

STRATS = [
    "walk", "tough", "nfsp", "mappo", "soft", "ppo", "psro",
    "ef1_bargainer", "openai_5.2_none", "openai_5.2_low",
    "openai_5.4_low", "openai_5.4_medium", "openai_5.2_medium",
]
CROSSPLAY = ROOT / "data" / "crossplay"
OUT = ROOT / "scripts" / "avg_game_loo_13.json"


def build_average_matrices(grouped, names):
    """Mean payoff/UW matrix over ALL rollouts per pair (no bootstrap)."""
    n = len(names)
    idx = {p: i for i, p in enumerate(names)}
    payoff_p1 = np.zeros((n, n))
    payoff_p2 = np.zeros((n, n))
    raw_payoff_p1 = np.zeros((n, n))
    raw_payoff_p2 = np.zeros((n, n))
    uw = np.zeros((n, n))
    for (pi, pj), data in grouped.items():
        if pi not in idx or pj not in idx:
            continue
        i, j = idx[pi], idx[pj]
        if data["n_games"] == 0:
            continue
        payoff_p1[i, j] = data["payoff_i"].mean()
        payoff_p2[i, j] = data["payoff_j"].mean()
        raw_payoff_p1[i, j] = data["raw_payoff_i"].mean()
        raw_payoff_p2[i, j] = data["raw_payoff_j"].mean()
        uw[i, j] = (data["raw_payoff_i"] + data["raw_payoff_j"]).mean()

    # Symmetrize: M[i,j] = mean of (i as P1 vs j) and (i as P2 vs j) over both rollouts.
    # i's payoff as P2 in the j_p1_vs_i matchup is raw_payoff_p2[j, i].
    sym = 0.5 * (raw_payoff_p1 + raw_payoff_p2.T)
    uw_sym = 0.5 * (uw + uw.T)
    return {"payoff": sym, "uw": uw_sym}


def uw_value(sigma: np.ndarray, M_uw: np.ndarray) -> float:
    """Symmetric UW at mixture σ on UW matrix."""
    return float(sigma @ M_uw @ sigma)


def main():
    print(f"loading {len(STRATS)}-strategy crossplay data from {CROSSPLAY}…")
    t0 = time.perf_counter()
    grouped = load_and_preprocess_data(CROSSPLAY, STRATS)
    print(f"  loaded {len(grouped)} pairs in {time.perf_counter()-t0:.1f}s")

    matrices = build_average_matrices(grouped, STRATS)
    payoff = matrices["payoff"]
    uw = matrices["uw"]
    n = payoff.shape[0]
    print(f"  payoff shape {payoff.shape}; payoff[ppo,psro]={payoff[5,6]:.3f}")

    solver = MENESolver(discrete_factors=100)

    sigma_full = solver.solve(payoff)
    uw_full = uw_value(sigma_full, uw)
    print(f"\nFull game: UW = {uw_full:.3f}")
    for k, v in sorted(zip(STRATS, sigma_full), key=lambda x: -x[1]):
        if v > 1e-3:
            print(f"  {k:25s} {v*100:6.2f}%")

    results = []
    for i, name in enumerate(STRATS):
        keep = [j for j in range(n) if j != i]
        payoff_loo = payoff[np.ix_(keep, keep)]
        uw_loo = uw[np.ix_(keep, keep)]
        names_loo = [STRATS[j] for j in keep]
        try:
            sigma_loo = solver.solve(payoff_loo)
            uw_loo_val = uw_value(sigma_loo, uw_loo)
            d = uw_full - uw_loo_val
            top = [(k, v) for k, v in zip(names_loo, sigma_loo) if v > 1e-3]
            top.sort(key=lambda x: -x[1])
            top_s = ", ".join(f"{k}={v*100:.0f}%" for k, v in top[:5])
            print(f"\nremove {name:25s} UW={uw_loo_val:7.3f}  ΔUW={d:+7.3f}")
            print(f"   post-NE: {top_s}")
        except Exception as e:
            uw_loo_val = None
            d = None
            print(f"\nremove {name:25s} FAILED: {e}")
        results.append({
            "strategy": name,
            "full_support": float(sigma_full[i]),
            "uw_full": uw_full,
            "uw_loo": uw_loo_val,
            "d_uw": d,
        })

    out = {
        "strategy_names": STRATS,
        "n": n,
        "uw_full": uw_full,
        "full_support": [float(x) for x in sigma_full],
        "loo": results,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {OUT}")


if __name__ == "__main__":
    main()
