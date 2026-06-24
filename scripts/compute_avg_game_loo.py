"""Compute LOO ΔUW on the bargaining average game (MENE solver).

Loads the precomputed average payoff matrix from data/analysis/curb_results.pkl,
solves MENE on the full game and on every leave-one-out subgame, and dumps the
results to scripts/avg_game_loo.json for the figure to consume.

UW (utilitarian welfare) on a symmetric game with payoff M and equilibrium σ:
    UW(σ) = σᵀ M σ + σᵀ Mᵀ σ
LOO effect:
    ΔUW(s_i) = UW_full − UW_LOO_i  (positive → presence helps welfare)

Usage:
    uv run --with cvxpy --with numpy --with scipy --with tqdm \\
        python scripts/compute_avg_game_loo.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Bypass package __init__ to avoid heavy import chain (causal_analysis → polarix).
import importlib.util  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_utils = _load(
    "iga_utils", ROOT / "src" / "iterative_game_analysis" / "utils.py",
)
# Stub out the package + solvers subpackage so .base and ..utils resolve.
import types  # noqa: E402

pkg = types.ModuleType("iga_pkg")
pkg.__path__ = [str(ROOT / "src" / "iterative_game_analysis")]
sys.modules["iga_pkg"] = pkg
sys.modules["iga_pkg.utils"] = _utils

solvers_pkg = types.ModuleType("iga_pkg.solvers")
solvers_pkg.__path__ = [str(ROOT / "src" / "iterative_game_analysis" / "solvers")]
sys.modules["iga_pkg.solvers"] = solvers_pkg

_base = _load(
    "iga_pkg.solvers.base",
    ROOT / "src" / "iterative_game_analysis" / "solvers" / "base.py",
)
_mene_src = (ROOT / "src" / "iterative_game_analysis" / "solvers" / "mene.py").read_text()
# Patch the relative imports to point at our stub package.
_mene_src = _mene_src.replace(
    "from .base import register_solver", "from iga_pkg.solvers.base import register_solver"
).replace(
    "from ..utils import simplex_projection", "from iga_pkg.utils import simplex_projection"
)
mene_mod = types.ModuleType("iga_pkg.solvers.mene")
exec(compile(_mene_src, str(ROOT / "src/iterative_game_analysis/solvers/mene.py"), "exec"),
     mene_mod.__dict__)
MENESolver = mene_mod.MENESolver  # noqa: E402

CURB_PKL = ROOT / "data" / "analysis" / "curb_results.pkl"
OUT_JSON = ROOT / "scripts" / "avg_game_loo.json"


def uw(sigma: np.ndarray, M: np.ndarray) -> float:
    """Symmetric utilitarian welfare at mixture σ on payoff matrix M."""
    return float(sigma @ M @ sigma + sigma @ M.T @ sigma)


def main():
    print(f"loading {CURB_PKL}…")
    with open(CURB_PKL, "rb") as f:
        d = pickle.load(f)

    # Find the point-estimate payoff matrix + strategy names
    M = np.asarray(d["point_estimate"]["payoff_matrix"], dtype=float)
    names = list(d["config"]["strategy_names"])
    n = M.shape[0]
    assert len(names) == n, f"names {len(names)} != matrix {n}"
    print(f"  payoff_matrix: {M.shape}, strategies: {n}")

    solver = MENESolver(discrete_factors=100)

    t0 = time.perf_counter()
    sigma_full = solver.solve(M)
    uw_full = uw(sigma_full, M)
    print(f"\nFull game ({time.perf_counter() - t0:.1f}s):  UW = {uw_full:.3f}")
    print("  support:")
    for k, v in sorted(zip(names, sigma_full), key=lambda x: -x[1]):
        if v > 1e-3:
            print(f"    {k:25s} {v*100:6.2f}%")

    results = []
    for i, name in enumerate(names):
        keep = [j for j in range(n) if j != i]
        M_loo = M[np.ix_(keep, keep)]
        names_loo = [names[j] for j in keep]

        t1 = time.perf_counter()
        try:
            sigma_loo = solver.solve(M_loo)
            uw_loo = uw(sigma_loo, M_loo)
            d_uw = uw_full - uw_loo  # +ve = presence helps welfare
            t_solve = time.perf_counter() - t1
            print(f"\n  remove {name:25s} UW_loo = {uw_loo:7.3f}  ΔUW = {d_uw:+7.3f}  ({t_solve:.1f}s)")
            top = sorted(zip(names_loo, sigma_loo), key=lambda x: -x[1])[:5]
            for k, v in top:
                if v > 1e-3:
                    print(f"      {k:25s} {v*100:6.2f}%")
        except Exception as e:
            d_uw = None
            uw_loo = None
            print(f"  remove {name:25s} FAILED: {e}")

        results.append({
            "strategy": name,
            "full_support": float(sigma_full[i]),
            "uw_full": uw_full,
            "uw_loo": uw_loo,
            "d_uw": d_uw,
        })

    print(f"\nTotal: {time.perf_counter() - t0:.1f}s")

    out = {
        "strategy_names": names,
        "n": n,
        "uw_full": uw_full,
        "full_support": [float(x) for x in sigma_full],
        "loo": results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"saved → {OUT_JSON}")


if __name__ == "__main__":
    main()
