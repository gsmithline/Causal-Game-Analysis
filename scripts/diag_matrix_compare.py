"""Diagnostic: reconcile the average-game MENE (PPO/PSRO) with the
bootstrap-mean equilibrium (LLM-dominated).

Builds the deterministic-mean payoff matrix two ways:
  (A) my construction: 0.5*(raw_payoff_p1 + raw_payoff_p2.T)
  (B) build_matrices_fast with a no-op "bootstrap" (indices = arange)
Solves MENE on each and reports the equilibrium + matrix diff.
"""

from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# --- load MENESolver without the heavy package __init__ ---
pkg = types.ModuleType("iga_pkg"); pkg.__path__ = [str(ROOT / "src/iterative_game_analysis")]
sys.modules["iga_pkg"] = pkg
_us = importlib.util.spec_from_file_location("iga_pkg.utils", ROOT / "src/iterative_game_analysis/utils.py")
_u = importlib.util.module_from_spec(_us); sys.modules["iga_pkg.utils"] = _u; _us.loader.exec_module(_u)
sp = types.ModuleType("iga_pkg.solvers"); sp.__path__ = [str(ROOT / "src/iterative_game_analysis/solvers")]
sys.modules["iga_pkg.solvers"] = sp
_bs = importlib.util.spec_from_file_location("iga_pkg.solvers.base", ROOT / "src/iterative_game_analysis/solvers/base.py")
_b = importlib.util.module_from_spec(_bs); sys.modules["iga_pkg.solvers.base"] = _b; _bs.loader.exec_module(_b)
_src = (ROOT / "src/iterative_game_analysis/solvers/mene.py").read_text() \
    .replace("from .base import register_solver", "from iga_pkg.solvers.base import register_solver") \
    .replace("from ..utils import simplex_projection", "from iga_pkg.utils import simplex_projection")
_m = types.ModuleType("iga_pkg.solvers.mene"); exec(compile(_src, "mene.py", "exec"), _m.__dict__)
MENESolver = _m.MENESolver

# --- stub heavy deps for the data loader ---
for name in ["polarix", "jax", "jax.numpy"]:
    sys.modules[name] = types.ModuleType(name)
sys.modules["jax"].numpy = sys.modules["jax.numpy"]
_srcpkg = types.ModuleType("src"); _srcpkg.__path__ = [str(ROOT / "src")]; sys.modules["src"] = _srcpkg
_iga = types.ModuleType("src.iterative_game_analysis"); _iga.__path__ = [str(ROOT / "src/iterative_game_analysis")]
sys.modules["src.iterative_game_analysis"] = _iga
_mg = types.ModuleType("src.iterative_game_analysis.metagame"); _mg.MetaGame = type("MetaGame", (), {})
sys.modules["src.iterative_game_analysis.metagame"] = _mg
_iu = types.ModuleType("src.iterative_game_analysis.utils"); _iu.compute_regret = lambda *a, **k: (0, 0, 0)
sys.modules["src.iterative_game_analysis.utils"] = _iu

from evaluation.original_paper_analysis import load_and_preprocess_data, build_matrices_fast  # noqa: E402

STRATS = ["walk", "tough", "nfsp", "mappo", "soft", "ppo", "psro",
          "ef1_bargainer", "openai_5.2_none", "openai_5.2_low",
          "openai_5.4_low", "openai_5.4_medium", "openai_5.2_medium"]


class IdentityRNG:
    """rng.choice(n, size=n, replace=True) -> arange(n) (deterministic mean)."""
    def choice(self, n, size=None, replace=True):
        return np.arange(n)


def build_mine(grouped, names):
    n = len(names); idx = {p: i for i, p in enumerate(names)}
    rp1 = np.zeros((n, n)); rp2 = np.zeros((n, n))
    for (pi, pj), data in grouped.items():
        if pi not in idx or pj not in idx or data["n_games"] == 0:
            continue
        i, j = idx[pi], idx[pj]
        rp1[i, j] = data["raw_payoff_i"].mean()
        rp2[i, j] = data["raw_payoff_j"].mean()
    return 0.5 * (rp1 + rp2.T)


def show(sigma, names, tag):
    print(f"\n{tag}:")
    for k, v in sorted(zip(names, sigma), key=lambda x: -x[1]):
        if v > 1e-3:
            print(f"   {k:22s} {v*100:6.2f}%")


def main():
    grouped = load_and_preprocess_data(ROOT / "data" / "crossplay", STRATS)
    solver = MENESolver(discrete_factors=100)

    M_mine = build_mine(grouped, STRATS)
    mats = build_matrices_fast(grouped, STRATS, IdentityRNG(), raw_utility=True)
    M_fast = np.asarray(mats["raw_payoff"], dtype=float)

    print(f"matrix max abs diff (mine vs fast): {np.abs(M_mine - M_fast).max():.4f}")
    print(f"mine[ppo,psro]={M_mine[5,6]:.3f}  fast[ppo,psro]={M_fast[5,6]:.3f}")
    print(f"are they symmetric? mine:{np.allclose(M_mine,M_mine.T)}  fast:{np.allclose(M_fast,M_fast.T)}")

    show(solver.solve(M_mine), STRATS, "MENE on MY average matrix")
    show(solver.solve(M_fast), STRATS, "MENE on build_matrices_fast average matrix")


if __name__ == "__main__":
    main()
