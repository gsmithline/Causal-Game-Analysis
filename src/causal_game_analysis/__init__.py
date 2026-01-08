"""Causal Meta-Game Analysis Framework.

A framework for analyzing multi-agent systems using Structural Causal Models (SCM)
and do-calculus, with three levels of analysis:

- Level 1: Interaction-level (no re-equilibration)
- Level 2: Ecosystem-level (re-equilibration)
- Level 3: Attribution (Shapley/Banzhaf values)

Example usage:

    import pandas as pd
    from causal_game_analysis import MetaGame, Bootstrap, level1_analysis, ecosystem_lift

    # Load cross-play data
    df = pd.read_csv("crossplay_results.csv")

    # Build meta-game from raw data
    game = MetaGame.from_dataframe(df)

    # Level 1: Partner lift analysis
    result = level1_analysis(game, baseline=["policy_a", "policy_b"], candidate="policy_c")

    # Level 2: Ecosystem lift with re-equilibration
    eco_result = ecosystem_lift(game, baseline=["policy_a", "policy_b"], candidate="policy_c")

    # Bootstrap for uncertainty quantification
    bootstrap = Bootstrap(df, n_samples=1000)
    bootstrap_results = bootstrap.run(lambda g: g.solve("mene"))
"""

__version__ = "0.1.0"

from causal_game_analysis.metagame import MetaGame
from causal_game_analysis.bootstrap import Bootstrap
from causal_game_analysis.analysis import (
    # Level 1
    baseline_value,
    partner_lift,
    level1_analysis,
    # Level 2
    ecosystem_value,
    ecosystem_lift,
    # Level 3
    shapley_value,
    banzhaf_value,
    # EF1 (bargaining fairness)
    ef1_frequency,
    ef1_frequency_matrix,
    aggregate_ef1_between_groups,
)
from causal_game_analysis.solvers import Solver, get_solver, MENESolver

__all__ = [
    # Core classes
    "MetaGame",
    "Bootstrap",
    # Solvers
    "Solver",
    "get_solver",
    "MENESolver",
    # Level 1 analysis
    "baseline_value",
    "partner_lift",
    "level1_analysis",
    # Level 2 analysis
    "ecosystem_value",
    "ecosystem_lift",
    # Level 3 analysis
    "shapley_value",
    "banzhaf_value",
    # EF1 fairness
    "ef1_frequency",
    "ef1_frequency_matrix",
    "aggregate_ef1_between_groups",
]
