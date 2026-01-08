"""MetaGame class for empirical game-theoretic analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from causal_game_analysis.solvers import get_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MetaGame:
    """Empirical meta-game representation.

    A meta-game is constructed from cross-play outcomes between policies.
    The payoff matrix M[i,j] represents the expected outcome for policy i
    when paired with policy j.

    Attributes:
        policies: List of policy names.
        payoff_matrix: Square matrix of expected payoffs.
        n_policies: Number of policies in the game.
    """

    def __init__(
        self,
        policies: list[str],
        payoff_matrix: NDArray[np.floating],
        counts_matrix: NDArray[np.integer] | None = None,
    ):
        """Initialize a MetaGame.

        Args:
            policies: List of policy names (in order matching matrix indices).
            payoff_matrix: Square matrix where M[i,j] is expected payoff for
                policy i when paired with policy j.
            counts_matrix: Optional matrix of sample counts per pair.
        """
        self.policies = list(policies)
        self.payoff_matrix = np.asarray(payoff_matrix, dtype=np.float64)
        self.counts_matrix = counts_matrix
        self.n_policies = len(policies)

        if self.payoff_matrix.shape != (self.n_policies, self.n_policies):
            raise ValueError(
                f"Payoff matrix shape {self.payoff_matrix.shape} doesn't match "
                f"number of policies ({self.n_policies})"
            )

        # Create policy name to index mapping
        self._policy_to_idx = {p: i for i, p in enumerate(policies)}

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        policy_i_col: str = "policy_i",
        policy_j_col: str = "policy_j",
        outcome_col: str = "outcome",
        policies: list[str] | None = None,
    ) -> MetaGame:
        """Build a MetaGame from raw cross-play data.

        Args:
            df: DataFrame with cross-play results.
            policy_i_col: Column name for row policy (the policy being evaluated).
            policy_j_col: Column name for column policy (the opponent/partner).
            outcome_col: Column name for the outcome value.
            policies: Optional explicit list of policies. If None, inferred from data.

        Returns:
            MetaGame instance.
        """
        if policies is None:
            # Infer policies from unique values in both columns
            all_policies = set(df[policy_i_col].unique()) | set(df[policy_j_col].unique())
            policies = sorted(all_policies)

        n = len(policies)
        policy_to_idx = {p: i for i, p in enumerate(policies)}

        # Aggregate outcomes by (policy_i, policy_j) pairs
        grouped = df.groupby([policy_i_col, policy_j_col])[outcome_col].agg(["mean", "count"])

        payoff_matrix = np.full((n, n), np.nan)
        counts_matrix = np.zeros((n, n), dtype=np.int64)

        for (pi, pj), row in grouped.iterrows():
            if pi in policy_to_idx and pj in policy_to_idx:
                i, j = policy_to_idx[pi], policy_to_idx[pj]
                payoff_matrix[i, j] = row["mean"]
                counts_matrix[i, j] = row["count"]

        return cls(policies, payoff_matrix, counts_matrix)

    def policy_index(self, policy: str) -> int:
        """Get the index of a policy by name."""
        if policy not in self._policy_to_idx:
            raise ValueError(f"Unknown policy: {policy}")
        return self._policy_to_idx[policy]

    def pairwise_payoff(self, policy_i: str, policy_j: str) -> float:
        """Get the expected payoff μ(π_i, π_j).

        Args:
            policy_i: Row policy (being evaluated).
            policy_j: Column policy (opponent/partner).

        Returns:
            Expected payoff for policy_i when paired with policy_j.
        """
        i = self.policy_index(policy_i)
        j = self.policy_index(policy_j)
        return float(self.payoff_matrix[i, j])

    def subset(self, policies: list[str]) -> MetaGame:
        """Create a sub-game restricted to given policies.

        Args:
            policies: List of policy names to include.

        Returns:
            New MetaGame with only the specified policies.
        """
        indices = [self.policy_index(p) for p in policies]
        sub_matrix = self.payoff_matrix[np.ix_(indices, indices)]

        counts = None
        if self.counts_matrix is not None:
            counts = self.counts_matrix[np.ix_(indices, indices)]

        return MetaGame(policies, sub_matrix, counts)

    def solve(self, solver: str = "mene") -> NDArray[np.floating]:
        """Compute equilibrium mixture over policies.

        Args:
            solver: Name of solver to use ("mene", "uniform").

        Returns:
            Equilibrium strategy (probability distribution over policies).
        """
        solver_instance = get_solver(solver)
        return solver_instance.solve(self.payoff_matrix)

    def expected_value(
        self, policy: str, opponent_mixture: NDArray[np.floating]
    ) -> float:
        """Compute expected payoff for a policy against an opponent mixture.

        V(π_i) = Σ_j σ(π_j) * μ(π_i, π_j)

        Args:
            policy: The policy to evaluate.
            opponent_mixture: Probability distribution over opponent policies.

        Returns:
            Expected payoff.
        """
        i = self.policy_index(policy)
        return float(self.payoff_matrix[i] @ opponent_mixture)

    def welfare(
        self, mixture: NDArray[np.floating], welfare_fn: str = "utilitarian"
    ) -> float:
        """Compute ecosystem welfare for a mixture.

        Args:
            mixture: Equilibrium mixture over policies.
            welfare_fn: Welfare function ("utilitarian", "nash", "egalitarian").

        Returns:
            Welfare value.
        """
        # Expected payoff for each policy under the mixture
        values = self.payoff_matrix @ mixture

        # Weighted by equilibrium probability
        weighted_values = mixture * values

        if welfare_fn == "utilitarian":
            # Average welfare (weighted by equilibrium mass)
            return float(np.sum(weighted_values))
        elif welfare_fn == "nash":
            # Nash welfare (product of utilities) - use log for numerical stability
            positive_vals = weighted_values[weighted_values > 0]
            if len(positive_vals) == 0:
                return 0.0
            return float(np.exp(np.sum(np.log(positive_vals))))
        elif welfare_fn == "egalitarian":
            # Minimum welfare among policies with positive mass
            active = mixture > 1e-10
            if not active.any():
                return 0.0
            return float(np.min(values[active]))
        else:
            raise ValueError(f"Unknown welfare function: {welfare_fn}")

    def __repr__(self) -> str:
        return f"MetaGame(policies={self.policies}, shape={self.payoff_matrix.shape})"
