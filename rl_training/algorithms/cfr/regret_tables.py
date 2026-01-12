from typing import Any, Dict, List
import numpy as np
from numpy.typing import NDArray


class RegretTables:
    """
    Tabular storage for CFR regrets and strategy accumulation.

    Supports vanilla CFR, CFR+, Linear CFR, and Discounted CFR.
    """

    def __init__(self, num_players: int, variant: str = "vanilla"):
        self.num_players = num_players
        self.variant = variant

        # Regrets indexed by info_state string
        self._regrets: Dict[str, NDArray[np.floating]] = {}
        # Cumulative strategy for average policy
        self._strategy_sum: Dict[str, NDArray[np.floating]] = {}
        # Number of actions per info state
        self._num_actions: Dict[str, int] = {}

    def get_strategy(
        self, info_state: str, legal_actions: List[int]
    ) -> NDArray[np.floating]:
        """
        Get current strategy via regret matching.

        Returns uniform distribution if info state is unseen or all regrets <= 0.
        """
        num_actions = len(legal_actions)

        if info_state not in self._regrets:
            self._regrets[info_state] = np.zeros(num_actions, dtype=np.float64)
            self._strategy_sum[info_state] = np.zeros(num_actions, dtype=np.float64)
            self._num_actions[info_state] = num_actions

        regrets = self._regrets[info_state]

        # Regret matching: positive regrets normalized
        if self.variant == "plus":
            # CFR+: use max(0, regret)
            positive_regrets = np.maximum(regrets, 0)
        else:
            positive_regrets = np.maximum(regrets, 0)

        regret_sum = positive_regrets.sum()

        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            strategy = np.ones(num_actions, dtype=np.float64) / num_actions

        return strategy

    def update_regrets(
        self,
        info_state: str,
        legal_actions: List[int],
        regrets: NDArray[np.floating],
        iteration: int = 1,
    ) -> None:
        """
        Update regrets for an info state.

        Args:
            info_state: Information state string.
            legal_actions: List of legal action indices.
            regrets: Counterfactual regrets for each action.
            iteration: Current iteration (for weighted variants).
        """
        if info_state not in self._regrets:
            self._regrets[info_state] = np.zeros(len(legal_actions), dtype=np.float64)
            self._strategy_sum[info_state] = np.zeros(len(legal_actions), dtype=np.float64)
            self._num_actions[info_state] = len(legal_actions)

        if self.variant == "plus":
            # CFR+: regrets can't go negative
            self._regrets[info_state] = np.maximum(
                self._regrets[info_state] + regrets, 0
            )
        elif self.variant == "linear":
            # Linear CFR: weight by iteration
            self._regrets[info_state] += iteration * regrets
        elif self.variant == "discounted":
            # Discounted CFR: weight updates
            # Note: full discounted CFR also discounts accumulated regrets
            self._regrets[info_state] += regrets
        else:
            # Vanilla CFR
            self._regrets[info_state] += regrets

    def accumulate_strategy(
        self,
        info_state: str,
        legal_actions: List[int],
        weighted_strategy: NDArray[np.floating],
        iteration: int = 1,
    ) -> None:
        """
        Accumulate strategy for computing average policy.

        Args:
            info_state: Information state string.
            legal_actions: List of legal action indices.
            weighted_strategy: Strategy weighted by reach probability.
            iteration: Current iteration (for weighted variants).
        """
        if info_state not in self._strategy_sum:
            self._strategy_sum[info_state] = np.zeros(len(legal_actions), dtype=np.float64)
            self._num_actions[info_state] = len(legal_actions)

        if self.variant == "linear":
            # Linear CFR: weight by iteration
            self._strategy_sum[info_state] += iteration * weighted_strategy
        else:
            self._strategy_sum[info_state] += weighted_strategy

    def get_average_strategy(self, info_state: str) -> NDArray[np.floating]:
        """Get average strategy (normalized cumulative strategy)."""
        if info_state not in self._strategy_sum:
            # Return uniform if never seen
            return np.ones(1, dtype=np.float64)

        strategy_sum = self._strategy_sum[info_state]
        total = strategy_sum.sum()

        if total > 0:
            return strategy_sum / total
        else:
            num_actions = len(strategy_sum)
            return np.ones(num_actions, dtype=np.float64) / num_actions

    @property
    def info_states(self) -> List[str]:
        """List all known info states."""
        return list(self._regrets.keys())

    def state_dict(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "regrets": {k: v.copy() for k, v in self._regrets.items()},
            "strategy_sum": {k: v.copy() for k, v in self._strategy_sum.items()},
            "num_actions": self._num_actions.copy(),
            "variant": self.variant,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load from checkpoint."""
        self._regrets = {k: v.copy() for k, v in state_dict["regrets"].items()}
        self._strategy_sum = {k: v.copy() for k, v in state_dict["strategy_sum"].items()}
        self._num_actions = state_dict["num_actions"].copy()
        self.variant = state_dict.get("variant", self.variant)

    def clear(self) -> None:
        """Clear all tables."""
        self._regrets.clear()
        self._strategy_sum.clear()
        self._num_actions.clear()
