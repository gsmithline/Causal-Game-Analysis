"""
CFR Trainer - Counterfactual Regret Minimization

Tabular algorithm for computing Nash equilibria in extensive-form games.
"""
from typing import Any, Callable, Dict, Optional
import numpy as np

from rl_training.core.base_trainer import BaseTrainer
from rl_training.core.base_policy import BaseTabularPolicy
from rl_training.core.registry import register_algorithm
from rl_training.algorithms.cfr.config import CFRConfig
from rl_training.algorithms.cfr.regret_tables import RegretTables


@register_algorithm("cfr")
class CFRTrainer(BaseTrainer[CFRConfig]):
    """
    Counterfactual Regret Minimization trainer.

    Computes Nash equilibrium strategies through iterative regret minimization.
    Supports vanilla CFR, CFR+, Linear CFR, and Discounted CFR variants.

    Args:
        config: CFR configuration.
        game_name: OpenSpiel game name (e.g., "kuhn_poker").
        logger: Optional logger instance.
    """

    def __init__(
        self,
        config: CFRConfig,
        env_fn: Optional[Callable] = None,
        game_name: Optional[str] = None,
        logger: Optional[Any] = None,
    ):
        super().__init__(config, env_fn, logger)
        self.game_name = game_name or config.env_id

        # Will be initialized in _setup
        self.game = None
        self.regret_tables: Optional[RegretTables] = None
        self.average_policy: Optional[BaseTabularPolicy] = None
        self._iteration: int = 0

    def _setup(self) -> None:
        """Initialize CFR components."""
        import pyspiel

        self.game = pyspiel.load_game(self.game_name)

        # Initialize regret tables
        self.regret_tables = RegretTables(
            num_players=self.game.num_players(),
            variant=self.config.cfr_variant,
        )

        # Average policy (what we output)
        self.average_policy = BaseTabularPolicy(
            num_actions=self.game.num_distinct_actions()
        )

    def _train_step(self) -> Dict[str, Any]:
        """
        One CFR iteration.

        For vanilla CFR: traverse game tree, update regrets.
        """
        self._iteration += 1

        # Run CFR iteration for each player
        if self.config.alternating_updates:
            # Alternating updates (more stable)
            player = (self._iteration - 1) % self.game.num_players()
            self._cfr_iteration(player)
        else:
            # Simultaneous updates
            for player in range(self.game.num_players()):
                self._cfr_iteration(player)

        # Update average policy
        self._update_average_policy()

        # Compute exploitability periodically
        metrics = {
            "iteration": self._iteration,
            "timesteps": 1,  # For compatibility with BaseTrainer
            "num_info_states": len(self.regret_tables.info_states),
        }

        if self._iteration % 100 == 0:
            try:
                exp = self._compute_exploitability()
                metrics["exploitability"] = exp
            except Exception:
                pass  # exploitability computation may fail for some games

        return metrics

    def _cfr_iteration(self, traversing_player: int) -> float:
        """Run CFR for one player."""
        return self._cfr_recursive(
            self.game.new_initial_state(),
            traversing_player,
            reach_probs=np.ones(self.game.num_players(), dtype=np.float64),
        )

    def _cfr_recursive(
        self,
        state,
        traversing_player: int,
        reach_probs: np.ndarray,
    ) -> float:
        """Recursive CFR tree traversal."""
        if state.is_terminal():
            return state.returns()[traversing_player]

        if state.is_chance_node():
            value = 0.0
            for action, prob in state.chance_outcomes():
                child = state.child(action)
                value += prob * self._cfr_recursive(
                    child, traversing_player, reach_probs
                )
            return value

        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        legal_actions = state.legal_actions()

        # Get current strategy from regret matching
        strategy = self.regret_tables.get_strategy(info_state, legal_actions)

        # Compute counterfactual values
        action_values = np.zeros(len(legal_actions), dtype=np.float64)
        for i, action in enumerate(legal_actions):
            child = state.child(action)
            new_reach = reach_probs.copy()
            new_reach[current_player] *= strategy[i]
            action_values[i] = self._cfr_recursive(
                child, traversing_player, new_reach
            )

        node_value = np.dot(strategy, action_values)

        # Update regrets for traversing player
        if current_player == traversing_player:
            # Counterfactual reach probability
            cf_reach = np.prod([
                reach_probs[p] for p in range(len(reach_probs))
                if p != current_player
            ])

            regrets = cf_reach * (action_values - node_value)
            self.regret_tables.update_regrets(
                info_state, legal_actions, regrets, self._iteration
            )

        # Accumulate strategy for average policy
        self.regret_tables.accumulate_strategy(
            info_state,
            legal_actions,
            strategy * reach_probs[current_player],
            self._iteration,
        )

        return node_value

    def _update_average_policy(self) -> None:
        """Update average policy from accumulated strategies."""
        for info_state in self.regret_tables.info_states:
            avg_strategy = self.regret_tables.get_average_strategy(info_state)
            self.average_policy.update_strategy(info_state, avg_strategy)

    def _compute_exploitability(self) -> float:
        """Compute exploitability using OpenSpiel."""
        from open_spiel.python.algorithms import exploitability
        from open_spiel.python import policy as policy_module

        # Convert to OpenSpiel policy format
        def policy_callable(state):
            info_state = state.information_state_string()
            legal_actions = state.legal_actions()
            strategy = self.average_policy.get_strategy(info_state)

            # Handle case where strategy length doesn't match legal actions
            if len(strategy) != len(legal_actions):
                strategy = np.ones(len(legal_actions)) / len(legal_actions)

            return list(zip(legal_actions, strategy))

        tabular_policy = policy_module.tabular_policy_from_callable(
            self.game, policy_callable
        )
        return exploitability.exploitability(self.game, tabular_policy)

    def _get_checkpoint_data(self) -> Dict[str, Any]:
        return {
            "regret_tables": self.regret_tables.state_dict(),
            "average_policy": self.average_policy.state_dict(),
            "iteration": self._iteration,
        }

    def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
        self.regret_tables.load_state_dict(data["regret_tables"])
        self.average_policy.load_state_dict(data["average_policy"])
        self._iteration = data["iteration"]

    def get_policy(self) -> BaseTabularPolicy:
        """Return trained average policy."""
        return self.average_policy
