from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from rl_training.envs.env_wrapper import BaseEnvWrapper, StepResult, MultiAgentStepResult


class OpenSpielWrapper(BaseEnvWrapper):
    """
    Wrapper that adapts OpenSpiel games to unified interface.

    Handles:
    - Sequential vs simultaneous games
    - Information state tensors
    - Legal action masking
    - Multi-player reward distribution

    Usage:
        wrapper = OpenSpielWrapper("kuhn_poker")
        obs, info = wrapper.reset()
        result = wrapper.step(action)
    """

    def __init__(
        self,
        game_name: str,
        game_params: Optional[Dict[str, Any]] = None,
        observation_type: str = "info_state",  # "info_state" or "observation"
    ):
        super().__init__()
        import pyspiel

        self.game_name = game_name
        self.game_params = game_params or {}
        self.observation_type = observation_type

        # Create game
        if game_params:
            self.game = pyspiel.load_game(game_name, game_params)
        else:
            self.game = pyspiel.load_game(game_name)

        # Extract properties
        self._num_players = self.game.num_players()
        self._num_actions = self.game.num_distinct_actions()

        # Observation shape depends on observation type
        if observation_type == "info_state":
            self._observation_shape = (self.game.information_state_tensor_size(),)
        else:
            self._observation_shape = (self.game.observation_tensor_size(),)

        self._state: Optional[Any] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[NDArray, Dict[str, Any]]:
        """Reset game to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self._state = self.game.new_initial_state()

        # Handle chance nodes
        while self._state.is_chance_node():
            outcomes = self._state.chance_outcomes()
            action = np.random.choice(
                [o[0] for o in outcomes],
                p=[o[1] for o in outcomes]
            )
            self._state.apply_action(action)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> StepResult:
        """Apply action and return result."""
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        current_player = self._state.current_player()

        # Apply action
        self._state.apply_action(action)

        # Handle chance nodes
        while self._state.is_chance_node():
            outcomes = self._state.chance_outcomes()
            chance_action = np.random.choice(
                [o[0] for o in outcomes],
                p=[o[1] for o in outcomes]
            )
            self._state.apply_action(chance_action)

        terminated = self._state.is_terminal()

        if terminated:
            returns = self._state.returns()
            reward = returns[current_player]
        else:
            reward = 0.0

        obs = self._get_observation() if not terminated else np.zeros(self._observation_shape, dtype=np.float32)
        info = self._get_info()

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=False,  # OpenSpiel games don't truncate
            info=info,
        )

    def step_multi_agent(self, action: int) -> MultiAgentStepResult:
        """Multi-agent step variant returning per-player info."""
        result = self.step(action)

        observations = {}
        rewards = {}

        if result.terminated:
            returns = self._state.returns()
            for p in range(self._num_players):
                observations[p] = np.zeros(self._observation_shape, dtype=np.float32)
                rewards[p] = returns[p]
        else:
            current = self.current_player
            for p in range(self._num_players):
                if p == current:
                    observations[p] = result.observation
                else:
                    observations[p] = self._get_observation(player=p)
                rewards[p] = 0.0

        return MultiAgentStepResult(
            observations=observations,
            rewards=rewards,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
            current_player=self.current_player if not result.terminated else -1,
        )

    def _get_observation(self, player: Optional[int] = None) -> NDArray:
        """Get observation tensor for player (default: current player)."""
        if player is None:
            player = self._state.current_player()

        if self.observation_type == "info_state":
            return np.array(self._state.information_state_tensor(player), dtype=np.float32)
        else:
            return np.array(self._state.observation_tensor(player), dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state."""
        if self._state.is_terminal():
            return {
                "terminal": True,
                "returns": self._state.returns(),
            }

        return {
            "terminal": False,
            "current_player": self._state.current_player(),
            "legal_actions": self._state.legal_actions(),
            "legal_actions_mask": self._get_legal_actions_mask(),
            "info_state_string": self._state.information_state_string(),
        }

    def _get_legal_actions_mask(self) -> NDArray:
        """Get binary mask of legal actions."""
        mask = np.zeros(self._num_actions, dtype=np.float32)
        for action in self._state.legal_actions():
            mask[action] = 1.0
        return mask

    @property
    def current_player(self) -> int:
        """Get current player."""
        if self._state is None or self._state.is_terminal():
            return -1
        return self._state.current_player()

    @property
    def info_state_string(self) -> str:
        """Get information state string (for tabular algorithms)."""
        if self._state is None:
            return ""
        return self._state.information_state_string()

    @property
    def legal_actions(self) -> List[int]:
        """Get list of legal actions."""
        if self._state is None or self._state.is_terminal():
            return []
        return self._state.legal_actions()

    @property
    def state(self):
        """Access underlying OpenSpiel state."""
        return self._state

    @property
    def game_instance(self):
        """Access underlying OpenSpiel game."""
        return self.game
