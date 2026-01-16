"""
Wrapper for the CUDA Bargaining Game Environment.

Adapts the BargainEnv to the rl_training framework interface.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import numpy as np
from numpy.typing import NDArray

from .env_wrapper import BaseEnvWrapper, StepResult, MultiAgentStepResult


@dataclass
class VectorizedStepResult:
    """Step result for vectorized (batched) environments."""
    observations: torch.Tensor          # [num_envs, obs_dim]
    rewards: torch.Tensor               # [num_envs, num_players]
    terminated: torch.Tensor            # [num_envs] bool
    truncated: torch.Tensor             # [num_envs] bool
    info: Dict[str, Any]                # Contains action_mask, current_player, etc.

    @property
    def dones(self) -> torch.Tensor:
        """Convenience property for done = terminated | truncated."""
        return self.terminated | self.truncated


class BargainEnvWrapper(BaseEnvWrapper):
    """
    Wrapper adapting CUDA BargainEnv to rl_training interface.

    This wrapper provides both single-env and vectorized interfaces
    for the GPU-accelerated bargaining game.

    The bargaining game:
        - 2 players negotiate division of items: [7, 4, 1] quantities
        - Each player has private values (1-100) for each item type
        - Each player has a private outside offer (fallback value)
        - 3 rounds maximum
        - Actions: 80 counteroffers + ACCEPT + WALK = 82 total

    Observation Space (92 floats):
        - [0-2]: Player's normalized item values
        - [3]: Normalized outside offer
        - [4-6]: Current offer items (-1 if none)
        - [7]: Offer valid flag
        - [8]: Normalized round
        - [9]: Current player (0 or 1)
        - [10-91]: Action mask (82 booleans)

    Action Space (82 discrete actions):
        - 0-79: Counteroffer (encoded as offer[0]*10 + offer[1]*2 + offer[2])
        - 80: ACCEPT
        - 81: WALK
    """

    # Constants
    NUM_ACTIONS = 82
    OBS_DIM = 92
    ACTION_ACCEPT = 80
    ACTION_WALK = 81
    NUM_ITEM_TYPES = 3
    MAX_ROUNDS = 3
    ITEM_QUANTITIES = (7, 4, 1)

    def __init__(
        self,
        num_envs: int = 1,
        self_play: bool = True,
        device: int = 0,
        seed: Optional[int] = None
    ):
        """
        Initialize the environment.

        Args:
            num_envs: Number of parallel game instances (default 1 for single-env)
            self_play: If True, both players controlled by RL agent
            device: CUDA device ID
            seed: Random seed for reproducibility
        """
        super().__init__()

        # Import here to avoid import errors if CUDA not available
        from cuda_bargain import BargainEnv

        self.num_envs = num_envs
        self.self_play = self_play
        self.cuda_device = device

        # Initialize the CUDA environment
        self._env = BargainEnv(
            num_envs=num_envs,
            self_play=self_play,
            device=device,
            seed=seed
        )

        # Set base class properties
        self._num_players = 2
        self._observation_shape = (self.OBS_DIM,)
        self._num_actions = self.NUM_ACTIONS

        # Track state
        self._current_obs: Optional[torch.Tensor] = None
        self._current_info: Optional[Dict[str, Any]] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Reset all environments.

        Args:
            seed: Optional new random seed

        Returns:
            observations: [num_envs, OBS_DIM] float32 tensor (on GPU)
            info: dict containing:
                - 'action_mask': [num_envs, NUM_ACTIONS] float32 tensor
                - 'current_player': [num_envs] uint8 tensor
        """
        obs, info = self._env.reset(seed=seed)
        self._current_obs = obs
        self._current_info = info
        return obs, info

    def step(self, actions: Union[torch.Tensor, np.ndarray, int]) -> VectorizedStepResult:
        """
        Step all environments with given actions.

        Args:
            actions: Actions to take. Can be:
                - torch.Tensor [num_envs] int32
                - np.ndarray [num_envs]
                - int (for single env)

        Returns:
            VectorizedStepResult containing:
                - observations: [num_envs, OBS_DIM]
                - rewards: [num_envs, 2] (rewards for both players)
                - terminated: [num_envs] bool
                - truncated: [num_envs] bool
                - info: dict with action_mask, current_player
        """
        # Convert actions to torch tensor if needed
        if isinstance(actions, int):
            actions = torch.tensor([actions], dtype=torch.int32)
        elif isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(torch.int32)

        # Ensure on GPU
        if not actions.is_cuda:
            actions = actions.cuda(self.cuda_device)

        # Step the environment
        obs, rewards, terminated, truncated, info = self._env.step(actions)

        self._current_obs = obs
        self._current_info = info

        return VectorizedStepResult(
            observations=obs,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info
        )

    def step_single(self, action: int) -> StepResult:
        """
        Single-environment step interface (for compatibility).

        This extracts results for the first environment only.
        Use step() for vectorized training.

        Args:
            action: Single action index

        Returns:
            StepResult for the first environment
        """
        result = self.step(torch.tensor([action], dtype=torch.int32))

        return StepResult(
            observation=result.observations[0].cpu().numpy(),
            reward=result.rewards[0].cpu().numpy(),  # [2] for both players
            terminated=result.terminated[0].item(),
            truncated=result.truncated[0].item(),
            info={
                'action_mask': result.info['action_mask'][0].cpu().numpy(),
                'current_player': result.info['current_player'][0].item(),
            }
        )

    def auto_reset(self) -> None:
        """
        Automatically reset any environments that are done.

        Call this after step() for continuous training rollouts.
        """
        self._env.auto_reset()

    def get_current_player(self) -> torch.Tensor:
        """
        Get which player's turn it is for each environment.

        Returns:
            current_player: [num_envs] uint8 tensor (0 or 1)
        """
        return self._env.get_current_player()

    def get_action_mask(self) -> torch.Tensor:
        """
        Get current action mask for all environments.

        Returns:
            action_mask: [num_envs, NUM_ACTIONS] float32 tensor
        """
        if self._current_info is not None:
            return self._current_info['action_mask']
        # If no cached info, do a reset to get it
        obs, info = self.reset()
        return info['action_mask']

    def get_opponent_actions(self, opponent_player: int = 1) -> torch.Tensor:
        """
        Get random valid actions for the opponent player.

        Args:
            opponent_player: Which player is the opponent (0 or 1)

        Returns:
            actions: [num_envs] int32 tensor (-1 where not opponent's turn)
        """
        return self._env.get_opponent_actions(opponent_player)

    def sample_valid_actions(self) -> torch.Tensor:
        """
        Sample random valid actions for all environments.

        Returns:
            actions: [num_envs] int32 tensor of valid actions
        """
        return self._env.sample_valid_actions()

    def get_dones(self) -> torch.Tensor:
        """
        Get done flags for all environments.

        Returns:
            dones: [num_envs] bool tensor
        """
        return self._env.get_dones()

    def close(self) -> None:
        """Clean up resources."""
        # The CUDA env handles cleanup in its destructor
        pass

    # =========================================================================
    # Utility methods
    # =========================================================================

    @staticmethod
    def decode_action(action_idx: int) -> Tuple[str, Optional[Tuple[int, int, int]]]:
        """
        Decode action index to human-readable format.

        Args:
            action_idx: Action index (0-81)

        Returns:
            (action_type, offer): Tuple of action type and offer
        """
        if action_idx == BargainEnvWrapper.ACTION_ACCEPT:
            return ("ACCEPT", None)
        elif action_idx == BargainEnvWrapper.ACTION_WALK:
            return ("WALK", None)
        elif 0 <= action_idx < 80:
            item2 = action_idx % 2
            action_idx //= 2
            item1 = action_idx % 5
            item0 = action_idx // 5
            return ("COUNTEROFFER", (item0, item1, item2))
        else:
            raise ValueError(f"Invalid action index: {action_idx}")

    @staticmethod
    def encode_offer(offer: Tuple[int, int, int]) -> int:
        """
        Encode offer tuple to action index.

        Args:
            offer: (n0, n1, n2) items to offer

        Returns:
            action_idx: Action index (0-79)
        """
        n0, n1, n2 = offer
        if not (0 <= n0 <= 7 and 0 <= n1 <= 4 and 0 <= n2 <= 1):
            raise ValueError(f"Invalid offer: {offer}")
        return n0 * 10 + n1 * 2 + n2

    def __repr__(self) -> str:
        return (
            f"BargainEnvWrapper(num_envs={self.num_envs}, "
            f"self_play={self.self_play}, device={self.cuda_device})"
        )
