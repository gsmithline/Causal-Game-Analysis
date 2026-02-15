"""
High-level Python wrapper for CUDA Bargaining Game Environment.

Provides a clean API compatible with standard RL training loops.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union

# Import the CUDA extension
from . import cuda_bargain_core


class BargainEnv:
    """
    CUDA-accelerated bargaining game environment for deep RL training.

    The environment runs thousands of parallel game instances on GPU.

    Game Rules:
        - 2 players negotiate division of items: [7, 4, 1] quantities
        - Each player has private values (1-100) for each item type
        - Each player has a private outside offer (fallback value)
        - 3 rounds maximum (6 total player actions)
        - Actions: 80 counteroffers + ACCEPT + WALK = 82 total
        - Final round Player 2 can only ACCEPT or WALK

    Observation Space (92 floats):
        - [0-2]: Player's normalized item values
        - [3]: Normalized outside offer
        - [4-6]: Current offer items (-1 if none)
        - [7]: Offer valid flag
        - [8]: Normalized round
        - [9]: Current player (0 or 1)
        - [10-91]: Action mask (82 booleans)

    Action Space:
        - 0-79: Counteroffer (encoded as offer[0]*10 + offer[1]*2 + offer[2])
        - 80: ACCEPT
        - 81: WALK
    """

    # Class-level constants from CUDA module
    NUM_ACTIONS = cuda_bargain_core.NUM_ACTIONS
    OBS_DIM = cuda_bargain_core.OBS_DIM
    ACTION_ACCEPT = cuda_bargain_core.ACTION_ACCEPT
    ACTION_WALK = cuda_bargain_core.ACTION_WALK
    NUM_ITEM_TYPES = cuda_bargain_core.NUM_ITEM_TYPES
    MAX_ROUNDS = cuda_bargain_core.MAX_ROUNDS
    ITEM_QUANTITIES = cuda_bargain_core.ITEM_QUANTITIES

    def __init__(
        self,
        num_envs: int,
        self_play: bool = True,
        device: int = 0,
        seed: Optional[int] = None
    ):
        """
        Initialize the environment.

        Args:
            num_envs: Number of parallel game instances
            self_play: If True, both players are controlled by RL agent.
                      If False, use get_opponent_actions() for opponent.
            device: CUDA device ID
            seed: Random seed for reproducibility
        """
        self.num_envs = num_envs
        self.self_play = self_play
        self.device = device
        self._seed = seed if seed is not None else np.random.randint(0, 2**32)

        # Create the CUDA environment
        self._env = cuda_bargain_core.BargainGameEnv(num_envs, self_play, device)

        # Track state
        self._needs_reset = True

    def reset(
        self,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Reset all environments.

        Args:
            seed: Optional new random seed

        Returns:
            observations: [num_envs, OBS_DIM] float32 tensor
            info: dict containing:
                - 'action_mask': [num_envs, NUM_ACTIONS] float32 tensor
                - 'current_player': [num_envs] uint8 tensor
        """
        if seed is not None:
            self._seed = seed

        obs, masks = self._env.reset(self._seed)
        self._seed += 1
        self._needs_reset = False

        current_player = self._env.get_current_player()

        return obs, {
            'action_mask': masks,
            'current_player': current_player
        }

    def step(
        self,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Step all environments with given actions.

        Args:
            actions: [num_envs] int32 tensor of action indices

        Returns:
            observations: [num_envs, OBS_DIM] float32
            rewards: [num_envs, 2] float32 - normalized rewards for both players
            terminated: [num_envs] bool
            truncated: [num_envs] bool
            info: dict containing:
                - 'action_mask': [num_envs, NUM_ACTIONS] float32
                - 'current_player': [num_envs] uint8
        """
        if self._needs_reset:
            raise RuntimeError("Environment needs reset before stepping")

        # Ensure actions are correct type and device
        if not actions.is_cuda:
            actions = actions.cuda(self.device)
        if actions.dtype != torch.int32:
            actions = actions.to(torch.int32)

        obs, masks, rewards, dones, truncateds = self._env.step(actions)
        current_player = self._env.get_current_player()

        return obs, rewards, dones, truncateds, {
            'action_mask': masks,
            'current_player': current_player
        }

    def get_current_player(self) -> torch.Tensor:
        """
        Get which player's turn it is for each environment.

        Returns:
            current_player: [num_envs] uint8 tensor (0 or 1)
        """
        return self._env.get_current_player()

    def get_opponent_actions(self, opponent_player: int = 1) -> torch.Tensor:
        """
        Get random valid actions for the opponent player.

        Useful for vs-random training mode.

        Args:
            opponent_player: Which player is the opponent (0 or 1)

        Returns:
            actions: [num_envs] int32 tensor
                     -1 for envs where it's not opponent's turn
        """
        return self._env.get_random_actions(opponent_player)

    def auto_reset(self) -> None:
        """
        Automatically reset any environments that are done.

        Call this after step() if you want continuous rollouts.
        """
        self._env.auto_reset()

    def get_dones(self) -> torch.Tensor:
        """
        Get done flags for all environments.

        Returns:
            dones: [num_envs] bool tensor
        """
        return self._env.get_dones()

    @staticmethod
    def decode_action(action_idx: int) -> Tuple[str, Optional[Tuple[int, int, int]]]:
        """
        Decode action index to human-readable format.

        Args:
            action_idx: Action index (0-81)

        Returns:
            (action_type, offer): Tuple of action type string and offer tuple
                - ("ACCEPT", None)
                - ("WALK", None)
                - ("COUNTEROFFER", (n0, n1, n2))
        """
        if action_idx == BargainEnv.ACTION_ACCEPT:
            return ("ACCEPT", None)
        elif action_idx == BargainEnv.ACTION_WALK:
            return ("WALK", None)
        elif 0 <= action_idx < 80:
            # Decode counteroffer
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
            offer: (n0, n1, n2) items to offer to opponent

        Returns:
            action_idx: Action index (0-79)
        """
        n0, n1, n2 = offer
        if not (0 <= n0 <= 7 and 0 <= n1 <= 4 and 0 <= n2 <= 1):
            raise ValueError(f"Invalid offer: {offer}")
        return n0 * 10 + n1 * 2 + n2

    def sample_valid_actions(self) -> torch.Tensor:
        """
        Sample random valid actions for all environments.

        This respects the action mask for each environment.

        Returns:
            actions: [num_envs] int32 tensor of valid action indices
        """
        # Use the current player to get valid random actions
        current_players = self._env.get_current_player()
        # Get actions for player 0
        actions_p0 = self._env.get_random_actions(0)
        # Get actions for player 1
        actions_p1 = self._env.get_random_actions(1)

        # Combine based on current player
        actions = torch.where(
            current_players == 0,
            actions_p0,
            actions_p1
        )

        return actions

    def __repr__(self) -> str:
        return (
            f"BargainEnv(num_envs={self.num_envs}, "
            f"self_play={self.self_play}, device={self.device})"
        )
