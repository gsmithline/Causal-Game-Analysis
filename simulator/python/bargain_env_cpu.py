"""
Pure Python/NumPy CPU implementation of the Bargaining Game Environment.

Provides the same API as the CUDA BargainEnv for running on CPU-only machines.
Useful for:
- Generating game data with trained policies
- Running meta-game analysis
- Testing and debugging
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


# Game Constants (matching CUDA version)
NUM_ITEM_TYPES = 3
MAX_ROUNDS = 3
NUM_PLAYERS = 2
ITEM_QUANTITIES = np.array([7, 4, 1], dtype=np.int32)

NUM_COUNTEROFFER_ACTIONS = 80
ACTION_ACCEPT = 80
ACTION_WALK = 81
NUM_ACTIONS = 82

MIN_VALUE = 1
MAX_VALUE = 100

OBS_DIM = 92

# Outcome codes
OUTCOME_ONGOING = 0
OUTCOME_ACCEPT = 1
OUTCOME_WALK = 2


class BargainEnvCPU:
    """
    CPU implementation of the bargaining game environment.

    Mirrors the CUDA BargainEnv API for compatibility with trained policies.

    Game Rules:
        - 2 players negotiate division of items: [7, 4, 1] quantities
        - Each player has private values (1-100) for each item type
        - Each player has a private outside option (fallback value)
        - 3 rounds maximum (6 total player actions)
        - Actions: 80 counteroffers + ACCEPT + WALK = 82 total
        - Final turn Player 2 can only ACCEPT or WALK

    Observation Space (92 floats):
        - [0-2]: Player's normalized item values
        - [3]: Normalized outside option
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

    # Class-level constants for API compatibility
    NUM_ACTIONS = NUM_ACTIONS
    OBS_DIM = OBS_DIM
    ACTION_ACCEPT = ACTION_ACCEPT
    ACTION_WALK = ACTION_WALK
    NUM_ITEM_TYPES = NUM_ITEM_TYPES
    MAX_ROUNDS = MAX_ROUNDS
    ITEM_QUANTITIES = ITEM_QUANTITIES

    def __init__(
        self,
        num_envs: int,
        self_play: bool = True,
        device: Optional[str] = None,  # Ignored, for API compatibility
        seed: Optional[int] = None
    ):
        """
        Initialize the environment.

        Args:
            num_envs: Number of parallel game instances
            self_play: If True, both players are controlled by RL agent
            device: Ignored (CPU only), for API compatibility
            seed: Random seed for reproducibility
        """
        self.num_envs = num_envs
        self.self_play = self_play
        self._seed = seed if seed is not None else np.random.randint(0, 2**31)
        self._rng = np.random.default_rng(self._seed)

        # Allocate state arrays
        self._player_values = np.zeros((num_envs, NUM_PLAYERS, NUM_ITEM_TYPES), dtype=np.float32)
        self._outside_options = np.zeros((num_envs, NUM_PLAYERS), dtype=np.float32)
        self._max_possible_values = np.zeros((num_envs, NUM_PLAYERS), dtype=np.float32)

        self._current_offer = np.zeros((num_envs, NUM_ITEM_TYPES), dtype=np.int8)
        self._offer_valid = np.zeros(num_envs, dtype=np.uint8)
        self._current_round = np.zeros(num_envs, dtype=np.uint8)
        self._current_player = np.zeros(num_envs, dtype=np.uint8)
        self._action_count = np.zeros(num_envs, dtype=np.uint8)

        self._done = np.zeros(num_envs, dtype=bool)
        self._outcome = np.zeros(num_envs, dtype=np.uint8)
        self._rewards = np.zeros((num_envs, NUM_PLAYERS), dtype=np.float32)

        # Output buffers
        self._observations = np.zeros((num_envs, OBS_DIM), dtype=np.float32)
        self._action_masks = np.zeros((num_envs, NUM_ACTIONS), dtype=np.float32)

        self._needs_reset = True

    def reset(
        self,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reset all environments.

        Args:
            seed: Optional new random seed

        Returns:
            observations: [num_envs, OBS_DIM] float32 array
            info: dict containing:
                - 'action_mask': [num_envs, NUM_ACTIONS] float32 array
                - 'current_player': [num_envs] uint8 array
        """
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(self._seed)

        self._reset_all_games()
        self._needs_reset = False

        self._build_observations()

        return self._observations.copy(), {
            'action_mask': self._action_masks.copy(),
            'current_player': self._current_player.copy()
        }

    def _reset_all_games(self):
        """Reset all game states."""
        # Generate random values for each player (1-100)
        self._player_values[:] = self._rng.integers(
            MIN_VALUE, MAX_VALUE + 1,
            size=(self.num_envs, NUM_PLAYERS, NUM_ITEM_TYPES)
        ).astype(np.float32)

        # Generate random outside options
        # Outside option is typically less than max possible value
        for i in range(NUM_PLAYERS):
            max_vals = np.sum(self._player_values[:, i, :] * ITEM_QUANTITIES, axis=1)
            self._max_possible_values[:, i] = max_vals
            # Outside option between 10% and 50% of max possible
            self._outside_options[:, i] = self._rng.uniform(
                0.1 * max_vals, 0.5 * max_vals
            ).astype(np.float32)

        # Reset game state
        self._current_offer[:] = -1
        self._offer_valid[:] = 0
        self._current_round[:] = 0
        self._current_player[:] = 0  # P1 starts
        self._action_count[:] = 0

        self._done[:] = False
        self._outcome[:] = OUTCOME_ONGOING
        self._rewards[:] = 0.0

    def _reset_games(self, mask: np.ndarray):
        """Reset specific games indicated by mask."""
        n_reset = mask.sum()
        if n_reset == 0:
            return

        # Generate new values
        self._player_values[mask] = self._rng.integers(
            MIN_VALUE, MAX_VALUE + 1,
            size=(n_reset, NUM_PLAYERS, NUM_ITEM_TYPES)
        ).astype(np.float32)

        # Generate new outside options
        for i in range(NUM_PLAYERS):
            max_vals = np.sum(self._player_values[mask, i, :] * ITEM_QUANTITIES, axis=1)
            self._max_possible_values[mask, i] = max_vals
            self._outside_options[mask, i] = self._rng.uniform(
                0.1 * max_vals, 0.5 * max_vals
            ).astype(np.float32)

        # Reset game state for these environments
        self._current_offer[mask] = -1
        self._offer_valid[mask] = 0
        self._current_round[mask] = 0
        self._current_player[mask] = 0
        self._action_count[mask] = 0

        self._done[mask] = False
        self._outcome[mask] = OUTCOME_ONGOING
        self._rewards[mask] = 0.0

    def _build_observations(self):
        """Build observation vectors for all environments."""
        self._observations[:] = 0.0
        self._action_masks[:] = 0.0

        for env_idx in range(self.num_envs):
            player = self._current_player[env_idx]

            # Normalize values to 0-1
            max_val = self._max_possible_values[env_idx, player]
            if max_val > 0:
                self._observations[env_idx, 0:3] = self._player_values[env_idx, player] / MAX_VALUE
                self._observations[env_idx, 3] = self._outside_options[env_idx, player] / max_val

            # Current offer
            if self._offer_valid[env_idx]:
                self._observations[env_idx, 4:7] = self._current_offer[env_idx] / ITEM_QUANTITIES
            else:
                self._observations[env_idx, 4:7] = -1.0

            # Offer valid flag
            self._observations[env_idx, 7] = float(self._offer_valid[env_idx])

            # Round (normalized)
            self._observations[env_idx, 8] = self._current_round[env_idx] / (MAX_ROUNDS - 1)

            # Current player
            self._observations[env_idx, 9] = float(player)

            # Action mask
            if not self._done[env_idx]:
                mask = self._get_action_mask(env_idx)
                self._observations[env_idx, 10:92] = mask
                self._action_masks[env_idx] = mask

    def _get_action_mask(self, env_idx: int) -> np.ndarray:
        """Get valid action mask for a single environment."""
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)

        if self._done[env_idx]:
            return mask

        action_count = self._action_count[env_idx]
        offer_valid = self._offer_valid[env_idx]

        # Final turn (action 5): only ACCEPT or WALK allowed
        is_final_turn = (action_count == 5)

        if is_final_turn:
            if offer_valid:
                mask[ACTION_ACCEPT] = 1.0
            mask[ACTION_WALK] = 1.0
        else:
            # Can make any counteroffer
            mask[0:NUM_COUNTEROFFER_ACTIONS] = 1.0
            # Can accept if there's an offer
            if offer_valid:
                mask[ACTION_ACCEPT] = 1.0
            # Can always walk
            mask[ACTION_WALK] = 1.0

        return mask

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Step all environments with given actions.

        Args:
            actions: [num_envs] int32 array of action indices

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

        actions = np.asarray(actions, dtype=np.int32)

        # Process each environment
        for env_idx in range(self.num_envs):
            if self._done[env_idx]:
                continue
            self._step_single(env_idx, actions[env_idx])

        self._build_observations()

        truncated = np.zeros(self.num_envs, dtype=bool)

        return (
            self._observations.copy(),
            self._rewards.copy(),
            self._done.copy(),
            truncated,
            {
                'action_mask': self._action_masks.copy(),
                'current_player': self._current_player.copy()
            }
        )

    def _step_single(self, env_idx: int, action: int):
        """Process a single action for one environment."""
        if action == ACTION_WALK:
            self._handle_walk(env_idx)
        elif action == ACTION_ACCEPT:
            self._handle_accept(env_idx)
        else:
            self._handle_counteroffer(env_idx, action)

    def _handle_walk(self, env_idx: int):
        """Handle WALK action - both players get outside options."""
        self._done[env_idx] = True
        self._outcome[env_idx] = OUTCOME_WALK

        # Rewards are normalized outside options
        for p in range(NUM_PLAYERS):
            max_val = self._max_possible_values[env_idx, p]
            if max_val > 0:
                self._rewards[env_idx, p] = self._outside_options[env_idx, p] / max_val
            else:
                self._rewards[env_idx, p] = 0.0

    def _handle_accept(self, env_idx: int):
        """Handle ACCEPT action - split items according to offer."""
        if not self._offer_valid[env_idx]:
            # Invalid accept, treat as walk
            self._handle_walk(env_idx)
            return

        self._done[env_idx] = True
        self._outcome[env_idx] = OUTCOME_ACCEPT

        # Calculate rewards based on item allocation
        offer = self._current_offer[env_idx]  # Items offered to the other player

        # Determine who made the offer (the previous player)
        current_player = self._current_player[env_idx]
        offering_player = 1 - current_player
        accepting_player = current_player

        # Offering player keeps: ITEM_QUANTITIES - offer
        # Accepting player gets: offer
        items_to_offerer = ITEM_QUANTITIES - offer
        items_to_accepter = offer

        for p in range(NUM_PLAYERS):
            if p == offering_player:
                items = items_to_offerer
            else:
                items = items_to_accepter

            value = np.sum(self._player_values[env_idx, p] * items)
            max_val = self._max_possible_values[env_idx, p]
            if max_val > 0:
                self._rewards[env_idx, p] = value / max_val
            else:
                self._rewards[env_idx, p] = 0.0

    def _handle_counteroffer(self, env_idx: int, action: int):
        """Handle counteroffer action."""
        # Decode action to offer
        offer = self._decode_action(action)

        # Update game state
        self._current_offer[env_idx] = offer
        self._offer_valid[env_idx] = 1
        self._action_count[env_idx] += 1

        # Switch player
        self._current_player[env_idx] = 1 - self._current_player[env_idx]

        # Update round (increments after both players have acted)
        if self._action_count[env_idx] % 2 == 0:
            self._current_round[env_idx] += 1

    @staticmethod
    def _decode_action(action: int) -> np.ndarray:
        """Decode action index to offer array."""
        offer = np.zeros(NUM_ITEM_TYPES, dtype=np.int8)
        offer[2] = action % 2
        action //= 2
        offer[1] = action % 5
        offer[0] = action // 5
        return offer

    def get_current_player(self) -> np.ndarray:
        """Get which player's turn it is for each environment."""
        return self._current_player.copy()

    def auto_reset(self) -> None:
        """Automatically reset any environments that are done."""
        if self._done.any():
            self._reset_games(self._done)
            self._build_observations()

    def get_dones(self) -> np.ndarray:
        """Get done flags for all environments."""
        return self._done.copy()

    def sample_valid_actions(self) -> np.ndarray:
        """Sample random valid actions for all environments."""
        actions = np.zeros(self.num_envs, dtype=np.int32)

        for env_idx in range(self.num_envs):
            if self._done[env_idx]:
                actions[env_idx] = 0
                continue

            mask = self._action_masks[env_idx]
            valid_actions = np.where(mask > 0)[0]
            if len(valid_actions) > 0:
                actions[env_idx] = self._rng.choice(valid_actions)
            else:
                actions[env_idx] = ACTION_WALK

        return actions

    @staticmethod
    def decode_action(action_idx: int) -> Tuple[str, Optional[Tuple[int, int, int]]]:
        """
        Decode action index to human-readable format.

        Args:
            action_idx: Action index (0-81)

        Returns:
            (action_type, offer): Tuple of action type string and offer tuple
        """
        if action_idx == ACTION_ACCEPT:
            return ("ACCEPT", None)
        elif action_idx == ACTION_WALK:
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
        """Encode offer tuple to action index."""
        n0, n1, n2 = offer
        if not (0 <= n0 <= 7 and 0 <= n1 <= 4 and 0 <= n2 <= 1):
            raise ValueError(f"Invalid offer: {offer}")
        return n0 * 10 + n1 * 2 + n2

    def get_game_data(self) -> Dict[str, np.ndarray]:
        """
        Get detailed game data for all environments.

        Useful for meta-game analysis.

        Returns:
            dict with:
                - 'player_values': [num_envs, 2, 3] values per player per item
                - 'outside_options': [num_envs, 2] outside option per player
                - 'rewards': [num_envs, 2] final rewards
                - 'outcome': [num_envs] outcome code (0=ongoing, 1=accept, 2=walk)
                - 'done': [num_envs] whether game is done
        """
        return {
            'player_values': self._player_values.copy(),
            'outside_options': self._outside_options.copy(),
            'rewards': self._rewards.copy(),
            'outcome': self._outcome.copy(),
            'done': self._done.copy(),
        }

    def __repr__(self) -> str:
        return (
            f"BargainEnvCPU(num_envs={self.num_envs}, "
            f"self_play={self.self_play})"
        )


def run_games_with_policy(
    env: BargainEnvCPU,
    policy_fn,
    num_games: int,
    collect_data: bool = True
) -> Dict[str, Any]:
    """
    Run games using a policy function and collect results.

    Args:
        env: BargainEnvCPU environment
        policy_fn: Function that takes (obs, action_mask) and returns actions
        num_games: Total number of games to complete
        collect_data: Whether to collect detailed game data

    Returns:
        dict with:
            - 'rewards_p0': list of player 0 rewards
            - 'rewards_p1': list of player 1 rewards
            - 'outcomes': list of outcome codes
            - 'game_data': list of game data dicts (if collect_data=True)
    """
    results = {
        'rewards_p0': [],
        'rewards_p1': [],
        'outcomes': [],
    }
    if collect_data:
        results['game_data'] = []

    games_completed = 0
    obs, info = env.reset()

    while games_completed < num_games:
        # Get actions from policy
        actions = policy_fn(obs, info['action_mask'])

        # Step environment
        obs, rewards, dones, _, info = env.step(actions)

        # Collect results for completed games
        done_indices = np.where(dones)[0]
        for idx in done_indices:
            if games_completed >= num_games:
                break
            results['rewards_p0'].append(rewards[idx, 0])
            results['rewards_p1'].append(rewards[idx, 1])
            results['outcomes'].append(env._outcome[idx])

            if collect_data:
                results['game_data'].append({
                    'player_values': env._player_values[idx].copy(),
                    'outside_options': env._outside_options[idx].copy(),
                    'rewards': rewards[idx].copy(),
                    'outcome': env._outcome[idx],
                })

            games_completed += 1

        # Auto-reset completed games
        env.auto_reset()

    return results
