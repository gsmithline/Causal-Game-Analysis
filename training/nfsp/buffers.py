"""
Replay buffers for NFSP.

Two types of buffers:
1. CircularBuffer: For Q-learning transitions (FIFO)
2. ReservoirBuffer: For average policy samples (uniform sampling from history)
"""

import torch
import numpy as np
from collections import namedtuple, deque
import random


# Transition tuple for Q-learning
TransitionTuple = namedtuple('TransitionTuple', [
    'obs',           # Current observation
    'action_mask',   # Valid actions mask
    'action',        # Action taken
    'reward',        # Reward received
    'next_obs',      # Next observation
    'next_action_mask',  # Next valid actions mask
    'done',          # Whether episode ended
])

# Policy tuple for average policy learning
PolicyTuple = namedtuple('PolicyTuple', [
    'obs',           # Observation
    'action_mask',   # Valid actions mask
    'action',        # Action taken (one-hot or index)
])

# History-based transition tuple for Q-learning
HistoryTransitionTuple = namedtuple('HistoryTransitionTuple', [
    'history',           # History tensor (seq_len, token_dim)
    'seq_len',           # Current sequence length
    'action_mask',       # Valid actions mask
    'action',            # Action taken
    'reward',            # Reward received
    'next_history',      # Next history tensor
    'next_seq_len',      # Next sequence length
    'next_action_mask',  # Next valid actions mask
    'done',              # Whether episode ended
])

# History-based policy tuple for average policy learning
HistoryPolicyTuple = namedtuple('HistoryPolicyTuple', [
    'history',       # History tensor (seq_len, token_dim)
    'seq_len',       # Sequence length
    'action_mask',   # Valid actions mask
    'action',        # Action taken
])


class CircularBuffer:
    """
    Circular (FIFO) replay buffer.

    Used for Q-learning transitions - recent experiences are more relevant.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, item):
        """Add an item to the buffer."""
        self.buffer.append(item)

    def sample(self, batch_size: int):
        """Sample a random batch from the buffer."""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)


class ReservoirBuffer:
    """
    Reservoir sampling buffer.

    Used for average policy learning - maintains uniform sample over all history.
    Each item has equal probability of being in the buffer regardless of when added.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.total_seen = 0

    def push(self, item):
        """Add an item using reservoir sampling."""
        self.total_seen += 1

        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            # Reservoir sampling: replace with probability capacity/total_seen
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = item

    def sample(self, batch_size: int):
        """Sample a random batch from the buffer."""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)


class NFSPBuffers:
    """
    Combined buffer management for NFSP.

    Maintains separate buffers for each player:
    - Q-buffers: CircularBuffer for Q-learning (recent transitions)
    - Policy-buffers: ReservoirBuffer for average policy (uniform over history)
    """

    def __init__(
        self,
        num_players: int = 2,
        q_buffer_size: int = 200_000_000,
        policy_buffer_size: int = 1_000_000_000,
    ):
        self.num_players = num_players

        # Separate buffers for each player
        self.q_buffers = [CircularBuffer(q_buffer_size) for _ in range(num_players)]
        self.policy_buffers = [ReservoirBuffer(policy_buffer_size) for _ in range(num_players)]

    def add_transition(self, player: int, transition: TransitionTuple):
        """Add a transition to the Q-buffer for a player."""
        self.q_buffers[player].push(transition)

    def add_policy_sample(self, player: int, sample: PolicyTuple):
        """Add a policy sample to the policy buffer for a player."""
        self.policy_buffers[player].push(sample)

    def sample_transitions(self, player: int, batch_size: int):
        """Sample transitions for Q-learning."""
        return self.q_buffers[player].sample(batch_size)

    def sample_policy(self, player: int, batch_size: int):
        """Sample policy data for average policy learning."""
        return self.policy_buffers[player].sample(batch_size)

    def q_buffer_size(self, player: int):
        return len(self.q_buffers[player])

    def policy_buffer_size(self, player: int):
        return len(self.policy_buffers[player])


class NFSPBuffersHistory:
    """
    Combined buffer management for history-based NFSP.

    Maintains separate buffers for each player using history sequences.
    """

    def __init__(
        self,
        num_players: int = 2,
        q_buffer_size: int = 500_000,
        policy_buffer_size: int = 2_000_000,
    ):
        self.num_players = num_players

        # Separate buffers for each player
        self.q_buffers = [CircularBuffer(q_buffer_size) for _ in range(num_players)]
        self.policy_buffers = [ReservoirBuffer(policy_buffer_size) for _ in range(num_players)]

    def add_transition(self, player: int, transition: HistoryTransitionTuple):
        """Add a history-based transition to the Q-buffer."""
        self.q_buffers[player].push(transition)

    def add_policy_sample(self, player: int, sample: HistoryPolicyTuple):
        """Add a history-based policy sample to the policy buffer."""
        self.policy_buffers[player].push(sample)

    def sample_transitions(self, player: int, batch_size: int):
        """Sample history-based transitions for Q-learning."""
        return self.q_buffers[player].sample(batch_size)

    def sample_policy(self, player: int, batch_size: int):
        """Sample history-based policy data for average policy learning."""
        return self.policy_buffers[player].sample(batch_size)

    def q_buffer_size(self, player: int):
        return len(self.q_buffers[player])

    def policy_buffer_size(self, player: int):
        return len(self.policy_buffers[player])
