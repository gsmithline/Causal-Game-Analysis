from __future__ import annotations
from typing import Any, Dict, List
import numpy as np


class ReplayBuffer:
    """
    Standard circular replay buffer for off-policy learning.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._storage: List[Dict[str, Any]] = []
        self._position = 0

    def add(self, transition: Dict[str, Any]) -> None:
        """Add transition to buffer."""
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._position] = transition
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of transitions."""
        indices = np.random.randint(0, len(self._storage), size=batch_size)
        batch = [self._storage[i] for i in indices]

        # Stack into arrays
        return {
            key: np.array([t[key] for t in batch])
            for key in batch[0].keys()
        }

    def __len__(self) -> int:
        return len(self._storage)

    def clear(self) -> None:
        """Clear the buffer."""
        self._storage.clear()
        self._position = 0


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ):
        super().__init__(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._max_priority = 1.0
        self._frame = 0

    def add(self, transition: Dict[str, Any]) -> None:
        """Add with max priority."""
        self._priorities[self._position] = self._max_priority
        super().add(transition)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample with priority-based probability."""
        self._frame += 1

        # Compute probabilities
        priorities = self._priorities[:len(self._storage)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample
        indices = np.random.choice(
            len(self._storage), batch_size, p=probs, replace=False
        )
        batch = [self._storage[i] for i in indices]

        # Compute importance weights
        beta = min(
            1.0,
            self.beta_start + self._frame * (1.0 - self.beta_start) / self.beta_frames
        )
        weights = (len(self._storage) * probs[indices]) ** (-beta)
        weights /= weights.max()

        result = {
            key: np.array([t[key] for t in batch])
            for key in batch[0].keys()
        }
        result["indices"] = indices
        result["weights"] = weights

        return result

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self._priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)


class ReservoirBuffer:
    """
    Reservoir sampling buffer for NFSP average policy training.

    Maintains a uniform sample of all data seen so far.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._storage: List[Dict[str, Any]] = []
        self._count = 0

    def add(self, item: Dict[str, Any]) -> None:
        """Add item using reservoir sampling."""
        self._count += 1
        if len(self._storage) < self.capacity:
            self._storage.append(item)
        else:
            # Replace with probability capacity/count
            idx = np.random.randint(0, self._count)
            if idx < self.capacity:
                self._storage[idx] = item

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch uniformly."""
        indices = np.random.randint(0, len(self._storage), size=batch_size)
        batch = [self._storage[i] for i in indices]

        return {
            key: np.array([t[key] for t in batch])
            for key in batch[0].keys()
        }

    def __len__(self) -> int:
        return len(self._storage)

    def clear(self) -> None:
        """Clear the buffer."""
        self._storage.clear()
        self._count = 0
