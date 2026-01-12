from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable
import numpy as np
from numpy.typing import NDArray


@dataclass
class StepResult:
    """Unified step result across environment types."""
    observation: Any                          # Next observation
    reward: Union[float, NDArray]             # Reward(s)
    terminated: bool                          # Episode ended naturally
    truncated: bool                           # Episode ended by time limit
    info: Dict[str, Any]                      # Additional info

    @property
    def done(self) -> bool:
        """Convenience property for done = terminated or truncated."""
        return self.terminated or self.truncated


@dataclass
class MultiAgentStepResult:
    """Step result for multi-agent environments."""
    observations: Dict[int, Any]              # {player_id: observation}
    rewards: Dict[int, float]                 # {player_id: reward}
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    current_player: int                       # Whose turn it is (-1 if terminal)


@runtime_checkable
class EnvWrapper(Protocol):
    """
    Unified environment interface protocol.

    Supports both single-agent and multi-agent environments.
    """

    @property
    def num_players(self) -> int:
        """Number of players (1 for single-agent)."""
        ...

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Shape of observations."""
        ...

    @property
    def num_actions(self) -> int:
        """Number of discrete actions (or action space dimension)."""
        ...

    @property
    def is_multi_agent(self) -> bool:
        """Whether this is a multi-agent environment."""
        ...

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment, return (observation, info)."""
        ...

    def step(self, action: Any) -> StepResult:
        """Take action, return StepResult."""
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...


class BaseEnvWrapper(ABC):
    """
    Abstract base class for environment wrappers.

    Provides common utilities and enforces interface.
    """

    def __init__(self):
        self._num_players: int = 1
        self._observation_shape: Tuple[int, ...] = ()
        self._num_actions: int = 0

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._observation_shape

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def is_multi_agent(self) -> bool:
        return self._num_players > 1

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        ...

    @abstractmethod
    def step(self, action: Any) -> StepResult:
        ...

    def close(self) -> None:
        pass
