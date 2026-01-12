from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable
import torch as th
from torch import nn
import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class PolicyProtocol(Protocol):
    """
    Minimal protocol for any policy (neural or tabular).

    Enables duck-typing across different policy implementations.
    """

    def get_action(
        self,
        observation: Any,
        deterministic: bool = False
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Select action given observation.

        Args:
            observation: Environment observation.
            deterministic: If True, select greedy action.

        Returns:
            (action, info_dict) where info_dict contains policy outputs
            like log_prob, value, etc.
        """
        ...

    def get_action_probs(self, observation: Any) -> Any:
        """Return action probabilities/distribution for observation."""
        ...


@runtime_checkable
class NeuralPolicyProtocol(PolicyProtocol, Protocol):
    """Protocol for neural network-based policies."""

    def state_dict(self) -> Dict[str, th.Tensor]:
        """Return network parameters."""
        ...

    def load_state_dict(self, state_dict: Dict[str, th.Tensor]) -> None:
        """Load network parameters."""
        ...

    def to(self, device: Union[str, th.device]) -> "NeuralPolicyProtocol":
        """Move to device."""
        ...

    def parameters(self):
        """Return parameters for optimizer."""
        ...


@runtime_checkable
class TabularPolicyProtocol(PolicyProtocol, Protocol):
    """Protocol for tabular policies (like CFR)."""

    def get_strategy(self, info_state: str) -> NDArray[np.floating]:
        """Return strategy (action distribution) for information state."""
        ...

    def update_strategy(self, info_state: str, strategy: NDArray[np.floating]) -> None:
        """Update strategy for information state."""
        ...


class BaseNeuralPolicy(nn.Module, ABC):
    """
    Base class for neural network policies.

    Provides common interface expected by trainers.
    """

    @abstractmethod
    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False
    ) -> Tuple[th.distributions.Distribution, th.Tensor, Dict[str, th.Tensor]]:
        """
        Forward pass.

        Args:
            obs: Observation tensor [B, *obs_shape]
            deterministic: If True, return mode of distribution

        Returns:
            (action_distribution, value, extras_dict)
        """
        ...

    def get_action(
        self,
        observation: Union[np.ndarray, th.Tensor],
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get action for a single observation (inference mode).
        """
        was_numpy = isinstance(observation, np.ndarray)
        if was_numpy:
            observation = th.as_tensor(observation, dtype=th.float32)

        # Add batch dimension if needed
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)

        # Move to same device as model
        device = next(self.parameters()).device
        observation = observation.to(device)

        with th.no_grad():
            dist, value, extras = self.forward(observation, deterministic=deterministic)

            if deterministic:
                action = dist.mode if hasattr(dist, 'mode') else dist.mean
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            if log_prob.ndim > 1:
                log_prob = log_prob.sum(dim=-1)

        info = {
            "log_prob": log_prob.squeeze(0).cpu().numpy(),
            "value": value.squeeze(0).cpu().numpy(),
            **{k: v.squeeze(0).cpu().numpy() for k, v in extras.items()}
        }

        action_np = action.squeeze(0).cpu().numpy()
        return action_np, info

    def get_action_probs(self, observation: th.Tensor) -> th.Tensor:
        """Get action probabilities (for discrete actions)."""
        with th.no_grad():
            dist, _, _ = self.forward(observation)
            if hasattr(dist, 'probs'):
                return dist.probs
            else:
                raise NotImplementedError("get_action_probs requires discrete distribution")


class BaseTabularPolicy(ABC):
    """
    Base class for tabular policies (CFR, etc.).

    Stores strategies indexed by information state strings.
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self._strategies: Dict[str, NDArray[np.floating]] = {}

    def get_strategy(self, info_state: str) -> NDArray[np.floating]:
        """Get current strategy for info state (uniform if unseen)."""
        if info_state not in self._strategies:
            return np.ones(self.num_actions, dtype=np.float64) / self.num_actions
        return self._strategies[info_state]

    def update_strategy(self, info_state: str, strategy: NDArray[np.floating]) -> None:
        """Update strategy for info state."""
        self._strategies[info_state] = strategy.copy()

    def get_action(
        self,
        observation: str,  # Info state string for tabular
        deterministic: bool = False
    ) -> Tuple[int, Dict[str, Any]]:
        """Sample action from strategy."""
        strategy = self.get_strategy(observation)

        if deterministic:
            action = int(np.argmax(strategy))
        else:
            action = int(np.random.choice(len(strategy), p=strategy))

        return action, {"strategy": strategy}

    def get_action_probs(self, observation: str) -> NDArray[np.floating]:
        """Return action probabilities."""
        return self.get_strategy(observation)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {"strategies": self._strategies.copy()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load from dictionary."""
        self._strategies = state_dict["strategies"].copy()

    @property
    def info_states(self) -> list:
        """List all known info states."""
        return list(self._strategies.keys())
