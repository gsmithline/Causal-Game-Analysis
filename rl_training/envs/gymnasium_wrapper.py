from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np

from rl_training.envs.env_wrapper import BaseEnvWrapper, StepResult


class GymnasiumWrapper(BaseEnvWrapper):
    """
    Wrapper that adapts Gymnasium environments to unified interface.

    Usage:
        wrapper = GymnasiumWrapper("CartPole-v1")
        obs, info = wrapper.reset()
        result = wrapper.step(action)
    """

    def __init__(self, env_id: str, **env_kwargs):
        super().__init__()
        import gymnasium as gym

        self.env_id = env_id
        self.env = gym.make(env_id, **env_kwargs)

        # Extract properties
        self._num_players = 1

        # Handle observation space
        obs_space = self.env.observation_space
        if hasattr(obs_space, 'shape'):
            self._observation_shape = obs_space.shape
        else:
            self._observation_shape = (obs_space.n,)

        # Handle action space
        act_space = self.env.action_space
        if hasattr(act_space, 'n'):
            self._num_actions = act_space.n
        elif hasattr(act_space, 'shape'):
            self._num_actions = act_space.shape[0]
        else:
            self._num_actions = 1

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        obs, info = self.env.reset(seed=seed)
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action: Any) -> StepResult:
        """Take action in environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        return StepResult(
            observation=np.asarray(obs, dtype=np.float32),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    @property
    def unwrapped(self):
        """Access underlying gymnasium environment."""
        return self.env
