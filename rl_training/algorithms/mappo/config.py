"""
Configuration for MAPPO (Multi-Agent PPO).
"""

from dataclasses import dataclass
from typing import Tuple

from rl_training.core.base_config import BaseConfig, TrainingParadigm


@dataclass
class MAPPOConfig(BaseConfig):
    """
    Configuration for MAPPO training.

    MAPPO uses centralized training with decentralized execution:
    - Each agent has its own actor network (decentralized)
    - Shared critic network observes global state (centralized)
    """

    algorithm_name: str = "mappo"

    # Environment
    num_envs: int = 4096
    cuda_device: int = 0
    lr: float = 3e-4

    # PPO hyperparameters
    rollout_steps: int = 64
    ppo_epochs: int = 4
    minibatch_size: int = 2048
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network architecture
    actor_hidden_dims: Tuple[int, ...] = (256, 256)
    critic_hidden_dims: Tuple[int, ...] = (256, 256)
    share_actor: bool = False  # Share actor weights between players

    # Centralized critic
    use_centralized_critic: bool = True  # Use global state for critic
    critic_input_dim: int = 184  # 92 * 2 for both players' obs

    # Value normalization
    use_value_norm: bool = True
    value_norm_clip: float = 10.0

    # Advantage normalization
    use_advantage_norm: bool = True

    @property
    def paradigm(self) -> TrainingParadigm:
        return TrainingParadigm.ON_POLICY

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "rollout_steps": self.rollout_steps,
            "ppo_epochs": self.ppo_epochs,
            "clip_eps": self.clip_eps,
            "share_actor": self.share_actor,
            "use_centralized_critic": self.use_centralized_critic,
        })
        return d
