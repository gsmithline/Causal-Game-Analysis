"""
PPO configuration for the CUDA bargaining game.
"""

from dataclasses import dataclass
from typing import Tuple

from rl_training.core.base_config import BaseConfig, TrainingParadigm
from rl_training.core.registry import register_config


@register_config("ppo_bargain")
@dataclass
class PPOBargainConfig(BaseConfig):
    """
    PPO configuration for CUDA bargaining game self-play training.

    Inherits common parameters from BaseConfig and adds PPO-specific ones.
    """

    # Environment (override defaults)
    num_envs: int = 4096          # Number of parallel environments (CUDA)
    cuda_device: int = 0          # CUDA device ID

    # Network architecture
    network_type: str = "transformer"  # "transformer" or "mlp"
    d_model: int = 128                 # Transformer model dimension
    nhead: int = 4                     # Transformer attention heads
    num_layers: int = 2                # Transformer encoder layers
    hidden_dims: Tuple[int, ...] = (256, 256)  # MLP hidden dimensions

    # PPO hyperparameters
    lr: float = 3e-4              # Learning rate
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda for advantage estimation
    clip_eps: float = 0.2         # PPO clipping parameter
    entropy_coef: float = 0.01    # Entropy bonus coefficient
    value_coef: float = 0.5       # Value loss coefficient
    max_grad_norm: float = 0.5    # Gradient clipping norm

    # Rollout parameters
    rollout_steps: int = 64       # Steps per rollout before update
    ppo_epochs: int = 4           # Epochs per PPO update
    minibatch_size: int = 512     # Minibatch size for PPO updates

    # Training
    total_timesteps: int = 10_000_000  # Total environment steps

    @property
    def paradigm(self) -> TrainingParadigm:
        return TrainingParadigm.ON_POLICY

    def validate(self) -> None:
        super().validate()

        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if self.clip_eps <= 0:
            raise ValueError("clip_eps must be positive")
        if self.network_type not in ("transformer", "mlp"):
            raise ValueError(f"Unknown network_type: {self.network_type}")
