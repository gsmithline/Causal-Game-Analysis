"""
NFSP configuration for the CUDA bargaining game.
"""

from dataclasses import dataclass
from typing import Tuple

from rl_training.core.base_config import BaseConfig, TrainingParadigm
from rl_training.core.registry import register_config


@register_config("nfsp_bargain")
@dataclass
class NFSPBargainConfig(BaseConfig):
    """
    NFSP configuration for CUDA bargaining game.

    Adapts Neural Fictitious Self-Play for vectorized GPU environments.
    """

    # Environment
    num_envs: int = 4096
    cuda_device: int = 0

    # Network architecture
    hidden_dims: Tuple[int, ...] = (256, 256)

    # Best Response (DQN) parameters
    br_lr: float = 0.1
    br_buffer_size: int = 200_000
    br_batch_size: int = 128
    br_target_update_freq: int = 300
    discount: float = 1.0  # Usually 1.0 for games

    # Epsilon decay for BR exploration
    epsilon_start: float = 0.06
    epsilon_end: float = 0.001
    epsilon_decay_steps: int = 2_000_000

    # Average Policy parameters
    avg_lr: float = 0.01
    avg_buffer_size: int = 2_000_000
    avg_batch_size: int = 128
    avg_train_freq: int = 64  # Train avg policy every N steps

    # Mixing parameter (probability of using BR vs average)
    eta: float = 0.1

    # Training
    total_timesteps: int = 5_000_000

    @property
    def paradigm(self) -> TrainingParadigm:
        return TrainingParadigm.HYBRID

    def validate(self) -> None:
        super().validate()
        if not 0 <= self.eta <= 1:
            raise ValueError("eta must be between 0 and 1")
