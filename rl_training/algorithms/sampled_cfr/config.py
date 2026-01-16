"""
Sampled CFR configuration for the bargaining game.

Uses Deep CFR approach with neural network function approximation.
"""

from dataclasses import dataclass
from typing import Tuple

from rl_training.core.base_config import BaseConfig, TrainingParadigm
from rl_training.core.registry import register_config


@register_config("sampled_cfr")
@dataclass
class SampledCFRConfig(BaseConfig):
    """
    Deep CFR / Sampled CFR configuration for bargaining game.

    Since the bargaining game has continuous private values,
    we cannot enumerate all information states. Instead, we
    sample trajectories and train neural networks to approximate
    regrets and strategies.
    """

    # Environment
    num_envs: int = 4096
    cuda_device: int = 0

    # Network architecture
    hidden_dims: Tuple[int, ...] = (256, 256)

    # Training parameters
    lr: float = 1e-3
    advantage_batch_size: int = 256
    strategy_batch_size: int = 256

    # Memory parameters
    advantage_memory_size: int = 1_000_000
    strategy_memory_size: int = 1_000_000

    # CFR parameters
    num_traversals: int = 100  # Traversals per iteration
    strategy_train_freq: int = 10  # Train strategy every N iterations

    # Total iterations
    total_timesteps: int = 1000  # Here this means iterations

    @property
    def paradigm(self) -> TrainingParadigm:
        return TrainingParadigm.TABULAR_ITERATIVE

    def validate(self) -> None:
        super().validate()
        if self.num_traversals <= 0:
            raise ValueError("num_traversals must be positive")
