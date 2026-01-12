from dataclasses import dataclass
from typing import Tuple
from rl_training.core.base_config import BaseConfig, TrainingParadigm
from rl_training.core.registry import register_config


@register_config("nfsp")
@dataclass
class NFSPConfig(BaseConfig):
    """Configuration for Neural Fictitious Self-Play."""

    # Network architecture
    hidden_dims: Tuple[int, ...] = (128, 128)

    # Best response (DQN) parameters
    br_lr: float = 0.1
    br_buffer_size: int = 200_000
    br_batch_size: int = 128
    br_target_update_freq: int = 300
    epsilon_start: float = 0.06
    epsilon_end: float = 0.001
    epsilon_decay_steps: int = 2_000_000

    # Average policy (supervised learning) parameters
    avg_lr: float = 0.01
    avg_buffer_size: int = 2_000_000
    avg_batch_size: int = 128
    avg_train_freq: int = 64

    # Anticipatory parameter (probability of best response)
    eta: float = 0.1  # Lower = more exploitation

    # Discount factor
    discount: float = 1.0  # Usually 1.0 for games

    @property
    def paradigm(self) -> TrainingParadigm:
        return TrainingParadigm.HYBRID

    def validate(self) -> None:
        super().validate()
        if not 0 <= self.eta <= 1:
            raise ValueError("eta must be in [0, 1]")
        if self.br_buffer_size <= 0:
            raise ValueError("br_buffer_size must be positive")
        if self.avg_buffer_size <= 0:
            raise ValueError("avg_buffer_size must be positive")
