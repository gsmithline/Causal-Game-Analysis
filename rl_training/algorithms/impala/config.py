from dataclasses import dataclass
from typing import Optional
from rl_training.core.base_config import BaseConfig, TrainingParadigm
from rl_training.core.registry import register_config


@register_config("impala")
@dataclass
class ImpalaConfig(BaseConfig):
    """IMPALA-specific configuration."""

    # V-trace parameters
    unroll_length: int = 80
    batch_size: int = 4  # How many rollouts per learner step
    discount: float = 0.99
    clip_rho_threshold: float = 1.0
    clip_c_threshold: float = 1.0

    # Optimization
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 40.0

    # Distributed training
    actor_queue_size: int = 64
    num_actors: int = 8
    refresh_interval: int = 50_000  # Env steps before actor refreshes weights

    # Device configuration (override base)
    learner_device: str = "auto"
    actor_device: str = "cpu"

    @property
    def paradigm(self) -> TrainingParadigm:
        return TrainingParadigm.ACTOR_LEARNER

    def __post_init__(self) -> None:
        # Use learner_device as the main device
        self.device = self.learner_device
        super().__post_init__()
        self.learner_device = self.device

    def validate(self) -> None:
        super().validate()
        if self.unroll_length <= 0:
            raise ValueError("unroll_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_actors <= 0:
            raise ValueError("num_actors must be positive")
