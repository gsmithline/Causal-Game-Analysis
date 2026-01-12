from dataclasses import dataclass
from rl_training.core.base_config import BaseConfig, TrainingParadigm
from rl_training.core.registry import register_config


@register_config("cfr")
@dataclass
class CFRConfig(BaseConfig):
    """Configuration for CFR algorithms."""

    # CFR-specific parameters
    num_iterations: int = 10_000
    cfr_variant: str = "vanilla"  # "vanilla", "plus", "linear", "discounted"
    alternating_updates: bool = True  # Alternate between players

    # Linear/Discounted CFR parameters
    alpha: float = 1.5   # For discounted CFR
    beta: float = 0.0    # For discounted CFR
    gamma: float = 2.0   # For discounted CFR

    @property
    def paradigm(self) -> TrainingParadigm:
        return TrainingParadigm.TABULAR_ITERATIVE

    @property
    def total_timesteps(self) -> int:
        """CFR measures progress in iterations, not timesteps."""
        return self.num_iterations

    def validate(self) -> None:
        super().validate()
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if self.cfr_variant not in ("vanilla", "plus", "linear", "discounted"):
            raise ValueError(f"Unknown CFR variant: {self.cfr_variant}")
