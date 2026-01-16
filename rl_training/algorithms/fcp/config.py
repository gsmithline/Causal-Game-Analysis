"""
Configuration for FCP (Fictitious Co-Play).
"""

from dataclasses import dataclass
from typing import Tuple

from rl_training.core.base_config import BaseConfig, TrainingParadigm


@dataclass
class FCPConfig(BaseConfig):
    """
    Configuration for Fictitious Co-Play training.

    FCP trains agents by playing against historical snapshots
    of partner policies, encouraging robustness and diversity.
    """

    algorithm_name: str = "fcp"

    # Environment
    num_envs: int = 4096
    cuda_device: int = 0
    lr: float = 3e-4

    # Population settings
    population_size: int = 10  # Max policies in population
    snapshot_interval: int = 10000  # Steps between snapshots
    prioritized_sampling: bool = False  # Prioritize recent policies

    # Sampling distribution
    uniform_prob: float = 0.5  # Probability of uniform vs recent
    recent_window: int = 3  # Window for recent policies

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
    hidden_dims: Tuple[int, ...] = (256, 256)
    network_type: str = "mlp"

    @property
    def paradigm(self) -> TrainingParadigm:
        return TrainingParadigm.ON_POLICY

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "population_size": self.population_size,
            "snapshot_interval": self.snapshot_interval,
            "prioritized_sampling": self.prioritized_sampling,
            "uniform_prob": self.uniform_prob,
        })
        return d
