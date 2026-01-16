"""
Configuration for PSRO (Policy Space Response Oracles).
"""

from dataclasses import dataclass
from typing import Tuple

from rl_training.core.base_config import BaseConfig, TrainingParadigm


@dataclass
class PSROConfig(BaseConfig):
    """
    Configuration for PSRO training.

    PSRO iteratively builds a population of policies by:
    1. Computing Nash equilibrium over current population
    2. Training best-response policies against Nash mixture
    3. Adding new policies to population
    """

    algorithm_name: str = "psro"

    # Environment
    num_envs: int = 4096
    cuda_device: int = 0
    lr: float = 3e-4

    # Population settings
    max_policies: int = 20  # Maximum number of policies per player
    initial_policies: int = 1  # Number of random initial policies

    # Meta-game settings
    num_eval_games: int = 1000  # Games to estimate payoff matrix entries
    nash_solver: str = "replicator"  # "replicator", "linear_program", "fictitious_play"
    replicator_iterations: int = 10000  # Iterations for replicator dynamics
    replicator_dt: float = 0.01  # Step size for replicator dynamics

    # Best response training (inner loop)
    br_training_steps: int = 100000  # Steps to train each BR policy
    br_rollout_steps: int = 64  # Rollout length for BR training
    br_ppo_epochs: int = 4
    br_minibatch_size: int = 2048
    br_clip_eps: float = 0.2
    br_gamma: float = 0.99
    br_gae_lambda: float = 0.95
    br_entropy_coef: float = 0.01
    br_value_coef: float = 0.5
    br_max_grad_norm: float = 0.5

    # Network architecture
    hidden_dims: Tuple[int, ...] = (256, 256)
    network_type: str = "mlp"  # "mlp" or "transformer"

    # PSRO-specific
    psro_iterations: int = 20  # Number of PSRO iterations
    rectified: bool = False  # Use rectified Nash (PSRO-rN)

    @property
    def paradigm(self) -> TrainingParadigm:
        return TrainingParadigm.ON_POLICY

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "max_policies": self.max_policies,
            "initial_policies": self.initial_policies,
            "num_eval_games": self.num_eval_games,
            "nash_solver": self.nash_solver,
            "br_training_steps": self.br_training_steps,
            "psro_iterations": self.psro_iterations,
            "rectified": self.rectified,
        })
        return d
