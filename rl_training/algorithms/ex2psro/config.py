"""
Configuration for Ex²PSRO (Explicit Exploration PSRO).

Based on "Explicit Exploration for High-Welfare Equilibria in
Game-Theoretic Multiagent Reinforcement Learning" (OpenReview 2025).
"""

from dataclasses import dataclass
from typing import Tuple, Literal

from rl_training.core.base_config import BaseConfig, TrainingParadigm


@dataclass
class Ex2PSROConfig(BaseConfig):
    """
    Configuration for Ex²PSRO training.

    Ex²PSRO extends PSRO to find high-welfare equilibria by:
    1. Creating exploration policies that imitate high-welfare behavior
    2. Regularizing best response training toward the exploration policy
    3. Biasing equilibrium selection toward prosocial outcomes
    """

    algorithm_name: str = "ex2psro"

    # Environment
    num_envs: int = 4096
    cuda_device: int = 0
    lr: float = 3e-4

    # Population settings (same as PSRO)
    max_policies: int = 20
    initial_policies: int = 1

    # Meta-game settings
    num_eval_games: int = 1000
    nash_solver: str = "replicator"
    replicator_iterations: int = 10000
    replicator_dt: float = 0.01

    # Best response training (inner loop)
    br_training_steps: int = 100000
    br_rollout_steps: int = 64
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
    network_type: str = "mlp"

    # PSRO iteration settings
    psro_iterations: int = 20
    rectified: bool = False

    # ============================================
    # Ex²PSRO-specific parameters
    # ============================================

    # Welfare function for ranking policies
    welfare_fn: Literal["utilitarian", "nash", "egalitarian"] = "utilitarian"

    # Exploration policy generation
    exploration_top_k: int = 3  # Top-K highest welfare policies for exploration
    exploration_temperature: float = 1.0  # Softmax temperature for welfare-weighted sampling
    use_welfare_weighted_sampling: bool = True  # Weight by welfare vs uniform over top-k

    # KL regularization toward exploration policy
    kl_coef: float = 0.1  # Coefficient for KL divergence regularization
    kl_target: float = 0.01  # Target KL divergence (for adaptive KL)
    use_adaptive_kl: bool = True  # Adapt kl_coef to target KL
    kl_horizon: int = 10  # Window for adaptive KL adjustment

    # Imitation learning for exploration policy
    imitation_epochs: int = 5  # Epochs to train exploration policy
    imitation_batch_size: int = 1024
    imitation_lr: float = 1e-3
    imitation_data_size: int = 10000  # Trajectories to collect for imitation

    # Welfare tracking
    track_welfare_history: bool = True
    welfare_eval_games: int = 500  # Games to evaluate welfare

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
            # Ex²PSRO specific
            "welfare_fn": self.welfare_fn,
            "exploration_top_k": self.exploration_top_k,
            "kl_coef": self.kl_coef,
            "use_adaptive_kl": self.use_adaptive_kl,
        })
        return d
