"""
Configuration for IS-MCTS (Information-Set Monte Carlo Tree Search).
"""

from dataclasses import dataclass
from typing import Tuple

from rl_training.core.base_config import BaseConfig, TrainingParadigm


@dataclass
class ISMCTSConfig(BaseConfig):
    """
    Configuration for IS-MCTS search.

    IS-MCTS is a search algorithm for imperfect information games.
    It can be used to improve any base policy at inference time.
    """

    algorithm_name: str = "is_mcts"

    # Environment
    num_envs: int = 32
    cuda_device: int = 0
    lr: float = 3e-4  # Not used for search, but needed by base

    # Search parameters
    num_simulations: int = 100  # MCTS simulations per move
    c_puct: float = 1.5  # Exploration constant
    temperature: float = 1.0  # Action selection temperature
    dirichlet_alpha: float = 0.3  # Root noise for exploration
    dirichlet_epsilon: float = 0.25  # Weight of root noise

    # Gumbel search parameters
    use_gumbel: bool = True  # Use Gumbel AlphaZero variant
    gumbel_scale: float = 1.0  # Scale for Gumbel noise
    max_num_considered: int = 16  # Max actions to consider in Gumbel

    # Value estimation
    discount: float = 0.99
    use_value_network: bool = True  # Use value network vs rollout

    # Base policy
    base_policy_path: str = ""  # Path to trained policy checkpoint
    network_type: str = "mlp"  # "mlp" or "transformer"
    hidden_dims: Tuple[int, ...] = (256, 256)

    # Parallelization
    num_parallel_envs: int = 32  # Envs for leaf evaluation

    @property
    def paradigm(self) -> TrainingParadigm:
        return TrainingParadigm.ON_POLICY  # Closest match

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "num_simulations": self.num_simulations,
            "c_puct": self.c_puct,
            "temperature": self.temperature,
            "use_gumbel": self.use_gumbel,
        })
        return d
