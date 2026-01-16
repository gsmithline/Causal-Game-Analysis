"""PPO algorithm for CUDA bargaining game self-play."""

from rl_training.algorithms.ppo_bargain.config import PPOBargainConfig
from rl_training.algorithms.ppo_bargain.trainer import PPOBargainTrainer

__all__ = ["PPOBargainConfig", "PPOBargainTrainer"]
