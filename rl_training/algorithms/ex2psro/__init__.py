"""
Ex²PSRO (Explicit Exploration PSRO) algorithm.

Based on "Explicit Exploration for High-Welfare Equilibria in
Game-Theoretic Multiagent Reinforcement Learning" (OpenReview 2025).
"""

from rl_training.algorithms.ex2psro.config import Ex2PSROConfig
from rl_training.algorithms.ex2psro.trainer import Ex2PSROTrainer

__all__ = ["Ex2PSROConfig", "Ex2PSROTrainer"]
