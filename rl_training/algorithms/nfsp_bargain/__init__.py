"""NFSP algorithm adapted for CUDA bargaining game."""

from rl_training.algorithms.nfsp_bargain.config import NFSPBargainConfig
from rl_training.algorithms.nfsp_bargain.trainer import NFSPBargainTrainer

__all__ = ["NFSPBargainConfig", "NFSPBargainTrainer"]
