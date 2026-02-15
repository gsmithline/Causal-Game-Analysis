"""
MAPPO (Multi-Agent PPO) training module.

MAPPO uses Centralized Training with Decentralized Execution (CTDE):
- Decentralized actors: Each agent only sees its own observation
- Centralized critic: Value function sees joint state from all agents

This provides better credit assignment in multi-agent settings while
maintaining decentralized execution at inference time.
"""

from .policy import MAPPOPolicy, MAPPOCritic, MAPPOHistoryPolicy

__all__ = [
    "MAPPOPolicy",
    "MAPPOCritic",
    "MAPPOHistoryPolicy",
]

# Training functions require cuda_bargain, import conditionally
try:
    from .train import train_mappo, train_mappo_history
    __all__.extend(["train_mappo", "train_mappo_history"])
except ImportError:
    pass
