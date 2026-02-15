"""
Magnetic Mirror Descent (MMD) training module.

MMD = PPO + KL penalty to a magnet (reference) distribution.
Reference: https://arxiv.org/abs/2206.05825
"""

from .policy import HistoryTransformerPolicy

__all__ = [
    "train_mmd",
    "train_mmd_scheduled",
    "HistoryTransformerPolicy",
]


def __getattr__(name):
    """Lazy-import training functions that depend on cuda_bargain."""
    if name in ("train_mmd", "train_mmd_scheduled"):
        from .train import train_mmd, train_mmd_scheduled
        return {"train_mmd": train_mmd, "train_mmd_scheduled": train_mmd_scheduled}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
