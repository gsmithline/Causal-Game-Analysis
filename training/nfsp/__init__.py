"""
NFSP (Neural Fictitious Self-Play) training module.

NFSP combines:
1. Best Response learning via DQN (Q-network)
2. Average Policy learning via supervised learning (Policy network)

The average policy converges to an approximate Nash equilibrium.

Supports both MLP and History Transformer architectures.
"""

from .policy import (
    QNetwork,
    AveragePolicyNetwork,
    NFSPAgent,
    QNetworkHistory,
    AveragePolicyNetworkHistory,
    NFSPAgentHistory,
)
from .buffers import (
    NFSPBuffers,
    TransitionTuple,
    PolicyTuple,
    CircularBuffer,
    ReservoirBuffer,
    NFSPBuffersHistory,
    HistoryTransitionTuple,
    HistoryPolicyTuple,
)

__all__ = [
    # MLP networks
    "QNetwork",
    "AveragePolicyNetwork",
    "NFSPAgent",
    # History Transformer networks
    "QNetworkHistory",
    "AveragePolicyNetworkHistory",
    "NFSPAgentHistory",
    # MLP buffers
    "NFSPBuffers",
    "TransitionTuple",
    "PolicyTuple",
    "CircularBuffer",
    "ReservoirBuffer",
    # History buffers
    "NFSPBuffersHistory",
    "HistoryTransitionTuple",
    "HistoryPolicyTuple",
]

# Training functions require cuda_bargain, import conditionally
try:
    from .train import train_nfsp, train_nfsp_history
    __all__.extend(["train_nfsp", "train_nfsp_history"])
except ImportError:
    pass
