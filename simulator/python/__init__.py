# CPU implementation (always available)
from .bargain_env_cpu import BargainEnvCPU, run_games_with_policy

# CUDA implementation (requires GPU and built extension)
try:
    from .bargain_env import BargainEnv
    CUDA_AVAILABLE = True
except ImportError:
    # CUDA not available, alias CPU version
    BargainEnv = BargainEnvCPU
    CUDA_AVAILABLE = False

# Constants
try:
    from . import cuda_bargain_core
    NUM_ACTIONS = cuda_bargain_core.NUM_ACTIONS
    OBS_DIM = cuda_bargain_core.OBS_DIM
    ACTION_ACCEPT = cuda_bargain_core.ACTION_ACCEPT
    ACTION_WALK = cuda_bargain_core.ACTION_WALK
    NUM_ITEM_TYPES = cuda_bargain_core.NUM_ITEM_TYPES
    MAX_ROUNDS = cuda_bargain_core.MAX_ROUNDS
    ITEM_QUANTITIES = cuda_bargain_core.ITEM_QUANTITIES
except ImportError:
    # Fallback values if CUDA module not built
    NUM_ACTIONS = 82
    OBS_DIM = 92
    ACTION_ACCEPT = 80
    ACTION_WALK = 81
    NUM_ITEM_TYPES = 3
    MAX_ROUNDS = 3
    ITEM_QUANTITIES = (7, 4, 1)

__all__ = [
    'BargainEnv',
    'BargainEnvCPU',
    'run_games_with_policy',
    'CUDA_AVAILABLE',
    'NUM_ACTIONS',
    'OBS_DIM',
    'ACTION_ACCEPT',
    'ACTION_WALK',
    'NUM_ITEM_TYPES',
    'MAX_ROUNDS',
    'ITEM_QUANTITIES',
]
