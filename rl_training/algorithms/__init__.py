# Algorithm implementations
# Import algorithms to register them with the registry

# Existing algorithms
from rl_training.algorithms import impala
from rl_training.algorithms import cfr
from rl_training.algorithms import nfsp

# Bargaining game algorithms
from rl_training.algorithms import ppo_bargain
from rl_training.algorithms import nfsp_bargain
from rl_training.algorithms import sampled_cfr

# New algorithms from meta-game evaluation paper
from rl_training.algorithms import psro  # Policy Space Response Oracles
from rl_training.algorithms import mappo  # Multi-Agent PPO
from rl_training.algorithms import fcp  # Fictitious Co-Play
from rl_training.algorithms import is_mcts  # Information-Set MCTS (search)
