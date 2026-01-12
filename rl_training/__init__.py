"""
RL Training Framework

A unified framework for training RL agents with multiple algorithm families:
- Deep RL: IMPALA, PPO, DQN, etc.
- Game-theoretic: CFR, NFSP

Example usage:

    # CFR for poker
    from rl_training.algorithms.cfr import CFRTrainer, CFRConfig

    config = CFRConfig(env_id="kuhn_poker", num_iterations=10000)
    trainer = CFRTrainer(config)
    trainer.train()

    # IMPALA for larger environments
    from rl_training.algorithms.impala import ImpalaTrainer, ImpalaConfig
    from rl_training.envs import OpenSpielWrapper

    config = ImpalaConfig(num_actors=8)
    trainer = ImpalaTrainer(
        config=config,
        env_fn=lambda: OpenSpielWrapper("leduc_poker"),
        policy_fn=make_policy,
    )
    trainer.train()
"""

# Core abstractions
from rl_training.core import (
    BaseConfig,
    TrainingParadigm,
    BaseTrainer,
    PolicyProtocol,
    NeuralPolicyProtocol,
    TabularPolicyProtocol,
    BaseNeuralPolicy,
    BaseTabularPolicy,
    register_algorithm,
    register_config,
    get_trainer,
    get_config,
    list_algorithms,
)

# Environment wrappers
from rl_training.envs import (
    StepResult,
    MultiAgentStepResult,
    EnvWrapper,
    BaseEnvWrapper,
    OpenSpielWrapper,
    GymnasiumWrapper,
)

# Utilities
from rl_training.utils import (
    Logger,
    ConsoleLogger,
    TensorBoardLogger,
    CompositeLogger,
    CheckpointManager,
    ReplayBuffer,
    PrioritizedReplayBuffer,
)

# Networks
from rl_training.networks import SimpleCategoricalMLP

# Import algorithms to register them
from rl_training import algorithms

__all__ = [
    # Core
    "BaseConfig",
    "TrainingParadigm",
    "BaseTrainer",
    "PolicyProtocol",
    "NeuralPolicyProtocol",
    "TabularPolicyProtocol",
    "BaseNeuralPolicy",
    "BaseTabularPolicy",
    "register_algorithm",
    "register_config",
    "get_trainer",
    "get_config",
    "list_algorithms",
    # Environments
    "StepResult",
    "MultiAgentStepResult",
    "EnvWrapper",
    "BaseEnvWrapper",
    "OpenSpielWrapper",
    "GymnasiumWrapper",
    # Utilities
    "Logger",
    "ConsoleLogger",
    "TensorBoardLogger",
    "CompositeLogger",
    "CheckpointManager",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    # Networks
    "SimpleCategoricalMLP",
]
