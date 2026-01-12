from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Any, Dict, Optional
import json


class TrainingParadigm(Enum):
    """Categorizes fundamental training loop patterns."""
    ACTOR_LEARNER = auto()      # IMPALA: async actors, central learner
    ON_POLICY = auto()          # PPO, A2C: collect -> update -> discard
    OFF_POLICY = auto()         # DQN, SAC: replay buffer based
    TABULAR_ITERATIVE = auto()  # CFR: no neural network, iteration-based
    HYBRID = auto()             # NFSP: combines multiple paradigms


@dataclass
class BaseConfig(ABC):
    """
    Base configuration with common hyperparameters shared across algorithms.

    Subclass this for each algorithm, adding algorithm-specific parameters.
    """
    # Training control
    seed: int = 42
    total_timesteps: int = 1_000_000

    # Logging and checkpointing
    log_interval: int = 100
    checkpoint_interval: int = 10_000
    checkpoint_dir: Optional[str] = None

    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.

    # Environment
    env_id: str = ""  # OpenSpiel game name or Gym env ID
    num_envs: int = 1  # For vectorized environments

    @property
    @abstractmethod
    def paradigm(self) -> TrainingParadigm:
        """Each algorithm must declare its paradigm."""
        ...

    def __post_init__(self) -> None:
        """Resolve device and perform validation."""
        self._resolve_device()
        self.validate()

    def _resolve_device(self) -> None:
        """Resolve 'auto' device to concrete device."""
        if self.device == "auto":
            import torch as th
            self.device = "cuda" if th.cuda.is_available() else "cpu"
        elif self.device == "cuda":
            import torch as th
            if th.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"

    def validate(self) -> None:
        """Override to add algorithm-specific validation."""
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (JSON-compatible)."""
        d = asdict(self)
        d["paradigm"] = self.paradigm.name
        d["_config_class"] = self.__class__.__name__
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseConfig":
        """Deserialize from dictionary."""
        d = d.copy()
        d.pop("paradigm", None)
        d.pop("_config_class", None)
        return cls(**d)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseConfig":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
