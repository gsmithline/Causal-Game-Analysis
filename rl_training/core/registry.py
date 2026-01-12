from typing import Dict, Type
from rl_training.core.base_trainer import BaseTrainer
from rl_training.core.base_config import BaseConfig


_TRAINERS: Dict[str, Type[BaseTrainer]] = {}
_CONFIGS: Dict[str, Type[BaseConfig]] = {}


def register_algorithm(name: str):
    """Decorator to register algorithm trainer."""
    def decorator(trainer_cls: Type[BaseTrainer]):
        _TRAINERS[name] = trainer_cls
        return trainer_cls
    return decorator


def register_config(name: str):
    """Decorator to register config class."""
    def decorator(config_cls: Type[BaseConfig]):
        _CONFIGS[name] = config_cls
        return config_cls
    return decorator


def get_trainer(name: str) -> Type[BaseTrainer]:
    """Get trainer class by algorithm name."""
    if name not in _TRAINERS:
        available = ", ".join(_TRAINERS.keys()) if _TRAINERS else "none"
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")
    return _TRAINERS[name]


def get_config(name: str) -> Type[BaseConfig]:
    """Get config class by algorithm name."""
    if name not in _CONFIGS:
        available = ", ".join(_CONFIGS.keys()) if _CONFIGS else "none"
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    return _CONFIGS[name]


def list_algorithms() -> list:
    """List all registered algorithms."""
    return list(_TRAINERS.keys())
