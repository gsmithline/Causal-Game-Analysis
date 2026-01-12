from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, TYPE_CHECKING
import signal

from rl_training.core.base_config import BaseConfig, TrainingParadigm

if TYPE_CHECKING:
    from rl_training.utils.logging import Logger
    from rl_training.utils.checkpointing import CheckpointManager

C = TypeVar("C", bound=BaseConfig)


class BaseTrainer(ABC, Generic[C]):
    """
    Abstract base class for all RL trainers.

    Provides common infrastructure:
    - Configuration management
    - Logging integration
    - Checkpointing
    - Graceful shutdown handling
    - Training loop skeleton

    Subclasses implement algorithm-specific:
    - _setup(): Initialize algorithm components
    - _train_step(): Single training iteration
    - _get_checkpoint_data(): Data to save in checkpoint
    - _load_checkpoint_data(): Restore from checkpoint
    """

    def __init__(
        self,
        config: C,
        env_fn: Optional[Callable] = None,
        logger: Optional["Logger"] = None,
    ):
        self.config = config
        self.env_fn = env_fn

        # Lazy import to avoid circular dependencies
        if logger is None:
            from rl_training.utils.logging import ConsoleLogger
            self.logger = ConsoleLogger(log_interval=config.log_interval)
        else:
            self.logger = logger

        # Checkpoint manager (initialized if checkpoint_dir is set)
        self.checkpoint_manager: Optional["CheckpointManager"] = None
        if config.checkpoint_dir:
            from rl_training.utils.checkpointing import CheckpointManager
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=config.checkpoint_dir,
                max_to_keep=5,
            )

        # Training state
        self.global_step: int = 0
        self.episodes_completed: int = 0
        self._shutdown_requested: bool = False
        self._is_setup: bool = False

    @property
    def paradigm(self) -> TrainingParadigm:
        """Return the training paradigm from config."""
        return self.config.paradigm

    # === Abstract Methods (must implement) ===

    @abstractmethod
    def _setup(self) -> None:
        """
        Initialize algorithm-specific components.

        Called once before training starts. Should create:
        - Environments
        - Networks/policies
        - Optimizers
        - Buffers
        - Workers (for distributed algorithms)
        """
        ...

    @abstractmethod
    def _train_step(self) -> Dict[str, Any]:
        """
        Execute one training step/iteration.

        The definition of "step" varies by paradigm:
        - ACTOR_LEARNER: One learner update (consuming batch from queue)
        - ON_POLICY: Collect rollout + update
        - OFF_POLICY: Sample batch from buffer + update
        - TABULAR_ITERATIVE: One CFR iteration
        - HYBRID: Algorithm-specific

        Returns:
            Dictionary of metrics from this step.
        """
        ...

    @abstractmethod
    def _get_checkpoint_data(self) -> Dict[str, Any]:
        """
        Return all data needed to resume training.

        Should include:
        - Network state dicts
        - Optimizer state dicts
        - Training state (steps, episodes)
        - Any algorithm-specific state
        """
        ...

    @abstractmethod
    def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
        """
        Restore trainer state from checkpoint data.
        """
        ...

    # === Optional Hooks (override if needed) ===

    def _on_training_start(self) -> None:
        """Called at the start of train(), after _setup()."""
        pass

    def _on_training_end(self) -> None:
        """Called at the end of train(), before cleanup."""
        pass

    def _on_step_end(self, metrics: Dict[str, Any]) -> None:
        """Called after each _train_step()."""
        pass

    def _cleanup(self) -> None:
        """
        Clean up resources (close envs, join workers, etc.).
        Override in subclasses with distributed components.
        """
        pass

    # === Public API ===

    def setup(self) -> None:
        """Public setup method, ensures single initialization."""
        if not self._is_setup:
            self._setup()
            self._is_setup = True

    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            resume_from: Path to checkpoint to resume from.

        Returns:
            Final training metrics.
        """
        self._setup_signal_handlers()
        self.setup()

        if resume_from:
            self.load_checkpoint(resume_from)

        self._on_training_start()
        self.logger.info(f"Starting training: {self.config.paradigm.name}")
        self.logger.info(f"Total timesteps: {self.config.total_timesteps}")

        final_metrics: Dict[str, Any] = {}

        try:
            while self.global_step < self.config.total_timesteps:
                if self._shutdown_requested:
                    self.logger.info("Shutdown requested, stopping training.")
                    break

                # Execute one training step
                metrics = self._train_step()
                self.global_step += metrics.get("timesteps", 1)

                # Log metrics
                self.logger.log(metrics, step=self.global_step)

                # Hook for subclass actions
                self._on_step_end(metrics)

                # Checkpoint
                if self._should_checkpoint():
                    self.save_checkpoint()

                final_metrics = metrics

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        finally:
            self._on_training_end()
            self._cleanup()

            # Final checkpoint
            if self.checkpoint_manager:
                self.save_checkpoint(is_final=True)

        return final_metrics

    def save_checkpoint(self, is_final: bool = False) -> str:
        """Save current training state to checkpoint."""
        if not self.checkpoint_manager:
            return ""

        data = {
            "global_step": self.global_step,
            "episodes_completed": self.episodes_completed,
            "config": self.config.to_dict(),
            "algorithm_data": self._get_checkpoint_data(),
        }

        path = self.checkpoint_manager.save(
            data,
            step=self.global_step,
            is_final=is_final
        )
        self.logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load training state from checkpoint."""
        if not self.checkpoint_manager:
            raise ValueError("Cannot load checkpoint without checkpoint_dir")

        data = self.checkpoint_manager.load(path)
        self.global_step = data["global_step"]
        self.episodes_completed = data["episodes_completed"]
        self._load_checkpoint_data(data["algorithm_data"])

        self.logger.info(f"Resumed from checkpoint: {path} (step {self.global_step})")

    def _should_checkpoint(self) -> bool:
        """Check if we should save a checkpoint."""
        return (
            self.checkpoint_manager is not None
            and self.global_step > 0
            and self.global_step % self.config.checkpoint_interval == 0
        )

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown on SIGINT/SIGTERM."""
        def handler(signum, frame):
            self.logger.info("\nShutdown signal received...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
