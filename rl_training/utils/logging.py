from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from collections import deque
import time


@runtime_checkable
class Logger(Protocol):
    """Protocol for logging implementations."""

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics at given step."""
        ...

    def info(self, message: str) -> None:
        """Log info message."""
        ...

    def close(self) -> None:
        """Clean up logger resources."""
        ...


class ConsoleLogger:
    """Simple console logger with moving average smoothing."""

    def __init__(
        self,
        log_interval: int = 100,
        smoothing_window: int = 100,
    ):
        self.log_interval = log_interval
        self.smoothing_window = smoothing_window
        self._step_count = 0
        self._metric_history: Dict[str, deque] = {}
        self._last_log_time = time.time()

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        self._step_count = step

        # Update histories
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in self._metric_history:
                    self._metric_history[key] = deque(maxlen=self.smoothing_window)
                self._metric_history[key].append(value)

        # Print periodically
        if step % self.log_interval == 0:
            self._print_metrics(step, metrics)

    def _print_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        elapsed = time.time() - self._last_log_time
        self._last_log_time = time.time()

        parts = [f"[Step {step:,}]"]

        # Add SPS (steps per second)
        if elapsed > 0:
            sps = self.log_interval / elapsed
            parts.append(f"SPS: {sps:.0f}")

        # Add smoothed metrics
        for key in sorted(metrics.keys()):
            if key in self._metric_history and len(self._metric_history[key]) > 0:
                avg = sum(self._metric_history[key]) / len(self._metric_history[key])
                if isinstance(avg, float):
                    parts.append(f"{key}: {avg:.4f}")
                else:
                    parts.append(f"{key}: {avg}")

        print(" | ".join(parts))

    def info(self, message: str) -> None:
        print(f"[INFO] {message}")

    def close(self) -> None:
        pass


class TensorBoardLogger:
    """TensorBoard logging wrapper."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

    def info(self, message: str) -> None:
        print(f"[INFO] {message}")

    def close(self) -> None:
        self.writer.close()


class WandbLogger:
    """Weights & Biases logging wrapper."""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        resume: bool = False,
        log_interval: int = 1,
    ):
        """
        Initialize Weights & Biases logger.

        Args:
            project: W&B project name
            name: Run name (auto-generated if None)
            config: Configuration dict to log
            tags: Tags for the run
            group: Group name for organizing runs
            resume: Whether to resume a previous run
            log_interval: Only log every N steps to reduce overhead
        """
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Install with: pip install wandb"
            )

        self.log_interval = log_interval
        self._step_count = 0

        # Initialize wandb run
        self._run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            group=group,
            resume="allow" if resume else None,
            reinit=True,
        )

        print(f"[W&B] Initialized run: {self._run.name}")
        print(f"[W&B] View at: {self._run.url}")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to W&B."""
        self._step_count = step

        if step % self.log_interval == 0:
            # Filter to only log numeric values
            log_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            self._wandb.log(log_metrics, step=step)

    def log_histogram(self, name: str, values: Any, step: int) -> None:
        """Log histogram to W&B."""
        self._wandb.log({name: self._wandb.Histogram(values)}, step=step)

    def log_image(self, name: str, image: Any, step: int) -> None:
        """Log image to W&B."""
        self._wandb.log({name: self._wandb.Image(image)}, step=step)

    def log_artifact(self, name: str, path: str, type: str = "model") -> None:
        """Log artifact (model checkpoint, etc.) to W&B."""
        artifact = self._wandb.Artifact(name, type=type)
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def save_model(self, model_path: str) -> None:
        """Save model file to W&B."""
        self._wandb.save(model_path)

    def info(self, message: str) -> None:
        """Log info message."""
        print(f"[W&B] {message}")

    def close(self) -> None:
        """Finish W&B run."""
        self._wandb.finish()

    @property
    def run(self):
        """Get the underlying wandb run object."""
        return self._run

    @property
    def run_id(self) -> str:
        """Get the run ID."""
        return self._run.id

    @property
    def run_name(self) -> str:
        """Get the run name."""
        return self._run.name


class CompositeLogger:
    """Combine multiple loggers."""

    def __init__(self, loggers: List[Logger]):
        self.loggers = loggers

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        for logger in self.loggers:
            logger.log(metrics, step)

    def info(self, message: str) -> None:
        for logger in self.loggers:
            logger.info(message)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


def create_logger(
    console: bool = True,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
    wandb_tags: Optional[List[str]] = None,
    tensorboard_dir: Optional[str] = None,
    log_interval: int = 100,
) -> Logger:
    """
    Create a logger with the specified backends.

    Args:
        console: Enable console logging
        wandb_project: W&B project name (enables W&B if set)
        wandb_name: W&B run name
        wandb_config: Config to log to W&B
        wandb_tags: Tags for W&B run
        tensorboard_dir: TensorBoard log directory (enables TB if set)
        log_interval: Logging interval for console

    Returns:
        Logger instance (possibly composite)
    """
    loggers = []

    if console:
        loggers.append(ConsoleLogger(log_interval=log_interval))

    if wandb_project:
        loggers.append(WandbLogger(
            project=wandb_project,
            name=wandb_name,
            config=wandb_config,
            tags=wandb_tags,
            log_interval=1,  # W&B handles its own rate limiting
        ))

    if tensorboard_dir:
        loggers.append(TensorBoardLogger(log_dir=tensorboard_dir))

    if len(loggers) == 0:
        return ConsoleLogger(log_interval=log_interval)
    elif len(loggers) == 1:
        return loggers[0]
    else:
        return CompositeLogger(loggers)
