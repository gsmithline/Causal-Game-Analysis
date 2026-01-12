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
