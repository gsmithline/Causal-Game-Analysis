"""
Results Manager for organizing training outputs.

Provides structured storage for:
- Trained neural network policies
- Training metrics and curves
- Hyperparameters and configuration
- Environment information
- Evaluation results

Directory structure:
    results/
    ├── index.json                    # Master index of all runs
    ├── ppo/
    │   ├── run_20240115_143022_seed42/
    │   │   ├── metadata.json         # Full run metadata
    │   │   ├── config.json           # Hyperparameters
    │   │   ├── metrics.json          # Training metrics history
    │   │   ├── final_eval.json       # Final evaluation results
    │   │   ├── policy.pt             # Trained policy weights
    │   │   ├── checkpoints/          # Intermediate checkpoints
    │   │   │   ├── step_100000.pt
    │   │   │   └── step_200000.pt
    │   │   └── plots/                # Training curves (optional)
    │   └── run_20240115_150000_seed123/
    │       └── ...
    ├── mappo/
    │   └── ...
    └── psro/
        └── ...
"""

from __future__ import annotations
import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np


@dataclass
class RunMetadata:
    """Metadata for a training run."""

    # Identifiers
    run_id: str = ""
    algorithm: str = ""
    environment: str = "bargain"
    timestamp: str = ""

    # Configuration
    seed: int = 42
    cuda_device: int = 0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Training info
    total_timesteps: int = 0
    actual_timesteps: int = 0
    training_time_seconds: float = 0.0
    num_updates: int = 0

    # Results
    final_metrics: Dict[str, float] = field(default_factory=dict)
    best_metrics: Dict[str, float] = field(default_factory=dict)

    # Environment info
    env_config: Dict[str, Any] = field(default_factory=dict)

    # System info
    pytorch_version: str = ""
    cuda_available: bool = False
    gpu_name: str = ""

    # Status
    status: str = "running"  # running, completed, failed, interrupted
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunMetadata":
        return cls(**data)


class ResultsManager:
    """
    Manages organized storage of training results.

    Usage:
        manager = ResultsManager("results")

        # Start a new run
        run = manager.create_run("ppo", seed=42, config=config.to_dict())

        # During training, log metrics
        run.log_metrics({"loss": 0.5, "reward": 10.0}, step=1000)

        # Save checkpoints
        run.save_checkpoint(policy.state_dict(), step=1000)

        # At the end, save final results
        run.save_final_policy(policy.state_dict())
        run.set_final_metrics({"avg_reward_p0": 5.0, "avg_reward_p1": 4.5})
        run.complete()

        # Later, load and compare results
        results = manager.list_runs("ppo")
        best_run = manager.get_best_run("ppo", metric="final_metrics.avg_reward_p0")
        policy = manager.load_policy(best_run.run_id)
    """

    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        self._load_index()

    def _load_index(self):
        """Load or create the master index."""
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                self._index = json.load(f)
        else:
            self._index = {"runs": {}, "created": datetime.now().isoformat()}
            self._save_index()

    def _save_index(self):
        """Save the master index."""
        self._index["updated"] = datetime.now().isoformat()
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def create_run(
        self,
        algorithm: str,
        seed: int = 42,
        config: Optional[Dict[str, Any]] = None,
        environment: str = "bargain",
        tags: Optional[List[str]] = None,
    ) -> "RunContext":
        """
        Create a new training run.

        Args:
            algorithm: Algorithm name (ppo, mappo, psro, etc.)
            seed: Random seed
            config: Hyperparameter configuration
            environment: Environment name
            tags: Optional tags for filtering

        Returns:
            RunContext for managing this run
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{algorithm}_{timestamp}_seed{seed}"

        # Create run directory
        run_dir = self.base_dir / algorithm / f"run_{timestamp}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

        # Create metadata
        metadata = RunMetadata(
            run_id=run_id,
            algorithm=algorithm,
            environment=environment,
            timestamp=timestamp,
            seed=seed,
            hyperparameters=config or {},
            pytorch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            env_config={"name": environment, "num_actions": 82, "obs_dim": 92},
        )

        # Add to index
        self._index["runs"][run_id] = {
            "algorithm": algorithm,
            "seed": seed,
            "timestamp": timestamp,
            "path": str(run_dir),
            "status": "running",
            "tags": tags or [],
        }
        self._save_index()

        return RunContext(self, run_id, run_dir, metadata)

    def get_run(self, run_id: str) -> Optional["RunContext"]:
        """Get an existing run by ID."""
        if run_id not in self._index["runs"]:
            return None

        run_info = self._index["runs"][run_id]
        run_dir = Path(run_info["path"])

        if not run_dir.exists():
            return None

        # Load metadata
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = RunMetadata.from_dict(json.load(f))
        else:
            metadata = RunMetadata(run_id=run_id)

        return RunContext(self, run_id, run_dir, metadata)

    def list_runs(
        self,
        algorithm: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List runs with optional filtering.

        Args:
            algorithm: Filter by algorithm
            status: Filter by status (completed, running, failed)
            tags: Filter by tags (any match)

        Returns:
            List of run info dictionaries
        """
        runs = []
        for run_id, info in self._index["runs"].items():
            if algorithm and info["algorithm"] != algorithm:
                continue
            if status and info.get("status") != status:
                continue
            if tags and not any(t in info.get("tags", []) for t in tags):
                continue

            # Load full metadata if available
            run_dir = Path(info["path"])
            metadata_path = run_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    full_info = json.load(f)
                    info = {**info, **full_info}

            runs.append({"run_id": run_id, **info})

        return sorted(runs, key=lambda x: x.get("timestamp", ""), reverse=True)

    def get_best_run(
        self,
        algorithm: str,
        metric: str = "final_metrics.total_reward",
        minimize: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run for an algorithm based on a metric.

        Args:
            algorithm: Algorithm name
            metric: Metric path (e.g., "final_metrics.avg_reward_p0")
            minimize: If True, lower is better

        Returns:
            Best run info or None
        """
        runs = self.list_runs(algorithm=algorithm, status="completed")

        if not runs:
            return None

        def get_metric(run):
            parts = metric.split(".")
            value = run
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return float("-inf") if not minimize else float("inf")
            return value if isinstance(value, (int, float)) else 0

        return min(runs, key=get_metric) if minimize else max(runs, key=get_metric)

    def load_policy(
        self,
        run_id: str,
        checkpoint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a trained policy from a run.

        Args:
            run_id: Run identifier
            checkpoint: Specific checkpoint name, or None for final policy

        Returns:
            Policy state dict or None
        """
        if run_id not in self._index["runs"]:
            return None

        run_dir = Path(self._index["runs"][run_id]["path"])

        if checkpoint:
            policy_path = run_dir / "checkpoints" / checkpoint
        else:
            policy_path = run_dir / "policy.pt"

        if not policy_path.exists():
            return None

        return torch.load(policy_path, map_location="cpu")

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metrics: Specific metrics to compare

        Returns:
            Comparison dictionary
        """
        comparison = {}

        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                run_data = {
                    "algorithm": run.metadata.algorithm,
                    "seed": run.metadata.seed,
                    "hyperparameters": run.metadata.hyperparameters,
                    "final_metrics": run.metadata.final_metrics,
                    "training_time": run.metadata.training_time_seconds,
                    "status": run.metadata.status,
                }
                comparison[run_id] = run_data

        return comparison

    def export_results(
        self,
        output_path: str,
        algorithm: Optional[str] = None,
        format: str = "csv",
    ):
        """Export results to CSV or JSON for external analysis."""
        runs = self.list_runs(algorithm=algorithm, status="completed")

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(runs, f, indent=2)
        elif format == "csv":
            import csv

            if not runs:
                return

            # Flatten nested dicts for CSV
            flat_runs = []
            for run in runs:
                flat = {"run_id": run["run_id"]}
                for key, value in run.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            flat[f"{key}.{k}"] = v
                    else:
                        flat[key] = value
                flat_runs.append(flat)

            # Get all keys
            all_keys = set()
            for run in flat_runs:
                all_keys.update(run.keys())

            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                writer.writeheader()
                writer.writerows(flat_runs)

    def cleanup_failed_runs(self, delete: bool = False):
        """Remove or mark failed/incomplete runs."""
        for run_id, info in list(self._index["runs"].items()):
            run_dir = Path(info["path"])

            if not run_dir.exists():
                del self._index["runs"][run_id]
                continue

            if info.get("status") == "running":
                # Check if actually still running (no recent updates)
                metadata_path = run_dir / "metadata.json"
                if metadata_path.exists():
                    import time
                    age = time.time() - metadata_path.stat().st_mtime
                    if age > 3600 * 24:  # Older than 24 hours
                        info["status"] = "abandoned"

            if delete and info.get("status") in ["failed", "abandoned"]:
                shutil.rmtree(run_dir)
                del self._index["runs"][run_id]

        self._save_index()


class RunContext:
    """Context manager for a single training run."""

    def __init__(
        self,
        manager: ResultsManager,
        run_id: str,
        run_dir: Path,
        metadata: RunMetadata,
    ):
        self.manager = manager
        self.run_id = run_id
        self.run_dir = run_dir
        self.metadata = metadata
        self._metrics_history: List[Dict[str, Any]] = []
        self._start_time = datetime.now()

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log training metrics at a given step."""
        entry = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}
        self._metrics_history.append(entry)

        # Update best metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in self.metadata.best_metrics:
                    self.metadata.best_metrics[key] = value
                elif "loss" in key.lower() or "error" in key.lower():
                    self.metadata.best_metrics[key] = min(
                        self.metadata.best_metrics[key], value
                    )
                else:
                    self.metadata.best_metrics[key] = max(
                        self.metadata.best_metrics[key], value
                    )

        # Periodically save metrics
        if len(self._metrics_history) % 100 == 0:
            self._save_metrics()

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        step: int,
        is_best: bool = False,
    ):
        """Save a training checkpoint."""
        checkpoint_path = self.run_dir / "checkpoints" / f"step_{step}.pt"
        torch.save(state_dict, checkpoint_path)

        if is_best:
            best_path = self.run_dir / "checkpoints" / "best.pt"
            torch.save(state_dict, best_path)

    def save_final_policy(self, state_dict: Dict[str, Any]):
        """Save the final trained policy."""
        policy_path = self.run_dir / "policy.pt"
        torch.save(state_dict, policy_path)

    def set_final_metrics(self, metrics: Dict[str, float]):
        """Set the final evaluation metrics."""
        self.metadata.final_metrics = metrics

    def set_actual_timesteps(self, timesteps: int):
        """Set actual timesteps completed."""
        self.metadata.actual_timesteps = timesteps

    def complete(self):
        """Mark the run as completed and save all data."""
        self.metadata.status = "completed"
        self.metadata.training_time_seconds = (
            datetime.now() - self._start_time
        ).total_seconds()
        self._save_all()

        # Update index
        self.manager._index["runs"][self.run_id]["status"] = "completed"
        self.manager._save_index()

    def fail(self, error_message: str = ""):
        """Mark the run as failed."""
        self.metadata.status = "failed"
        self.metadata.error_message = error_message
        self.metadata.training_time_seconds = (
            datetime.now() - self._start_time
        ).total_seconds()
        self._save_all()

        self.manager._index["runs"][self.run_id]["status"] = "failed"
        self.manager._save_index()

    def _save_metrics(self):
        """Save metrics history to file."""
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._metrics_history, f, indent=2)

    def _save_all(self):
        """Save all run data."""
        # Save metadata
        metadata_path = self.run_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        # Save config separately for easy viewing
        config_path = self.run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.metadata.hyperparameters, f, indent=2)

        # Save metrics
        self._save_metrics()

        # Save final evaluation
        if self.metadata.final_metrics:
            eval_path = self.run_dir / "final_eval.json"
            with open(eval_path, "w") as f:
                json.dump(self.metadata.final_metrics, f, indent=2)


def create_results_manager(base_dir: str = "results") -> ResultsManager:
    """Create a results manager instance."""
    return ResultsManager(base_dir)
