from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch as th


class CheckpointManager:
    """
    Manages checkpoint saving and loading.

    Features:
    - Automatic cleanup (keep N most recent)
    - Metadata tracking
    - Support for both torch and JSON data
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 5,
        checkpoint_prefix: str = "checkpoint",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_to_keep = max_to_keep
        self.checkpoint_prefix = checkpoint_prefix

        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track checkpoints
        self._checkpoints: List[Path] = self._discover_checkpoints()

    def _discover_checkpoints(self) -> List[Path]:
        """Find existing checkpoints in directory."""
        pattern = f"{self.checkpoint_prefix}_*.pt"
        checkpoints = sorted(
            self.checkpoint_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime
        )
        return list(checkpoints)

    def save(
        self,
        data: Dict[str, Any],
        step: int,
        is_final: bool = False,
    ) -> str:
        """
        Save checkpoint.

        Args:
            data: Dictionary of data to save.
            step: Current training step.
            is_final: If True, mark as final checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        if is_final:
            filename = f"{self.checkpoint_prefix}_final.pt"
        else:
            filename = f"{self.checkpoint_prefix}_{step:08d}.pt"

        path = self.checkpoint_dir / filename

        # Add metadata
        data["_checkpoint_meta"] = {
            "step": step,
            "is_final": is_final,
        }

        th.save(data, path)

        # Track and cleanup
        if not is_final:
            self._checkpoints.append(path)
            self._cleanup_old_checkpoints()

        return str(path)

    def load(self, path: str) -> Dict[str, Any]:
        """Load checkpoint from path."""
        return th.load(path, map_location="cpu", weights_only=False)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint."""
        if not self._checkpoints:
            return None
        return self.load(str(self._checkpoints[-1]))

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_to_keep."""
        while len(self._checkpoints) > self.max_to_keep:
            old_path = self._checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()

    def get_latest_path(self) -> Optional[str]:
        """Get path to most recent checkpoint."""
        if not self._checkpoints:
            return None
        return str(self._checkpoints[-1])
