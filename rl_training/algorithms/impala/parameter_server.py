import threading
from typing import Dict, Any, Optional

class ParameterServer:
    """Thread-safe parameter server for sharing weights between learner and actors."""

    def __init__(self, initial_state_dict: Optional[Dict[str, Any]] = None):
        self._lock = threading.Lock()
        self._params: Dict[str, Any] = {}
        self._version: int = 0
        if initial_state_dict is not None:
            self._params = {k: v.cpu().clone() for k, v in initial_state_dict.items()}

    def update(self, state_dict):
        with self._lock:
            self._params = {k: v.cpu().clone() for k, v in state_dict.items()}
            self._version += 1

    def get(self):
        with self._lock:
            return self._version, {k: v.clone() for k, v in self._params.items()}

    def is_initialized(self) -> bool:
        with self._lock:
            return len(self._params) > 0
