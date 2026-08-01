"""Hardware auto-detection: figure out whether real, working CUDA-accelerated
XGBoost is available, and fall back to CPU cleanly (never crash) if not.

Two independent checks have to both pass before we ever hand device="cuda" to
a real training run:
  1. An NVIDIA GPU is actually present (nvidia-smi succeeds).
  2. The installed XGBoost build can actually fit a model on that device (a
     GPU can be present while the XGBoost wheel installed is CPU-only).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class HardwareInfo:
    device: str          # "cuda" or "cpu"
    gpu_name: str | None
    cpu_count: int

    @property
    def label(self) -> str:
        if self.device == "cuda":
            return f"GPU ({self.gpu_name})"
        return f"CPU ({self.cpu_count} cores)"


def _detect_nvidia_gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    name = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else None
    return name or None


def _xgboost_supports_cuda() -> bool:
    try:
        import numpy as np
        from xgboost import XGBClassifier
    except ImportError:
        return False

    try:
        X = np.random.rand(32, 4)
        y = np.random.randint(0, 2, size=32)
        model = XGBClassifier(
            n_estimators=2,
            max_depth=2,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        model.fit(X, y)
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def detect() -> HardwareInfo:
    """Detect once per process and cache the result."""
    cpu_count = os.cpu_count() or 1
    gpu_name = _detect_nvidia_gpu_name()

    if gpu_name and _xgboost_supports_cuda():
        return HardwareInfo(device="cuda", gpu_name=gpu_name, cpu_count=cpu_count)

    return HardwareInfo(device="cpu", gpu_name=None, cpu_count=cpu_count)
