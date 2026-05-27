"""Small numerical helpers shared across the flocking simulator."""

from __future__ import annotations

from typing import Optional

import numpy as np


def as_2d_array(x: np.ndarray, name: str, expected_rows: Optional[int] = None) -> np.ndarray:
    """Validate and return a finite float array with shape [N, 2]."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2], got {arr.shape}")
    if expected_rows is not None and arr.shape[0] != expected_rows:
        raise ValueError(f"{name} must have shape [{expected_rows}, 2], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf values")
    return arr


def as_vector2(x: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a finite float vector with shape [2]."""
    arr = np.asarray(x, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape [2], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf values")
    return arr


def clip_by_norm(x: np.ndarray, max_norm: Optional[float], eps: float = 1e-12) -> np.ndarray:
    """Clip vectors along the last axis to a maximum Euclidean norm."""
    arr = np.asarray(x, dtype=float)
    if max_norm is None or max_norm <= 0:
        return arr.copy()
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / (norms + eps))
    return arr * scale


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    import os

    os.makedirs(path, exist_ok=True)
