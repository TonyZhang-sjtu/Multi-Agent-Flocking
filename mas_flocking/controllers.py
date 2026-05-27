"""Simple Layer 0 controllers used to validate simulator plumbing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .obstacles import CircleObstacle
from .utils import as_2d_array, as_vector2


def zero_control(q: np.ndarray, p: Optional[np.ndarray] = None) -> np.ndarray:
    """Return zero acceleration for every agent."""
    q_arr = as_2d_array(q, "q")
    return np.zeros_like(q_arr)


def goal_pd_control(
    q: np.ndarray,
    p: np.ndarray,
    goal: np.ndarray,
    k_p: float = 0.8,
    k_d: float = 1.2,
) -> np.ndarray:
    """Simple point-goal PD controller for smoke-testing Layer 0 dynamics."""
    q_arr = as_2d_array(q, "q")
    p_arr = as_2d_array(p, "p", expected_rows=q_arr.shape[0])
    goal_vec = as_vector2(goal, "goal")
    return float(k_p) * (goal_vec[None, :] - q_arr) - float(k_d) * p_arr


def controller_template(
    q: np.ndarray,
    p: np.ndarray,
    obstacles: Optional[List[CircleObstacle]],
    goal: Optional[np.ndarray],
    params: Optional[Dict[str, Any]],
) -> np.ndarray:
    """Document the common controller signature expected by later layers."""
    del obstacles, goal, params
    return zero_control(q, p)
