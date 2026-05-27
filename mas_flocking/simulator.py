"""Core 2-D double-integrator simulator for Layer 0 flocking experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .obstacles import CircleObstacle
from .utils import as_2d_array, as_vector2, clip_by_norm


@dataclass
class FlockingState:
    """Snapshot of the simulator state."""

    q: np.ndarray
    p: np.ndarray
    t: float
    step_count: int
    obstacles: List[Dict[str, object]]


class FlockingEnv:
    """Lightweight 2-D multi-agent simulator with second-order dynamics.

    Agents follow the double-integrator model used by Olfati-Saber-style
    flocking controllers: q_dot = p, p_dot = u. The environment owns only
    physics, limits, obstacles, and bookkeeping; concrete flocking controllers
    are intentionally kept outside this class.
    """

    def __init__(
        self,
        n_agents: int = 30,
        dt: float = 0.02,
        world_size: Tuple[float, float] = (20.0, 12.0),
        v_max: float = 3.0,
        u_max: float = 8.0,
        seed: int = 0,
        boundary_mode: str = "reflect",
        obstacles: Optional[Sequence[CircleObstacle]] = None,
    ) -> None:
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if v_max <= 0 or u_max <= 0:
            raise ValueError("v_max and u_max must be positive")
        if boundary_mode not in {"reflect", "clip", "none"}:
            raise ValueError("boundary_mode must be one of: reflect, clip, none")

        self.n = int(n_agents)
        self.dt = float(dt)
        self.world_size = as_vector2(np.asarray(world_size, dtype=float), "world_size")
        if np.any(self.world_size <= 0):
            raise ValueError("world_size values must be positive")
        self.v_max = float(v_max)
        self.u_max = float(u_max)
        self.boundary_mode = boundary_mode
        self.rng = np.random.default_rng(seed)
        self.obstacles: List[CircleObstacle] = list(obstacles or [])

        self.q: Optional[np.ndarray] = None
        self.p: Optional[np.ndarray] = None
        self.last_u: Optional[np.ndarray] = None
        self.t = 0.0
        self.step_count = 0

    def reset(
        self,
        init_mode: str = "random_left",
        q0: Optional[np.ndarray] = None,
        p0: Optional[np.ndarray] = None,
    ) -> FlockingState:
        """Reset agent state using a named initializer or custom arrays."""
        if init_mode == "custom":
            if q0 is None:
                raise ValueError("q0 is required when init_mode='custom'")
            q = as_2d_array(q0, "q0", expected_rows=self.n).copy()
            p = np.zeros((self.n, 2), dtype=float) if p0 is None else as_2d_array(p0, "p0", expected_rows=self.n).copy()
        elif init_mode == "random_left":
            x_high = max(1.0, min(5.0, self.world_size[0] * 0.35))
            x = self.rng.uniform(1.0, x_high, size=(self.n, 1))
            y = self.rng.uniform(1.0, max(1.0, self.world_size[1] - 1.0), size=(self.n, 1))
            q = np.hstack([x, y])
            p = self.rng.normal(loc=0.0, scale=0.2, size=(self.n, 2)) if p0 is None else as_2d_array(p0, "p0", expected_rows=self.n).copy()
        elif init_mode == "random_center":
            low = np.array([self.world_size[0] * 0.35, self.world_size[1] * 0.25])
            high = np.array([self.world_size[0] * 0.65, self.world_size[1] * 0.75])
            q = self.rng.uniform(low=low, high=high, size=(self.n, 2))
            p = self.rng.normal(loc=0.0, scale=0.2, size=(self.n, 2)) if p0 is None else as_2d_array(p0, "p0", expected_rows=self.n).copy()
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        self.q = q
        self.p = clip_by_norm(p, self.v_max)
        self.last_u = np.zeros((self.n, 2), dtype=float)
        self.t = 0.0
        self.step_count = 0
        self._handle_agent_boundaries()
        return self.get_state()

    def step(self, u: np.ndarray) -> FlockingState:
        """Advance the simulation by one semi-implicit Euler step."""
        if self.q is None or self.p is None:
            raise RuntimeError("Environment must be reset before calling step")

        u_arr = as_2d_array(u, "u", expected_rows=self.n)
        u_clipped = clip_by_norm(u_arr, self.u_max)

        self.p = self.p + u_clipped * self.dt
        self.p = clip_by_norm(self.p, self.v_max)
        self.q = self.q + self.p * self.dt
        self._handle_agent_boundaries()

        for obstacle in self.obstacles:
            obstacle.step(self.dt, world_size=self.world_size, boundary_mode=self.boundary_mode)

        self.last_u = u_clipped
        self.t += self.dt
        self.step_count += 1
        return self.get_state()

    def get_state(self) -> FlockingState:
        """Return a defensive-copy state snapshot."""
        if self.q is None or self.p is None:
            raise RuntimeError("Environment must be reset before reading state")
        return FlockingState(
            q=self.q.copy(),
            p=self.p.copy(),
            t=float(self.t),
            step_count=int(self.step_count),
            obstacles=[obs.as_dict() for obs in self.obstacles],
        )

    def add_obstacle(self, obstacle: CircleObstacle) -> None:
        """Add a circular obstacle to the environment."""
        self.obstacles.append(obstacle)

    def set_obstacles(self, obstacles: Sequence[CircleObstacle]) -> None:
        """Replace all obstacles."""
        self.obstacles = list(obstacles)

    def _handle_agent_boundaries(self) -> None:
        if self.q is None or self.p is None or self.boundary_mode == "none":
            return
        for dim in range(2):
            low_hit = self.q[:, dim] < 0.0
            high_hit = self.q[:, dim] > self.world_size[dim]
            hit = low_hit | high_hit
            if not np.any(hit):
                continue
            self.q[low_hit, dim] = 0.0
            self.q[high_hit, dim] = self.world_size[dim]
            if self.boundary_mode == "reflect":
                self.p[low_hit, dim] = np.abs(self.p[low_hit, dim])
                self.p[high_hit, dim] = -np.abs(self.p[high_hit, dim])
            elif self.boundary_mode == "clip":
                self.p[hit, dim] = 0.0
