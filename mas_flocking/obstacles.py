"""Obstacle data structures for Layer 0 and later dynamic-avoidance layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .utils import as_vector2


@dataclass
class CircleObstacle:
    """Circular static or dynamic obstacle in a 2-D world.

    The velocity field is intentionally part of Layer 0 because later MRF-IAPF
    obstacle-avoidance terms need obstacle motion to compute relative velocity.
    """

    center: np.ndarray
    radius: float
    velocity: np.ndarray
    name: str = "obstacle"
    dynamic: bool = True

    def __init__(
        self,
        center: Tuple[float, float],
        radius: float,
        velocity: Tuple[float, float] = (0.0, 0.0),
        name: str = "obstacle",
        dynamic: Optional[bool] = None,
    ) -> None:
        self.center = as_vector2(np.asarray(center, dtype=float), "center")
        self.radius = float(radius)
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        self.velocity = as_vector2(np.asarray(velocity, dtype=float), "velocity")
        self.name = name
        self.dynamic = bool(np.linalg.norm(self.velocity) > 0.0) if dynamic is None else bool(dynamic)

    def step(self, dt: float, world_size: Optional[np.ndarray] = None, boundary_mode: str = "reflect") -> None:
        """Advance obstacle state and optionally keep it inside the world."""
        if self.dynamic:
            self.center = self.center + self.velocity * float(dt)
        if world_size is not None:
            self.handle_boundary(world_size, boundary_mode=boundary_mode)

    def handle_boundary(self, world_size: np.ndarray, boundary_mode: str = "reflect") -> None:
        """Apply obstacle boundary handling using its radius as margin."""
        world = as_vector2(np.asarray(world_size, dtype=float), "world_size")
        if boundary_mode not in {"reflect", "clip", "none"}:
            raise ValueError(f"Unknown obstacle boundary_mode: {boundary_mode}")
        if boundary_mode == "none":
            return
        for dim in range(2):
            low = self.radius
            high = world[dim] - self.radius
            if self.center[dim] < low:
                self.center[dim] = low
                if boundary_mode == "reflect":
                    self.velocity[dim] = abs(self.velocity[dim])
            elif self.center[dim] > high:
                self.center[dim] = high
                if boundary_mode == "reflect":
                    self.velocity[dim] = -abs(self.velocity[dim])

    def as_dict(self) -> Dict[str, object]:
        """Return a serializable obstacle snapshot."""
        return {
            "name": self.name,
            "center": self.center.copy(),
            "radius": self.radius,
            "velocity": self.velocity.copy(),
            "dynamic": self.dynamic,
        }
