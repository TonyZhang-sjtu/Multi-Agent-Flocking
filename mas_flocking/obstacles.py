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


class ScriptedCircleObstacle(CircleObstacle):
    """Circular obstacle with analytic time-varying center and velocity."""

    def __init__(
        self,
        center: Tuple[float, float],
        radius: float,
        base_velocity: Tuple[float, float] = (0.0, 0.0),
        acceleration: Tuple[float, float] = (0.0, 0.0),
        sine_amplitude: Tuple[float, float] = (0.0, 0.0),
        sine_omega: float = 0.0,
        sine_phase: float = 0.0,
        circle_radius: float = 0.0,
        circle_omega: float = 0.0,
        circle_phase: float = 0.0,
        name: str = "scripted_obstacle",
    ) -> None:
        super().__init__(center=center, radius=radius, velocity=base_velocity, name=name, dynamic=True)
        self.origin = self.center.copy()
        self.base_velocity = as_vector2(np.asarray(base_velocity, dtype=float), "base_velocity")
        self.acceleration = as_vector2(np.asarray(acceleration, dtype=float), "acceleration")
        self.sine_amplitude = as_vector2(np.asarray(sine_amplitude, dtype=float), "sine_amplitude")
        self.sine_omega = float(sine_omega)
        self.sine_phase = float(sine_phase)
        self.circle_radius = float(circle_radius)
        self.circle_omega = float(circle_omega)
        self.circle_phase = float(circle_phase)
        self.script_time = 0.0
        self.center = self._center_at(0.0)
        self.velocity = self._velocity_at(0.0)

    def step(self, dt: float, world_size: Optional[np.ndarray] = None, boundary_mode: str = "reflect") -> None:
        """Advance scripted obstacle using its analytic trajectory."""
        self.script_time += float(dt)
        self.center = self._center_at(self.script_time)
        self.velocity = self._velocity_at(self.script_time)
        if world_size is not None:
            self.handle_boundary(world_size, boundary_mode=boundary_mode)

    def _center_at(self, t: float) -> np.ndarray:
        t_float = float(t)
        linear = self.base_velocity * t_float + 0.5 * self.acceleration * t_float * t_float
        sine = self.sine_amplitude * (np.sin(self.sine_omega * t_float + self.sine_phase) - np.sin(self.sine_phase))
        circular = np.zeros(2, dtype=float)
        if self.circle_radius != 0.0 and self.circle_omega != 0.0:
            angle = self.circle_omega * t_float + self.circle_phase
            start = np.array([np.cos(self.circle_phase), np.sin(self.circle_phase)])
            current = np.array([np.cos(angle), np.sin(angle)])
            circular = self.circle_radius * (current - start)
        return self.origin + linear + sine + circular

    def _velocity_at(self, t: float) -> np.ndarray:
        t_float = float(t)
        velocity = self.base_velocity + self.acceleration * t_float
        velocity = velocity + self.sine_amplitude * self.sine_omega * np.cos(self.sine_omega * t_float + self.sine_phase)
        if self.circle_radius != 0.0 and self.circle_omega != 0.0:
            angle = self.circle_omega * t_float + self.circle_phase
            velocity = velocity + self.circle_radius * self.circle_omega * np.array([-np.sin(angle), np.cos(angle)])
        return velocity

    def as_dict(self) -> Dict[str, object]:
        """Return a serializable scripted obstacle snapshot."""
        snapshot = super().as_dict()
        snapshot["script_time"] = float(self.script_time)
        snapshot["scripted"] = True
        return snapshot
