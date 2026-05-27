"""Static obstacle beta-agent avoidance for Olfati-Saber Algorithm 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from .alpha_flocking import ParamsLike as AlphaParamsLike, alpha_flocking_control, bump_function
from .gamma_navigation import GammaAgent, GammaParamsLike, gamma_navigation_control
from .obstacles import CircleObstacle
from .utils import as_2d_array


@dataclass(frozen=True)
class BetaObstacleParams:
    """Parameters for static circular-obstacle beta-agent avoidance."""

    epsilon: float = 0.1
    h: float = 0.2
    r_beta: float = 2.5
    c1_beta: float = 8.0
    c2_beta: float = 3.0
    agent_radius: float = 0.12
    beta_velocity_mode: str = "projected"

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0.0 <= self.h < 1.0:
            raise ValueError("h must satisfy 0 <= h < 1")
        if self.r_beta <= 0:
            raise ValueError("r_beta must be positive")
        if self.c1_beta < 0 or self.c2_beta < 0:
            raise ValueError("c1_beta and c2_beta must be non-negative")
        if self.agent_radius < 0:
            raise ValueError("agent_radius must be non-negative")
        if self.beta_velocity_mode not in {"projected", "zero"}:
            raise ValueError("beta_velocity_mode must be 'projected' or 'zero'")


BetaParamsLike = Optional[Union[BetaObstacleParams, Mapping[str, Union[float, str]]]]


def _sigma_norm_scalar(x: np.ndarray, epsilon: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return (np.sqrt(1.0 + float(epsilon) * arr * arr) - 1.0) / float(epsilon)


def as_beta_params(params: BetaParamsLike = None) -> BetaObstacleParams:
    """Normalize None/dict/dataclass inputs to a BetaObstacleParams object."""
    if params is None:
        return BetaObstacleParams()
    if isinstance(params, BetaObstacleParams):
        return params
    return BetaObstacleParams(**dict(params))


def _safe_normals(q: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = q - center[None, :]
    dists = np.linalg.norm(diff, axis=1)
    normals = np.zeros_like(diff)
    valid = dists > 1e-12
    normals[valid] = diff[valid] / dists[valid, None]
    normals[~valid] = np.array([1.0, 0.0])
    return normals, dists


def project_to_obstacle_boundary(
    q: np.ndarray,
    obstacle: CircleObstacle,
    agent_radius: float = 0.0,
) -> np.ndarray:
    """Project agent positions to the inflated circular obstacle boundary."""
    q_arr = as_2d_array(q, "q")
    normals, _ = _safe_normals(q_arr, obstacle.center)
    effective_radius = obstacle.radius + float(agent_radius)
    return obstacle.center[None, :] + effective_radius * normals


def obstacle_clearances(q: np.ndarray, obstacle: CircleObstacle, agent_radius: float = 0.0) -> np.ndarray:
    """Signed Euclidean clearance from agents to an inflated circular obstacle."""
    q_arr = as_2d_array(q, "q")
    _, dists = _safe_normals(q_arr, obstacle.center)
    return dists - obstacle.radius - float(agent_radius)


def beta_action(clearance_sigma: np.ndarray, params: BetaParamsLike = None) -> np.ndarray:
    """One-sided repulsive beta action that smoothly vanishes at r_beta."""
    prm = as_beta_params(params)
    r_beta_sigma = float(_sigma_norm_scalar(np.asarray(prm.r_beta), epsilon=prm.epsilon))
    ratio = np.asarray(clearance_sigma, dtype=float) / r_beta_sigma
    clipped = np.clip(ratio, 0.0, 1.0)
    return -bump_function(ratio, h=prm.h) * (1.0 - clipped)


def beta_obstacle_control(
    q: np.ndarray,
    p: np.ndarray,
    obstacles: Optional[Sequence[CircleObstacle]],
    params: BetaParamsLike = None,
) -> np.ndarray:
    """Compute static circular-obstacle beta-agent avoidance acceleration."""
    prm = as_beta_params(params)
    q_arr = as_2d_array(q, "q")
    p_arr = as_2d_array(p, "p", expected_rows=q_arr.shape[0])
    u = np.zeros_like(q_arr)
    if not obstacles:
        return u

    r_beta_sigma = float(_sigma_norm_scalar(np.asarray(prm.r_beta), epsilon=prm.epsilon))
    for obstacle in obstacles:
        normals, dists = _safe_normals(q_arr, obstacle.center)
        clearances = dists - obstacle.radius - prm.agent_radius
        active_clearance = np.maximum(clearances, 0.0)
        clearance_sigma = _sigma_norm_scalar(active_clearance, epsilon=prm.epsilon)
        ratio = clearance_sigma / r_beta_sigma
        weights = bump_function(ratio, h=prm.h)
        active = ratio < 1.0
        if not np.any(active):
            continue

        # Direction from an outside agent toward its beta-agent lies inward;
        # a negative beta action along this direction therefore repels outward.
        inward_dirs = -normals / np.sqrt(1.0 + prm.epsilon * active_clearance[:, None] * active_clearance[:, None])
        action = beta_action(clearance_sigma, prm)
        gradient_term = prm.c1_beta * action[:, None] * inward_dirs

        if prm.beta_velocity_mode == "projected":
            normal_speed = np.sum(p_arr * normals, axis=1, keepdims=True)
            p_beta = p_arr - normal_speed * normals
        else:
            p_beta = np.zeros_like(p_arr)
        damping_term = prm.c2_beta * weights[:, None] * (p_beta - p_arr)

        contribution = gradient_term + damping_term
        contribution[~active] = 0.0
        u += contribution
    return u


def flocking_with_static_obstacle_control(
    q: np.ndarray,
    p: np.ndarray,
    obstacles: Optional[Sequence[CircleObstacle]],
    alpha_params: AlphaParamsLike = None,
    gamma: Optional[GammaAgent] = None,
    gamma_params: GammaParamsLike = None,
    beta_params: BetaParamsLike = None,
) -> np.ndarray:
    """Combine alpha-agent, beta-agent, and gamma-agent controls."""
    target = GammaAgent() if gamma is None else gamma
    u_alpha = alpha_flocking_control(q, p, alpha_params)
    u_beta = beta_obstacle_control(q, p, obstacles, beta_params)
    u_gamma = gamma_navigation_control(q, p, target, gamma_params)
    return u_alpha + u_beta + u_gamma
