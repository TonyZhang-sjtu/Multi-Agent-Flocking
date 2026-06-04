"""Shao-inspired dynamic IAPF obstacle avoidance for Layer 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from .alpha_flocking import ParamsLike as AlphaParamsLike, alpha_flocking_control
from .beta_obstacle import BetaParamsLike, beta_obstacle_control
from .gamma_navigation import GammaAgent, GammaParamsLike, gamma_navigation_control
from .obstacles import CircleObstacle
from .utils import as_2d_array, as_vector2, clip_by_norm


@dataclass(frozen=True)
class DynamicIAPFParams:
    """Parameters for prediction-based dynamic obstacle avoidance."""

    prediction_horizon: float = 3.0
    influence_distance: float = 3.0
    safe_distance: float = 0.35
    k_repulse: float = 1.2
    k_velocity: float = 0.8
    k_tangent: float = 0.6
    k_obs: float = 1.5
    max_obs_speed: float = 2.0
    agent_radius: float = 0.0
    eps: float = 1e-6
    use_tangent: bool = True

    def __post_init__(self) -> None:
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        if self.influence_distance <= 0:
            raise ValueError("influence_distance must be positive")
        if self.safe_distance < 0:
            raise ValueError("safe_distance must be non-negative")
        if self.k_repulse < 0 or self.k_velocity < 0 or self.k_tangent < 0 or self.k_obs < 0:
            raise ValueError("dynamic IAPF gains must be non-negative")
        if self.max_obs_speed <= 0:
            raise ValueError("max_obs_speed must be positive")
        if self.agent_radius < 0:
            raise ValueError("agent_radius must be non-negative")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


DynamicParamsLike = Optional[Union[DynamicIAPFParams, Mapping[str, Union[float, bool]]]]


def as_dynamic_iapf_params(params: DynamicParamsLike = None) -> DynamicIAPFParams:
    """Normalize None/dict/dataclass inputs to DynamicIAPFParams."""
    if params is None:
        return DynamicIAPFParams()
    if isinstance(params, DynamicIAPFParams):
        return params
    return DynamicIAPFParams(**dict(params))


def closest_approach(
    q: np.ndarray,
    p: np.ndarray,
    obstacle: CircleObstacle,
    params: DynamicParamsLike = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute CPA time, predicted relative position, clearance, and closing speed."""
    prm = as_dynamic_iapf_params(params)
    q_arr = as_2d_array(q, "q")
    p_arr = as_2d_array(p, "p", expected_rows=q_arr.shape[0])
    rel_pos = q_arr - obstacle.center[None, :]
    rel_vel = p_arr - obstacle.velocity[None, :]
    rel_speed_sq = np.sum(rel_vel * rel_vel, axis=1)
    t_star = -np.sum(rel_pos * rel_vel, axis=1) / (rel_speed_sq + prm.eps)
    t_star = np.clip(t_star, 0.0, prm.prediction_horizon)
    r_pred = rel_pos + t_star[:, None] * rel_vel
    d_pred = np.linalg.norm(r_pred, axis=1) - obstacle.radius - prm.agent_radius - prm.safe_distance
    closing_speed = np.maximum(0.0, -np.sum(rel_pos * rel_vel, axis=1) / (np.linalg.norm(rel_pos, axis=1) + prm.eps))
    return t_star, r_pred, d_pred, closing_speed


def dynamic_obstacle_risk(d_pred: np.ndarray, params: DynamicParamsLike = None) -> np.ndarray:
    """Smooth risk weight that is zero outside the dynamic influence distance."""
    prm = as_dynamic_iapf_params(params)
    d = np.asarray(d_pred, dtype=float)
    positive_d = np.maximum(d, 0.0)
    raw = (1.0 / (positive_d + prm.eps)) - (1.0 / prm.influence_distance)
    raw = np.maximum(raw, 0.0)
    raw[d >= prm.influence_distance] = 0.0
    # Keep the velocity command bounded when predicted clearance is near/under zero.
    return np.minimum(raw, 1.0 / prm.eps**0.25)


def dynamic_inhibiting_velocity(
    q: np.ndarray,
    p: np.ndarray,
    obstacles: Optional[Sequence[CircleObstacle]],
    goal: np.ndarray,
    params: DynamicParamsLike = None,
) -> np.ndarray:
    """Compute dynamic obstacle inhibiting velocity for each agent."""
    prm = as_dynamic_iapf_params(params)
    q_arr = as_2d_array(q, "q")
    p_arr = as_2d_array(p, "p", expected_rows=q_arr.shape[0])
    goal_vec = as_vector2(goal, "goal")
    v_obs = np.zeros_like(q_arr)
    if not obstacles:
        return v_obs

    for obstacle in obstacles:
        if not obstacle.dynamic and np.linalg.norm(obstacle.velocity) <= prm.eps:
            continue
        _, r_pred, d_pred, closing_speed = closest_approach(q_arr, p_arr, obstacle, prm)
        risk = dynamic_obstacle_risk(d_pred, prm)
        active = risk > 0.0
        if not np.any(active):
            continue

        pred_norms = np.linalg.norm(r_pred, axis=1, keepdims=True)
        normals = r_pred / (pred_norms + prm.eps)
        tangent = np.column_stack([-normals[:, 1], normals[:, 0]])
        if prm.use_tangent and prm.k_tangent > 0.0:
            to_goal = goal_vec[None, :] - q_arr
            signs = np.sign(np.sum(tangent * to_goal, axis=1, keepdims=True))
            signs[signs == 0.0] = 1.0
            tangent = tangent * signs
            tangent_term = prm.k_tangent * tangent
        else:
            tangent_term = np.zeros_like(q_arr)

        speed_term = prm.k_velocity * closing_speed[:, None] * normals
        repulse_term = prm.k_repulse * normals
        contribution = risk[:, None] * (repulse_term + speed_term + tangent_term)
        contribution[~active] = 0.0
        v_obs += contribution
    return clip_by_norm(v_obs, prm.max_obs_speed)


def dynamic_iapf_control(
    q: np.ndarray,
    p: np.ndarray,
    obstacles: Optional[Sequence[CircleObstacle]],
    goal: np.ndarray,
    params: DynamicParamsLike = None,
) -> np.ndarray:
    """Convert dynamic inhibiting velocity to an acceleration-like control term."""
    prm = as_dynamic_iapf_params(params)
    return prm.k_obs * dynamic_inhibiting_velocity(q, p, obstacles, goal, prm)


def dynamic_iapf_diagnostics(
    q: np.ndarray,
    p: np.ndarray,
    obstacles: Optional[Sequence[CircleObstacle]],
    params: DynamicParamsLike = None,
) -> dict[str, float]:
    """Return scalar diagnostics for dynamic obstacle risk logging."""
    prm = as_dynamic_iapf_params(params)
    q_arr = as_2d_array(q, "q")
    p_arr = as_2d_array(p, "p", expected_rows=q_arr.shape[0])
    if not obstacles:
        return {"active_dynamic_risk_count": 0.0, "min_predicted_obstacle_clearance": float("inf")}
    active_count = 0
    min_clearance = float("inf")
    for obstacle in obstacles:
        _, _, d_pred, _ = closest_approach(q_arr, p_arr, obstacle, prm)
        risk = dynamic_obstacle_risk(d_pred, prm)
        active_count += int(np.sum(risk > 0.0))
        min_clearance = min(min_clearance, float(np.min(d_pred)))
    return {"active_dynamic_risk_count": float(active_count), "min_predicted_obstacle_clearance": min_clearance}


def flocking_with_dynamic_iapf_control(
    q: np.ndarray,
    p: np.ndarray,
    obstacles: Optional[Sequence[CircleObstacle]],
    goal: np.ndarray,
    alpha_params: AlphaParamsLike = None,
    gamma: Optional[GammaAgent] = None,
    gamma_params: GammaParamsLike = None,
    beta_params: BetaParamsLike = None,
    dynamic_params: DynamicParamsLike = None,
    include_beta: bool = True,
) -> np.ndarray:
    """Combine alpha, beta, gamma, and dynamic IAPF controls."""
    target = GammaAgent(q=goal) if gamma is None else gamma
    u_alpha = alpha_flocking_control(q, p, alpha_params)
    u_gamma = gamma_navigation_control(q, p, target, gamma_params)
    u_beta = beta_obstacle_control(q, p, obstacles, beta_params) if include_beta else np.zeros_like(u_alpha)
    u_dyn = dynamic_iapf_control(q, p, obstacles, goal, dynamic_params)
    return u_alpha + u_beta + u_gamma + u_dyn
