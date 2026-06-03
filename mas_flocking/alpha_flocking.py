"""Olfati-Saber alpha-agent free-space flocking controller.

This module implements the Algorithm 1 interaction terms from
Olfati-Saber (2006): a gradient-based distance regulation term plus a
velocity-consensus damping term on a weighted proximity graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Union

import numpy as np

from .utils import as_2d_array


@dataclass(frozen=True)
class AlphaFlockingParams:
    """Parameters for Olfati-Saber alpha-agent free-space flocking."""

    epsilon: float = 0.1
    h: float = 0.2
    d: float = 1.2
    r: float = 3.0
    a: float = 5.0
    b: float = 5.0
    c1_alpha: float = 1.0
    c2_alpha: float = 2.0
    # ---- 新增安全屏障可调参数 ----
    d_safe_agent: float = 0.40
    k_barrier: float = 5.0

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0.0 <= self.h < 1.0:
            raise ValueError("h must satisfy 0 <= h < 1")
        if self.d <= 0 or self.r <= 0:
            raise ValueError("d and r must be positive")
        if self.d >= self.r:
            raise ValueError("d must be smaller than sensing radius r")
        if self.a <= 0 or self.b <= 0:
            raise ValueError("a and b must be positive")
        if self.c1_alpha < 0 or self.c2_alpha < 0:
            raise ValueError("c1_alpha and c2_alpha must be non-negative")
        # ---- 新增参数校验 ----
        if self.d_safe_agent < 0:
            raise ValueError("d_safe_agent must be non-negative")
        if self.k_barrier < 0:
            raise ValueError("k_barrier must be non-negative")


ParamsLike = Optional[Union[AlphaFlockingParams, Mapping[str, float]]]


def as_alpha_params(params: ParamsLike = None) -> AlphaFlockingParams:
    """Normalize None/dict/dataclass inputs to an AlphaFlockingParams object."""
    if params is None:
        return AlphaFlockingParams()
    if isinstance(params, AlphaFlockingParams):
        return params
    return AlphaFlockingParams(**dict(params))


def sigma_norm(z: np.ndarray, epsilon: float = 0.1, axis: int = -1) -> np.ndarray:
    """Smooth sigma norm: (sqrt(1 + eps * ||z||^2) - 1) / eps.

    Scalars are interpreted as Euclidean distances. Arrays whose last dimension
    stores vector coordinates are interpreted as vectors and reduced over axis.
    """
    arr = np.asarray(z, dtype=float)
    if arr.ndim == 0:
        norm_sq = arr * arr
    else:
        norm_sq = np.sum(arr * arr, axis=axis)
    return (np.sqrt(1.0 + float(epsilon) * norm_sq) - 1.0) / float(epsilon)


def sigma_1(z: np.ndarray) -> np.ndarray:
    """Uneven sigmoid helper z / sqrt(1 + z^2)."""
    arr = np.asarray(z, dtype=float)
    return arr / np.sqrt(1.0 + arr * arr)


def bump_function(z: np.ndarray, h: float = 0.2) -> np.ndarray:
    """Olfati-Saber smooth bump rho_h(z)."""
    if not 0.0 <= h < 1.0:
        raise ValueError("h must satisfy 0 <= h < 1")
    arr = np.asarray(z, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    out[arr < h] = 1.0
    middle = (arr >= h) & (arr <= 1.0)
    out[middle] = 0.5 * (1.0 + np.cos(np.pi * (arr[middle] - h) / (1.0 - h)))
    return out


def phi(z: np.ndarray, a: float = 5.0, b: float = 5.0) -> np.ndarray:
    """Olfati-Saber action function phi(z)."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    c = abs(a - b) / np.sqrt(4.0 * a * b)
    return 0.5 * ((a + b) * sigma_1(np.asarray(z, dtype=float) + c) + (a - b))


def phi_alpha(z: np.ndarray, params: ParamsLike = None) -> np.ndarray:
    """Finite-cutoff alpha action function phi_alpha(z)."""
    p = as_alpha_params(params)
    r_alpha = sigma_norm(np.asarray(p.r), epsilon=p.epsilon)
    d_alpha = sigma_norm(np.asarray(p.d), epsilon=p.epsilon)
    return bump_function(np.asarray(z, dtype=float) / r_alpha, h=p.h) * phi(np.asarray(z, dtype=float) - d_alpha, a=p.a, b=p.b)


def pairwise_sigma_distances(q: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    """Pairwise sigma-norm distances for all agent pairs."""
    q_arr = as_2d_array(q, "q")
    diff = q_arr[None, :, :] - q_arr[:, None, :]
    return sigma_norm(diff, epsilon=epsilon, axis=-1)


def sigma_unit_vectors(q: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    """Smooth direction vectors n_ij from agent i to agent j."""
    q_arr = as_2d_array(q, "q")
    diff = q_arr[None, :, :] - q_arr[:, None, :]
    euclid_sq = np.sum(diff * diff, axis=-1, keepdims=True)
    return diff / np.sqrt(1.0 + float(epsilon) * euclid_sq)


def alpha_adjacency_matrix(q: np.ndarray, params: ParamsLike = None) -> np.ndarray:
    """Weighted proximity adjacency a_ij(q) using sigma-norm distances."""
    p = as_alpha_params(params)
    sigma_dists = pairwise_sigma_distances(q, epsilon=p.epsilon)
    r_alpha = sigma_norm(np.asarray(p.r), epsilon=p.epsilon)
    adjacency = bump_function(sigma_dists / r_alpha, h=p.h)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def alpha_flocking_control(q: np.ndarray, p: np.ndarray, params: ParamsLike = None) -> np.ndarray:
    """Compute Algorithm 1 alpha-agent free-space flocking acceleration with hard-core safety barrier."""
    prm = as_alpha_params(params)
    q_arr = as_2d_array(q, "q")
    p_arr = as_2d_array(p, "p", expected_rows=q_arr.shape[0])

    sigma_dists = pairwise_sigma_distances(q_arr, epsilon=prm.epsilon)
    action = phi_alpha(sigma_dists, prm)
    np.fill_diagonal(action, 0.0)

    directions = sigma_unit_vectors(q_arr, epsilon=prm.epsilon)
    gradient_term = prm.c1_alpha * np.sum(action[:, :, None] * directions, axis=1)

    adjacency = alpha_adjacency_matrix(q_arr, prm)
    velocity_diff = p_arr[None, :, :] - p_arr[:, None, :]
    consensus_term = prm.c2_alpha * np.sum(adjacency[:, :, None] * velocity_diff, axis=1)

    # ---------------- 核心修改：使用动态参数 ----------------
    # 1. 计算真实的 pairwise 欧氏距离
    diff = q_arr[None, :, :] - q_arr[:, None, :]
    euclid_dists = np.linalg.norm(diff, axis=-1)

    # 2. 设定硬安全距离阈值与排斥增益（从配置中读取）
    d_safe_agent = prm.d_safe_agent
    k_barrier = prm.k_barrier  # 屏障排斥增益

    barrier_force = np.zeros_like(q_arr)
    n_agents = q_arr.shape[0]

    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            dist = euclid_dists[i, j]
            # 当两智能体距离突破硬安全边界时，触发强排斥
            if dist < d_safe_agent:
                dir_vec = (q_arr[i] - q_arr[j]) / (dist + 1e-12)
                force_mag = k_barrier * (1.0 / (dist + 1e-12) - 1.0 / d_safe_agent) ** 2
                barrier_force[i] += force_mag * dir_vec

    # 3. 将原控制律与硬屏障力叠加
    return gradient_term + consensus_term + barrier_force
