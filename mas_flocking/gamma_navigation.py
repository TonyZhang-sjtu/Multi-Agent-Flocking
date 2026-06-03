"""Gamma-agent target navigation for Olfati-Saber Algorithm 2.

Layer 2 combines the alpha-agent interaction terms from Algorithm 1 with a
virtual gamma-agent that represents the desired rendezvous/target state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from .alpha_flocking import AlphaFlockingParams, ParamsLike as AlphaParamsLike, alpha_flocking_control
from .utils import as_2d_array, as_vector2


@dataclass(frozen=True)
class GammaNavigationParams:
    """Navigation gains for the gamma-agent feedback term."""

    c1_gamma: float = 1.0
    c2_gamma: float = 1.2

    def __post_init__(self) -> None:
        if self.c1_gamma < 0 or self.c2_gamma < 0:
            raise ValueError("c1_gamma and c2_gamma must be non-negative")


GammaParamsLike = Optional[Union[GammaNavigationParams, Mapping[str, float]]]


def as_gamma_params(params: GammaParamsLike = None) -> GammaNavigationParams:
    """Normalize None/dict/dataclass inputs to a GammaNavigationParams object."""
    if params is None:
        return GammaNavigationParams()
    if isinstance(params, GammaNavigationParams):
        return params
    return GammaNavigationParams(**dict(params))


@dataclass
class GammaAgent:
    """Virtual target agent state used by the gamma navigation term."""

    q: np.ndarray
    p: np.ndarray

    def __init__(self, q: Sequence[float] = (18.0, 6.0), p: Sequence[float] = (0.0, 0.0)) -> None:
        self.q = as_vector2(q, "gamma.q").copy()
        self.p = as_vector2(p, "gamma.p").copy()

    def step(self, dt: float) -> None:
        """Advance a possibly dynamic gamma-agent by one Euler step."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.q = self.q + self.p * float(dt)

    def as_dict(self) -> dict:
        """Return a serializable snapshot of the target state."""
        return {"q": self.q.copy(), "p": self.p.copy()}


def vector_sigma_1(z: np.ndarray) -> np.ndarray:
    """Vector saturation z / sqrt(1 + ||z||^2).

    This is the gamma-agent navigational sigma_1. It differs from the scalar
    action-function helper used by the alpha controller.
    """
    arr = np.asarray(z, dtype=float)
    if arr.ndim == 1:
        if arr.shape != (2,):
            raise ValueError(f"z must have shape [2] or [N, 2], got {arr.shape}")
        return arr / np.sqrt(1.0 + float(np.dot(arr, arr)))
    if arr.ndim == 2 and arr.shape[1] == 2:
        denom = np.sqrt(1.0 + np.sum(arr * arr, axis=1, keepdims=True))
        return arr / denom
    raise ValueError(f"z must have shape [2] or [N, 2], got {arr.shape}")


def gamma_navigation_control(
    q: np.ndarray,
    p: np.ndarray,
    gamma: GammaAgent,
    params: GammaParamsLike = None,
) -> np.ndarray:
    """Compute the Algorithm 2 gamma-agent target navigation acceleration with target decay."""
    prm = as_gamma_params(params)
    q_arr = as_2d_array(q, "q")
    p_arr = as_2d_array(p, "p", expected_rows=q_arr.shape[0])
    if not isinstance(gamma, GammaAgent):
        raise TypeError("gamma must be a GammaAgent")

    pos_error = q_arr - gamma.q[None, :]
    vel_error = p_arr - gamma.p[None, :]

    # ---------------- 核心修改部分 ----------------
    # 计算每个智能体到目标点的欧氏距离
    dists = np.linalg.norm(pos_error, axis=1, keepdims=True)

    # 设定一个衰减半径（例如：期望晶格间距 d 的 2 倍，约 2.4 米）
    r_decay = 2.4

    # 使用指数函数平滑衰减：距离越近，decay_scale 越接近 0
    decay_scale = 1.0 - np.exp(-(dists / r_decay) ** 2)

    # 将衰减因子应用到位置误差引力项上
    return -prm.c1_gamma * decay_scale * vector_sigma_1(pos_error) - prm.c2_gamma * vel_error
    # ---------------------------------------------


def free_flocking_with_navigation_control(
    q: np.ndarray,
    p: np.ndarray,
    alpha_params: AlphaParamsLike = None,
    gamma: Optional[GammaAgent] = None,
    gamma_params: GammaParamsLike = None,
) -> np.ndarray:
    """Combine alpha-agent flocking and gamma-agent navigation controls."""
    target = GammaAgent() if gamma is None else gamma
    u_alpha = alpha_flocking_control(q, p, alpha_params)
    u_gamma = gamma_navigation_control(q, p, target, gamma_params)
    return u_alpha + u_gamma
