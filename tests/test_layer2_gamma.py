"""Tests for Layer 2 gamma-agent target navigation."""

from __future__ import annotations

import unittest

import numpy as np

from mas_flocking.alpha_flocking import AlphaFlockingParams, alpha_flocking_control
from mas_flocking.gamma_navigation import (
    GammaAgent,
    GammaNavigationParams,
    free_flocking_with_navigation_control,
    gamma_navigation_control,
    vector_sigma_1,
)
from mas_flocking.metrics import center_of_mass_goal_distance, mean_goal_distance, min_agent_distance
from mas_flocking.simulator import FlockingEnv


class TestGammaNavigation(unittest.TestCase):
    def test_vector_sigma_1_zero_and_saturates(self) -> None:
        np.testing.assert_allclose(vector_sigma_1(np.array([0.0, 0.0])), np.array([0.0, 0.0]))
        values = vector_sigma_1(np.array([[3.0, 4.0], [1.0, 0.0]]))
        self.assertLess(float(np.linalg.norm(values[0])), 1.0)
        self.assertLess(float(np.linalg.norm(values[1])), 1.0)

    def test_agent_left_of_goal_is_accelerated_right(self) -> None:
        gamma = GammaAgent(q=np.array([10.0, 0.0]), p=np.zeros(2))
        q = np.array([[0.0, 0.0]])
        p = np.zeros((1, 2))
        u = gamma_navigation_control(q, p, gamma, GammaNavigationParams(c1_gamma=1.0, c2_gamma=0.0))
        self.assertGreater(float(u[0, 0]), 0.0)
        self.assertAlmostEqual(float(u[0, 1]), 0.0)

    def test_velocity_damping_opposes_velocity_error(self) -> None:
        gamma = GammaAgent(q=np.array([0.0, 0.0]), p=np.zeros(2))
        q = np.array([[0.0, 0.0]])
        p = np.array([[2.0, -1.0]])
        u = gamma_navigation_control(q, p, gamma, GammaNavigationParams(c1_gamma=0.0, c2_gamma=1.5))
        self.assertLess(float(u[0, 0]), 0.0)
        self.assertGreater(float(u[0, 1]), 0.0)

    def test_composed_control_matches_alpha_plus_gamma(self) -> None:
        alpha_params = AlphaFlockingParams(d=1.0, r=3.0)
        gamma_params = GammaNavigationParams(c1_gamma=0.7, c2_gamma=0.9)
        gamma = GammaAgent(q=np.array([4.0, 2.0]), p=np.array([0.1, 0.0]))
        q = np.array([[0.0, 0.0], [1.2, 0.0], [0.6, 1.0]])
        p = np.array([[0.2, 0.0], [0.0, 0.1], [-0.1, 0.0]])
        expected = alpha_flocking_control(q, p, alpha_params) + gamma_navigation_control(q, p, gamma, gamma_params)
        actual = free_flocking_with_navigation_control(q, p, alpha_params, gamma, gamma_params)
        np.testing.assert_allclose(actual, expected)

    def test_short_integrated_run_moves_group_toward_goal(self) -> None:
        alpha_params = AlphaFlockingParams(d=1.0, r=3.0, c1_alpha=1.0, c2_alpha=2.0)
        gamma_params = GammaNavigationParams(c1_gamma=1.0, c2_gamma=1.2)
        gamma = GammaAgent(q=np.array([14.0, 6.0]), p=np.zeros(2))
        q0 = np.array(
            [
                [1.0, 4.8], [2.0, 4.8], [3.0, 4.8],
                [1.0, 6.0], [2.0, 6.0], [3.0, 6.0],
                [1.0, 7.2], [2.0, 7.2], [3.0, 7.2],
            ],
            dtype=float,
        )
        p0 = np.zeros((q0.shape[0], 2))
        env = FlockingEnv(
            n_agents=q0.shape[0],
            dt=0.02,
            world_size=(20.0, 12.0),
            v_max=3.0,
            u_max=20.0,
            boundary_mode="none",
        )
        state = env.reset(init_mode="custom", q0=q0, p0=p0)
        start_mean_goal_distance = mean_goal_distance(state.q, gamma.q)
        start_com_goal_distance = center_of_mass_goal_distance(state.q, gamma.q)
        for _ in range(400):
            u = free_flocking_with_navigation_control(state.q, state.p, alpha_params, gamma, gamma_params)
            state = env.step(u)
        self.assertTrue(np.all(np.isfinite(state.q)))
        self.assertTrue(np.all(np.isfinite(state.p)))
        self.assertLess(mean_goal_distance(state.q, gamma.q), start_mean_goal_distance)
        self.assertLess(center_of_mass_goal_distance(state.q, gamma.q), start_com_goal_distance)
        self.assertGreater(min_agent_distance(state.q), 0.05)


if __name__ == "__main__":
    unittest.main()
