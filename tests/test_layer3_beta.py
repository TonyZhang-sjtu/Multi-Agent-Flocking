"""Tests for Layer 3 static beta-agent obstacle avoidance."""

from __future__ import annotations

import unittest

import numpy as np

from mas_flocking.alpha_flocking import AlphaFlockingParams
from mas_flocking.beta_obstacle import (
    BetaObstacleParams,
    beta_obstacle_control,
    flocking_with_static_obstacle_control,
    project_to_obstacle_boundary,
)
from mas_flocking.gamma_navigation import GammaAgent, GammaNavigationParams
from mas_flocking.layer3_static_obstacles import build_obstacles, sample_random_left_with_min_separation
from mas_flocking.metrics import collision_count, mean_goal_distance, min_obstacle_clearance
from mas_flocking.obstacles import CircleObstacle
from mas_flocking.simulator import FlockingEnv


class TestBetaObstacleAvoidance(unittest.TestCase):
    def test_projection_lands_on_inflated_boundary(self) -> None:
        obstacle = CircleObstacle(center=(2.0, 3.0), radius=1.0, dynamic=False)
        q = np.array([[4.0, 3.0], [2.0, 5.0], [0.0, 3.0]])
        projected = project_to_obstacle_boundary(q, obstacle, agent_radius=0.2)
        dists = np.linalg.norm(projected - obstacle.center[None, :], axis=1)
        np.testing.assert_allclose(dists, np.full(3, 1.2))

    def test_far_agent_has_zero_beta_control(self) -> None:
        obstacle = CircleObstacle(center=(0.0, 0.0), radius=1.0, dynamic=False)
        params = BetaObstacleParams(r_beta=2.0, c1_beta=8.0, c2_beta=3.0)
        q = np.array([[5.0, 0.0]])
        p = np.zeros((1, 2))
        u = beta_obstacle_control(q, p, [obstacle], params)
        np.testing.assert_allclose(u, np.zeros((1, 2)), atol=1e-12)

    def test_agent_left_of_obstacle_is_repulsed_left(self) -> None:
        obstacle = CircleObstacle(center=(0.0, 0.0), radius=1.0, dynamic=False)
        params = BetaObstacleParams(r_beta=2.5, c1_beta=8.0, c2_beta=0.0, agent_radius=0.0)
        q = np.array([[-1.4, 0.0]])
        p = np.zeros((1, 2))
        u = beta_obstacle_control(q, p, [obstacle], params)
        self.assertLess(float(u[0, 0]), 0.0)
        self.assertAlmostEqual(float(u[0, 1]), 0.0)

    def test_projected_velocity_damps_normal_not_tangent(self) -> None:
        obstacle = CircleObstacle(center=(0.0, 0.0), radius=1.0, dynamic=False)
        params = BetaObstacleParams(r_beta=2.5, c1_beta=0.0, c2_beta=3.0, agent_radius=0.0)
        q = np.array([[1.5, 0.0], [1.5, 0.0]])
        p = np.array([[-1.0, 0.0], [0.0, 1.0]])
        u = beta_obstacle_control(q, p, [obstacle], params)
        self.assertGreater(float(u[0, 0]), 0.0)
        np.testing.assert_allclose(u[1], np.zeros(2), atol=1e-12)

    def test_collision_count_and_clearance(self) -> None:
        obstacle = CircleObstacle(center=(0.0, 0.0), radius=1.0, dynamic=False)
        q = np.array([[0.5, 0.0], [1.5, 0.0], [3.0, 0.0]])
        self.assertEqual(collision_count(q, [obstacle], agent_radius=0.2), 1)
        self.assertLess(min_obstacle_clearance(q, [obstacle], agent_radius=0.2), 0.0)

    def test_short_integrated_run_avoids_static_obstacle_better_than_baseline(self) -> None:
        obstacle = CircleObstacle(center=(5.0, 0.0), radius=0.8, dynamic=False)
        alpha_params = AlphaFlockingParams(d=1.0, r=3.0, c1_alpha=1.0, c2_alpha=2.0)
        gamma = GammaAgent(q=(10.0, 0.0), p=(0.0, 0.0))
        gamma_params = GammaNavigationParams(c1_gamma=1.0, c2_gamma=1.2)
        beta_params = BetaObstacleParams(r_beta=2.0, c1_beta=8.0, c2_beta=3.0, agent_radius=0.12)
        q0 = np.array([[1.0, -0.5], [1.0, 0.0], [1.0, 0.5], [2.0, -0.5], [2.0, 0.0], [2.0, 0.5]])
        p0 = np.zeros((q0.shape[0], 2))

        def run(use_beta: bool) -> tuple[int, float]:
            env = FlockingEnv(
                n_agents=q0.shape[0],
                dt=0.02,
                world_size=(12.0, 4.0),
                v_max=3.0,
                u_max=25.0,
                boundary_mode="none",
                obstacles=[CircleObstacle(center=(5.0, 0.0), radius=0.8, dynamic=False)],
            )
            state = env.reset(init_mode="custom", q0=q0, p0=p0)
            total_collisions = 0
            min_clearance = float("inf")
            for _ in range(350):
                if use_beta:
                    u = flocking_with_static_obstacle_control(state.q, state.p, env.obstacles, alpha_params, gamma, gamma_params, beta_params)
                else:
                    u = flocking_with_static_obstacle_control(
                        state.q,
                        state.p,
                        [],
                        alpha_params,
                        gamma,
                        gamma_params,
                        beta_params,
                    )
                state = env.step(u)
                total_collisions += collision_count(state.q, env.obstacles, agent_radius=beta_params.agent_radius)
                min_clearance = min(min_clearance, min_obstacle_clearance(state.q, env.obstacles, agent_radius=beta_params.agent_radius))
            return total_collisions, min_clearance

        baseline_collisions, baseline_clearance = run(use_beta=False)
        beta_collisions, beta_clearance = run(use_beta=True)
        self.assertLess(beta_collisions, baseline_collisions)
        self.assertGreater(beta_clearance, baseline_clearance)

    def test_layer3_default_beta_tuning_keeps_clearance_positive(self) -> None:
        world_size = np.array([20.0, 12.0])
        rng = np.random.default_rng(3)
        q0 = sample_random_left_with_min_separation(20, world_size, rng, min_sep=0.75)
        p0 = rng.normal(loc=0.0, scale=0.05, size=(20, 2))
        env = FlockingEnv(
            n_agents=20,
            dt=0.01,
            world_size=tuple(world_size),
            v_max=2.5,
            u_max=25.0,
            seed=3,
            boundary_mode="none",
            obstacles=build_obstacles("layer0_static"),
        )
        state = env.reset(init_mode="custom", q0=q0, p0=p0)
        alpha_params = AlphaFlockingParams(d=1.2, r=3.0, c1_alpha=1.0, c2_alpha=2.0, d_safe_agent=0.40, k_barrier=15.0)
        beta_params = BetaObstacleParams(r_beta=1.5, c1_beta=6.0, c2_beta=3.0, agent_radius=0.12)
        gamma = GammaAgent(q=(18.0, 6.0), p=(0.0, 0.0))
        gamma_params = GammaNavigationParams(c1_gamma=1.5, c2_gamma=1.2)

        min_clearance = float("inf")
        total_collisions = 0
        for _ in range(2400):
            u = flocking_with_static_obstacle_control(state.q, state.p, env.obstacles, alpha_params, gamma, gamma_params, beta_params)
            state = env.step(u)
            gamma.step(0.01)
            min_clearance = min(min_clearance, min_obstacle_clearance(state.q, env.obstacles, agent_radius=beta_params.agent_radius))
            total_collisions += collision_count(state.q, env.obstacles, agent_radius=beta_params.agent_radius)

        self.assertGreater(min_clearance, 0.0)
        self.assertEqual(total_collisions, 0)
        self.assertLess(mean_goal_distance(state.q, gamma.q), 1.0)


if __name__ == "__main__":
    unittest.main()
