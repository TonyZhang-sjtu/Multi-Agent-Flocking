"""Layer 0 smoke and numerical tests."""

from __future__ import annotations

import unittest

import numpy as np

from mas_flocking.controllers import goal_pd_control, zero_control
from mas_flocking.metrics import (
    adjacency_matrix,
    algebraic_connectivity,
    average_flocking_error,
    cohesion_radius,
    min_agent_distance,
    normalized_velocity_mismatch,
    num_connected_components,
    relative_connectivity,
    speed_statistics,
)
from mas_flocking.obstacles import CircleObstacle, ScriptedCircleObstacle
from mas_flocking.simulator import FlockingEnv


class TestLayer0(unittest.TestCase):
    def test_zero_control_keeps_zero_velocity_agents_stationary(self) -> None:
        env = FlockingEnv(n_agents=2, dt=0.1, world_size=(5.0, 5.0), seed=1)
        q0 = np.array([[1.0, 1.0], [2.0, 2.0]])
        p0 = np.zeros((2, 2))
        state = env.reset(init_mode="custom", q0=q0, p0=p0)
        state = env.step(zero_control(state.q, state.p))
        np.testing.assert_allclose(state.q, q0)
        np.testing.assert_allclose(state.p, p0)

    def test_velocity_and_acceleration_limits(self) -> None:
        env = FlockingEnv(n_agents=3, dt=0.1, world_size=(10.0, 10.0), v_max=1.0, u_max=2.0, seed=2)
        state = env.reset(init_mode="random_center", p0=np.zeros((3, 2)))
        state = env.step(np.full((3, 2), 100.0))
        self.assertLessEqual(float(np.max(np.linalg.norm(state.p, axis=1))), 1.0 + 1e-9)
        self.assertLessEqual(float(np.max(np.linalg.norm(env.last_u, axis=1))), 2.0 + 1e-9)
        self.assertTrue(np.all(state.q >= 0.0))
        self.assertTrue(np.all(state.q <= env.world_size[None, :]))

    def test_reflect_boundary(self) -> None:
        env = FlockingEnv(n_agents=1, dt=1.0, world_size=(2.0, 2.0), v_max=10.0, u_max=10.0, boundary_mode="reflect")
        state = env.reset(init_mode="custom", q0=np.array([[1.9, 1.0]]), p0=np.array([[1.0, 0.0]]))
        state = env.step(np.zeros((1, 2)))
        self.assertAlmostEqual(float(state.q[0, 0]), 2.0)
        self.assertLess(float(state.p[0, 0]), 0.0)

    def test_goal_pd_reduces_goal_distance(self) -> None:
        env = FlockingEnv(n_agents=5, dt=0.02, world_size=(20.0, 12.0), seed=3)
        state = env.reset(init_mode="random_left")
        goal = np.array([18.0, 6.0])
        start_dist = float(np.mean(np.linalg.norm(state.q - goal[None, :], axis=1)))
        for _ in range(250):
            state = env.step(goal_pd_control(state.q, state.p, goal))
        end_dist = float(np.mean(np.linalg.norm(state.q - goal[None, :], axis=1)))
        self.assertLess(end_dist, start_dist)

    def test_graph_metrics(self) -> None:
        q = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
        adjacency = adjacency_matrix(q, radius=1.5)
        np.testing.assert_allclose(adjacency, adjacency.T)
        np.testing.assert_allclose(np.diag(adjacency), np.zeros(3))
        self.assertEqual(num_connected_components(q, radius=1.5), 2)
        self.assertAlmostEqual(algebraic_connectivity(q, radius=1.5), 0.0)
        self.assertAlmostEqual(relative_connectivity(q, radius=1.5), 0.5)
        self.assertAlmostEqual(min_agent_distance(q), 1.0)

    def test_layer4_reusable_flocking_metrics(self) -> None:
        q_ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        q_same_distances = q_ref + np.array([2.0, 3.0])
        q_deformed = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
        p_consensus = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        p_mismatch = np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0]])

        self.assertAlmostEqual(average_flocking_error(q_same_distances, q_ref), 0.0)
        self.assertGreater(average_flocking_error(q_deformed, q_ref), 0.0)
        self.assertGreater(cohesion_radius(q_ref), 0.0)
        self.assertAlmostEqual(normalized_velocity_mismatch(p_consensus), 0.0)
        self.assertGreater(normalized_velocity_mismatch(p_mismatch), 0.0)
        mean_speed, max_speed, speed_std = speed_statistics(p_mismatch)
        self.assertGreater(mean_speed, 0.0)
        self.assertAlmostEqual(max_speed, 2.0)
        self.assertGreater(speed_std, 0.0)

    def test_dynamic_obstacle_steps_and_reflects(self) -> None:
        obs = CircleObstacle(center=(1.8, 1.0), radius=0.2, velocity=(1.0, 0.0), name="dyn")
        obs.step(dt=1.0, world_size=np.array([2.0, 2.0]), boundary_mode="reflect")
        self.assertAlmostEqual(float(obs.center[0]), 1.8)
        self.assertLess(float(obs.velocity[0]), 0.0)

    def test_scripted_obstacle_updates_position_and_velocity(self) -> None:
        obs = ScriptedCircleObstacle(
            center=(1.0, 2.0),
            radius=0.2,
            base_velocity=(0.5, 0.0),
            acceleration=(0.2, 0.0),
            sine_amplitude=(0.0, 1.0),
            sine_omega=1.0,
            name="scripted",
        )
        obs.step(dt=1.0, world_size=None, boundary_mode="none")
        np.testing.assert_allclose(obs.center, np.array([1.6, 2.0 + np.sin(1.0)]))
        np.testing.assert_allclose(obs.velocity, np.array([0.7, np.cos(1.0)]))


if __name__ == "__main__":
    unittest.main()
