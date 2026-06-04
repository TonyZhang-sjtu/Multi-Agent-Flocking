"""Tests for Layer 4 dynamic IAPF obstacle avoidance."""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np

from mas_flocking.alpha_flocking import AlphaFlockingParams
from mas_flocking.beta_obstacle import BetaObstacleParams
from mas_flocking.dynamic_iapf import (
    DynamicIAPFParams,
    closest_approach,
    dynamic_iapf_control,
    dynamic_inhibiting_velocity,
    dynamic_obstacle_risk,
    flocking_with_dynamic_iapf_control,
)
from mas_flocking.gamma_navigation import GammaAgent, GammaNavigationParams
from mas_flocking.layer4_dynamic_iapf import LAYER4_METRIC_KEYS, build_arg_parser, run_demo
from mas_flocking.metrics import collision_count
from mas_flocking.obstacles import CircleObstacle
from mas_flocking.simulator import FlockingEnv


class TestDynamicIAPF(unittest.TestCase):
    def test_far_or_receding_obstacle_has_zero_inhibiting_velocity(self) -> None:
        params = DynamicIAPFParams(influence_distance=2.0)
        obstacle = CircleObstacle(center=(10.0, 0.0), radius=0.5, velocity=(1.0, 0.0), dynamic=True)
        q = np.array([[0.0, 0.0]])
        p = np.array([[0.0, 0.0]])
        v_obs = dynamic_inhibiting_velocity(q, p, [obstacle], goal=np.array([20.0, 0.0]), params=params)
        np.testing.assert_allclose(v_obs, np.zeros((1, 2)), atol=1e-12)

    def test_closest_approach_for_crossing_case(self) -> None:
        params = DynamicIAPFParams(prediction_horizon=5.0, safe_distance=0.2)
        obstacle = CircleObstacle(center=(2.0, 1.0), radius=0.5, velocity=(0.0, -1.0), dynamic=True)
        q = np.array([[2.0, 0.0]])
        p = np.zeros((1, 2))
        t_star, r_pred, d_pred, closing_speed = closest_approach(q, p, obstacle, params)
        self.assertAlmostEqual(float(t_star[0]), 1.0, places=5)
        np.testing.assert_allclose(r_pred[0], np.zeros(2), atol=1e-5)
        self.assertLess(float(d_pred[0]), 0.0)
        self.assertGreater(float(closing_speed[0]), 0.0)

    def test_closest_approach_clearance_includes_agent_radius(self) -> None:
        params = DynamicIAPFParams(prediction_horizon=5.0, safe_distance=0.2, agent_radius=0.12)
        obstacle = CircleObstacle(center=(2.0, 1.0), radius=0.5, velocity=(0.0, -1.0), dynamic=True)
        q = np.array([[2.0, 0.0]])
        p = np.zeros((1, 2))
        _, _, d_pred, _ = closest_approach(q, p, obstacle, params)
        self.assertAlmostEqual(float(d_pred[0]), -0.82, places=5)

    def test_approaching_obstacle_has_higher_risk_than_receding(self) -> None:
        params = DynamicIAPFParams(influence_distance=3.0)
        close = dynamic_obstacle_risk(np.array([0.2]), params)
        far = dynamic_obstacle_risk(np.array([5.0]), params)
        self.assertGreater(float(close[0]), float(far[0]))
        self.assertAlmostEqual(float(far[0]), 0.0)

    def test_tangent_term_can_be_disabled(self) -> None:
        obstacle = CircleObstacle(center=(2.0, 1.0), radius=0.5, velocity=(0.0, -1.0), dynamic=True)
        q = np.array([[2.0, 0.0]])
        p = np.zeros((1, 2))
        goal = np.array([6.0, 1.0])
        with_tangent = dynamic_inhibiting_velocity(
            q,
            p,
            [obstacle],
            goal,
            DynamicIAPFParams(k_repulse=0.0, k_velocity=0.0, k_tangent=1.0, max_obs_speed=10.0, use_tangent=True),
        )
        without_tangent = dynamic_inhibiting_velocity(
            q,
            p,
            [obstacle],
            goal,
            DynamicIAPFParams(k_repulse=0.0, k_velocity=0.0, k_tangent=1.0, max_obs_speed=10.0, use_tangent=False),
        )
        self.assertGreater(float(np.linalg.norm(with_tangent[0])), 0.0)
        np.testing.assert_allclose(without_tangent, np.zeros((1, 2)), atol=1e-12)

    def test_max_obs_speed_clipping(self) -> None:
        obstacle = CircleObstacle(center=(2.0, 1.0), radius=0.5, velocity=(0.0, -1.0), dynamic=True)
        q = np.array([[2.0, 0.0]])
        p = np.zeros((1, 2))
        params = DynamicIAPFParams(k_repulse=100.0, k_velocity=100.0, k_tangent=100.0, max_obs_speed=0.25)
        v_obs = dynamic_inhibiting_velocity(q, p, [obstacle], np.array([6.0, 1.0]), params)
        self.assertLessEqual(float(np.linalg.norm(v_obs[0])), 0.250001)
        u_dyn = dynamic_iapf_control(q, p, [obstacle], np.array([6.0, 1.0]), params)
        self.assertLessEqual(float(np.linalg.norm(u_dyn[0])), params.k_obs * 0.250001)

    def test_short_integrated_run_with_dynamic_iapf_is_finite(self) -> None:
        obstacle = CircleObstacle(center=(5.0, 2.0), radius=0.55, velocity=(0.0, -0.35), dynamic=True)
        alpha_params = AlphaFlockingParams(d=1.0, r=3.0, c1_alpha=1.0, c2_alpha=2.0)
        beta_params = BetaObstacleParams(r_beta=1.5, c1_beta=3.0, c2_beta=2.0, agent_radius=0.12)
        gamma = GammaAgent(q=(10.0, 0.0), p=(0.0, 0.0))
        gamma_params = GammaNavigationParams(c1_gamma=1.5, c2_gamma=1.2)
        dynamic_params = DynamicIAPFParams(prediction_horizon=3.0, influence_distance=3.0)
        q0 = np.array([[1.0, -0.4], [1.0, 0.4], [2.0, -0.4], [2.0, 0.4]])
        p0 = np.zeros((q0.shape[0], 2))
        env = FlockingEnv(
            n_agents=q0.shape[0],
            dt=0.02,
            world_size=(12.0, 4.0),
            v_max=3.0,
            u_max=30.0,
            boundary_mode="none",
            obstacles=[obstacle],
        )
        state = env.reset(init_mode="custom", q0=q0, p0=p0)
        for _ in range(250):
            u = flocking_with_dynamic_iapf_control(
                state.q,
                state.p,
                env.obstacles,
                gamma.q,
                alpha_params,
                gamma,
                gamma_params,
                beta_params,
                dynamic_params,
            )
            state = env.step(u)
        self.assertTrue(np.all(np.isfinite(state.q)))
        self.assertTrue(np.all(np.isfinite(state.p)))
        self.assertGreaterEqual(collision_count(state.q, env.obstacles, agent_radius=beta_params.agent_radius), 0)

    def test_layer4_dynamic_scenarios_share_metric_schema(self) -> None:
        parser = build_arg_parser()
        with TemporaryDirectory() as tmpdir:
            schemas = []
            for scenario in (
                "layer3_same",
                "complex_dynamic",
                "multi_curved_dynamic",
                "mixed_accel_dynamic",
                "multi_curved_dynamic_v2",
                "mixed_accel_dynamic_v2",
            ):
                args = parser.parse_args(
                    [
                        "--scenario",
                        scenario,
                        "--method",
                        "dynamic_iapf",
                        "--n-agents",
                        "6",
                        "--n-steps",
                        "8",
                        "--skip-animation",
                        "--output-dir",
                        tmpdir,
                    ]
                )
                logs = run_demo(args)
                schemas.append(list(logs.keys()))
                self.assertTrue(all(len(values) == args.n_steps for values in logs.values()))
            for schema in schemas:
                self.assertEqual(schema, LAYER4_METRIC_KEYS)


if __name__ == "__main__":
    unittest.main()
