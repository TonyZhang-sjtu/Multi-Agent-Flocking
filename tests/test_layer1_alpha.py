"""Tests for Olfati-Saber Layer 1 alpha-agent free-space flocking."""

from __future__ import annotations

import unittest

import numpy as np

from mas_flocking.alpha_flocking import (
    AlphaFlockingParams,
    alpha_adjacency_matrix,
    alpha_flocking_control,
    bump_function,
    phi_alpha,
    sigma_norm,
)
from mas_flocking.metrics import lattice_deviation_energy, velocity_consensus_error
from mas_flocking.simulator import FlockingEnv


class TestAlphaFlocking(unittest.TestCase):
    def test_sigma_norm_is_finite_and_nonnegative(self) -> None:
        self.assertAlmostEqual(float(sigma_norm(np.array([0.0, 0.0]))), 0.0)
        values = sigma_norm(np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 4.0]]), epsilon=0.1)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values >= 0.0))
        self.assertGreater(float(values[-1]), float(values[1]))

    def test_bump_function_regions(self) -> None:
        values = bump_function(np.array([0.0, 0.1, 0.2, 0.6, 1.0, 1.2]), h=0.2)
        self.assertAlmostEqual(float(values[0]), 1.0)
        self.assertAlmostEqual(float(values[1]), 1.0)
        self.assertAlmostEqual(float(values[2]), 1.0)
        self.assertGreater(float(values[3]), 0.0)
        self.assertLess(float(values[3]), 1.0)
        self.assertAlmostEqual(float(values[4]), 0.0)
        self.assertAlmostEqual(float(values[5]), 0.0)

    def test_alpha_adjacency_matrix_is_weighted_symmetric_and_cutoff(self) -> None:
        params = AlphaFlockingParams(d=1.2, r=3.0)
        q = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
        adjacency = alpha_adjacency_matrix(q, params)
        np.testing.assert_allclose(adjacency, adjacency.T)
        np.testing.assert_allclose(np.diag(adjacency), np.zeros(3))
        self.assertGreater(float(adjacency[0, 1]), 0.0)
        self.assertAlmostEqual(float(adjacency[0, 2]), 0.0)

    def test_phi_alpha_equilibrium_at_desired_distance(self) -> None:
        params = AlphaFlockingParams(d=1.2, r=3.0)
        d_alpha = sigma_norm(np.asarray(params.d), epsilon=params.epsilon)
        self.assertAlmostEqual(float(phi_alpha(np.asarray(d_alpha), params)), 0.0, places=12)

    def test_too_close_agents_repel(self) -> None:
        params = AlphaFlockingParams(d=1.2, r=3.0, c1_alpha=1.0, c2_alpha=0.0)
        q = np.array([[0.0, 0.0], [0.6, 0.0]])
        p = np.zeros((2, 2))
        u = alpha_flocking_control(q, p, params)
        self.assertLess(float(u[0, 0]), 0.0)
        self.assertGreater(float(u[1, 0]), 0.0)

    def test_agents_inside_radius_beyond_desired_distance_attract(self) -> None:
        params = AlphaFlockingParams(d=1.2, r=3.0, c1_alpha=1.0, c2_alpha=0.0)
        q = np.array([[0.0, 0.0], [2.0, 0.0]])
        p = np.zeros((2, 2))
        u = alpha_flocking_control(q, p, params)
        self.assertGreater(float(u[0, 0]), 0.0)
        self.assertLess(float(u[1, 0]), 0.0)

    def test_agents_at_desired_distance_with_equal_velocity_have_zero_control(self) -> None:
        params = AlphaFlockingParams(d=1.2, r=3.0)
        q = np.array([[0.0, 0.0], [params.d, 0.0]])
        p = np.array([[0.4, -0.2], [0.4, -0.2]])
        u = alpha_flocking_control(q, p, params)
        np.testing.assert_allclose(u, np.zeros((2, 2)), atol=1e-10)

    def test_short_integrated_run_is_finite_and_improves_consensus(self) -> None:
        params = AlphaFlockingParams(d=1.0, r=2.8, c1_alpha=1.0, c2_alpha=2.0)
        grid = np.array(
            [
                [8.8, 5.4], [10.0, 5.4], [11.2, 5.4],
                [8.8, 6.6], [10.0, 6.6], [11.2, 6.6],
                [8.8, 7.8], [10.0, 7.8], [11.2, 7.8],
            ],
            dtype=float,
        )
        rng = np.random.default_rng(4)
        p0 = rng.normal(0.0, 0.4, size=(grid.shape[0], 2))
        env = FlockingEnv(n_agents=grid.shape[0], dt=0.02, world_size=(20.0, 12.0), v_max=4.0, u_max=20.0, boundary_mode="none")
        state = env.reset(init_mode="custom", q0=grid, p0=p0)
        start_consensus = velocity_consensus_error(state.p)
        start_energy = lattice_deviation_energy(state.q, params.r, params.d)
        for _ in range(250):
            state = env.step(alpha_flocking_control(state.q, state.p, params))
        self.assertTrue(np.all(np.isfinite(state.q)))
        self.assertTrue(np.all(np.isfinite(state.p)))
        self.assertLess(velocity_consensus_error(state.p), start_consensus)
        self.assertLess(lattice_deviation_energy(state.q, params.r, params.d), start_energy)


if __name__ == "__main__":
    unittest.main()
