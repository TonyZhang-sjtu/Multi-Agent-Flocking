"""Layer 1 demo: Olfati-Saber alpha-agent free-space flocking."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from .alpha_flocking import AlphaFlockingParams, alpha_flocking_control
from .metrics import (
    algebraic_connectivity,
    lattice_deviation_energy,
    mean_neighbor_count,
    min_agent_distance,
    num_connected_components,
    velocity_consensus_error,
)
from .simulator import FlockingEnv
from .visualize import animate_trajectories, plot_metrics, plot_trajectories


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Layer 1 Olfati-Saber alpha-agent free-space flocking.")
    parser.add_argument("--n-agents", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init-min-sep", type=float, default=0.75)
    parser.add_argument("--init-velocity-scale", type=float, default=0.05)
    parser.add_argument("--world-width", type=float, default=20.0)
    parser.add_argument("--world-height", type=float, default=12.0)
    parser.add_argument("--v-max", type=float, default=2.0)
    parser.add_argument("--u-max", type=float, default=15.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--h", type=float, default=0.2)
    parser.add_argument("--d", type=float, default=1.2)
    parser.add_argument("--r", type=float, default=3.0)
    parser.add_argument("--a", type=float, default=5.0)
    parser.add_argument("--b", type=float, default=5.0)
    parser.add_argument("--c1-alpha", type=float, default=1.0)
    parser.add_argument("--c2-alpha", type=float, default=2.0)
    parser.add_argument("--d-safe-agent", type=float, default=0.40)
    parser.add_argument("--k-barrier", type=float, default=5.0)
    parser.add_argument("--skip-animation", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    return parser


def write_metrics_csv(logs: Dict[str, List[float]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(logs.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + keys)
        for idx in range(len(next(iter(logs.values())))):
            writer.writerow([idx] + [logs[key][idx] for key in keys])


def sample_random_center_with_min_separation(
    n_agents: int,
    world_size: np.ndarray,
    rng: np.random.Generator,
    min_sep: float,
    max_attempts: int = 20000,
) -> np.ndarray:
    """Sample random-center positions while avoiding near-overlap starts."""
    low = np.array([world_size[0] * 0.35, world_size[1] * 0.25])
    high = np.array([world_size[0] * 0.65, world_size[1] * 0.75])
    positions: List[np.ndarray] = []
    attempts = 0
    while len(positions) < n_agents and attempts < max_attempts:
        attempts += 1
        candidate = rng.uniform(low=low, high=high)
        if not positions:
            positions.append(candidate)
            continue
        existing = np.vstack(positions)
        if np.min(np.linalg.norm(existing - candidate[None, :], axis=1)) >= min_sep:
            positions.append(candidate)
    if len(positions) != n_agents:
        raise RuntimeError(
            f"Could not sample {n_agents} agents with min_sep={min_sep}; "
            "try reducing --n-agents or --init-min-sep."
        )
    return np.vstack(positions)


def run_demo(args: argparse.Namespace) -> Dict[str, List[float]]:
    world_size = np.array([args.world_width, args.world_height], dtype=float)
    params = AlphaFlockingParams(
        epsilon=args.epsilon,
        h=args.h,
        d=args.d,
        r=args.r,
        a=args.a,
        b=args.b,
        c1_alpha=args.c1_alpha,
        c2_alpha=args.c2_alpha,
        d_safe_agent=args.d_safe_agent,
        k_barrier=args.k_barrier,
    )

    env = FlockingEnv(
        n_agents=args.n_agents,
        dt=args.dt,
        world_size=tuple(world_size),
        v_max=args.v_max,
        u_max=args.u_max,
        seed=args.seed,
        boundary_mode="none",
    )
    rng = np.random.default_rng(args.seed)
    q0 = sample_random_center_with_min_separation(args.n_agents, world_size, rng, args.init_min_sep)
    p0 = rng.normal(loc=0.0, scale=args.init_velocity_scale, size=(args.n_agents, 2))
    state = env.reset(init_mode="custom", q0=q0, p0=p0)

    traj: List[np.ndarray] = []
    velocities: List[np.ndarray] = []
    logs: Dict[str, List[float]] = {
        "velocity_consensus_error": [],
        "min_agent_distance": [],
        "lattice_deviation_energy": [],
        "mean_neighbor_count": [],
        "connected_components": [],
        "lambda_2": [],
    }

    for _ in tqdm(range(args.n_steps), desc="Layer 1 alpha flocking", leave=False):
        u = alpha_flocking_control(state.q, state.p, params)
        state = env.step(u)
        traj.append(state.q.copy())
        velocities.append(state.p.copy())
        logs["velocity_consensus_error"].append(velocity_consensus_error(state.p))
        logs["min_agent_distance"].append(min_agent_distance(state.q))
        logs["lattice_deviation_energy"].append(lattice_deviation_energy(state.q, params.r, params.d))
        logs["mean_neighbor_count"].append(mean_neighbor_count(state.q, params.r))
        logs["connected_components"].append(float(num_connected_components(state.q, params.r)))
        logs["lambda_2"].append(algebraic_connectivity(state.q, params.r))

    figures_dir = os.path.join(args.output_dir, "figures", "layer1")
    animations_dir = os.path.join(args.output_dir, "animations", "layer1")
    logs_dir = os.path.join(args.output_dir, "logs", "layer1")

    plot_trajectories(
        traj,
        world_size=world_size,
        save_path=os.path.join(figures_dir, "layer1_alpha_trajectories.png"),
        title="Layer 1 Olfati-Saber Alpha Flocking",
    )
    plot_metrics(logs, save_dir=figures_dir)
    write_metrics_csv(logs, os.path.join(logs_dir, "layer1_alpha_metrics.csv"))

    if not args.skip_animation:
        animate_trajectories(
            traj,
            velocities=velocities,
            world_size=world_size,
            save_path=os.path.join(animations_dir, "layer1_alpha_flocking.gif"),
            stride=max(1, args.n_steps // 160),
            title="Layer 1 Alpha-Agent Free-Space Flocking",
        )

    print("Layer 1 simulation finished.")
    print(f"Final velocity consensus error: {logs['velocity_consensus_error'][-1]:.4f}")
    print(f"Final min agent distance: {logs['min_agent_distance'][-1]:.4f}")
    print(f"Final lattice deviation energy: {logs['lattice_deviation_energy'][-1]:.4f}")
    print(f"Final connected components: {int(logs['connected_components'][-1])}")
    return logs


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
