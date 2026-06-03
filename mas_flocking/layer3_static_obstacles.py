"""Layer 3 demo: Olfati-Saber flocking with static beta-agent obstacle avoidance."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from .alpha_flocking import AlphaFlockingParams
from .beta_obstacle import BetaObstacleParams, flocking_with_static_obstacle_control
from .gamma_navigation import GammaAgent, GammaNavigationParams, free_flocking_with_navigation_control
from .metrics import (
    algebraic_connectivity,
    center_of_mass_goal_distance,
    collision_count,
    lattice_deviation_energy,
    mean_goal_distance,
    mean_neighbor_count,
    min_agent_distance,
    min_obstacle_clearance,
    num_connected_components,
    velocity_consensus_error,
)
from .obstacles import CircleObstacle
from .simulator import FlockingEnv
from .visualize import animate_trajectories, plot_metrics, plot_trajectories


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Layer 3 alpha+beta+gamma static-obstacle flocking.")
    parser.add_argument("--n-agents", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=2400)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--init-min-sep", type=float, default=0.75)
    parser.add_argument("--init-velocity-scale", type=float, default=0.05)
    parser.add_argument("--world-width", type=float, default=20.0)
    parser.add_argument("--world-height", type=float, default=12.0)
    parser.add_argument("--v-max", type=float, default=2.5)
    parser.add_argument("--u-max", type=float, default=25.0)
    parser.add_argument("--goal-x", type=float, default=18.0)
    parser.add_argument("--goal-y", type=float, default=6.0)
    parser.add_argument("--goal-vx", type=float, default=0.0)
    parser.add_argument("--goal-vy", type=float, default=0.0)
    parser.add_argument("--scenario", choices=["layer0_static", "narrow_passage"], default="layer0_static")
    parser.add_argument("--disable-beta", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--h", type=float, default=0.2)
    parser.add_argument("--d", type=float, default=1.2)
    parser.add_argument("--r", type=float, default=3.0)
    parser.add_argument("--a", type=float, default=5.0)
    parser.add_argument("--b", type=float, default=5.0)
    parser.add_argument("--c1-alpha", type=float, default=1.0)
    parser.add_argument("--c2-alpha", type=float, default=2.0)
    parser.add_argument("--c1-gamma", type=float, default=1.5)
    parser.add_argument("--c2-gamma", type=float, default=1.2)
    parser.add_argument("--r-beta", type=float, default=1.5)
    parser.add_argument("--c1-beta", type=float, default=3.0)
    parser.add_argument("--c2-beta", type=float, default=2.0)
    parser.add_argument("--agent-radius", type=float, default=0.12)
    parser.add_argument("--beta-velocity-mode", choices=["projected", "zero"], default="projected")
    parser.add_argument("--d-safe-agent", type=float, default=0.40)
    parser.add_argument("--k-barrier", type=float, default=15.0)  # 静态障碍物下默认高防御力
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


def sample_random_left_with_min_separation(
    n_agents: int,
    world_size: np.ndarray,
    rng: np.random.Generator,
    min_sep: float,
    max_attempts: int = 40000,
) -> np.ndarray:
    """Sample random-left positions while avoiding near-overlap starts."""
    low = np.array([1.0, 1.0])
    high = np.array([min(5.0, world_size[0] * 0.35), max(1.0, world_size[1] - 1.0)])
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


def build_obstacles(scenario: str) -> List[CircleObstacle]:
    """Create static circular obstacles for the requested Layer 3 scenario."""
    if scenario == "layer0_static":
        return [
            CircleObstacle(center=(10.0, 6.0), radius=0.8, velocity=(0.0, 0.0), name="static_demo", dynamic=False),
            CircleObstacle(center=(13.0, 9.5), radius=0.45, velocity=(0.0, 0.0), name="static_former_dynamic", dynamic=False),
        ]
    if scenario == "narrow_passage":
        return [
            CircleObstacle(center=(10.0, 4.0), radius=1.2, velocity=(0.0, 0.0), name="lower_gate", dynamic=False),
            CircleObstacle(center=(10.0, 8.0), radius=1.2, velocity=(0.0, 0.0), name="upper_gate", dynamic=False),
        ]
    raise ValueError(f"Unknown scenario: {scenario}")


def run_demo(args: argparse.Namespace) -> Dict[str, List[float]]:
    world_size = np.array([args.world_width, args.world_height], dtype=float)
    goal = np.array([args.goal_x, args.goal_y], dtype=float)
    goal_velocity = np.array([args.goal_vx, args.goal_vy], dtype=float)
    obstacles = build_obstacles(args.scenario)
    alpha_params = AlphaFlockingParams(
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
    gamma_params = GammaNavigationParams(c1_gamma=args.c1_gamma, c2_gamma=args.c2_gamma)
    beta_params = BetaObstacleParams(
        epsilon=args.epsilon,
        h=args.h,
        r_beta=args.r_beta,
        c1_beta=args.c1_beta,
        c2_beta=args.c2_beta,
        agent_radius=args.agent_radius,
        beta_velocity_mode=args.beta_velocity_mode,
    )
    gamma = GammaAgent(q=goal, p=goal_velocity)

    env = FlockingEnv(
        n_agents=args.n_agents,
        dt=args.dt,
        world_size=tuple(world_size),
        v_max=args.v_max,
        u_max=args.u_max,
        seed=args.seed,
        boundary_mode="none",
        obstacles=obstacles,
    )
    rng = np.random.default_rng(args.seed)
    q0 = sample_random_left_with_min_separation(args.n_agents, world_size, rng, args.init_min_sep)
    p0 = rng.normal(loc=0.0, scale=args.init_velocity_scale, size=(args.n_agents, 2))
    state = env.reset(init_mode="custom", q0=q0, p0=p0)

    traj: List[np.ndarray] = []
    velocities: List[np.ndarray] = []
    gamma_traj: List[np.ndarray] = []
    total_collision_steps = 0
    total_collision_count = 0
    logs: Dict[str, List[float]] = {
        "mean_goal_distance": [],
        "center_of_mass_goal_distance": [],
        "velocity_consensus_error": [],
        "min_agent_distance": [],
        "min_obstacle_clearance": [],
        "collision_count": [],
        "total_collision_steps": [],
        "total_collision_count": [],
        "lattice_deviation_energy": [],
        "mean_neighbor_count": [],
        "connected_components": [],
        "lambda_2": [],
    }

    for _ in tqdm(range(args.n_steps), desc="Layer 3 static obstacle flocking", leave=False):
        if args.disable_beta:
            u = free_flocking_with_navigation_control(state.q, state.p, alpha_params, gamma, gamma_params)
        else:
            u = flocking_with_static_obstacle_control(state.q, state.p, env.obstacles, alpha_params, gamma, gamma_params, beta_params)
        state = env.step(u)
        gamma.step(args.dt)
        current_goal = gamma.q.copy()
        current_collisions = collision_count(state.q, env.obstacles, agent_radius=args.agent_radius)
        total_collision_count += current_collisions
        if current_collisions > 0:
            total_collision_steps += 1

        traj.append(state.q.copy())
        velocities.append(state.p.copy())
        gamma_traj.append(current_goal)
        logs["mean_goal_distance"].append(mean_goal_distance(state.q, current_goal))
        logs["center_of_mass_goal_distance"].append(center_of_mass_goal_distance(state.q, current_goal))
        logs["velocity_consensus_error"].append(velocity_consensus_error(state.p))
        logs["min_agent_distance"].append(min_agent_distance(state.q))
        logs["min_obstacle_clearance"].append(min_obstacle_clearance(state.q, env.obstacles, agent_radius=args.agent_radius))
        logs["collision_count"].append(float(current_collisions))
        logs["total_collision_steps"].append(float(total_collision_steps))
        logs["total_collision_count"].append(float(total_collision_count))
        logs["lattice_deviation_energy"].append(lattice_deviation_energy(state.q, alpha_params.r, alpha_params.d))
        logs["mean_neighbor_count"].append(mean_neighbor_count(state.q, alpha_params.r))
        logs["connected_components"].append(float(num_connected_components(state.q, alpha_params.r)))
        logs["lambda_2"].append(algebraic_connectivity(state.q, alpha_params.r))

    figures_dir = os.path.join(args.output_dir, "figures", "layer3")
    animations_dir = os.path.join(args.output_dir, "animations", "layer3")
    logs_dir = os.path.join(args.output_dir, "logs", "layer3")
    suffix = "no_beta" if args.disable_beta else "beta"

    plot_trajectories(
        traj,
        goal=gamma_traj[-1] if gamma_traj else goal,
        obstacles=env.obstacles,
        world_size=world_size,
        save_path=os.path.join(figures_dir, f"layer3_static_obstacles_{suffix}_trajectories.png"),
        title="Layer 3 Static Obstacle Flocking" + (" Baseline" if args.disable_beta else ""),
    )
    plot_metrics(logs, save_dir=figures_dir)
    write_metrics_csv(logs, os.path.join(logs_dir, f"layer3_static_obstacles_{suffix}_metrics.csv"))

    if not args.skip_animation:
        animate_trajectories(
            traj,
            velocities=velocities,
            goal=gamma_traj[-1] if gamma_traj else goal,
            obstacles=env.obstacles,
            world_size=world_size,
            save_path=os.path.join(animations_dir, f"layer3_static_obstacles_{suffix}.gif"),
            stride=max(1, args.n_steps // 160),
            title="Layer 3 Static Beta-Agent Obstacle Avoidance" + (" Baseline" if args.disable_beta else ""),
        )

    print("Layer 3 simulation finished.")
    print(f"Final mean goal distance: {logs['mean_goal_distance'][-1]:.4f}")
    print(f"Final center-of-mass goal distance: {logs['center_of_mass_goal_distance'][-1]:.4f}")
    print(f"Final velocity consensus error: {logs['velocity_consensus_error'][-1]:.4f}")
    print(f"Final min obstacle clearance: {logs['min_obstacle_clearance'][-1]:.4f}")
    print(f"Total collision steps: {int(logs['total_collision_steps'][-1])}")
    print(f"Total collision count: {int(logs['total_collision_count'][-1])}")
    return logs


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
