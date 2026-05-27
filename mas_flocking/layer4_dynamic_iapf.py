"""Layer 4 demo: dynamic obstacle avoidance with Shao-inspired IAPF."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from .alpha_flocking import AlphaFlockingParams
from .beta_obstacle import BetaObstacleParams, flocking_with_static_obstacle_control
from .dynamic_iapf import DynamicIAPFParams, dynamic_iapf_diagnostics, flocking_with_dynamic_iapf_control
from .gamma_navigation import GammaAgent, GammaNavigationParams, free_flocking_with_navigation_control
from .layer3_static_obstacles import sample_random_left_with_min_separation
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
    parser = argparse.ArgumentParser(description="Run Layer 4 dynamic IAPF obstacle avoidance.")
    parser.add_argument("--method", choices=["dynamic_iapf", "static_beta", "no_tangent", "no_avoidance"], default="dynamic_iapf")
    parser.add_argument(
        "--scenario",
        choices=["layer3_same", "single_crossing", "layer0_dynamic", "multi_dynamic"],
        default="layer3_same",
    )
    parser.add_argument("--n-agents", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=1800)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--init-min-sep", type=float, default=0.75)
    parser.add_argument("--init-velocity-scale", type=float, default=0.05)
    parser.add_argument("--world-width", type=float, default=20.0)
    parser.add_argument("--world-height", type=float, default=12.0)
    parser.add_argument("--v-max", type=float, default=2.5)
    parser.add_argument("--u-max", type=float, default=30.0)
    parser.add_argument("--goal-x", type=float, default=18.0)
    parser.add_argument("--goal-y", type=float, default=6.0)
    parser.add_argument("--goal-vx", type=float, default=0.0)
    parser.add_argument("--goal-vy", type=float, default=0.0)
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
    parser.add_argument("--prediction-horizon", type=float, default=3.0)
    parser.add_argument("--influence-distance", type=float, default=3.0)
    parser.add_argument("--safe-distance", type=float, default=0.35)
    parser.add_argument("--k-repulse", type=float, default=1.2)
    parser.add_argument("--k-velocity", type=float, default=0.8)
    parser.add_argument("--k-tangent", type=float, default=0.6)
    parser.add_argument("--k-obs", type=float, default=1.5)
    parser.add_argument("--max-obs-speed", type=float, default=2.0)
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


def build_obstacles(scenario: str) -> List[CircleObstacle]:
    if scenario == "layer3_same":
        return [
            CircleObstacle(center=(10.0, 6.0), radius=0.8, velocity=(0.0, 0.0), name="static_demo", dynamic=False),
            CircleObstacle(center=(13.0, 9.5), radius=0.45, velocity=(0.0, -0.35), name="dynamic_demo_same_start", dynamic=True),
        ]
    if scenario == "single_crossing":
        return [CircleObstacle(center=(10.0, 10.5), radius=0.65, velocity=(0.0, -0.45), name="crossing_dynamic", dynamic=True)]
    if scenario == "layer0_dynamic":
        return [CircleObstacle(center=(13.0, 9.5), radius=0.45, velocity=(0.0, -0.35), name="layer0_dynamic", dynamic=True)]
    if scenario == "multi_dynamic":
        return [
            CircleObstacle(center=(9.0, 10.5), radius=0.55, velocity=(0.0, -0.42), name="upper_crossing", dynamic=True),
            CircleObstacle(center=(12.0, 1.5), radius=0.55, velocity=(0.0, 0.35), name="lower_crossing", dynamic=True),
            CircleObstacle(center=(15.5, 6.0), radius=0.45, velocity=(-0.35, 0.0), name="head_on", dynamic=True),
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
    )
    gamma_params = GammaNavigationParams(c1_gamma=args.c1_gamma, c2_gamma=args.c2_gamma)
    beta_params = BetaObstacleParams(
        epsilon=args.epsilon,
        h=args.h,
        r_beta=args.r_beta,
        c1_beta=args.c1_beta,
        c2_beta=args.c2_beta,
        agent_radius=args.agent_radius,
        beta_velocity_mode="projected",
    )
    dynamic_params = DynamicIAPFParams(
        prediction_horizon=args.prediction_horizon,
        influence_distance=args.influence_distance,
        safe_distance=args.safe_distance,
        k_repulse=args.k_repulse,
        k_velocity=args.k_velocity,
        k_tangent=args.k_tangent,
        k_obs=args.k_obs,
        max_obs_speed=args.max_obs_speed,
        use_tangent=args.method != "no_tangent",
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
    obstacle_frames: List[List[Dict[str, object]]] = []
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
        "active_dynamic_risk_count": [],
        "min_predicted_obstacle_clearance": [],
        "control_effort": [],
        "lattice_deviation_energy": [],
        "mean_neighbor_count": [],
        "connected_components": [],
        "lambda_2": [],
    }

    for _ in tqdm(range(args.n_steps), desc=f"Layer 4 {args.method}", leave=False):
        if args.method == "no_avoidance":
            u = free_flocking_with_navigation_control(state.q, state.p, alpha_params, gamma, gamma_params)
        elif args.method == "static_beta":
            u = flocking_with_static_obstacle_control(state.q, state.p, env.obstacles, alpha_params, gamma, gamma_params, beta_params)
        else:
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
                include_beta=True,
            )
        diagnostics = dynamic_iapf_diagnostics(state.q, state.p, env.obstacles, dynamic_params)
        state = env.step(u)
        gamma.step(args.dt)
        current_collisions = collision_count(state.q, env.obstacles, agent_radius=args.agent_radius)
        total_collision_count += current_collisions
        if current_collisions > 0:
            total_collision_steps += 1

        traj.append(state.q.copy())
        velocities.append(state.p.copy())
        obstacle_frames.append(state.obstacles)
        logs["mean_goal_distance"].append(mean_goal_distance(state.q, gamma.q))
        logs["center_of_mass_goal_distance"].append(center_of_mass_goal_distance(state.q, gamma.q))
        logs["velocity_consensus_error"].append(velocity_consensus_error(state.p))
        logs["min_agent_distance"].append(min_agent_distance(state.q))
        logs["min_obstacle_clearance"].append(min_obstacle_clearance(state.q, env.obstacles, agent_radius=args.agent_radius))
        logs["collision_count"].append(float(current_collisions))
        logs["total_collision_steps"].append(float(total_collision_steps))
        logs["total_collision_count"].append(float(total_collision_count))
        logs["active_dynamic_risk_count"].append(diagnostics["active_dynamic_risk_count"])
        logs["min_predicted_obstacle_clearance"].append(diagnostics["min_predicted_obstacle_clearance"])
        logs["control_effort"].append(float(np.mean(np.sum(u * u, axis=1))))
        logs["lattice_deviation_energy"].append(lattice_deviation_energy(state.q, alpha_params.r, alpha_params.d))
        logs["mean_neighbor_count"].append(mean_neighbor_count(state.q, alpha_params.r))
        logs["connected_components"].append(float(num_connected_components(state.q, alpha_params.r)))
        logs["lambda_2"].append(algebraic_connectivity(state.q, alpha_params.r))

    figures_dir = os.path.join(args.output_dir, "figures", "layer4")
    animations_dir = os.path.join(args.output_dir, "animations", "layer4")
    logs_dir = os.path.join(args.output_dir, "logs", "layer4")
    stem = f"layer4_{args.scenario}_{args.method}"

    plot_trajectories(
        traj,
        goal=gamma.q,
        obstacles=env.obstacles,
        world_size=world_size,
        save_path=os.path.join(figures_dir, f"{stem}_trajectories.png"),
        title=f"Layer 4 Dynamic IAPF ({args.method})",
    )
    plot_metrics(logs, save_dir=figures_dir)
    write_metrics_csv(logs, os.path.join(logs_dir, f"{stem}_metrics.csv"))

    if not args.skip_animation:
        animate_trajectories(
            traj,
            velocities=velocities,
            goal=gamma.q,
            obstacle_trajectories=obstacle_frames,
            world_size=world_size,
            save_path=os.path.join(animations_dir, f"{stem}.gif"),
            stride=max(1, args.n_steps // 160),
            title=f"Layer 4 Dynamic IAPF ({args.method})",
        )

    print("Layer 4 simulation finished.")
    print(f"Method: {args.method}")
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
