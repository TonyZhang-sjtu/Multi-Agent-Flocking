"""Minimal Layer 0 demo: random agents move toward a goal with PD control."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from .controllers import goal_pd_control
from .metrics import (
    algebraic_connectivity,
    min_agent_distance,
    min_obstacle_clearance,
    num_connected_components,
    velocity_consensus_error,
    mean_goal_distance,
)
from .obstacles import CircleObstacle
from .simulator import FlockingEnv
from .visualize import animate_trajectories, plot_metrics, plot_trajectories


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Layer 0 flocking simulator demo.")
    parser.add_argument("--n-agents", type=int, default=30)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighbor-radius", type=float, default=3.0)
    parser.add_argument("--agent-radius", type=float, default=0.12)
    parser.add_argument("--skip-animation", action="store_true", help="Skip GIF generation for faster smoke tests.")
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


def run_demo(args: argparse.Namespace) -> Dict[str, List[float]]:
    world_size = np.array([20.0, 12.0], dtype=float)
    goal = np.array([18.0, 6.0], dtype=float)
    obstacles = [
        CircleObstacle(center=(10.0, 6.0), radius=0.8, velocity=(0.0, 0.0), name="static_demo"),
        CircleObstacle(center=(13.0, 9.5), radius=0.45, velocity=(0.0, -0.35), name="dynamic_demo"),
    ]

    env = FlockingEnv(
        n_agents=args.n_agents,
        dt=args.dt,
        world_size=tuple(world_size),
        v_max=3.0,
        u_max=8.0,
        seed=args.seed,
        boundary_mode="reflect",
        obstacles=obstacles,
    )
    state = env.reset(init_mode="random_left")

    traj: List[np.ndarray] = []
    velocities: List[np.ndarray] = []
    logs: Dict[str, List[float]] = {
        "velocity_consensus_error": [],
        "min_agent_distance": [],
        "mean_goal_distance": [],
        "min_obstacle_clearance": [],
        "connected_components": [],
        "lambda_2": [],
    }

    for _ in tqdm(range(args.n_steps), desc="Layer 0 demo", leave=False):
        u = goal_pd_control(state.q, state.p, goal, k_p=0.8, k_d=1.2)
        state = env.step(u)
        traj.append(state.q.copy())
        velocities.append(state.p.copy())
        logs["velocity_consensus_error"].append(velocity_consensus_error(state.p))
        logs["min_agent_distance"].append(min_agent_distance(state.q))
        logs["mean_goal_distance"].append(mean_goal_distance(state.q, goal))
        logs["min_obstacle_clearance"].append(min_obstacle_clearance(state.q, env.obstacles, agent_radius=args.agent_radius))
        logs["connected_components"].append(float(num_connected_components(state.q, args.neighbor_radius)))
        logs["lambda_2"].append(algebraic_connectivity(state.q, args.neighbor_radius))

    figures_dir = os.path.join(args.output_dir, "figures")
    animations_dir = os.path.join(args.output_dir, "animations")
    logs_dir = os.path.join(args.output_dir, "logs")

    plot_trajectories(
        traj,
        goal=goal,
        obstacles=env.obstacles,
        world_size=world_size,
        save_path=os.path.join(figures_dir, "layer0_trajectories.png"),
    )
    plot_metrics(logs, save_dir=figures_dir)
    write_metrics_csv(logs, os.path.join(logs_dir, "layer0_metrics.csv"))
    if not args.skip_animation:
        animate_trajectories(
            traj,
            velocities=velocities,
            goal=goal,
            obstacles=env.obstacles,
            world_size=world_size,
            save_path=os.path.join(animations_dir, "layer0_demo.gif"),
            stride=max(1, args.n_steps // 160),
        )

    print("Simulation finished.")
    print(f"Final mean goal distance: {logs['mean_goal_distance'][-1]:.4f}")
    print(f"Final min agent distance: {logs['min_agent_distance'][-1]:.4f}")
    print(f"Final min obstacle clearance: {logs['min_obstacle_clearance'][-1]:.4f}")
    return logs


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
