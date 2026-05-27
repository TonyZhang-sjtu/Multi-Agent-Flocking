"""Visualization helpers for trajectories, metrics, and simple animations."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

# Use a headless-safe backend for servers and remote terminals.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from .obstacles import CircleObstacle


def _prepare_traj(traj: Iterable[np.ndarray]) -> np.ndarray:
    arr = np.asarray(list(traj), dtype=float)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"traj must have shape [T, N, 2], got {arr.shape}")
    return arr


def _draw_obstacles(ax: plt.Axes, obstacles: Optional[List[CircleObstacle]]) -> None:
    if not obstacles:
        return
    for obs in obstacles:
        circle = plt.Circle(obs.center, obs.radius, fill=False, linewidth=2.0, color="tab:red")
        ax.add_patch(circle)
        ax.scatter(obs.center[0], obs.center[1], s=16, color="tab:red")


def _draw_obstacle_snapshots(ax: plt.Axes, obstacle_snapshots: Optional[Sequence[Dict[str, object]]]) -> List[object]:
    artists: List[object] = []
    if not obstacle_snapshots:
        return artists
    for obs in obstacle_snapshots:
        center = np.asarray(obs["center"], dtype=float)
        radius = float(obs["radius"])
        dynamic = bool(obs.get("dynamic", False))
        color = "tab:red" if dynamic else "firebrick"
        circle = plt.Circle(center, radius, fill=False, linewidth=2.0, color=color)
        point = ax.scatter(center[0], center[1], s=18, color=color)
        ax.add_patch(circle)
        artists.extend([circle, point])
    return artists


def plot_trajectories(
    traj: Iterable[np.ndarray],
    goal: Optional[np.ndarray] = None,
    obstacles: Optional[List[CircleObstacle]] = None,
    world_size: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Layer 0 Agent Trajectories",
) -> None:
    """Plot agent paths with start/end markers."""
    traj_arr = _prepare_traj(traj)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(traj_arr.shape[1]):
        ax.plot(traj_arr[:, i, 0], traj_arr[:, i, 1], linewidth=1.0, alpha=0.8)
        ax.scatter(traj_arr[0, i, 0], traj_arr[0, i, 1], s=14, marker="o", color="tab:green")
        ax.scatter(traj_arr[-1, i, 0], traj_arr[-1, i, 1], s=18, marker="x", color="tab:blue")

    if goal is not None:
        goal_arr = np.asarray(goal, dtype=float)
        ax.scatter(goal_arr[0], goal_arr[1], s=140, marker="*", color="gold", edgecolor="black", label="Goal")

    _draw_obstacles(ax, obstacles)

    if world_size is not None:
        world = np.asarray(world_size, dtype=float)
        ax.set_xlim(0.0, world[0])
        ax.set_ylim(0.0, world[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if goal is not None:
        ax.legend(loc="best")
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(logs: Dict[str, List[float]], save_dir: Optional[str] = None) -> None:
    """Save one line plot per scalar metric."""
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    for key, values in logs.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(values, linewidth=1.5)
        ax.set_title(key)
        ax.set_xlabel("step")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, f"{key}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def animate_trajectories(
    traj: Iterable[np.ndarray],
    velocities: Optional[Iterable[np.ndarray]] = None,
    goal: Optional[np.ndarray] = None,
    obstacles: Optional[List[CircleObstacle]] = None,
    obstacle_trajectories: Optional[Iterable[Sequence[Dict[str, object]]]] = None,
    world_size: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    interval_ms: int = 40,
    stride: int = 5,
    title: str = "Layer 0 Flocking Environment",
) -> None:
    """Create a lightweight GIF animation of agent positions and velocities."""
    traj_arr = _prepare_traj(traj)
    vel_arr = None if velocities is None else np.asarray(list(velocities), dtype=float)
    if vel_arr is not None and vel_arr.shape != traj_arr.shape:
        raise ValueError(f"velocities must have shape {traj_arr.shape}, got {vel_arr.shape}")
    obstacle_frames = None if obstacle_trajectories is None else list(obstacle_trajectories)
    if obstacle_frames is not None and len(obstacle_frames) != traj_arr.shape[0]:
        raise ValueError(f"obstacle_trajectories must have length {traj_arr.shape[0]}, got {len(obstacle_frames)}")

    frames = list(range(0, traj_arr.shape[0], max(1, int(stride))))
    if frames[-1] != traj_arr.shape[0] - 1:
        frames.append(traj_arr.shape[0] - 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    if world_size is not None:
        world = np.asarray(world_size, dtype=float)
        ax.set_xlim(0.0, world[0])
        ax.set_ylim(0.0, world[1])
    else:
        pad = 1.0
        ax.set_xlim(float(np.min(traj_arr[:, :, 0]) - pad), float(np.max(traj_arr[:, :, 0]) + pad))
        ax.set_ylim(float(np.min(traj_arr[:, :, 1]) - pad), float(np.max(traj_arr[:, :, 1]) + pad))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if obstacle_frames is None:
        _draw_obstacles(ax, obstacles)
    if goal is not None:
        goal_arr = np.asarray(goal, dtype=float)
        ax.scatter(goal_arr[0], goal_arr[1], s=140, marker="*", color="gold", edgecolor="black")

    scatter = ax.scatter([], [], s=35, color="tab:blue")
    quiver = ax.quiver([], [], [], [], color="tab:orange", angles="xy", scale_units="xy", scale=1.0, width=0.004)
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes)
    obstacle_artists: List[object] = []

    def update(frame_idx: int):
        nonlocal quiver, obstacle_artists
        q = traj_arr[frame_idx]
        scatter.set_offsets(q)
        quiver.remove()
        for artist in obstacle_artists:
            artist.remove()
        obstacle_artists = []
        if obstacle_frames is not None:
            obstacle_artists = _draw_obstacle_snapshots(ax, obstacle_frames[frame_idx])
        if vel_arr is not None:
            v = vel_arr[frame_idx]
            quiver = ax.quiver(q[:, 0], q[:, 1], v[:, 0], v[:, 1], color="tab:orange", angles="xy", scale_units="xy", scale=8.0, width=0.004)
        else:
            quiver = ax.quiver([], [], [], [], color="tab:orange")
        time_text.set_text(f"step={frame_idx}")
        return (scatter, quiver, time_text, *obstacle_artists)

    animation = FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        animation.save(save_path, writer=PillowWriter(fps=max(1, int(1000 / interval_ms))))
    plt.close(fig)
