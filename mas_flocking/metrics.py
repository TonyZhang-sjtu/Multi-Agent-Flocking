"""Metrics and graph utilities for Layer 0 validation and later ablations."""

from __future__ import annotations

from typing import List, Optional

import networkx as nx
import numpy as np

from .obstacles import CircleObstacle
from .utils import as_2d_array, as_vector2


def pairwise_distances(q: np.ndarray) -> np.ndarray:
    """Return pairwise Euclidean distances between all agents."""
    q_arr = as_2d_array(q, "q")
    diff = q_arr[:, None, :] - q_arr[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def adjacency_matrix(q: np.ndarray, radius: float) -> np.ndarray:
    """Build an undirected binary proximity graph adjacency matrix."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    dists = pairwise_distances(q)
    adjacency = (dists < float(radius)).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def neighbor_lists(q: np.ndarray, radius: float) -> List[List[int]]:
    """Return neighbor indices for every agent under a distance threshold."""
    adjacency = adjacency_matrix(q, radius)
    return [list(np.where(adjacency[i] > 0)[0]) for i in range(adjacency.shape[0])]


def mean_velocity(p: np.ndarray) -> np.ndarray:
    """Mean velocity vector across all agents."""
    p_arr = as_2d_array(p, "p")
    return np.mean(p_arr, axis=0)


def velocity_consensus_error(p: np.ndarray) -> float:
    """Mean distance of each velocity from the group mean velocity."""
    p_arr = as_2d_array(p, "p")
    p_bar = np.mean(p_arr, axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(p_arr - p_bar, axis=1)))


def normalized_velocity_mismatch(p: np.ndarray, eps: float = 1e-12) -> float:
    """Scale-free velocity disagreement around the group mean velocity."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    p_arr = as_2d_array(p, "p")
    p_bar = np.mean(p_arr, axis=0, keepdims=True)
    numerator = float(np.sum((p_arr - p_bar) ** 2))
    denominator = float(np.sum(p_arr**2)) + eps
    return numerator / denominator


def speed_statistics(p: np.ndarray) -> tuple[float, float, float]:
    """Return mean, max, and standard deviation of agent speed magnitudes."""
    p_arr = as_2d_array(p, "p")
    speeds = np.linalg.norm(p_arr, axis=1)
    return float(np.mean(speeds)), float(np.max(speeds)), float(np.std(speeds))


def min_agent_distance(q: np.ndarray) -> float:
    """Minimum pairwise agent distance, excluding self-distances."""
    q_arr = as_2d_array(q, "q")
    if q_arr.shape[0] < 2:
        return float("inf")
    dists = pairwise_distances(q_arr)
    np.fill_diagonal(dists, np.inf)
    return float(np.min(dists))


def mean_goal_distance(q: np.ndarray, goal: np.ndarray) -> float:
    """Mean distance from agents to a target point."""
    q_arr = as_2d_array(q, "q")
    goal_vec = as_vector2(goal, "goal")
    return float(np.mean(np.linalg.norm(q_arr - goal_vec[None, :], axis=1)))


def center_of_mass(q: np.ndarray) -> np.ndarray:
    """Mean agent position."""
    q_arr = as_2d_array(q, "q")
    return np.mean(q_arr, axis=0)


def cohesion_radius(q: np.ndarray) -> float:
    """Maximum distance from any agent to the group center of mass."""
    q_arr = as_2d_array(q, "q")
    q_bar = np.mean(q_arr, axis=0, keepdims=True)
    return float(np.max(np.linalg.norm(q_arr - q_bar, axis=1)))


def center_of_mass_goal_distance(q: np.ndarray, goal: np.ndarray) -> float:
    """Distance from the group center of mass to a target point."""
    goal_vec = as_vector2(goal, "goal")
    return float(np.linalg.norm(center_of_mass(q) - goal_vec))


def min_obstacle_clearance(q: np.ndarray, obstacles: Optional[List[CircleObstacle]], agent_radius: float = 0.0) -> float:
    """Minimum signed clearance from agents to circular obstacles."""
    q_arr = as_2d_array(q, "q")
    if not obstacles:
        return float("inf")
    clearances = []
    for obs in obstacles:
        center = as_vector2(obs.center, "obstacle.center")
        dist = np.linalg.norm(q_arr - center[None, :], axis=1) - obs.radius - float(agent_radius)
        clearances.append(dist)
    return float(np.min(np.concatenate(clearances)))


def collision_count(q: np.ndarray, obstacles: Optional[List[CircleObstacle]], agent_radius: float = 0.0) -> int:
    """Count agent-obstacle overlaps using inflated circular obstacles."""
    q_arr = as_2d_array(q, "q")
    if not obstacles:
        return 0
    total = 0
    for obs in obstacles:
        center = as_vector2(obs.center, "obstacle.center")
        clearances = np.linalg.norm(q_arr - center[None, :], axis=1) - obs.radius - float(agent_radius)
        total += int(np.sum(clearances < 0.0))
    return total


def num_connected_components(q: np.ndarray, radius: float) -> int:
    """Number of connected components in the proximity graph."""
    adjacency = adjacency_matrix(q, radius)
    graph = nx.from_numpy_array(adjacency)
    return nx.number_connected_components(graph)


def algebraic_connectivity(q: np.ndarray, radius: float) -> float:
    """Second-smallest Laplacian eigenvalue of the proximity graph."""
    adjacency = adjacency_matrix(q, radius)
    if adjacency.shape[0] < 2:
        return 0.0
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    eigvals = np.linalg.eigvalsh(laplacian)
    return float(max(eigvals[1], 0.0))


def relative_connectivity(q: np.ndarray, radius: float) -> float:
    """Rank-normalized graph connectivity in [0, 1]."""
    adjacency = adjacency_matrix(q, radius)
    n_agents = adjacency.shape[0]
    if n_agents < 2:
        return 1.0
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    rank = np.linalg.matrix_rank(laplacian, tol=1e-9)
    return float(rank / (n_agents - 1))


def lattice_deviation_energy(q: np.ndarray, radius: float, desired_distance: float) -> float:
    """Mean squared deviation from desired neighbor distance over graph edges."""
    if desired_distance <= 0:
        raise ValueError("desired_distance must be positive")
    q_arr = as_2d_array(q, "q")
    dists = pairwise_distances(q_arr)
    mask = np.triu((dists < float(radius)), k=1)
    if not np.any(mask):
        return 0.0
    deviations = dists[mask] - float(desired_distance)
    return float(np.mean(deviations * deviations))


def mean_neighbor_count(q: np.ndarray, radius: float) -> float:
    """Average number of Euclidean-radius neighbors per agent."""
    adjacency = adjacency_matrix(q, radius)
    return float(np.mean(np.sum(adjacency > 0.0, axis=1)))


def average_flocking_error(q: np.ndarray, q_ref: np.ndarray) -> float:
    """Shao-style average pairwise-distance error relative to a reference flock."""
    q_arr = as_2d_array(q, "q")
    q_ref_arr = as_2d_array(q_ref, "q_ref", expected_rows=q_arr.shape[0])
    dists = pairwise_distances(q_arr)
    ref_dists = pairwise_distances(q_ref_arr)
    return float(np.sum(np.abs(dists - ref_dists)) / (2.0 * q_arr.shape[0]))
