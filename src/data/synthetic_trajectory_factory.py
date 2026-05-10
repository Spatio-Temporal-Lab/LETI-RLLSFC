"""
Synthetic trajectory generator for experiments without access to real data.
"""
from typing import List, Tuple

import numpy as np

from src.common import SpatialBoundingBox


def generate_synthetic_trajectories(
    num_trajectories: int, 
    bbox: SpatialBoundingBox, 
    seed: int = 42,
    min_points: int = 10,
    max_points: int = 50,
    step_scale: float = 0.1
) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """
    Generate synthetic trajectories for testing and experiments.
    
    Args:
        num_trajectories: Number of trajectories to generate
        bbox: Spatial bounding box for trajectory generation
        seed: Random seed for reproducibility
        min_points: Minimum number of points per trajectory
        max_points: Maximum number of points per trajectory
        step_scale: Scale factor for random walk steps (relative to bbox size)
        
    Returns:
        List of (trajectory_id, points) tuples where points are (x, y) coordinates
    """
    np.random.seed(seed)
    trajectories: List[Tuple[int, List[Tuple[float, float]]]] = []

    for trajectory_id in range(num_trajectories):
        num_points = np.random.randint(min_points, max_points)
        points: List[Tuple[float, float]] = []

        # Random starting position
        x_position = np.random.uniform(bbox.min_x, bbox.max_x)
        y_position = np.random.uniform(bbox.min_y, bbox.max_y)

        # Generate trajectory as random walk
        for _ in range(num_points):
            x_position += np.random.uniform(-step_scale, step_scale) * (bbox.max_x - bbox.min_x)
            y_position += np.random.uniform(-step_scale, step_scale) * (bbox.max_y - bbox.min_y)

            # Clip to bounding box
            x_position = float(np.clip(x_position, bbox.min_x, bbox.max_x))
            y_position = float(np.clip(y_position, bbox.min_y, bbox.max_y))

            points.append((x_position, y_position))

        trajectories.append((trajectory_id, points))

    return trajectories


# Backward compatibility: keep the class for existing code
class SyntheticTrajectoryFactory:
    """
    **DEPRECATED**: Use `generate_synthetic_trajectories()` function instead.
    
    This class is kept for backward compatibility only.
    """

    @staticmethod
    def generate(num_trajectories: int, bbox: SpatialBoundingBox, seed: int = 42) -> List[Tuple[int, List[Tuple[float, float]]]]:
        """Generate synthetic trajectories (deprecated, use generate_synthetic_trajectories instead)."""
        return generate_synthetic_trajectories(num_trajectories, bbox, seed)
