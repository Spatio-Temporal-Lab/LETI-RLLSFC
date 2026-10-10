"""
Utility helpers for trajectory-related geometric operations.

Provides functions for computing bounding boxes, distances, and other
geometric properties of trajectories.
"""
from typing import Iterable, Tuple

from src.common import SpatialBoundingBox


def compute_trajectory_bounding_box(points: Iterable[Tuple[float, float]]) -> SpatialBoundingBox:
    """
    Compute the minimal bounding box that covers the provided trajectory points.
    
    Args:
        points: Iterable of (x, y) coordinate tuples
        
    Returns:
        SpatialBoundingBox covering all points
    """
    point_list = list(points)
    if not point_list:
        return SpatialBoundingBox(0.0, 0.0, 0.0, 0.0)

    xs = [p[0] for p in point_list]
    ys = [p[1] for p in point_list]

    return SpatialBoundingBox(min(xs), min(ys), max(xs), max(ys))
