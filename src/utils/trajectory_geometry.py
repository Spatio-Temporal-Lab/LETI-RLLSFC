"""
Utility helpers for trajectory-related geometric operations.

Provides functions for computing bounding boxes, distances, and other
geometric properties of trajectories.
"""
from typing import Iterable, List, Tuple
import math

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


def compute_trajectory_length(points: List[Tuple[float, float]]) -> float:
    """
    Compute the total Euclidean length of a trajectory.
    
    Args:
        points: List of (x, y) coordinate tuples
        
    Returns:
        Total length of the trajectory
    """
    if len(points) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        total_length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    return total_length


def compute_point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute Euclidean distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        Euclidean distance between the points
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def compute_trajectory_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Compute the centroid (center of mass) of a trajectory.
    
    Args:
        points: List of (x, y) coordinate tuples
        
    Returns:
        Centroid as (x, y) tuple
    """
    if not points:
        return (0.0, 0.0)
    
    x_sum = sum(p[0] for p in points)
    y_sum = sum(p[1] for p in points)
    n = len(points)
    
    return (x_sum / n, y_sum / n)


def simplify_trajectory(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    """
    Simplify a trajectory using the Douglas-Peucker algorithm.
    
    Args:
        points: List of (x, y) coordinate tuples
        epsilon: Maximum distance threshold for simplification
        
    Returns:
        Simplified list of points
    """
    if len(points) < 3:
        return points
    
    def perpendicular_distance(point: Tuple[float, float], 
                               line_start: Tuple[float, float], 
                               line_end: Tuple[float, float]) -> float:
        """Compute perpendicular distance from point to line segment."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        
        t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return math.sqrt((x0 - proj_x) ** 2 + (y0 - proj_y) ** 2)
    
    def douglas_peucker(pts: List[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
        """Recursive Douglas-Peucker implementation."""
        if len(pts) < 3:
            return pts
        
        # Find point with maximum distance from line
        max_dist = 0.0
        max_index = 0
        
        for i in range(1, len(pts) - 1):
            dist = perpendicular_distance(pts[i], pts[0], pts[-1])
            if dist > max_dist:
                max_dist = dist
                max_index = i
        
        # If max distance is greater than epsilon, recursively simplify
        if max_dist > eps:
            left = douglas_peucker(pts[:max_index + 1], eps)
            right = douglas_peucker(pts[max_index:], eps)
            return left[:-1] + right
        else:
            return [pts[0], pts[-1]]
    
    return douglas_peucker(points, epsilon)
