"""In-memory storage implementation: loads all trajectories into memory."""
import sys
from typing import Dict, List, Tuple

from src.common import SpatialBoundingBox
from src.storage.base import TrajectoryStorage
from src.utils.trajectory_geometry import compute_trajectory_bounding_box


class InMemoryTrajectoryStorage(TrajectoryStorage):
    """In-memory storage implementation.
    
    Stores all trajectory data in an in-memory dictionary, providing the fastest access speed.
    Suitable for small datasets (< 5GB memory) scenarios.
    
    Attributes:
        _trajectory_points: Mapping from trajectory ID to coordinate points
        _trajectory_mbrs: Mapping from trajectory ID to MBR
    """
    
    def __init__(self):
        """Initialize empty storage."""
        self._trajectory_points: Dict[int, List[Tuple[float, float]]] = {}
        self._trajectory_mbrs: Dict[int, SpatialBoundingBox] = {}
    
    def get_trajectory(self, tid: int) -> List[Tuple[float, float]]:
        """Get the coordinate point sequence of the specified trajectory.
        
        Args:
            tid: Trajectory ID
            
        Returns:
            List of trajectory coordinate points
            
        Raises:
            KeyError: Trajectory ID does not exist
        """
        if tid not in self._trajectory_points:
            raise KeyError(f"Trajectory {tid} not found")
        return self._trajectory_points[tid]
    
    def store_trajectory(self, tid: int, points: List[Tuple[float, float]]) -> None:
        """Store trajectory data.
        
        Args:
            tid: Trajectory ID
            points: List of trajectory coordinate points
        """
        self._trajectory_points[tid] = points
        self._trajectory_mbrs[tid] = compute_trajectory_bounding_box(points)
    
    def get_trajectory_mbr(self, tid: int) -> SpatialBoundingBox:
        """Get trajectory MBR.
        
        Args:
            tid: Trajectory ID
            
        Returns:
            Trajectory MBR
            
        Raises:
            KeyError: Trajectory ID does not exist
        """
        if tid not in self._trajectory_mbrs:
            raise KeyError(f"Trajectory {tid} not found")
        return self._trajectory_mbrs[tid]
    
    def get_many_trajectories(self, tids: List[int]) -> Dict[int, List[Tuple[float, float]]]:
        """Batch retrieve trajectories.
        
        In memory mode, directly query from dictionary.
        
        Args:
            tids: List of trajectory IDs
            
        Returns:
            Mapping from trajectory ID to coordinate points
        """
        return {tid: self._trajectory_points[tid] for tid in tids if tid in self._trajectory_points}
    
    def has_trajectory(self, tid: int) -> bool:
        """Check if trajectory exists."""
        return tid in self._trajectory_points
    
    def get_all_tids(self) -> List[int]:
        """Get all trajectory IDs."""
        return list(self._trajectory_points.keys())
    
    def close(self) -> None:
        """Clear storage and release memory."""
        self._trajectory_points.clear()
        self._trajectory_mbrs.clear()
    
    def __len__(self) -> int:
        """Return total number of trajectories."""
        return len(self._trajectory_points)
    
    @property
    def memory_usage_bytes(self) -> int:
        """Estimate current memory usage (bytes).
        
        Calculation method:
        - Dictionary overhead: approximately 72 bytes per entry (Python dict overhead)
        - Coordinate points: 2 float64 per point, 16 bytes
        - MBR: 4 float64 each, 32 bytes
        """
        total = sys.getsizeof(self._trajectory_points)
        total += sys.getsizeof(self._trajectory_mbrs)
        
        for points in self._trajectory_points.values():
            total += sys.getsizeof(points)
            total += len(points) * 16
        
        for mbr in self._trajectory_mbrs.values():
            total += sys.getsizeof(mbr)
        
        return total

    @property
    def trajectory_points(self):
        return self._trajectory_points

    @property
    def trajectory_mbrs(self):
        return self._trajectory_mbrs
