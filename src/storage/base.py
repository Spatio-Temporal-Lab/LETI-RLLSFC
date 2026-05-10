"""Trajectory storage abstract base class, defining unified access interface."""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

from src.common import SpatialBoundingBox


Trajectory = Tuple[int, List[Tuple[float, float]]]


class TrajectoryStorage(ABC):
    """Trajectory storage abstract base class.
    
    Provides unified trajectory data access interface, supporting both in-memory and disk mode implementations.
    All trajectory data operations should be performed through this class interface to avoid direct access to underlying storage structures.
    """
    
    @abstractmethod
    def get_trajectory(self, tid: int) -> List[Tuple[float, float]]:
        """Get the coordinate point sequence of the specified trajectory.
        
        Args:
            tid: Trajectory ID
            
        Returns:
            List of trajectory coordinate points, each point is an (x, y) tuple
            
        Raises:
            KeyError: Trajectory ID does not exist
        """
        pass
    
    @abstractmethod
    def store_trajectory(self, tid: int, points: List[Tuple[float, float]]) -> None:
        """Store trajectory data.
        
        This method is called in batch during the data loading phase to persist trajectories to storage.
        
        Args:
            tid: Trajectory ID
            points: List of trajectory coordinate points
        """
        pass
    
    @abstractmethod
    def get_trajectory_mbr(self, tid: int) -> SpatialBoundingBox:
        """Get the minimum bounding rectangle (MBR) of the trajectory.
        
        Args:
            tid: Trajectory ID
            
        Returns:
            MBR bounding box of the trajectory
            
        Raises:
            KeyError: Trajectory ID does not exist
        """
        pass
    
    @abstractmethod
    def get_many_trajectories(self, tids: List[int]) -> Dict[int, List[Tuple[float, float]]]:
        """Batch retrieve multiple trajectories.
        
        Used for scenarios requiring simultaneous access to multiple trajectories, such as similarity calculations.
        Disk storage implementations can perform batch pre-read optimizations.
        
        Args:
            tids: List of trajectory IDs
            
        Returns:
            Dictionary mapping trajectory IDs to coordinate points
        """
        pass
    
    @abstractmethod
    def has_trajectory(self, tid: int) -> bool:
        """Check if trajectory exists.
        
        Args:
            tid: Trajectory ID
            
        Returns:
            Whether the trajectory exists
        """
        pass
    
    @abstractmethod
    def get_all_tids(self) -> List[int]:
        """Get list of all trajectory IDs.
        
        Returns:
            List of all stored trajectory IDs
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close storage and release resources.
        
        Disk storage implementations need to close file handles.
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return total number of stored trajectories."""
        pass
    
    @property
    @abstractmethod
    def memory_usage_bytes(self) -> int:
        """Return current memory usage (bytes).
        
        Used for monitoring and automatic mode switching decisions.
        """
        pass
    
    def __enter__(self):
        """Support context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically close resources on exit."""
        self.close()
