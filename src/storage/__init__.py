"""Trajectory storage module: supports both in-memory and disk storage modes.

This module provides a unified trajectory data storage interface, supporting:
- InMemoryTrajectoryStorage: Full in-memory storage, suitable for small datasets
- DiskTrajectoryStorage: Disk storage + LRU cache, suitable for large datasets

Example usage:
    >>> from src.storage import create_storage
    >>> storage = create_storage("memory")
    >>> storage.store_trajectory(1, [(0.1, 0.2), (0.3, 0.4)])
    >>> points = storage.get_trajectory(1)
"""
from pathlib import Path
from typing import Literal, Optional, Union

from src.storage.base import TrajectoryStorage, Trajectory
from src.storage.memory_storage import InMemoryTrajectoryStorage
from src.storage.disk_storage import DiskTrajectoryStorage


def create_storage(
    mode: Literal["memory", "disk", "auto"] = "auto",
    storage_dir: Optional[Union[str, Path]] = None,
    cache_mb: int = 2048,
    estimated_data_gb: Optional[float] = None,
) -> TrajectoryStorage:
    """Create a trajectory storage instance.
    
    Args:
        mode: Storage mode
            - "memory": Use in-memory storage
            - "disk": Use disk storage
            - "auto": Automatically select based on estimated data size (default)
        storage_dir: Disk storage directory (required for disk mode)
        cache_mb: LRU cache size for disk mode (MB), default 2GB
        estimated_data_gb: Estimated data size (GB), used for auto mode decision
        
    Returns:
        Configured storage instance
        
    Raises:
        ValueError: Parameter configuration error
        
    Examples:
        >>> # Memory mode
        >>> storage = create_storage("memory")
        >>> 
        >>> # Disk mode
        >>> storage = create_storage("disk", storage_dir="./storage", cache_mb=4096)
        >>> 
        >>> # Auto mode (automatically selects disk if estimated > 10GB)
        >>> storage = create_storage("auto", estimated_data_gb=30.0)
    """
    if mode == "auto":
        threshold_gb = 8.0
        if estimated_data_gb and estimated_data_gb > threshold_gb:
            mode = "disk"
        else:
            mode = "memory"
    
    if mode == "memory":
        return InMemoryTrajectoryStorage()
    
    elif mode == "disk":
        if storage_dir is None:
            storage_dir = Path("resource") / "storage"
        
        return DiskTrajectoryStorage(Path(storage_dir), cache_mb=cache_mb)
    
    else:
        raise ValueError(f"Unknown storage mode: {mode}")


__all__ = [
    "TrajectoryStorage",
    "InMemoryTrajectoryStorage", 
    "DiskTrajectoryStorage",
    "create_storage",
    "Trajectory",
]
