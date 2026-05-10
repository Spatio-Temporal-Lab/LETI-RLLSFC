"""Disk storage implementation: binary files + memory cache."""
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

import numpy as np

from src.common import SpatialBoundingBox
from src.storage.base import TrajectoryStorage
from src.utils.trajectory_geometry import compute_trajectory_bounding_box


class DiskTrajectoryStorage(TrajectoryStorage):
    """Disk storage implementation.
    
    Persists trajectory data to binary files, uses LRU cache to accelerate hot data access.
    Suitable for large datasets (30GB+) scenarios with controlled memory usage.
    
    Storage format:
    - Index file (.npy): NumPy structured array containing tid, offset, size, mbr
    - Data file (.bin): Compact binary format [num_points: uint32][points: (x,y) * N]
    
    Attributes:
        _storage_dir: Storage directory path
        _cache_mb: LRU cache size (MB)
        _index: NumPy structured array index
        _data_file: Data file handle
    """
    
    INDEX_DTYPE = np.dtype([
        ('tid', np.int64),
        ('offset', np.int64),
        ('size', np.int32),
        ('mbr_min_x', np.float64),
        ('mbr_min_y', np.float64),
        ('mbr_max_x', np.float64),
        ('mbr_max_y', np.float64),
    ])
    
    HEADER_FMT = '<I'
    POINT_FMT = '<ff'
    
    def __init__(self, storage_dir: Path, cache_mb: int = 2048):
        """Initialize disk storage.
        
        Args:
            storage_dir: Storage directory path
            cache_mb: LRU cache size (MB), default 2GB
        """
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache_mb = cache_mb
        
        self._index_path = self._storage_dir / "traj_index.npy"
        self._data_path = self._storage_dir / "traj_data.bin"
        
        self._index: Optional[np.ndarray] = None
        self._data_file = None
        self._tid_to_row: Dict[int, int] = {}
        self._current_offset = 0
        self._num_trajectories = 0
    
    def _ensure_open(self) -> None:
        """Ensure file is opened."""
        if self._data_file is None:
            if self._data_path.exists():
                self._data_file = open(self._data_path, 'rb')
                self._load_index()
            else:
                self._data_file = open(self._data_path, 'wb')
    
    def _load_index(self) -> None:
        """Load index file."""
        if self._index_path.exists():
            self._index = np.load(self._index_path)
            self._tid_to_row = {tid: i for i, tid in enumerate(self._index['tid'])}
            self._num_trajectories = len(self._index)
    
    def _save_index(self) -> None:
        """Save index file."""
        if self._index is not None:
            np.save(self._index_path, self._index)
    
    @property
    def _cache_size(self) -> int:
        """Calculate cache entry count.
        
        Assuming average 100 points per trajectory, ~800 bytes + overhead ≈ 1000 bytes/entry
        """
        return self._cache_mb * 1024 * 1024 // 1000
    
    @lru_cache(maxsize=None)
    def _get_cached_trajectory(self, tid: int) -> Tuple[Tuple[float, float], ...]:
        """Trajectory read with LRU cache."""
        points = self._read_trajectory_from_disk(tid)
        return tuple(points)
    
    def _read_trajectory_from_disk(self, tid: int) -> List[Tuple[float, float]]:
        """Read trajectory from disk."""
        if self._index is None or tid not in self._tid_to_row:
            raise KeyError(f"Trajectory {tid} not found")
        
        row = self._tid_to_row[tid]
        offset = int(self._index['offset'][row])
        size = int(self._index['size'][row])
        
        self._data_file.seek(offset)
        data = self._data_file.read(size)
        
        num_points = struct.unpack(self.HEADER_FMT, data[:4])[0]
        points = []
        point_data = data[4:]
        
        for i in range(num_points):
            start = i * 8
            x, y = struct.unpack(self.POINT_FMT, point_data[start:start+8])
            points.append((x, y))
        
        return points
    
    def get_trajectory(self, tid: int) -> List[Tuple[float, float]]:
        """Get trajectory, prioritize reading from cache."""
        if self._index is None:
            self._ensure_open()
        
        cached = self._get_cached_trajectory(tid)
        return list(cached)
    
    def store_trajectory(self, tid: int, points: List[Tuple[float, float]]) -> None:
        """Store trajectory to disk."""
        if self._data_file is None:
            self._ensure_open()
        
        if self._data_file.mode != 'wb':
            raise RuntimeError("Cannot write to existing storage, create new instance")
        
        num_points = len(points)
        header = struct.pack(self.HEADER_FMT, num_points)
        point_data = b''.join(struct.pack(self.POINT_FMT, x, y) for x, y in points)
        data = header + point_data
        
        mbr = compute_trajectory_bounding_box(points)
        
        if not hasattr(self, '_pending_records'):
            self._pending_records = []
        
        self._pending_records.append({
            'tid': tid,
            'offset': self._current_offset,
            'size': len(data),
            'mbr_min_x': mbr.min_x,
            'mbr_min_y': mbr.min_y,
            'mbr_max_x': mbr.max_x,
            'mbr_max_y': mbr.max_y,
        })
        
        self._data_file.write(data)
        self._current_offset += len(data)
        self._num_trajectories += 1
    
    def finalize_writes(self) -> None:
        """Complete batch writes and save index file.
        
        Call after all trajectories have been written.
        """
        if hasattr(self, '_pending_records') and self._pending_records:
            self._index = np.array(self._pending_records, dtype=self.INDEX_DTYPE)
            self._index = np.sort(self._index, order='tid')
            self._save_index()
            
            self._tid_to_row = {tid: i for i, tid in enumerate(self._index['tid'])}
            
            del self._pending_records
        
        if self._data_file:
            self._data_file.close()
            self._data_file = open(self._data_path, 'rb')
    
    def get_trajectory_mbr(self, tid: int) -> SpatialBoundingBox:
        """Get MBR directly from index (no need to read trajectory data)."""
        if self._index is None:
            self._ensure_open()
        
        if tid not in self._tid_to_row:
            raise KeyError(f"Trajectory {tid} not found")
        
        row = self._tid_to_row[tid]
        return SpatialBoundingBox(
            self._index['mbr_min_x'][row],
            self._index['mbr_min_y'][row],
            self._index['mbr_max_x'][row],
            self._index['mbr_max_y'][row]
        )
    
    def get_many_trajectories(self, tids: List[int]) -> Dict[int, List[Tuple[float, float]]]:
        """Batch get trajectories with sequential read optimization."""
        if self._index is None:
            self._ensure_open()
        
        result = {}
        valid_tids = [tid for tid in tids if tid in self._tid_to_row]
        sorted_tids = sorted(valid_tids, key=lambda t: self._tid_to_row[t])
        
        for tid in sorted_tids:
            result[tid] = self.get_trajectory(tid)
        
        return result
    
    def has_trajectory(self, tid: int) -> bool:
        """Check if trajectory exists."""
        if self._index is None:
            self._ensure_open()
        return tid in self._tid_to_row
    
    def get_all_tids(self) -> List[int]:
        """Get all trajectory IDs."""
        if self._index is None:
            self._ensure_open()
        return list(self._index['tid'])
    
    def close(self) -> None:
        """Close file handle."""
        if self._data_file:
            if hasattr(self, '_pending_records') and self._pending_records:
                self.finalize_writes()
            self._data_file.close()
            self._data_file = None
    
    def __len__(self) -> int:
        """Return number of trajectories."""
        if self._index is None:
            self._ensure_open()
        return self._num_trajectories
    
    @property
    def memory_usage_bytes(self) -> int:
        """Estimate current memory usage.
        
        Includes:
        - Index array (resident in memory)
        - tid_to_row dictionary
        - LRU cache
        """
        total = 0
        if self._index is not None:
            total += self._index.nbytes
        total += len(self._tid_to_row) * 72
        
        cache_info = self._get_cached_trajectory.cache_info()
        
        return total
    
    def clear_cache(self) -> None:
        """Clear LRU cache to free memory."""
        self._get_cached_trajectory.cache_clear()
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get cache statistics."""
        info = self._get_cached_trajectory.cache_info()
        return {
            'hits': info.hits,
            'misses': info.misses,
            'maxsize': info.maxsize,
            'currsize': info.currsize,
        }
