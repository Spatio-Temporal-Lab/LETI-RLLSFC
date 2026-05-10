"""
Feature engineering for the reinforcement learning state representation.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.features.trajectory_statistics import TrajectoryStatistics
from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex


class StateFeatureBuilder:
    """
    Precompute feature vectors for all target QuadTreeCells and dynamically build states.

    Features include:
    1. Trajectory statistics features (4 dimensions):
       - N_ee_log: Number of trajectories covered by enlarged element (logarithmic)
       - N_ee / N_int: Coverage to intersection ratio
       - N_cov / N_int: Complete coverage ratio
       - d_center: Density center distance (logarithmic)
    2. Spatial features (4 dimensions):
       - Δx, Δy: Cell center offset relative to overall space center
       - Δdx, Δdy: Trajectory density center offset relative to enlarged element center
    3. Structural features (2 dimensions):
       - S: Log area
       - qcode: Quadtree encoding value
    4. Traversal features (4 dimensions):
       - visited_ratio: Visit ratio
       - Δx_rel, Δy_rel: Displacement relative to previous visited node
       - n_heat: Trajectory density of unvisited nodes among neighbors

    Total dimensions: 10 (static) + 4 (dynamic) = 14
    """

    EPSILON = 1e-6
    LOG_EPSILON = 1e-8
    DISTANCE_SCALE = 10.0
    MAX_NEIGHBORS = 8

    def __init__(
        self,
        quadtree: QuadTreeIndex,
        alpha: int,
        beta: int,
        target_cells: List[QuadTreeCell],
        cache_dir: Optional[str] = None,
    ):
        self.quadtree = quadtree
        self.alpha = alpha
        self.beta = beta
        self.state_dimensions = 14

        self.cache_dir = cache_dir
        self.target_cells = target_cells

        self.trajectory_statistics = TrajectoryStatistics(target_cells)
        self.ee_counts: Dict[QuadTreeCell, int] = {}
        self.intersect_counts: Dict[QuadTreeCell, int] = {}
        self.cover_counts: Dict[QuadTreeCell, int] = {}
        self.density_centres: Dict[QuadTreeCell, Tuple[float, float]] = {}
        self.spatial_features: Dict[QuadTreeCell, Tuple[float, float, float, float]] = {}
        self.structure_features: Dict[QuadTreeCell, Tuple[float, float]] = {}
        self.feature_vectors: Dict[QuadTreeCell, np.ndarray] = {}
        self.max_ee_count = 1

        self._precompute()

    def _precompute(self) -> None:
        self._compute_trajectory_statistics()
        self._compute_spatial_features()
        self._compute_structure_features()
        self._assemble_feature_vectors()

    def _compute_trajectory_statistics(self) -> None:
        self.trajectory_statistics.compute()
        self.ee_counts = self.trajectory_statistics.ee_counts
        self.intersect_counts = self.trajectory_statistics.intersect_counts
        self.cover_counts = self.trajectory_statistics.cover_counts
        self.density_centres = self.trajectory_statistics.density_centres
        self.max_ee_count = self.trajectory_statistics.max_ee_count

    def _compute_spatial_features(self) -> None:
        qbox = self.quadtree.bbox
        space_cx = (qbox.min_x + qbox.max_x) / 2
        space_cy = (qbox.min_y + qbox.max_y) / 2
        width = qbox.max_x - qbox.min_x
        height = qbox.max_y - qbox.min_y

        temp_den_offsets = []

        for cell in self.target_cells:
            cx, cy = cell.get_center()
            dx_pos = (cx - space_cx) / (width + self.EPSILON)
            dy_pos = (cy - space_cy) / (height + self.EPSILON)

            eb = cell.get_enlarged_element_bbox(self.alpha, self.beta)
            e_cx = (eb.min_x + eb.max_x) / 2
            e_cy = (eb.min_y + eb.max_y) / 2
            e_w = eb.max_x - eb.min_x
            e_h = eb.max_y - eb.min_y

            dcenter = self.density_centres[cell]
            dx_den = (dcenter[0] - e_cx) / (e_w + self.EPSILON)
            dy_den = (dcenter[1] - e_cy) / (e_h + self.EPSILON)

            self.spatial_features[cell] = (dx_pos, dy_pos, dx_den, dy_den)
            temp_den_offsets.append([dx_den, dy_den])

        den_arr = np.array(temp_den_offsets)
        self.den_mean = np.mean(den_arr, axis=0)
        self.den_std = np.std(den_arr, axis=0) + self.EPSILON

    def _compute_structure_features(self) -> None:
        """
        Compute structural features of cells.
        1. S: Logarithmic scale of cell area;
        2. q_code: Normalized encoding value at maximum level;
        """
        max_code_value = (4 ** (self.quadtree.max_level + 1) - 1) // 3

        all_log_sizes = []
        for cell in self.target_cells:
            w = cell.bbox.max_x - cell.bbox.min_x
            h = cell.bbox.max_y - cell.bbox.min_y
            all_log_sizes.append(np.log(w * h + self.LOG_EPSILON))

        self.ls_min, self.ls_max = min(all_log_sizes), max(all_log_sizes)

        for i, cell in enumerate(self.target_cells):
            log_size = all_log_sizes[i]
            size_norm = (log_size - self.ls_min) / (self.ls_max - self.ls_min + self.LOG_EPSILON)
            code_norm = cell.get_quadrant_code(self.quadtree.max_level) / (max_code_value + self.LOG_EPSILON)
            self.structure_features[cell] = (size_norm, code_norm)

    def _assemble_feature_vectors(self) -> None:
        """
        Concatenate statistics, spatial, and structural features into complete feature vectors.

        Feature order:
        1. Density statistics (3): N_ee_log, N_ee/N_int, N_cov/N_int
        2. Distance (1): d_center
        3. Spatial position (4): Δx, Δy, Δdx_norm, Δdy_norm
        4. Structure (2): log_size, quad_code
        """
        for cell in self.target_cells:
            ee = self.ee_counts[cell]
            inter = self.intersect_counts[cell]
            cov = self.cover_counts[cell]

            f_ee_log = np.log1p(ee) / np.log1p(self.max_ee_count)
            f_rel_ee = np.sqrt(ee / (inter + self.EPSILON))
            f_rel_cov = np.sqrt(cov / (inter + self.EPSILON))

            cx, cy = cell.get_center()
            dcx, dcy = self.density_centres[cell]
            dist = np.sqrt((dcx - cx) ** 2 + (dcy - cy) ** 2)
            f_dist_log = np.log1p(dist * self.DISTANCE_SCALE)

            dx_p, dy_p, dx_d, dy_d = self.spatial_features[cell]
            f_dx_den = (dx_d - self.den_mean[0]) / self.den_std[0]
            f_dy_den = (dy_d - self.den_mean[1]) / self.den_std[1]

            f_size, f_code = self.structure_features[cell]

            features = [
                f_ee_log,
                f_rel_ee,
                f_rel_cov,
                f_dist_log,
                dx_p,
                dy_p,
                f_dx_den,
                f_dy_den,
                f_size,
                f_code,
            ]
            self.feature_vectors[cell] = np.array(features, dtype=np.float32)

    def get_features(
        self,
        cell: QuadTreeCell,
        prev_cell: Optional[QuadTreeCell],
        visited_ratio: float,
        visited_cells: set,
    ) -> np.ndarray:
        """
        Dynamically build complete vector including traversal features.

        Args:
            cell: Current cell
            prev_cell: Previously visited cell
            visited_ratio: Visited ratio
            visited_cells: Set of visited cells

        Returns:
            Complete feature vector (14 dimensions)
        """
        base = self.feature_vectors.get(cell)
        if base is None:
            return np.zeros(self.state_dimensions, dtype=np.float32)

        dx_rel, dy_rel = self._compute_relative_position(cell, prev_cell)
        f_nb_heat = self._compute_neighbor_heat(cell, visited_cells)
        dynamic_features = [visited_ratio, dx_rel, dy_rel, f_nb_heat]
        return np.concatenate([base, dynamic_features]).astype(np.float32)

    def _compute_relative_position(
        self,
        cell: QuadTreeCell,
        prev_cell: Optional[QuadTreeCell],
    ) -> Tuple[float, float]:
        if not prev_cell:
            return 0.0, 0.0

        c1x, c1y = cell.get_center()
        c2x, c2y = prev_cell.get_center()
        qbox = self.quadtree.bbox
        dx_rel = (c1x - c2x) / (qbox.max_x - qbox.min_x)
        dy_rel = (c1y - c2y) / (qbox.max_y - qbox.min_y)
        return dx_rel, dy_rel

    def _compute_neighbor_heat(
        self,
        cell: QuadTreeCell,
        visited_cells: set,
    ) -> float:
        raw_neighbors = self.quadtree.get_eight_neighbor_cells(cell)
        total_nb_trajs = 0

        for nb in raw_neighbors:
            if not nb or nb.muted or nb in visited_cells:
                continue
            total_nb_trajs += self.ee_counts.get(nb, 0)

        max_possible = self.max_ee_count * self.MAX_NEIGHBORS
        f_nb_heat = np.log1p(total_nb_trajs) / np.log1p(max_possible)
        return float(f_nb_heat)
