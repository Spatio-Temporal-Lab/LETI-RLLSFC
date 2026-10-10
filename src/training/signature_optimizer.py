from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

from src.common import SpatialBoundingBox
from src.indexing.quadtree_cell import QuadTreeCell
from src.utils.signature import compute_signature_vectorized


@dataclass(frozen=True)
class PartitionMetrics:
    alpha: int
    beta: int
    occupancy_ratio: float
    discrimination: float
    score: float


@dataclass(frozen=True)
class WorkloadPartitionMetrics:
    alpha: int
    beta: int
    cost: float


class SignatureOptimizer:
    """Adaptive partition optimizer for a single enlarged element."""

    def __init__(
            self,
            alpha_range: Tuple[int, int] = (2, 8),
            beta_range: Tuple[int, int] = (2, 8),
            occupancy_weight: float = 0.3,
            discrimination_weight: float = 0.7,
            max_sample_trajs: int = 64,
    ):
        self.min_alpha, self.max_alpha = alpha_range
        self.min_beta, self.max_beta = beta_range
        self.occupancy_weight = occupancy_weight
        self.discrimination_weight = discrimination_weight
        self.max_sample_trajs = max_sample_trajs

    def _sample_trajectories(
            self,
            traj_points_list: List[np.ndarray],
            seed: int,
    ) -> List[np.ndarray]:
        if len(traj_points_list) <= self.max_sample_trajs:
            return traj_points_list

        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(
            len(traj_points_list),
            size=self.max_sample_trajs,
            replace=False,
        )
        return [traj_points_list[int(idx)] for idx in sample_indices]

    @staticmethod
    def _bit_count(value: int) -> int:
        return bin(int(value)).count("1")

    def evaluate_partition(
            self,
            alpha: int,
            beta: int,
            traj_points_list: List[np.ndarray],
            ee_bbox: SpatialBoundingBox,
            seed: int = 0,
    ) -> PartitionMetrics:
        """Score a partition by occupancy ratio and normalized Hamming discrimination."""
        if not traj_points_list:
            return PartitionMetrics(alpha, beta, 0.0, 0.0, 0.0)

        sampled_trajs = self._sample_trajectories(traj_points_list, seed)
        signatures = [
            compute_signature_vectorized(alpha, beta, points, ee_bbox)
            for points in sampled_trajs
        ]

        total_cells = alpha * beta
        occupied_mask = 0
        for signature in signatures:
            occupied_mask |= int(signature)
        occupancy_ratio = self._bit_count(occupied_mask) / total_cells if total_cells else 0.0

        if len(signatures) < 2 or total_cells == 0:
            discrimination = 0.0
        else:
            pair_scores = [
                self._bit_count(int(signatures[i]) ^ int(signatures[j])) / total_cells
                for i, j in combinations(range(len(signatures)), 2)
            ]
            discrimination = float(np.mean(pair_scores)) if pair_scores else 0.0

        score = (
                self.occupancy_weight * occupancy_ratio
                + self.discrimination_weight * discrimination
        )
        return PartitionMetrics(alpha, beta, occupancy_ratio, discrimination, score)

    def find_best_config(
            self,
            global_alpha: int,
            global_beta: int,
            traj_points_list: List[np.ndarray],
            cell: QuadTreeCell,
            workload_queries: Optional[List[SpatialBoundingBox]] = None,
            seek_cost: float = 2.0,
            row_cost: float = 1.0,
            trajectory_cost: float = 0.125,
            min_workload_queries: int = 8,
            min_relative_gain: float = 0.001,
            previous_ends: Optional[np.ndarray] = None,
            next_starts: Optional[np.ndarray] = None,
    ) -> Tuple[int, int]:
        """Find the best adaptive partition by grid search from global alpha/beta up to the configured max."""
        ee_bbox = cell.get_enlarged_element_bbox(global_alpha, global_beta)
        base_alpha = min(max(2, global_alpha), self.max_alpha)
        base_alpha = max(self.min_alpha, base_alpha)
        base_beta = min(max(2, global_beta), self.max_beta)
        base_beta = max(self.min_beta, base_beta)
        seed = int(cell.code * 1315423911) & 0xFFFFFFFF

        if workload_queries is not None and len(workload_queries) >= min_workload_queries:
            best_workload = self.evaluate_workload_partition(
                base_alpha, base_beta, traj_points_list, ee_bbox, workload_queries,
                seek_cost=seek_cost, row_cost=row_cost,
                trajectory_cost=trajectory_cost, previous_ends=previous_ends,
                next_starts=next_starts,
            )
            for alpha in range(base_alpha, self.max_alpha + 1):
                for beta in range(base_beta, self.max_beta + 1):
                    if alpha == base_alpha and beta == base_beta:
                        continue
                    candidate = self.evaluate_workload_partition(
                        alpha, beta, traj_points_list, ee_bbox, workload_queries,
                        seek_cost=seek_cost, row_cost=row_cost,
                        trajectory_cost=trajectory_cost, previous_ends=previous_ends,
                        next_starts=next_starts,
                    )
                    required_gain = best_workload.cost * min_relative_gain
                    if candidate.cost < best_workload.cost - required_gain:
                        best_workload = candidate
            return best_workload.alpha, best_workload.beta

        best = self.evaluate_partition(base_alpha, base_beta, traj_points_list, ee_bbox, seed)

        for alpha in range(base_alpha, self.max_alpha + 1):
            for beta in range(base_beta, self.max_beta + 1):
                if alpha == base_alpha and beta == base_beta:
                    continue

                candidate = self.evaluate_partition(alpha, beta, traj_points_list, ee_bbox, seed)
                if candidate.score > best.score * 1.1:
                    best = candidate

        return best.alpha, best.beta

    def evaluate_workload_partition(
            self,
            alpha: int,
            beta: int,
            traj_points_list: List[np.ndarray],
            ee_bbox: SpatialBoundingBox,
            workload_queries: List[SpatialBoundingBox],
            seek_cost: float = 2.0,
            row_cost: float = 1.0,
            trajectory_cost: float = 0.125,
            previous_ends: Optional[np.ndarray] = None,
            next_starts: Optional[np.ndarray] = None,
    ) -> WorkloadPartitionMetrics:
        """Estimate filter-and-refine cost for one local signature grid."""
        signatures = [
            compute_signature_vectorized(alpha, beta, points, ee_bbox)
            for points in traj_points_list
        ]
        total_cost = 0.0
        for query_index, query in enumerate(workload_queries):
            if not ee_bbox.intersects(query):
                continue

            grid_w = (ee_bbox.max_x - ee_bbox.min_x) / alpha
            grid_h = (ee_bbox.max_y - ee_bbox.min_y) / beta
            i_start = max(0, int(np.floor((query.min_x - ee_bbox.min_x) / grid_w)))
            i_end = min(alpha - 1, int(np.floor((query.max_x - ee_bbox.min_x) / grid_w)))
            j_start = max(0, int(np.floor((query.min_y - ee_bbox.min_y) / grid_h)))
            j_end = min(beta - 1, int(np.floor((query.max_y - ee_bbox.min_y) / grid_h)))
            query_signature = 0
            for i in range(i_start, i_end + 1):
                for j in range(j_start, j_end + 1):
                    query_signature |= 1 << (i * beta + j)

            matches = sum(1 for signature in signatures if signature & query_signature)
            if matches:
                connected = previous_ends is not None and bool(previous_ends[query_index])
                total_cost += (0.0 if connected else seek_cost) + row_cost
                total_cost += trajectory_cost * matches
            elif (previous_ends is not None and next_starts is not None
                  and bool(previous_ends[query_index]) and bool(next_starts[query_index])):
                total_cost += seek_cost

        return WorkloadPartitionMetrics(alpha, beta, total_cost)
