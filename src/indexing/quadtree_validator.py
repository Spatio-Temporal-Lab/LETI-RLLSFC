"""Quadtree data consistency validator."""
import logging
from typing import Dict, List, Set, Tuple

from src.indexing.quadtree_cell import QuadTreeCell
from src.utils.trajectory_geometry import compute_trajectory_bounding_box


class QuadTreeValidator:
    """Quadtree data consistency validator.
    
    Responsible for validating data integrity and consistency after quadtree operations (such as pruning, trajectory assignment).
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def validate_reassignment(
        self,
        traj_id: int,
        points: List[Tuple[float, float]],
        old_cell: QuadTreeCell,
        old_points: List[Tuple[float, float]],
        quadtree
    ) -> None:
        """Validate trajectory reassignment consistency.
        
        Args:
            traj_id: Trajectory ID
            points: New trajectory points
            old_cell: Original cell
            old_points: Original trajectory points
            quadtree: Quadtree index
            
        Raises:
            AssertionError: If geometric drift is detected
        """
        self.logger.info(f"[DIAGNOSE] Detected Re-assignment for TID: {traj_id}")
        points_match = (len(points) == len(old_points)) and all(
            p1 == p2 for p1, p2 in zip(points, old_points)
        )
        self.logger.info(f"  - Points Match: {points_match}")
        if not points_match:
            self.logger.warning(
                f"    Warning: Points content changed! Old len: {len(old_points)}, New len: {len(points)}"
            )

        new_bbox = compute_trajectory_bounding_box(points)
        new_lvl = quadtree.compute_target_level(new_bbox)
        new_cell = quadtree.get_cell_at(new_bbox.min_x, new_bbox.min_y, new_lvl)

        self.logger.info(
            f"  - Old Cell: Level {old_cell.level}, Path: {old_cell.quadrant_sequence}"
        )
        self.logger.info(
            f"  - New Cell: Level {new_lvl}, Path: {new_cell.quadrant_sequence if new_cell is not None else None}"
        )

        cell_match = (old_cell == new_cell)
        self.logger.info(f"  - Cell Consistency Match: {cell_match}")

        if not cell_match:
            self.logger.error(f"  [!] CRITICAL: Geometry drift detected for TID {traj_id}!")
            self.logger.error(f"    MBR: {new_bbox}")
            raise AssertionError("Reassignment Trajectory!")

    def validate_merge_results(
        self,
        initial_tids: Set[int],
        initial_count: int,
        min_threshold: int,
        active_cells: List[QuadTreeCell],
        trajectory_to_cell: Dict[int, QuadTreeCell],
        root_cell: QuadTreeCell
    ) -> None:
        """Validate consistency, completeness, and uniqueness of merged data.
        
        Args:
            initial_tids: Initial trajectory ID set
            initial_count: Initial trajectory total count
            min_threshold: Minimum trajectory count threshold
            active_cells: Active cell list
            trajectory_to_cells: Trajectory to cell mapping
            root_cell: Root node
            
        Raises:
            AssertionError: If validation fails
        """
        final_tids = {tid for cell in active_cells for tid in cell.trajectories}
        final_total_count = sum(len(cell.trajectories) for cell in active_cells)

        assert final_tids == initial_tids, \
            f"Trajectory ID mismatch! Missing: {initial_tids - final_tids}, Extra: {final_tids - initial_tids}"
        assert final_total_count == initial_count, \
            f"Trajectory total count changed! Expected {initial_count}, Actual {final_total_count}"

        for cell in active_cells:
            if cell != root_cell:
                assert len(cell.trajectories) >= min_threshold, \
                    f"Cell {cell.code} (Level {cell.level}) trajectory count {len(cell.trajectories)} below threshold {min_threshold}"

        for tid in initial_tids:
            mapped_cell = trajectory_to_cell.get(tid)
            assert mapped_cell is not None, f"Trajectory {tid} lost in trajectory_to_cells mapping"
            assert not mapped_cell.muted, f"Trajectory {tid} mapped to muted Cell {mapped_cell.code}"
            assert tid in mapped_cell.trajectories, \
                f"Mapping conflict: Trajectory {tid} points to Cell {mapped_cell.code}, but Cell does not contain this TID"

        tid_appearance_counts = {}
        for cell in active_cells:
            for tid in cell.trajectories:
                tid_appearance_counts[tid] = tid_appearance_counts.get(tid, 0) + 1

        has_duplicates = False
        for tid, count in tid_appearance_counts.items():
            if count > 1:
                has_duplicates = True
                self.logger.error(f"\n[ERROR] Uniqueness violation detected! Trajectory TID: {tid} appears in {count} Cells:")

                for cell in active_cells:
                    if tid in cell.trajectories:
                        path = "->".join(map(str, cell.quadrant_sequence))
                        sig = cell.signatures.get(tid)
                        self.logger.error(
                            f"  > Level: {cell.level} | Path: [{path}] | "
                            f"Muted: {cell.muted} | Signature: {bin(sig) if sig is not None else 'None'}"
                        )

                mapped = trajectory_to_cell.get(tid)
                if mapped:
                    self.logger.error(
                        f"  > Mapping table (trajectory_to_cells) currently points to: "
                        f"Level {mapped.level} [Path: {'->'.join(map(str, mapped.quadrant_sequence))}]"
                    )

        if has_duplicates:
            raise AssertionError("Uniqueness violation: Duplicate trajectory assignment detected, please check _mute_cell or trajectory promotion logic.")

        self.logger.info(
            f"[Validate] Pruning result validation passed: {len(active_cells)} Active Cells, {final_total_count} Trajectories."
        )
