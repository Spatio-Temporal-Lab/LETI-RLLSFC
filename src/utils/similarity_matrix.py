"""Similarity matrix pre-computation utility."""
import logging
import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Deque, Dict, Iterator, List, Optional, Tuple, Any

import numpy as np
from tqdm import tqdm

from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex
from src.common import TraversalCostEvaluator


_WORKER_COST_EVALUATOR: Optional[TraversalCostEvaluator] = None
_WORKER_ALL_CELLS: Optional[List[QuadTreeCell]] = None


def _init_similarity_worker(
    cost_evaluator: TraversalCostEvaluator,
    all_cells: List[QuadTreeCell]
) -> None:
    """Initialize shared read-only state once per child process."""
    global _WORKER_COST_EVALUATOR, _WORKER_ALL_CELLS
    _WORKER_COST_EVALUATOR = cost_evaluator
    _WORKER_ALL_CELLS = all_cells


def _compute_chunk_task(
    indices_pairs: List[Tuple[int, int]]
) -> List[Tuple[int, int, float]]:
    """Task function executed by child process: compute similarity for a batch of cell pairs.
    
    Args:
        task_data: (list of index pairs, cost evaluator, cell list)
        
    Returns:
        List of (i, j, similarity) tuples
    """
    if _WORKER_COST_EVALUATOR is None or _WORKER_ALL_CELLS is None:
        raise RuntimeError("Similarity worker is not initialized")

    results = []
    for i, j in indices_pairs:
        sim = _WORKER_COST_EVALUATOR.jaccard_similarity(
            _WORKER_ALL_CELLS[i],
            _WORKER_ALL_CELLS[j],
            use_cache=False,
        )
        results.append((i, j, float(sim)))
    return results


class SimilarityMatrix:
    """
    Similarity matrix pre-computation utility.
    
    Supports multi-process parallel computation and efficient matrix storage/loading.
    """
    
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_WORKERS = 12

    def __init__(self, quadtree: QuadTreeIndex, cost_evaluator: TraversalCostEvaluator):
        self.quadtree = quadtree
        self.cost_evaluator = cost_evaluator
        self.logger = logging.getLogger(__name__)

        self.matrix_array: Optional[np.ndarray] = None
        self.cell_to_index: Dict[QuadTreeCell, int] = {}
        self.all_cells: List[QuadTreeCell] = []
        self.is_computed: bool = False

    def _build_cell_index_mapping(self, all_cells: List[QuadTreeCell]) -> None:
        self.all_cells = all_cells
        self.cell_to_index = {cell: idx for idx, cell in enumerate(all_cells)}
        n = len(all_cells)
        self.matrix_array = np.zeros((n, n), dtype=np.float32)

    def compute(
        self,
        all_cells: List[QuadTreeCell],
        use_symmetric: bool = True,
        show_progress: bool = True,
        num_workers: Optional[int] = None,
        chunk_size: int = None
    ) -> None:
        """Pre-compute similarity matrix using multi-process chunked parallel computation.
        
        Args:
            all_cells: List of all cells
            use_symmetric: Whether to leverage symmetry to reduce computation
            show_progress: Whether to display progress bar
            num_workers: Number of worker processes, None means use CPU core count
            chunk_size: Size of each task chunk
        """
        n = len(all_cells)
        chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        num_workers = num_workers if num_workers is not None else self.DEFAULT_WORKERS
        
        self._build_cell_index_mapping(all_cells)

        total_pairs = self._count_task_pairs(n, use_symmetric)

        self.logger.info(
            f"Starting parallel pre-computation of similarity matrix (Workers: {num_workers}, Total: {total_pairs})"
        )
        start_time = time.perf_counter()

        if num_workers <= 1:
            self._compute_sequential(n, all_cells, use_symmetric, show_progress)
        else:
            self._compute_parallel(
                n, all_cells, use_symmetric, show_progress,
                num_workers, chunk_size, total_pairs
            )

        self.is_computed = True
        elapsed = time.perf_counter() - start_time
        self.logger.info(
            f"Pre-computation completed! Time: {elapsed:.2f}s | Speed: {total_pairs / elapsed:.1f} pairs/s"
        )

    def _count_task_pairs(self, n: int, use_symmetric: bool) -> int:
        """Count the number of cell pairs that need to be computed."""
        if use_symmetric:
            return n * (n + 1) // 2
        return n * n

    def _generate_task_pairs(self, n: int, use_symmetric: bool) -> Iterator[Tuple[int, int]]:
        """Generate cell pairs that need to be computed."""
        if use_symmetric:
            for i in range(n):
                for j in range(i, n):
                    yield (i, j)
        else:
            for i in range(n):
                for j in range(n):
                    yield (i, j)

    def _generate_task_chunks(
        self,
        n: int,
        use_symmetric: bool,
        chunk_size: int
    ) -> Iterator[List[Tuple[int, int]]]:
        """Generate cell pairs that need to be computed."""
        chunk: List[Tuple[int, int]] = []
        for pair in self._generate_task_pairs(n, use_symmetric):
            chunk.append(pair)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def _compute_sequential(
        self,
        n: int,
        all_cells: List[QuadTreeCell],
        use_symmetric: bool,
        show_progress: bool
    ) -> None:
        """Compute similarity matrix sequentially."""
        pairs = self._generate_task_pairs(n, use_symmetric)
        total_pairs = self._count_task_pairs(n, use_symmetric)
        for i, j in tqdm(pairs, total=total_pairs, disable=not show_progress, desc="Sequential computation"):
            sim = self.cost_evaluator.jaccard_similarity(all_cells[i], all_cells[j], use_cache=False)
            self.matrix_array[i, j] = sim
            if use_symmetric:
                self.matrix_array[j, i] = sim

    def _compute_parallel(
        self,
        n: int,
        all_cells: List[QuadTreeCell],
        use_symmetric: bool,
        show_progress: bool,
        num_workers: int,
        chunk_size: int,
        total_pairs: int
    ) -> None:
        """Compute similarity matrix in parallel."""
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp_context,
            initializer=_init_similarity_worker,
            initargs=(self.cost_evaluator, all_cells),
        ) as executor:
            chunk_iter = self._generate_task_chunks(n, use_symmetric, chunk_size)
            pending: Deque = deque()

            for _ in range(max(1, num_workers * 2)):
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    break
                pending.append(executor.submit(_compute_chunk_task, chunk))

            with tqdm(total=total_pairs, disable=not show_progress, desc="Parallel computation") as pbar:
                while pending:
                    done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.remove(future)
                        chunk_results = future.result()
                        for i, j, sim in chunk_results:
                            self.matrix_array[i, j] = sim
                            if use_symmetric:
                                self.matrix_array[j, i] = sim
                        pbar.update(len(chunk_results))

                        try:
                            chunk = next(chunk_iter)
                        except StopIteration:
                            continue
                        pending.append(executor.submit(_compute_chunk_task, chunk))

    def get_similarity(self, cell_a: QuadTreeCell, cell_b: QuadTreeCell) -> float:
        """Get similarity, supports O(1) matrix query."""
        if not self.is_computed:
            return self.cost_evaluator.jaccard_similarity(cell_a, cell_b)

        idx_a = self.cell_to_index.get(cell_a)
        idx_b = self.cell_to_index.get(cell_b)

        if idx_a is None or idx_b is None:
            return self.cost_evaluator.jaccard_similarity(cell_a, cell_b)

        return float(self.matrix_array[idx_a, idx_b])

    def save(self, filepath: str) -> None:
        """Save data in efficient .npz format.
        
        Args:
            filepath: Save path
            
        Raises:
            RuntimeError: If matrix has not been computed yet
        """
        if not self.is_computed or self.matrix_array is None:
            raise RuntimeError("Matrix has not been computed yet")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        cell_ids = np.array([
            ",".join(map(str, c.quadrant_sequence))
            for c in self.all_cells
        ], dtype=str)

        np.savez_compressed(
            filepath,
            matrix=self.matrix_array,
            cell_ids=cell_ids
        )
        
        size_mb = self.matrix_array.nbytes / 1024 ** 2
        self.logger.info(f"Matrix compressed and saved to: {filepath} ({size_mb:.2f} MB)")

    def load(self, filepath: str, all_cells: List[QuadTreeCell]) -> bool:
        """Load from file and perform strict cell order validation.
        
        Args:
            filepath: Load path
            all_cells: Current cell list
            
        Returns:
            Whether loading was successful
        """
        if not os.path.exists(filepath):
            return False

        try:
            data = np.load(filepath, allow_pickle=True)
            loaded_matrix = data['matrix']
            loaded_ids = data['cell_ids']

            if len(all_cells) != len(loaded_ids):
                self.logger.warning("Loading failed: cell count mismatch")
                return False

            current_ids = [
                ",".join(map(str, c.quadrant_sequence))
                for c in all_cells
            ]
            if not np.array_equal(loaded_ids, current_ids):
                self.logger.warning("Loading failed: cell identifier order mismatch")
                return False

            self.all_cells = all_cells
            self.cell_to_index = {cell: idx for idx, cell in enumerate(all_cells)}
            self.matrix_array = loaded_matrix
            self.is_computed = True
            return True
            
        except Exception as e:
            self.logger.error(f"Matrix loading exception: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Efficiently compute statistical information using NumPy."""
        if not self.is_computed or self.matrix_array is None:
            return {"status": "Not Computed"}

        return {
            "mean": float(np.mean(self.matrix_array)),
            "std": float(np.std(self.matrix_array)),
            "max": float(np.max(self.matrix_array)),
            "sparsity": float(np.mean(self.matrix_array == 0)),
            "memory_mb": self.matrix_array.nbytes / 1024 ** 2
        }
