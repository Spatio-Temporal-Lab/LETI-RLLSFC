"""Training component factory."""
import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

from src.config import TShapeConfig
from src.common import SpatialBoundingBox
from src.data import generate_synthetic_trajectories, normalize_trajectories, load_cleaned_dataset
from src.indexing import QuadTreeIndex
from src.common import TraversalCostEvaluator
from src.rl import TraversalEnvironment
from src.storage import create_storage
from src.utils.similarity_matrix import SimilarityMatrix


class TrainingComponentFactory:
    """Training component factory class.
    
    Responsible for creating and initializing various components required for training,
    reducing the complexity of the main training class.
    """

    def __init__(self, config: TShapeConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_quadtree(self) -> QuadTreeIndex:
        """Initialize bounding box and build quadtree index according to configuration.
        
        Automatically selects storage backend based on data.storage_mode configuration.
        
        Returns:
            Quadtree index instance
        """
        if self.config.index.use_original_bbox:
            bbox = self.config.get_original_bbox()
        else:
            bbox = SpatialBoundingBox(0.0, 0.0, 1.0, 1.0)
        
        storage = create_storage(
            mode=self.config.data.storage_mode,
            storage_dir=self.config.data.storage_dir,
            cache_mb=self.config.data.disk_cache_mb,
        )
        
        return QuadTreeIndex(
            bbox,
            max_level=self.config.index.max_level,
            alpha=self.config.index.alpha,
            beta=self.config.index.beta,
            storage=storage,
            parallel_signatures=self.config.index.parallel_signatures,
            signature_workers=self.config.index.signature_workers
        )

    def load_trajectories(
            self,
            quadtree: QuadTreeIndex
    ) -> List[Tuple[int, List[Tuple[float, float]]]]:
        """Load trajectory data, supports TDrive real data loading and synthetic data generation.
        
        Args:
            quadtree: Quadtree index
            
        Returns:
            Trajectory data list
        """
        bbox = quadtree.bbox

        dataset_path = self.config.get_dataset_trajectory_path()
        dataset_name = self.config.datasets.active

        if self.config.data.source == "dataset" and dataset_path is not None:
            print(f"Loading real trajectory data: {dataset_name} ({dataset_path})")
            trajectories = load_cleaned_dataset(
                str(dataset_path),
                max_trajectories=self.config.data.num_trajectories if self.config.data.num_trajectories > 0 else None,
            )

            if not trajectories:
                print(f"Valid dataset {dataset_name} not found, falling back to synthetic data.")
                trajectories = generate_synthetic_trajectories(self.config.data.num_trajectories, bbox)
            elif not self.config.index.use_original_bbox:
                print("Executing trajectory normalization...")
                trajectories = normalize_trajectories(trajectories, (0.0, 0.0, 1.0, 1.0))
        else:
            print("Generating synthetic trajectory data...")
            trajectories = generate_synthetic_trajectories(self.config.data.num_trajectories, bbox)

        return trajectories

    def create_environment(
            self,
            quadtree: QuadTreeIndex,
            cost_evaluator: TraversalCostEvaluator
    ) -> TraversalEnvironment:
        """Create traversal environment.
        
        Args:
            quadtree: Quadtree index
            cost_evaluator: Cost evaluator
            
        Returns:
            Traversal environment instance
        """
        train_queries, val_queries, test_queries = self._load_and_split_queries()

        env = TraversalEnvironment(
            quadtree=quadtree,
            cost_evaluator=cost_evaluator,
            reference_queries=train_queries,
            val_queries=val_queries,
            test_queries=test_queries,
            alpha=self.config.index.alpha,
            beta=self.config.index.beta,
            exclude_muted_cells=self.config.index.use_prune,
            quadcode_include_muted=self.config.index.quadcode_include_muted,
            local_reward_weight=self.config.reward.local_reward_weight,
            global_reward_weight=self.config.reward.global_reward_weight,
            reward_schedule_episodes=self.config.reward.reward_schedule_episodes,
            local_reward_start_scale=self.config.reward.local_reward_start_scale,
            global_reward_start_scale=self.config.reward.global_reward_start_scale,
            global_reward_scale=self.config.reward.global_reward_scale,
            global_reward_num_evals=self.config.reward.global_reward_num_evals,
            global_reward_query_sample_size=self.config.reward.global_reward_query_sample_size,
            global_reward_frontload_exponent=self.config.reward.global_reward_frontload_exponent,
        )

        return env

    def _load_and_split_queries(self) -> Tuple[List, List, List]:
        """Load query dataset and split into train/validation/test sets"""

        dataset_path = self.config.get_query_dataset_root()

        dist_type = self.config.reward.query_distribution_type

        category_dir = dataset_path / dist_type
        if category_dir.exists() and category_dir.is_dir():
            self.logger.info(f"Detected pre-split query set directory: {category_dir}")
            return self._load_pre_split_queries(category_dir, dist_type)

        raise FileNotFoundError(
            f"Pre-split query dataset directory not found: {dataset_path / dist_type}\n"
            f"Please generate query dataset first: python -m scripts.preprocess.generate_query_dataset"
        )

    def _load_pre_split_queries(self, category_dir: Path, dist_type: str) -> Tuple[List, List, List]:
        """Load train/validation/test sets from pre-split directory (JSON format)"""
        import json

        def load_json_queries(file_path: Path) -> List:
            """Load JSON query file"""
            if not file_path.exists():
                raise FileNotFoundError(f"Query file does not exist: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            queries = []
            for item in data:
                query_str = item['query']
                coords = [float(x.strip()) for x in query_str.split(',')]
                queries.append(SpatialBoundingBox(coords[0], coords[1], coords[2], coords[3]))

            return queries

        train_queries = load_json_queries(category_dir / "queries_train.json")
        val_queries = load_json_queries(category_dir / "queries_val.json")
        test_queries = load_json_queries(category_dir / "queries_test.json")

        sample_ratio = self.config.reward.query_sample_ratio
        if sample_ratio < 1.0:
            random.seed(42)
            
            train_sample_size = int(len(train_queries) * sample_ratio)
            if train_sample_size < len(train_queries):
                train_queries = random.sample(train_queries, train_sample_size)
            
            val_sample_size = int(len(val_queries) * sample_ratio)
            if val_sample_size < len(val_queries):
                val_queries = random.sample(val_queries, val_sample_size)
            
            print(f"  - Sampled at {sample_ratio * 100:.0f}%: train={len(train_queries)}, val={len(val_queries)}")

        total_loaded = len(train_queries) + len(val_queries) + len(test_queries)
        print(f"Loaded {dist_type} pre-split query set: total {total_loaded}")
        print(f"  - Train set: {len(train_queries)}")
        print(f"  - Validation set: {len(val_queries)}")
        print(f"  - Test set: {len(test_queries)}")

        return train_queries, val_queries, test_queries

    def setup_similarity_matrix(
            self,
            quadtree: QuadTreeIndex,
            cost_evaluator: TraversalCostEvaluator,
            all_cells: List
    ) -> Optional[SimilarityMatrix]:
        """Configure, load or compute trajectory similarity matrix.
        
        Args:
            quadtree: Quadtree index
            cost_evaluator: Cost evaluator
            all_cells: List of all cells
            
        Returns:
            Similarity matrix instance, or None if not enabled
        """
        if not self.config.data.use_similarity_matrix:
            return None

        similarity_matrix = SimilarityMatrix(quadtree, cost_evaluator)
        uses_explicit_path = bool(self.config.data.similarity_matrix_path)
        matrix_path = self.config.get_effective_similarity_matrix_path()

        if not uses_explicit_path:
            print(f"Using default similarity matrix path: {matrix_path}")

        if os.path.exists(matrix_path):
            print(f"Loading similarity matrix: {matrix_path}")
            if not similarity_matrix.load(matrix_path, all_cells):
                if uses_explicit_path:
                    print("Matrix dimensions mismatch, recomputing and overwriting specified file...")
                    self._compute_and_save_similarity_matrix(similarity_matrix, matrix_path, all_cells)
                else:
                    fallback_path = self.config.get_experiment_similarity_matrix_path()
                    print(
                        "Shared similarity matrix does not match current cell set, "
                        f"will use experiment private matrix: {fallback_path}"
                    )
                    if os.path.exists(fallback_path):
                        print(f"Attempting to load experiment private similarity matrix: {fallback_path}")
                        if not similarity_matrix.load(fallback_path, all_cells):
                            print("Experiment private matrix also mismatched, recomputing...")
                            self._compute_and_save_similarity_matrix(similarity_matrix, fallback_path, all_cells)
                    else:
                        self._compute_and_save_similarity_matrix(similarity_matrix, fallback_path, all_cells)
        else:
            self._compute_and_save_similarity_matrix(similarity_matrix, matrix_path, all_cells)

        return similarity_matrix

    def _compute_and_save_similarity_matrix(
            self,
            similarity_matrix: SimilarityMatrix,
            matrix_path: str,
            all_cells: List
    ) -> None:
        """Compute and persist similarity matrix.
        
        Args:
            similarity_matrix: Similarity matrix instance
            matrix_path: Save path
            all_cells: List of all cells
        """
        print(f"Starting similarity matrix computation (nodes: {len(all_cells)})...")
        num_workers = self.config.data.similarity_num_workers
        similarity_matrix.compute(all_cells, use_symmetric=True, show_progress=True, num_workers=num_workers)
        similarity_matrix.save(matrix_path)
