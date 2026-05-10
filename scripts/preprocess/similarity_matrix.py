"""Precompute a similarity matrix from a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import TShapeConfig
from src.common import TraversalCostEvaluator
from src.training.component_factory import TrainingComponentFactory
from src.utils.similarity_matrix import SimilarityMatrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute similarity matrix")
    parser.add_argument("--config", type=str, default="configs/experiments/default/config.yaml", help="YAML configuration file path")
    parser.add_argument("--dataset", type=str, default=None, help="Override active dataset in configuration, e.g. tdrive / cdtaxi")
    parser.add_argument("--output-file", type=str, default=None, help="Manually specify output matrix path")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers for similarity computation")
    parser.add_argument("--force", action="store_true", help="Recompute even if target file already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TShapeConfig.from_yaml(args.config)
    if args.dataset:
        config.datasets.active = args.dataset

    output_path = Path(args.output_file) if args.output_file else config.get_effective_similarity_matrix_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.force:
        print(f"Similarity matrix already exists, skipping computation: {output_path}")
        return

    factory = TrainingComponentFactory(config)
    quadtree = factory.create_quadtree()
    trajectories = factory.load_trajectories(quadtree)
    print(f"Starting trajectory assignment, total {len(trajectories)} trajectories")
    for trajectory_id, points in trajectories:
        quadtree.assign_trajectory(trajectory_id, points)

    if config.index.min_cell_trajs is not None:
        quadtree.post_prune_tree(config.index.min_cell_trajs)
    quadtree.compute_signatures(config.index.enable_sig_optimize)

    all_cells = [cell for cell in quadtree.get_all_cells() if not cell.muted]
    cost_evaluator = TraversalCostEvaluator(quadtree)
    matrix = SimilarityMatrix(quadtree, cost_evaluator)
    num_workers = args.num_workers if args.num_workers is not None else config.data.similarity_num_workers

    print("=" * 60)
    print(
        f"Precomputing similarity matrix | dataset={config.datasets.active} | "
        f"cells={len(all_cells)} | output={output_path}"
    )
    print("=" * 60)
    matrix.compute(all_cells, use_symmetric=True, show_progress=True, num_workers=num_workers)
    matrix.save(str(output_path))

    stats = matrix.get_statistics()
    print(f"matrix_size: {stats.get('matrix_size', len(all_cells))}")
    print(f"mean_similarity: {stats.get('mean_similarity', 0.0):.4f}")
    print(f"saved_to: {output_path}")


if __name__ == "__main__":
    main()
