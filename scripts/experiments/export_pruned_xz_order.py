"""Export a pruned + shape-optimized traversal order using the default XZ order."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.config import TShapeConfig
from src.indexing import TraversalOrderEncoder
from src.rl.order_formatter import TrajectoryOrderFormatter
from src.training.component_factory import TrainingComponentFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export default XZ order after pruning and shape optimization only")
    parser.add_argument("--config", type=str, default="configs/experiments/default/config.yaml", help="YAML configuration file path")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset, e.g. tdrive / cdtaxi")
    parser.add_argument("--max-level", type=int, default=None, help="Override quadtree level")
    parser.add_argument("--min-trajs", type=int, default=None, help="Override min_cell_trajs")
    parser.add_argument("--num-trajectories", type=int, default=None, help="Override trajectory count")
    parser.add_argument("--source", type=str, choices=["dataset", "synthetic"], default=None, help="Override data source")
    parser.add_argument("--disable-prune", action="store_true", help="Disable pruning")
    parser.add_argument("--disable-optimize", action="store_true", help="Disable shape optimization")
    parser.add_argument("--export-prefix", type=str, default="pruned_xz_order", help="Output file prefix")
    parser.add_argument("--output-file", type=str, default=None, help="Explicit output path")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TShapeConfig:
    config = TShapeConfig.from_yaml(args.config)
    if args.dataset:
        config.datasets.active = args.dataset
    if args.max_level is not None:
        config.index.max_level = args.max_level
    if args.min_trajs is not None:
        config.index.min_cell_trajs = args.min_trajs
        config.index.use_prune = args.min_trajs > 0
    if args.num_trajectories is not None:
        config.data.num_trajectories = args.num_trajectories
    if args.source is not None:
        config.data.source = args.source
    if args.disable_prune:
        config.index.use_prune = False
        config.index.min_cell_trajs = 0
    if args.disable_optimize:
        config.index.enable_sig_optimize = False
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)

    factory = TrainingComponentFactory(config)
    quadtree = factory.create_quadtree()
    trajectories = factory.load_trajectories(quadtree)

    print(f"Starting trajectory assignment, total {len(trajectories)} trajectories")
    for trajectory_id, points in trajectories:
        quadtree.assign_trajectory(trajectory_id, points)

    if config.index.use_prune and config.index.min_cell_trajs is not None:
        prune_stats = quadtree.post_prune_tree(config.index.min_cell_trajs)
        print(f"Pruning completed: active={prune_stats['after']}, muted={prune_stats['muted']}")
    else:
        print("Skipping pruning, keeping original XZ structure")

    optimize_stats = quadtree.compute_signatures(config.index.enable_sig_optimize)
    print(
        "Shape optimization completed: "
        f"alpha_shrunk={optimize_stats['shrunk_alpha']}, "
        f"beta_shrunk={optimize_stats['shrunk_beta']}, "
        f"both={optimize_stats['both_shrunk']}"
    )

    encoder = TraversalOrderEncoder(
        quadtree,
        alpha=config.index.alpha,
        beta=config.index.beta,
    )
    xz_order = encoder.z_curve_order(include_muted=False)

    formatter = TrajectoryOrderFormatter(config=config)
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        formatter.output_dir = output_path.parent
        filename = output_path.name
    else:
        filename = (
            f"{config.reward.query_distribution_type}_"
            f"r{config.index.max_level}_"
            f"min{config.index.min_cell_trajs}_"
            f"a{config.index.alpha}_"
            f"b{config.index.beta}.json"
        )

    _, save_path = formatter.generate_config_file_from_order(
        order=xz_order,
        quadtree=quadtree,
        filename=filename,
        global_alpha=config.index.alpha,
        global_beta=config.index.beta,
        order_source="pruned_default_xz_order",
    )

    print(f"active_order_length: {len(xz_order)}")
    print(f"order_file: {save_path}")


if __name__ == "__main__":
    main()
