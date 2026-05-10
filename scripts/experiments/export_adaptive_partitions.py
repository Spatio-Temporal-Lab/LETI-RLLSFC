"""Export signature-optimized adaptive partition information without traversal order."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.config import TShapeConfig
from src.rl.partition_formatter import AdaptivePartitionFormatter
from src.training.component_factory import TrainingComponentFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export adaptive partition information")
    parser.add_argument("--config", type=str, default="configs/experiments/default/config.yaml", help="YAML configuration file path")
    parser.add_argument("--dataset", type=str, default=None, help="Override active dataset, e.g. tdrive / cdtaxi")
    parser.add_argument("--max-level", type=int, default=None, help="Override quadtree level")
    parser.add_argument("--min-trajs", type=int, default=None, help="Override min_cell_trajs")
    parser.add_argument("--num-trajectories", type=int, default=None, help="Override trajectory count")
    parser.add_argument("--source", type=str, choices=["dataset", "synthetic"], default=None, help="Override data source")
    parser.add_argument("--disable-optimize", action="store_true", help="Disable signature optimization, only export global partition")
    parser.add_argument("--export-prefix", type=str, default="adaptive_partitions", help="Output file prefix")
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
        quadtree.post_prune_tree(config.index.min_cell_trajs)
    quadtree.compute_signatures(config.index.enable_sig_optimize)

    formatter = AdaptivePartitionFormatter(config=config)
    payload = formatter.export(quadtree)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        formatter.output_dir = output_path.parent
        filename = output_path.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{args.export_prefix}_{config.datasets.active}_"
            f"res{config.index.max_level}_min{config.index.min_cell_trajs}_{timestamp}.json"
        )

    save_path = formatter.save(payload, filename)
    print(f"adaptive_partition_file: {save_path}")


if __name__ == "__main__":
    main()
