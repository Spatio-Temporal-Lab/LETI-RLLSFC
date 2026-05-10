"""Unified experiment run script: uses LSFCPipeLine for model training and order export

Usage:
    python -m scripts.experiments.run_pipeline --config configs/experiments/test/config.yaml --name test
    python -m scripts.experiments.run_pipeline --config configs/experiments/formal/config.yaml --name formal
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

from src.config import NetworkConfig, TShapeConfig
from src.rl.pipeline import LSFCPipeLine
from src.utils.logger import setup_logging


def run_rl_indexing_experiment(config_path: str, exp_name: str, nt_config: NetworkConfig):
    """Execute complete RL index optimization experiment.

    Args:
        config_path: YAML configuration file path
        exp_name: Experiment name (e.g., test/formal)
        nt_config: Neural network configuration object
    """
    ts_config = TShapeConfig.from_yaml(config_path)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(f"Experiment_{exp_name.capitalize()}_{timestamp}")
    logger.info(f"=== Starting {exp_name} experiment ===")

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    if nt_config is None:
        nt_config = NetworkConfig(
            hidden_dims=[256, 256],
            device="auto"
        )

    export_prefix = f"quadorder_{exp_name}_{timestamp}"

    pipeline = LSFCPipeLine(
        ts_config,
        nt_config,
        logger=logger
    )

    try:
        logger.info("Step 1: Starting full training and export process...")
        start_time = time.time()

        results = pipeline.run_full_pipeline(export_prefix=export_prefix)

        end_time = time.time()
        duration = (end_time - start_time) / 60
        logger.info(f"Pipeline executed successfully, time elapsed: {duration:.2f} minutes")

        logger.info("Step 2: Validating output results...")
        orders_dir = Path(pipeline.resource_paths["results"])
        models_dir = Path(pipeline.resource_paths["checkpoints"])

        check_files = [
            f"{export_prefix}.json"
        ]

        print("\n" + "=" * 50)
        print(f" Experiment Summary - {timestamp} ")
        print("-" * 50)
        print(f"Model save path: {models_dir}")
        print(f"Total learned nodes: {results['quadorder_length']}")
        improvement_rate = results.get('improvement_rate', None)
        if improvement_rate is not None:
            print(f"Improvement rate (vs QuadCode): {improvement_rate:.2f}%")
        else:
            print(f"Improvement rate (vs QuadCode): N/A")

        print("\nExported file status:")
        for fname in check_files:
            fpath = orders_dir / fname
            status = "[OK]" if fpath.exists() else "[MISSING]"
            size = f"{fpath.stat().st_size / 1024:.1f} KB" if fpath.exists() else "0 KB"
            print(f"  {status} {fname} ({size})")

        meta_path = orders_dir / f"{export_prefix}_metadata.json"
        with open(meta_path, 'w') as f:
            meta_data = {
                "timestamp": timestamp,
                "duration_min": duration,
                "config": ts_config.to_dict(),
                "results": {
                    k: (str(v) if not isinstance(v, dict) else v) for k, v in results.items()
                    if k != 'export_results'
                }
            }
            json.dump(meta_data, f, indent=4)
        print(f"\nExperiment metadata saved to: {meta_path}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Error occurred during experiment execution: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL index optimization experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML configuration file path (e.g., configs/experiments/test/config.yaml)"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Experiment name (e.g., test/formal), used to organize output directory"
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Neural network hidden layer dimensions (default: 256 256)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computing device (auto/cuda/cpu, default: auto)"
    )

    args = parser.parse_args()

    network_config = NetworkConfig(
        hidden_dims=list(args.hidden_dims),
        device=args.device
    )

    run_rl_indexing_experiment(args.config, args.name, network_config)
