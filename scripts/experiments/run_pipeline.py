"""Unified experiment run script: uses LSFCPipeLine for model training and order export

Usage:
    python -m scripts.experiments.run_pipeline --config configs/experiments/default/config.yaml --name default
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import List, Optional

from src.config import TShapeConfig
from src.rl.pipeline import LSFCPipeLine
from src.utils.logger import setup_logging


def run_rl_indexing_experiment(config_path: str, exp_name: str,
                               hidden_dims: Optional[List[int]] = None,
                               device: Optional[str] = None):
    """Execute complete RL index optimization experiment.

    Args:
        config_path: YAML configuration file path
        exp_name: Experiment name (e.g., test/formal)
        hidden_dims: Overrides the network hidden dimensions from the configuration file
        device: Overrides the computing device from the configuration file
    """
    ts_config = TShapeConfig.from_yaml(config_path)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(f"Experiment_{exp_name.capitalize()}_{timestamp}")
    logger.info(f"=== Starting {exp_name} experiment ===")

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    nt_config = ts_config.network
    if hidden_dims:
        nt_config.hidden_dims = list(hidden_dims)
    if device:
        nt_config.device = device

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

        logger.info("Step 2: Evaluating selected model on the test query set...")
        test_metrics = pipeline.evaluate_on_test()

        logger.info("Step 3: Validating output results...")
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
        val_improvement = results.get('val_improvement', None)
        if val_improvement is not None:
            print(f"Val improvement (model selection): {val_improvement:.2f}%")
        else:
            print(f"Val improvement: N/A")
        print(f"Test improvement (vs QuadCode): {test_metrics['improvement_percent']:.2f}%")

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
        default=None,
        help="Neural network hidden layer dimensions (overrides the configuration file)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computing device (auto/cuda/cpu, overrides the configuration file)"
    )

    args = parser.parse_args()

    run_rl_indexing_experiment(args.config, args.name, args.hidden_dims, args.device)
