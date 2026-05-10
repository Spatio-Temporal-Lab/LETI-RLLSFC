# LETI: Learned Space-Filling Curve for Efficient Trajectory Indexing

## Overview

This repository contains the official source code for the paper:

**"LETI: Learned Space-Filling Curve for Efficient Trajectory Indexing in Key-Value Databases"**

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd RLLSFC-Public
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

### Environment Variables (Optional)

You can configure dataset paths using environment variables:

```bash
# Windows
set DATASET_TDRIVE_PATH=path\to\tdrive.txt
set DATASET_CHENGDU_PATH=path\to\chengdu_taxi.txt
set OUTPUT_DIR=path\to\outputs

# Linux/Mac
export DATASET_TDRIVE_PATH=path/to/tdrive.txt
export DATASET_CHENGDU_PATH=path/to/chengdu_taxi.txt
export OUTPUT_DIR=path/to/outputs
```

**Note**: When using predefined dataset configurations (e.g., `--dataset tdrive`), the system will look for trajectory paths in this order:
1. Command-line argument (`--traj-path`)
2. Environment variable (`DATASET_TDRIVE_PATH` or `DATASET_CHENGDU_PATH`)
3. If neither is provided, an error will be shown

## Dataset Format

### Trajectory Data

Trajectory files should be in pipe-delimited format:

```
trajectory_id|timestamp|longitude|latitude
1|2023-01-01 00:00:00|116.397|39.908
1|2023-01-01 00:01:00|116.398|39.909
2|2023-01-01 00:00:00|116.400|39.910
...
```

### Query Data

Query files are in JSON format:

```json
[
  {
    "min_lon": 116.3,
    "min_lat": 39.8,
    "max_lon": 116.5,
    "max_lat": 40.0
  }
]
```

Query datasets should be organized as:
```
resource/queries/<dataset>/<distribution>/
  queries_train.json
  queries_val.json
  queries_test.json
```

Where `<distribution>` is one of: `gaussian`, `skewed`, or `uniform`.

See `resource/README.md` for detailed format specifications.

## Learned Order Format

The system exports learned traversal orders as JSON files. These files define the optimized space-filling curve order for trajectory indexing.

### File Structure

A learned order file contains two main sections: `ordering` and `metadata`.

```json
{
  "ordering": [
    {
      "quad_code": 0,
      "order": 0,
      "parent": {
        "alpha": 2,
        "beta": 2,
        "level": 0,
        "element_code": 0,
        "xmin": 0.0,
        "ymin": 0.0,
        "xmax": 1.0,
        "ymax": 1.0
      },
      "coverage": {
        "effective_subtree_count": 15,
        "effective_subtree_orders": [1, 2, 3, ...]
      }
    },
    ...
  ],
  "metadata": {
    "total_cells": 256,
    "active_cells": 128,
    "spatial_boundary": {
      "xmin": 115.29,
      "ymin": 39.00,
      "xmax": 117.83,
      "ymax": 41.50
    },
    "quadtree_max_level": 7,
    "global_alpha": 2,
    "global_beta": 2,
    "generation_timestamp": "20xx-xx-xxTxx:xx:xx",
    "version": "1.4",
    "order_source": "rl_quadorder",
    "effective_subtree_contiguous": false,
    "partition_search": [[2, 2], [8, 8]],
    "max_partition": 64,
    "max_shape_count": 16,
    "min_trajs": 10
  }
}
```

### Field Descriptions

#### Ordering Array

Each entry in the `ordering` array represents a quadtree cell in the learned traversal order:

- **`quad_code`** (integer): Unique identifier for the quadtree cell, computed from its quadrant sequence
- **`order`** (integer): Position in the traversal order (0-indexed)
- **`parent`** (object): Cell properties and spatial information
  - **`alpha`** (integer): Horizontal partition parameter (cell width = parent_width / alpha)
  - **`beta`** (integer): Vertical partition parameter (cell height = parent_height / beta)
  - **`level`** (integer): Depth in the quadtree (0 = root)
  - **`element_code`** (integer): Same as `quad_code`, kept for compatibility
  - **`xmin`, `ymin`, `xmax`, `ymax`** (float): Spatial bounding box coordinates
- **`coverage`** (object): Subtree coverage information
  - **`effective_subtree_count`** (integer): Number of descendant cells in the active tree
  - **`effective_subtree_orders`** (array, optional): Order indices of all descendants (omitted for XZ orders to save space)

#### Metadata Object

Global information about the learned order:

- **`total_cells`** (integer): Total number of cells in the complete quadtree
- **`active_cells`** (integer): Number of non-pruned cells in the traversal order
- **`spatial_boundary`** (object): Global spatial extent
  - **`xmin`, `ymin`, `xmax`, `ymax`** (float): Bounding box of the entire dataset
- **`quadtree_max_level`** (integer): Maximum depth of the quadtree
- **`global_alpha`** (integer): Default horizontal partition parameter
- **`global_beta`** (integer): Default vertical partition parameter
- **`generation_timestamp`** (string): ISO 8601 timestamp of when the order was generated
- **`version`** (string): Format version (current: "1.4")
- **`order_source`** (string): How the order was generated
  - `"rl_quadorder"`: Learned via reinforcement learning
  - `"default_xz_order"`: Standard Z-order (Morton curve)
  - `"hilbert_order"`: Hilbert curve
- **`effective_subtree_contiguous`** (boolean): Whether subtrees are contiguous in the order
- **`partition_search`** (array): Range of partition parameters explored during optimization
- **`max_partition`** (integer): Maximum partition size (alpha × beta) used
- **`max_shape_count`** (integer): Maximum number of unique signatures per cell
- **`min_trajs`** (integer, nullable): Minimum trajectory count threshold for pruning

### Usage Example

Load and use a learned order:

```python
from src.indexing import TraversalOrderEncoder

# Create encoder and load learned order
encoder = TraversalOrderEncoder(quadtree, alpha=2, beta=2)
encoder.load_quadorder_mapping("path/to/learned_order.json")

# Get ordered cells
ordered_cells = encoder.quadorder(include_muted=False)

# Apply encoding
encoding = encoder.encode_with_quadorder(include_muted=False)
```

### Exporting Learned Orders

Export orders after training:

```bash
# Export RL-learned order
python -m scripts.experiments.run_pipeline \
  --config configs/experiments/default/config.yaml \
  --name my_experiment

# Export baseline XZ order (for comparison)
python -m scripts.experiments.export_pruned_xz_order \
  --config configs/experiments/default/config.yaml \
  --export-prefix baseline_xz
```

### Adaptive Partitions

For orders with adaptive partitioning (varying alpha/beta per cell), export partition metadata separately:

```bash
python -m scripts.experiments.export_adaptive_partitions \
  --config configs/experiments/default/config.yaml
```

This generates a companion file with per-cell partition parameters that can be merged with the order file using:

```bash
python -m scripts.experiments.merge_order_with_partitions \
  --order-file path/to/order.json \
  --partitions-file path/to/partitions.json \
  --output-file path/to/merged_order.json
```

## Quick Start

### 1. Generate Query Data

Generate queries for a dataset:

```bash
python scripts/preprocess/generate_queries.py --dataset tdrive
```

### 2. Run Training

Train the RL model with test configuration:

```bash
python -m scripts.experiments.run_pipeline --config configs/experiments/test/config.yaml --name test
```

### 3. Export Learned Orders

Export the learned traversal order:

```bash
python -m scripts.experiments.export_pruned_xz_order --config configs/experiments/default/config.yaml
```