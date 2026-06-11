# LETI: Learned Space-Filling Curve for Efficient Trajectory Indexing

Official implementation of **LETI: Learned Space-Filling Curve for Efficient Trajectory Indexing in Key-Value Databases**.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd LETI-RLLSFC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

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
[{"min_lon": 116.3, "min_lat": 39.8, "max_lon": 116.5, "max_lat": 40.0}]
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
      "quad_code": 0, "order": 0,
      "parent": {
        "alpha": 2, "beta": 2,
        "level": 0, "element_code": 0,
        "xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0
      },
      "coverage": {
        "effective_subtree_count": 15,
        "effective_subtree_orders": [1, 2, 3, "..."]
      }
    },
    "..."
  ],
  "metadata": {
    "total_cells": 256, "active_cells": 128,
    "spatial_boundary": {"xmin": 115.29, "ymin": 39.00, "xmax": 117.83, "ymax": 41.50},
    "quadtree_max_level": 7, "global_alpha": 2, "global_beta": 2,
    "generation_timestamp": "20xx-xx-xxTxx:xx:xx",
    "version": "1.0",
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

- **`quad_code`**: Unique identifier for the quadtree cell, computed from its quadrant sequence
- **`order`**: Position in the traversal order
- **`parent`**: Cell properties and spatial information
  - **`alpha`**: Horizontal partition parameter (cell width = parent_width / alpha)
  - **`beta`**: Vertical partition parameter (cell height = parent_height / beta)
  - **`level`**: Depth in the quadtree (0 = root)
  - **`element_code`**: Same as `quad_code`, kept for compatibility
  - **`xmin`, `ymin`, `xmax`, `ymax`**: Spatial bounding box coordinates
- **`coverage`**: Subtree coverage information
  - **`effective_subtree_count`**: Number of descendant cells in the active tree
  - **`effective_subtree_orders`**: Order indices of all descendants (omitted for XZ orders)

#### Metadata Object

Global information about the learned order:

- **`total_cells`**: Total number of cells in the complete quadtree
- **`active_cells`**: Number of non-pruned cells in the traversal order
- **`spatial_boundary`**: Global spatial extent
  - **`xmin`, `ymin`, `xmax`, `ymax`**: Bounding box of the entire dataset
- **`quadtree_max_level`**: Maximum depth of the quadtree
- **`global_alpha`**: Default horizontal partition parameter
- **`global_beta`**: Default vertical partition parameter
- **`generation_timestamp`**: ISO 8601 timestamp of when the order was generated
- **`version`**: Format version
- **`order_source`**: How the order was generated
  - `"rl_quadorder"`: Learned via reinforcement learning
  - `"default_xz_order"`: Standard Z-order
  - `"hilbert_order"`: Hilbert curve
- **`effective_subtree_contiguous`**: Whether subtrees are contiguous in the order (True for XZ Order)
- **`partition_search`**: Range of partition parameters explored during optimization
- **`max_partition`**: Maximum partition size (alpha × beta) used
- **`max_shape_count`** Maximum number of unique signatures per cell
- **`min_trajs`**: Minimum trajectory count threshold for pruning

## Quick Start

### 1. Generate Query Data

Generate queries for a dataset:
```bash
python scripts/preprocess/generate_queries.py --dataset tdrive
```

### 2. Run Training

Train the RL model with configuration:
```bash
python -m scripts.experiments.run_pipeline --config configs/experiments/default/config.yaml --name default
```