# LETI: Learned Space-Filling Curves for Efficient Trajectory Indexing

Official implementation of **LETI: Learned Space-Filling Curves for Efficient
Trajectory Indexing in Key-Value Databases**.

This repository contains the reinforcement-learning pipeline used to construct
the learned traversal order in LETI, together with compact-quadtree
construction, adaptive shape encoding, data preparation, evaluation, and order
export utilities.

## Repository Layout

```text
configs/                 Editable training configuration template
examples/                Example learned-order JSON
scripts/                 RL training and parameter sweeps
src/data/                Trajectory loading and normalization
src/indexing/            Compact quadtree and trajectory encoding
src/features/            RL state features
src/rl/                  Environment, policy, PPO, and order formatting
src/training/            Training orchestration and adaptive partitions
src/evaluation/          Traversal evaluation
resource/                Local data and generated artifacts (not tracked)
```

## Installation

Python 3.8 or later is required.

```bash
git clone <repository-url>
cd LSFC
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, install the project in editable mode:

```bash
pip install -e .
```

## Input Data

Data files are not included. A cleaned text file stores one trajectory per
line:

```text
trajectory_id|object_id|segment_id|lon,lat;lon,lat;...
```

Each trajectory must contain at least two points. GeoJSON Lines and WKT inputs
are also accepted by the loader; see `resource/README.md` for their formats and
the expected query layout.

Set the dataset path through the YAML configuration or an environment variable:

```bash
export DATASET_TDRIVE_PATH=/path/to/tdrive.txt
export DATASET_CHENGDU_PATH=/path/to/chengdu.txt
```

Training, validation, and test query workloads must be prepared separately and
placed under the query root described in `resource/README.md`.

## RL Training

Edit `configs/config.template.yaml` to select the dataset and set its local
paths, spatial bounds, and query distribution. Then run:

```bash
python -m scripts.run_pipeline \
  --config configs/config.template.yaml \
  --name leti-example
```

The pipeline constructs and prunes the quadtree, trains the Actor--Critic
policy with PPO, evaluates the learned traversal, and exports its order and
adaptive shape metadata.

To reproduce the depth and pruning-threshold parameter sweeps:

```bash
python -m scripts.generate_param_orders \
  --base-config configs/config.template.yaml \
  --distribution skewed \
  --num-episodes 400
```

The default output root is `outputs/`. It can be changed with `OUTPUT_DIR`;
shared resources can be redirected with `RESOURCE_BASE_DIR`.

## Exported Order

The exported JSON records the learned order of retained quadtree cells and the
metadata required by the trajectory index, including the spatial boundary,
quadtree depth, pruning threshold, enlarged-element partition, and local shape
encoding. The `order_source` of a trained model is `rl_quadorder`.
`examples/order.example.json` provides a small loadable example of this
format; it is illustrative and is not a reported experimental result.


## License

This project is licensed under the Apache License 2.0. See `LICENSE` for the
full license text.
