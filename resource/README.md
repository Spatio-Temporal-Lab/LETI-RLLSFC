# Data and Generated Resource Layout

The `resource` directory is a local workspace for datasets, query workloads,
similarity matrices, and exported traversal orders.
The repository does not provide dataset-preprocessing scripts.

## Cleaned Trajectory Data

The primary text format stores one complete trajectory per line:

```text
trajectory_id|object_id|segment_id|lon,lat;lon,lat;...
```

Example:

```text
1|taxi_001|0|116.397,39.908;116.398,39.909;116.400,39.910
2|taxi_002|0|116.410,39.920;116.412,39.921
```

Requirements:

- longitude followed by latitude;
- at least two valid points per trajectory;
- coordinates in the same reference system as the bounds in the selected
  configuration.

The loader also accepts:

- GeoJSON Lines, with `properties.tid` and a coordinate list;
- WKT records in the form
  `trajectory_id,object_id,segment_id,LINESTRING(lon lat, ...)`.

Dataset files may be stored anywhere. Point the configuration to them through
`trajectory_path`, `DATASET_TDRIVE_PATH`, or `DATASET_CHENGDU_PATH`.

## Query Workloads

Each query file is a JSON array. Every item contains a comma-separated bounding
box and its range in meters:

```json
[
  {
    "query": "116.300000, 39.800000, 116.500000, 40.000000",
    "range_meters": 1000
  }
]
```

The training pipeline expects the following layout:

```text
resource/
└── queries/
    ├── tdrive/
    │   ├── gaussian/
    │   │   ├── queries_train.json
    │   │   ├── queries_val.json
    │   │   └── queries_test.json
    │   ├── skewed/
    │   │   ├── queries_train.json
    │   │   ├── queries_val.json
    │   │   └── queries_test.json
    │   └── uniform/
    │       ├── queries_train.json
    │       ├── queries_val.json
    │       └── queries_test.json
    └── chengdu/
        └── ...
```

Prepare these workload files externally and set `query_root` in the training
configuration to their parent directory.

## Other Generated Artifacts

Depending on the configuration, the pipeline may create:

```text
resource/
├── matrices/            Cached trajectory-similarity matrices
└── orders/              Exported learned traversal orders

outputs/
├── checkpoints/         Trained model states
├── logs/                Training logs
└── results/             Evaluation summaries and order metadata
```

These files can be regenerated and must not be committed. The repository's
`.gitignore` excludes them by default.

A compact example of the exported order format is provided at
`examples/order.example.json`.
