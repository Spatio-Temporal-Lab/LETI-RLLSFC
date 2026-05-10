# Dataset Format Specification

## Trajectory Data Format

### File Format

Trajectory data should be provided as a pipe-delimited text file with the following columns:

```
trajectory_id|timestamp|longitude|latitude
```

### Column Descriptions

- **trajectory_id**: Unique identifier for the trajectory (integer or string)
- **timestamp**: Timestamp in format `YYYY-MM-DD HH:MM:SS`
- **longitude**: Longitude coordinate (float, typically in range -180 to 180)
- **latitude**: Latitude coordinate (float, typically in range -90 to 90)

### Example

```
1|2023-01-01 00:00:00|116.397|39.908
1|2023-01-01 00:01:00|116.398|39.909
1|2023-01-01 00:02:00|116.399|39.910
2|2023-01-01 00:00:00|116.400|39.910
2|2023-01-01 00:01:00|116.401|39.911
```

### Requirements

- File encoding: UTF-8
- Line separator: `\n` (Unix-style) or `\r\n` (Windows-style)
- Trajectories should be sorted by trajectory_id and timestamp
- Coordinates should be in WGS84 coordinate system
- Minimum 2 points per trajectory

## Query Data Format

### File Format

Query data is stored in JSON format with an array of bounding box queries:

```json
[
  {
    "min_lon": 116.3,
    "min_lat": 39.8,
    "max_lon": 116.5,
    "max_lat": 40.0
  },
  {
    "min_lon": 116.4,
    "min_lat": 39.9,
    "max_lon": 116.6,
    "max_lat": 40.1
  }
]
```

### Field Descriptions

- **min_lon**: Minimum longitude of the query bounding box
- **min_lat**: Minimum latitude of the query bounding box
- **max_lon**: Maximum longitude of the query bounding box
- **max_lat**: Maximum latitude of the query bounding box

### Requirements

- File encoding: UTF-8
- Valid JSON format
- Bounding boxes should be within the dataset bounds
- min_lon < max_lon and min_lat < max_lat

## Directory Structure

The expected directory structure for datasets:

```
resource/
├── queries/
│   ├── tdrive/
│   │   ├── gaussian/
│   │   │   ├── queries_train.json
│   │   │   ├── queries_val.json
│   │   │   └── queries_test.json
│   │   ├── skewed/
│   │   │   ├── queries_train.json
│   │   │   ├── queries_val.json
│   │   │   └── queries_test.json
│   │   └── uniform/
│   │       ├── queries_train.json
│   │       ├── queries_val.json
│   │       └── queries_test.json
│   └── chengdu/
│       └── [same structure as tdrive]
├── matrices/
│   └── similarity/
│       └── [precomputed similarity matrices]
└── orders/
    └── [exported traversal orders]
```

### Query Distribution Types

- **gaussian**: Queries follow a Gaussian distribution centered on high-density areas
- **skewed**: Queries are heavily concentrated in specific regions
- **uniform**: Queries are uniformly distributed across the spatial domain

### Dataset Splits

- **queries_train.json**: Training queries (typically 60% of total)
- **queries_val.json**: Validation queries (typically 20% of total)
- **queries_test.json**: Test queries (typically 20% of total)

## Preparing Custom Datasets

### Step 1: Prepare Trajectory Data

1. Convert your trajectory data to the pipe-delimited format
2. Ensure coordinates are in WGS84 (EPSG:4326)
3. Sort by trajectory_id and timestamp
4. Save as UTF-8 encoded text file

### Step 2: Generate Queries

Use the provided script to generate queries:

```bash
python scripts/preprocess/generate_queries.py \
  --min-lon <min_longitude> \
  --min-lat <min_latitude> \
  --max-lon <max_longitude> \
  --max-lat <max_latitude> \
  --traj-path <path_to_trajectory_file> \
  --output-dir resource/queries/<dataset_name>
```

This will generate queries for all three distributions (gaussian, skewed, uniform).

### Step 3: Split Query Datasets

Split the generated queries into train/val/test sets:

```bash
python scripts/preprocess/split_query_datasets.py \
  --queries-dir resource/queries/<dataset_name> \
  --output-dir resource/queries/<dataset_name> \
  --train-ratio 0.6 \
  --val-ratio 0.2
```

### Step 4: Update Configuration

Update the dataset configuration in `configs/experiments/default/config.yaml`:

```yaml
datasets:
  profiles:
    <dataset_name>:
      trajectory_path: "<path_to_trajectory_file>"
      bounds:
        min_lon: <min_longitude>
        min_lat: <min_latitude>
        max_lon: <max_longitude>
        max_lat: <max_latitude>
```

## Supported Datasets

The system has been tested with the following public datasets:

### T-Drive

- **Source**: Microsoft Research
- **Description**: GPS trajectories of taxis in Beijing
- **Size**: ~15 million points, ~10,000 trajectories
- **Spatial extent**: Beijing metropolitan area
- **Temporal extent**: February 2008

### Chengdu Taxi

- **Source**: Chengdu Transportation Research Institute
- **Description**: GPS trajectories of taxis in Chengdu
- **Size**: ~20 million points, ~12,000 trajectories
- **Spatial extent**: Chengdu metropolitan area

## Data Quality Guidelines

For best results, ensure your trajectory data meets these quality criteria:

1. **Sampling rate**: Regular sampling (e.g., every 30-60 seconds)
2. **Coverage**: Good spatial coverage of the study area
3. **Completeness**: Minimal missing data or gaps
4. **Accuracy**: GPS accuracy within 10-20 meters
5. **Volume**: At least 1,000 trajectories for meaningful training
