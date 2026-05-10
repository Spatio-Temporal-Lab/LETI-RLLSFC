r"""Query dataset generation script (universal version)

Supports multiple datasets, specify configuration through parameters.

Generates three distribution types of query datasets:
1. uniform: Uniform distribution (uniformly sampled from trajectory points)
2. skewed: Skewed distribution (80% concentrated in trajectory hotspot areas)
3. gaussian: Gaussian distribution (centered on trajectory point distribution)

Each type contains 5 ranges: 100m, 500m, 1000m, 1500m, 2000m
Each range generates 100 queries, 500 queries per type in total

Usage examples:
    # TDrive (Beijing) - use preset configuration
    python scripts/preprocess/generate_queries.py --dataset tdrive
    
    # Chengdu dataset - use preset configuration
    python scripts/preprocess/generate_queries.py --dataset chengdu
    
    # Custom parameters
    python scripts/preprocess/generate_queries.py \
        --min-lon 115.29 --min-lat 39.00 --max-lon 117.83 --max-lat 41.50 \
        --traj-path /path/to/trajectory/data.txt \
        --output-dir resource/queries_tdrive
"""
import argparse
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict
import json


#  Predefined dataset configurations
DATASET_CONFIGS = {
    'tdrive': {
        'min_lon': 115.29,
        'min_lat': 39.00,
        'max_lon': 117.83,
        'max_lat': 41.50,
        'traj_path': None,  # Set via environment variable DATASET_TDRIVE_PATH or command line
        'output_dir': 'resource/queries/tdrive',
    },
    'chengdu': {
        'min_lon': 104.04,
        'min_lat': 30.65,
        'max_lon': 104.13,
        'max_lat': 30.73,
        'traj_path': None,  # Set via environment variable DATASET_CHENGDU_PATH or command line
        'output_dir': 'resource/queries/chengdu',
    },
}

#  Query ranges (meters)
QUERY_RANGES = [100, 500, 1000, 1500, 2000]
QUERIES_PER_RANGE = 100


class QueryGenerator:
    """Query generator - samples and generates queries from trajectory data"""
    
    def __init__(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                 output_dir: Path, traj_path: Optional[str] = None):
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.output_dir = output_dir
        self.traj_path = traj_path
        
        self.traj_points = self._load_trajectory_points()
        if not self.traj_points:
            raise ValueError(f"Failed to load any points from trajectory file: {traj_path}")
        
        lons = [p[0] for p in self.traj_points]
        lats = [p[1] for p in self.traj_points]
        self.traj_center_lon = np.mean(lons)
        self.traj_center_lat = np.mean(lats)
        self.traj_std_lon = np.std(lons) if np.std(lons) > 0 else (max_lon - min_lon) / 4
        self.traj_std_lat = np.std(lats) if np.std(lats) > 0 else (max_lat - min_lat) / 4
        
        self.hotspot_points = self._compute_hotspot_points()
        
        print(f"Loaded {len(self.traj_points)} trajectory points")
        print(f"Trajectory center: ({self.traj_center_lon:.6f}, {self.traj_center_lat:.6f})")
        print(f"Trajectory std dev: ({self.traj_std_lon:.6f}, {self.traj_std_lat:.6f})")
        print(f"Hotspot area contains {len(self.hotspot_points)} points")
    
    def _load_trajectory_points(self) -> List[Tuple[float, float]]:
        """Load all trajectory points from trajectory file"""
        points = []
        if not self.traj_path or not Path(self.traj_path).exists():
            print(f"Warning: Trajectory file does not exist: {self.traj_path}")
            return points
        
        print(f"Loading trajectory data: {self.traj_path}")
        try:
            with open(self.traj_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('|')
                    if len(parts) >= 4:
                        coords_str = parts[3]
                        coord_pairs = coords_str.split(';')
                        for cp in coord_pairs:
                            if ',' in cp:
                                try:
                                    lon_str, lat_str = cp.split(',')
                                    lon, lat = float(lon_str), float(lat_str)
                                    if self.min_lon <= lon <= self.max_lon and \
                                       self.min_lat <= lat <= self.max_lat:
                                        points.append((lon, lat))
                                except ValueError:
                                    continue
        except Exception as e:
            print(f"Error loading trajectory file: {e}")
        
        return points
    
    def _compute_hotspot_points(self, grid_size: int = 10) -> List[Tuple[float, float]]:
        """Compute points in hotspot areas (top 20% highest trajectory density areas)"""
        if not self.traj_points:
            return []
        
        lon_bins = np.linspace(self.min_lon, self.max_lon, grid_size + 1)
        lat_bins = np.linspace(self.min_lat, self.max_lat, grid_size + 1)
        
        grid_counts = defaultdict(list)
        for lon, lat in self.traj_points:
            lon_idx = np.searchsorted(lon_bins, lon, side='right') - 1
            lat_idx = np.searchsorted(lat_bins, lat, side='right') - 1
            lon_idx = max(0, min(lon_idx, grid_size - 1))
            lat_idx = max(0, min(lat_idx, grid_size - 1))
            grid_counts[(lon_idx, lat_idx)].append((lon, lat))
        
        grid_densities = [(len(pts), pts) for pts in grid_counts.values()]
        grid_densities.sort(reverse=True)
        
        num_hotspot_grids = max(1, int(len(grid_densities) * 0.2))
        hotspot_points = []
        for _, pts in grid_densities[:num_hotspot_grids]:
            hotspot_points.extend(pts)
        
        return hotspot_points if hotspot_points else self.traj_points
    
    def meters_to_degrees(self, meters: float, latitude: float) -> Tuple[float, float]:
        """Convert meters to longitude/latitude offset"""
        lat_offset = meters / 111000.0
        lon_offset = meters / (111000.0 * np.cos(np.radians(latitude)))
        return lon_offset, lat_offset

    def generate_uniform_queries(self, num_queries: int, range_meters: float) -> List[str]:
        """Generate uniformly distributed queries - uniformly sampled from all trajectory points"""
        queries = []
        for _ in range(num_queries):
            center_lon, center_lat = random.choice(self.traj_points)
            lon_offset, lat_offset = self.meters_to_degrees(range_meters / 2, center_lat)
            min_lon = max(self.min_lon, center_lon - lon_offset)
            max_lon = min(self.max_lon, center_lon + lon_offset)
            min_lat = max(self.min_lat, center_lat - lat_offset)
            max_lat = min(self.max_lat, center_lat + lat_offset)
            queries.append(f"{min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}")
        return queries

    def generate_skewed_queries(self, num_queries: int, range_meters: float) -> List[str]:
        """Generate skewed distribution queries (80% concentrated in trajectory hotspot areas)"""
        queries = []
        
        for _ in range(num_queries):
            if random.random() < 0.8 and self.hotspot_points:
                center_lon, center_lat = random.choice(self.hotspot_points)
            else:
                center_lon, center_lat = random.choice(self.traj_points)
            
            lon_offset, lat_offset = self.meters_to_degrees(range_meters / 2, center_lat)
            min_lon = max(self.min_lon, center_lon - lon_offset)
            max_lon = min(self.max_lon, center_lon + lon_offset)
            min_lat = max(self.min_lat, center_lat - lat_offset)
            max_lat = min(self.max_lat, center_lat + lat_offset)
            queries.append(f"{min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}")
        return queries

    def generate_gaussian_queries(self, num_queries: int, range_meters: float) -> List[str]:
        """Generate Gaussian distribution queries - centered on trajectory distribution"""
        queries = []
        
        for _ in range(num_queries):
            query_center_lon = np.random.normal(self.traj_center_lon, self.traj_std_lon)
            query_center_lat = np.random.normal(self.traj_center_lat, self.traj_std_lat)
            
            query_center_lon = np.clip(query_center_lon, self.min_lon, self.max_lon)
            query_center_lat = np.clip(query_center_lat, self.min_lat, self.max_lat)
            
            lon_offset, lat_offset = self.meters_to_degrees(range_meters / 2, query_center_lat)
            min_lon = max(self.min_lon, query_center_lon - lon_offset)
            max_lon = min(self.max_lon, query_center_lon + lon_offset)
            min_lat = max(self.min_lat, query_center_lat - lat_offset)
            max_lat = min(self.max_lat, query_center_lat + lat_offset)
            queries.append(f"{min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}")
        return queries

    def run(self):
        """Run generation"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        distributions = {
            'gaussian': self.generate_gaussian_queries,
            'skewed': self.generate_skewed_queries,
            'uniform': self.generate_uniform_queries
        }
        
        for dist_name, generator_func in distributions.items():
            print(f"\nGenerating {dist_name} distribution queries...")
            all_queries = []
            
            for range_meters in QUERY_RANGES:
                print(f"  Range: {range_meters}m")
                queries = generator_func(QUERIES_PER_RANGE, range_meters)
                for q in queries:
                    all_queries.append({'query': q, 'range_meters': range_meters})
                
                range_dist_dir = self.output_dir / "range" / dist_name
                range_dist_dir.mkdir(parents=True, exist_ok=True)
                filepath = range_dist_dir / f"{dist_name}_{range_meters}m.txt"
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(queries))
                print(f"    Generated {len(queries)} queries -> range/{dist_name}/{filepath.name}")
            
            random.shuffle(all_queries)
            total = len(all_queries)
            train_size = int(total * 0.7)
            val_size = int(total * 0.15)
            
            splits = {
                'train': all_queries[:train_size],
                'val': all_queries[train_size:train_size + val_size],
                'test': all_queries[train_size + val_size:]
            }
            
            dist_dir = self.output_dir / dist_name
            dist_dir.mkdir(parents=True, exist_ok=True)
            
            for split_name, split_data in splits.items():
                filepath = dist_dir / f"queries_{split_name}.json"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, indent=2, ensure_ascii=False)
                print(f"    Saved {split_name}: {len(split_data)} queries")
            
            print(f"  ✓ {dist_name} completed: {total} queries")
        
        print(f"\n✓ Query dataset generation completed!")
        print(f"  Bounding box: [{self.min_lon}, {self.min_lat}, {self.max_lon}, {self.max_lat}]")
        print(f"  Output directory: {self.output_dir}")
        print(f"  File structure:")
        print(f"    range/[distribution]/[distribution]_[range]m.txt - By type and range files (100 each)")
        print(f"    gaussian/ | skewed/ | uniform/")
        print(f"      - queries_train.json (350 queries)")
        print(f"      - queries_val.json (75 queries)")
        print(f"      - queries_test.json (75 queries)")


def main():
    import os
    
    parser = argparse.ArgumentParser(description='Generate query dataset (based on trajectory data sampling)')
    parser.add_argument('--dataset', type=str, choices=['tdrive', 'chengdu'],
                        help='Use predefined dataset configuration (tdrive/chengdu)')
    parser.add_argument('--min-lon', type=float, help='Minimum longitude')
    parser.add_argument('--min-lat', type=float, help='Minimum latitude')
    parser.add_argument('--max-lon', type=float, help='Maximum longitude')
    parser.add_argument('--max-lat', type=float, help='Maximum latitude')
    parser.add_argument('--traj-path', type=str, help='Trajectory data file path')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    if args.dataset:
        config = DATASET_CONFIGS[args.dataset]
        min_lon = config['min_lon']
        min_lat = config['min_lat']
        max_lon = config['max_lon']
        max_lat = config['max_lat']
        
        # Try to get trajectory path from: 1) command line, 2) environment variable, 3) config
        traj_path = args.traj_path
        if not traj_path:
            env_var = f"DATASET_{args.dataset.upper()}_PATH"
            traj_path = os.environ.get(env_var, config['traj_path'])
        
        output_dir = Path(config['output_dir'])
        print(f"Using predefined configuration: {args.dataset}")
        
        if not traj_path:
            print(f"Error: Trajectory path not specified. Please provide via:")
            print(f"  1. Command line: --traj-path /path/to/data.txt")
            print(f"  2. Environment variable: {env_var}")
            return
    else:
        min_lon = args.min_lon or 115.29
        min_lat = args.min_lat or 39.00
        max_lon = args.max_lon or 117.83
        max_lat = args.max_lat or 41.50
        traj_path = args.traj_path
        output_dir = Path(args.output_dir or 'resource/queries/tdrive')
        
        if not traj_path:
            print("Error: Trajectory path required. Use --traj-path /path/to/data.txt")
            return
    
    generator = QueryGenerator(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        output_dir=output_dir,
        traj_path=traj_path
    )
    generator.run()


if __name__ == "__main__":
    main()
