"""
Query dataset splitting script.

Splits query files into training set, validation set, and test set,
saves to <category>/ subdirectories under specified output directory.

Usage:
    python scripts/split_query_datasets.py --categories gus ske uni --train-ratio 0.6 --val-ratio 0.2
    
Or split all categories:
    python scripts/split_query_datasets.py --all
"""
import argparse
import random
from pathlib import Path
from typing import List, Tuple


def load_queries_from_files(category: str, queries_dir: Path) -> List[Tuple[float, float, float, float]]:
    """Load all queries of a specific category from original files."""
    all_queries = []

    category_map = {
        'gus': 'gaussian',
        'ske': 'skewed',
        'uni': 'uniform',
        'gaussian': 'gaussian',
        'skewed': 'skewed',
        'uniform': 'uniform',
    }

    dist_type = category_map.get(category, category)

    for range_m in [100, 500, 1000, 1500, 2000]:
        query_file = queries_dir / "range" / dist_type / f"{dist_type}_{range_m}m.txt"

        if not query_file.exists():
            print(f"warning: file does not exist {query_file}")
            continue

        with open(query_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                continue

            queries_str = content.replace('\n', ';').split(';')
            for q_str in queries_str:
                q_str = q_str.strip()
                if not q_str:
                    continue

                try:
                    coords = [float(x.strip()) for x in q_str.split(',')]
                    if len(coords) >= 4:
                        all_queries.append((coords[0], coords[1], coords[2], coords[3]))
                except (ValueError, IndexError) as e:
                    print(f"parse query failure '{q_str}': {e}")
                    continue

    return all_queries


def save_queries(queries: List[Tuple[float, float, float, float]], output_path: Path) -> None:
    """Save queries to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        content = ';'.join(f"{x1},{y1},{x2},{y2}" for x1, y1, x2, y2 in queries)
        f.write(content)

    print(f"  Saved {len(queries)} queries to {output_path}")


def split_dataset(queries: List, train_ratio: float, val_ratio: float,
                  seed: int = 42) -> Tuple[List, List, List]:
    """Split dataset."""
    random.seed(seed)
    shuffled = queries.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_set = shuffled[:train_size]
    val_set = shuffled[train_size:train_size + val_size]
    test_set = shuffled[train_size + val_size:]

    return train_set, val_set, test_set


def split_category(category: str, queries_dir: Path, output_base: Path,
                   train_ratio: float, val_ratio: float, seed: int = 42) -> None:
    """Split query set for a single category."""
    print(f"\nProcessing category: {category}")

    queries = load_queries_from_files(category, queries_dir)
    print(f"  Total loaded {len(queries)} queries")

    if len(queries) == 0:
        print(f"  Warning: No query data found for {category}")
        return

    train_set, val_set, test_set = split_dataset(queries, train_ratio, val_ratio, seed)

    print(f"  Split result: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    category_map = {
        'gus': 'gaussian',
        'ske': 'skewed',
        'uni': 'uniform',
    }
    output_dir = output_base / category_map.get(category, category)

    save_queries(train_set, output_dir / "train.txt")
    save_queries(val_set, output_dir / "val.txt")
    save_queries(test_set, output_dir / "test.txt")
    save_queries(queries, output_dir / "all.txt")


def main():
    parser = argparse.ArgumentParser(description="Split query dataset")
    parser.add_argument('--categories', nargs='+', default=['gus', 'ske', 'uni'],
                        help='Category codes to split (gus=gaussian, ske=skewed, uni=uniform)')
    parser.add_argument('--all', action='store_true',
                        help='Split all categories')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                        help='Training set ratio (default: 0.6)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--queries-dir', type=str, default='resource/queries/tdrive',
                        help='Query file directory')
    parser.add_argument('--output-dir', type=str, default='resource/queries/tdrive',
                        help='Output directory')

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    queries_dir = project_root / args.queries_dir
    output_dir = project_root / args.output_dir

    print(f"Query file directory: {queries_dir}")
    print(f"Output directory: {output_dir}")
    print(
        f"Split ratio: train={args.train_ratio}, val={args.val_ratio}, test={1 - args.train_ratio - args.val_ratio:.1f}")

    if args.all:
        categories = ['gaussian', 'skewed', 'uniform']
    else:
        categories = args.categories

    for category in categories:
        split_category(category, queries_dir, output_dir,
                       args.train_ratio, args.val_ratio, args.seed)

    print("\nQuery dataset splitting completed!")


if __name__ == "__main__":
    main()
