"""Bounding box conversion tool: converts normalized quadrant_sequence to original coordinate bounding box."""

from typing import List, Tuple

from src.common import SpatialBoundingBox
from src.indexing.quadtree_cell import QuadTreeCell


def convert_sequences_to_original_bboxes(
    sequences: List[List[int]],
    levels: List[int],
    root_bbox: SpatialBoundingBox
) -> List[SpatialBoundingBox]:
    """
    Batch convert quadrant_sequence to original bounding boxes.
    
    Args:
        sequences: List of quadrant_sequences
        levels: Corresponding level list
        root_bbox: Original bounding box of root node
    
    Returns:
        List of original bounding boxes
    """
    return [
        QuadTreeCell.sequence_to_bbox(seq, root_bbox)
        for seq, level in zip(sequences, levels)
    ]


def get_bbox_center(bbox: SpatialBoundingBox) -> Tuple[float, float]:
    """Get center point coordinates of bounding box."""
    return bbox.get_center()


def format_bbox_string(bbox: SpatialBoundingBox, precision: int = 4) -> str:
    """
    Format bounding box as string, format: min_x min_y max_x max_y
    
    Args:
        bbox: Bounding box
        precision: Decimal precision
    
    Returns:
        Formatted string, e.g.: "115.7019 39.2008 117.5987 40.8490"
    """
    return (f"{bbox.min_x:.{precision}f} {bbox.min_y:.{precision}f} "
            f"{bbox.max_x:.{precision}f} {bbox.max_y:.{precision}f}")


def parse_bbox_string(bbox_str: str) -> SpatialBoundingBox:
    """
    Parse bounding box from string, format: min_x min_y max_x max_y
    
    Args:
        bbox_str: Bounding box string, e.g.: "115.7019 39.2008 117.5987 40.8490"
    
    Returns:
        SpatialBoundingBox object
    """
    parts = bbox_str.strip().split()
    if len(parts) != 4:
        raise ValueError(f"Invalid bbox string format: {bbox_str}, expected 4 values")
    
    return SpatialBoundingBox(
        min_x=float(parts[0]),
        min_y=float(parts[1]),
        max_x=float(parts[2]),
        max_y=float(parts[3])
    )
