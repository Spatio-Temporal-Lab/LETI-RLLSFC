"""Trajectory order formatting utility functions."""
from typing import Any, Dict, List, Tuple, Optional

from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex


def get_unique_active_nodes(order: List[QuadTreeCell], quadtree: QuadTreeIndex) -> List[QuadTreeCell]:
    """Deduplicate and extract active nodes.
    
    Args:
        order: Original visit order
        quadtree: Quadtree index
        
    Returns:
        Deduplicated list of active nodes
    """
    seen_codes = set()
    active_nodes = []

    root = quadtree.root
    if not root.muted:
        active_nodes.append(root)
        seen_codes.add(root.get_quadrant_code(quadtree.max_level))

    for cell in order:
        if not cell.muted:
            qc = cell.get_quadrant_code(quadtree.max_level)
            if qc not in seen_codes:
                active_nodes.append(cell)
                seen_codes.add(qc)
    
    return active_nodes


def build_active_map(active_nodes: List[QuadTreeCell], max_level: int) -> Tuple[Dict, Dict]:
    """Build mapping from Cell objects to configuration containers.
    
    Args:
        active_nodes: List of active nodes
        max_level: Maximum quadtree level
        
    Returns:
        (active_map, qc_to_info) tuple
        - active_map: Cell -> Info mapping
        - qc_to_info: QC -> Info mapping
    """
    active_map = {}
    qc_to_info = {}

    for idx, cell in enumerate(active_nodes):
        qc = cell.get_quadrant_code(max_level)
        info = {
            "order": idx,
            "cell": cell,
            "active_qc": qc,
            "muted_set": set()
        }
        active_map[cell] = info
        qc_to_info[qc] = info
    
    return active_map, qc_to_info


def build_parent_descriptor(cell: QuadTreeCell, max_level: int) -> Dict[str, Any]:
    """Build parent node geometry description.
    
    Args:
        cell: Cell object
        max_level: Maximum quadtree level
        
    Returns:
        Dictionary containing parent node geometry information
    """
    return {
        "alpha": int(cell.alpha),
        "beta": int(cell.beta),
        "level": int(cell.level),
        "element_code": int(cell.get_quadrant_code(max_level)),
        "xmin": float(cell.bbox.min_x),
        "ymin": float(cell.bbox.min_y),
        "xmax": float(cell.bbox.max_x),
        "ymax": float(cell.bbox.max_y)
    }


def find_active_parent(cell: QuadTreeCell) -> Optional[QuadTreeCell]:
    """Trace upward until finding a non-muted node.
    
    Args:
        cell: Starting cell
        
    Returns:
        First non-muted ancestor node, or None if none exists
    """
    curr = cell.parent
    while curr is not None:
        if not curr.muted:
            return curr
        curr = curr.parent
    return None


def assemble_ordering(qc_to_info: Dict, max_level: int) -> List[Dict]:
    """Construct final export structure.
    
    Args:
        qc_to_info: Mapping from QC to Info
        max_level: Maximum quadtree level
        
    Returns:
        Sorted configuration list
    """
    sorted_infos = sorted(qc_to_info.values(), key=lambda x: x["order"])
    ordering = []

    for info in sorted_infos:
        cell = info["cell"]
        combined_codes = [info["active_qc"]] + sorted(list(info["muted_set"]))

        ordering.append({
            "quad_code": combined_codes[0],
            "order": info["order"],
            "parent": build_parent_descriptor(cell, max_level)
        })
    
    return ordering
