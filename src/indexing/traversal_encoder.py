"""Generate traversal order and encoding based on quadtree index."""
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from src.indexing.lsfc_loader import LSFCMappingLoader
from src.indexing.quadtree_cell import QuadTreeCell
from src.indexing.quadtree_index import QuadTreeIndex


def encode_with_order(order: Iterable[QuadTreeCell]) -> Dict[QuadTreeCell, int]:
    """Encode cells according to given traversal order."""
    encoding: Dict[QuadTreeCell, int] = {}
    for index, cell in enumerate(order):
        cell.code = index
        encoding[cell] = index
    return encoding


class TraversalOrderEncoder:
    """Generate traversal order and corresponding encoding based on given logic."""

    def __init__(self, quadtree: QuadTreeIndex, alpha: int = 2, beta: int = 2):
        self.quadtree = quadtree
        self.alpha = alpha
        self.beta = beta
        self._order_loader: Optional[LSFCMappingLoader] = None

    def z_curve_order(self, include_muted: bool = False) -> List[QuadTreeCell]:
        """Return depth-first Z-curve traversal order."""
        ordered_cells: List[QuadTreeCell] = []

        def dfs(cell: QuadTreeCell) -> None:
            if include_muted or not cell.muted:
                ordered_cells.append(cell)
            if cell.level < self.quadtree.max_level:
                for child in cell.children:
                    if child:
                        dfs(child)

        dfs(self.quadtree.root)
        return ordered_cells

    def load_quadorder_mapping(self, filepath: Union[str, Path]) -> None:
        """Load learned order mapping file (JSON or CSV)"""
        self._order_loader = LSFCMappingLoader()
        filepath = Path(filepath)

        if filepath.suffix.lower() == '.json':
            self._order_loader.load_from_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}, please use .json")

    def get_order_loader(self) -> Optional[LSFCMappingLoader]:
        """Return currently loaded order mapping loader."""
        return self._order_loader

    def quadorder(self, include_muted: bool = False) -> Optional[List[QuadTreeCell]]:
        """Return sorted cell list based on loaded learned order mapping"""
        if self._order_loader is None or not self._order_loader.is_loaded():
            return None

        if include_muted:
            all_cells = list(self.quadtree.all_cells.values())
        else:
            all_cells = self.quadtree.get_active_cells()

        return self._order_loader.get_ordered_cells(all_cells)

    def encode_with_quadorder(self, include_muted: bool = False) -> Optional[Dict[QuadTreeCell, int]]:
        """Encode cells using learned order"""
        if self._order_loader is None or not self._order_loader.is_loaded():
            return None

        if include_muted:
            all_cells = list(self.quadtree.all_cells.values())
        else:
            all_cells = self.quadtree.get_active_cells()

        return self._order_loader.apply_order_to_cells(all_cells)
