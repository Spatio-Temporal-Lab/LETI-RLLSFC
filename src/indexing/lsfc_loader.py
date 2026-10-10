"""Efficient learned order mapping loader: loads from JSON and uses tuple keys for optimized queries."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.indexing.quadtree_cell import QuadTreeCell


class LSFCMappingLoader:
    """Efficient learned order mapping loader.
    
    Uses tuple keys to optimize query performance, supports JSON format only.
    """

    def __init__(self):
        """Initialize the mapping loader."""
        self.logger = logging.getLogger(__name__)
        self._tuple_to_order: Dict[Tuple[int, Tuple[int, ...]], int] = {}
        self._quad_code_to_order: Dict[int, int] = {}
        self._quad_code_to_coverage: Dict[int, Dict[str, Union[int, bool, List[int]]]] = {}
        self._max_level: Optional[int] = None
        self._metadata: Dict[str, Union[int, float, str, bool]] = {}
        self._loaded = False

    def load_from_json(self, filepath: Union[str, Path]) -> None:
        """Load mapping from JSON file.
        
        Args:
            filepath: JSON file path
            
        Raises:
            FileNotFoundError: File does not exist
            ValueError: Invalid JSON format
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Mapping file does not exist: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'ordering' not in data:
            raise ValueError("Invalid JSON format: missing 'ordering' field")

        if 'metadata' in data and 'quadtree_max_level' in data['metadata']:
            self._max_level = data['metadata']['quadtree_max_level']
            self._metadata = dict(data['metadata'])
        else:
            raise ValueError("Invalid JSON format: missing metadata.quadtree_max_level field")

        self._tuple_to_order.clear()
        self._quad_code_to_order.clear()
        self._quad_code_to_coverage.clear()

        for item in data['ordering']:
            if 'quad_code' not in item or 'order' not in item:
                self.logger.warning(f"Skipping invalid item: {item}")
                continue

            order = item['order']
            quad_codes = item['quad_code']
            coverage = item.get('coverage') or {}

            if isinstance(quad_codes, (int, float)):
                quad_codes = [int(quad_codes)]

            for q_code in quad_codes:
                q_code = int(q_code)
                try:
                    level, quadrant_sequence = self._decode_quad_code(q_code)
                    key = (level, tuple(quadrant_sequence))

                    self._tuple_to_order[key] = order
                    self._quad_code_to_order[q_code] = order
                    if coverage:
                        self._quad_code_to_coverage[q_code] = coverage
                except ValueError as e:
                    self.logger.warning(f"Failed to decode quad_code={q_code}: {e}")
                    continue

        self._loaded = True
        self.logger.info(f"Successfully loaded {len(self._tuple_to_order)} mapping items")


    def _decode_quad_code(self, quad_code: int) -> Tuple[int, List[int]]:
        """Decode quad_code to obtain level and quadrant_sequence.
        
        Encoding rules (TShape encoding):
        - Root (level=0): quad_code = 0
        - Other nodes: code = sum(quadrant[i] * ((4^(max_level - i + 1) - 1) // 3) + 1 for i in 1..level)
        
        Args:
            quad_code: Quadtree encoding
            
        Returns:
            (level, quadrant_sequence) tuple
            
        Raises:
            ValueError: Unable to decode
        """
        if self._max_level is None:
            raise RuntimeError("max_level not initialized, please load configuration file first")

        if quad_code == 0:
            return 0, []

        for target_level in range(1, self._max_level + 1):
            quadrant_sequence = []
            remaining = quad_code
            
            for i in range(1, target_level + 1):
                base = (4 ** (self._max_level - i + 1) - 1) // 3
                
                if remaining < 1:
                    break
                
                remaining -= 1
                
                quadrant = remaining // base
                
                if quadrant < 0 or quadrant > 3:
                    break
                
                quadrant_sequence.append(quadrant)
                
                remaining -= quadrant * base
            
            if remaining == 0 and len(quadrant_sequence) == target_level:
                return target_level, quadrant_sequence
        
        raise ValueError(
            f"Unable to decode quad_code={quad_code}, max_level={self._max_level}"
        )

    def get_order(self, cell: QuadTreeCell) -> Optional[int]:
        """Get the order of a cell.
        
        Args:
            cell: Quadtree cell
            
        Returns:
            Order value, or None if not found
            
        Raises:
            RuntimeError: Mapping not loaded
        """
        if not self._loaded:
            raise RuntimeError("Please call load_from_json() to load mapping file first")
        
        key = (cell.level, tuple(cell.quadrant_sequence))
        return self._tuple_to_order.get(key)

    def get_order_from_tuple(
        self,
        level: int,
        quadrant_sequence: Tuple[int, ...]
    ) -> Optional[int]:
        """Get order from tuple.
        
        Args:
            level: Level
            quadrant_sequence: Quadrant sequence
            
        Returns:
            Order value, or None if not found
            
        Raises:
            RuntimeError: Mapping not loaded
        """
        if not self._loaded:
            raise RuntimeError("Please call load_from_json() to load mapping file first")

        key = (level, quadrant_sequence)
        return self._tuple_to_order.get(key)

    def apply_order_to_cells(
        self,
        cells: List[QuadTreeCell]
    ) -> Dict[QuadTreeCell, int]:
        """Batch apply order to cell list.
        
        Args:
            cells: Cell list
            
        Returns:
            Mapping dictionary from cell to order
            
        Raises:
            RuntimeError: Mapping not loaded
        """
        if not self._loaded:
            raise RuntimeError("Please call load_from_json() to load mapping file first")

        result = {}
        for cell in cells:
            order = self.get_order(cell)
            if order is not None:
                result[cell] = order

        return result

    def get_ordered_cells(self, cells: List[QuadTreeCell]) -> List[QuadTreeCell]:
        """Sort cell list according to the mapping order.
        
        Args:
            cells: Cell list
            
        Returns:
            Sorted cell list
            
        Raises:
            RuntimeError: Mapping not loaded
        """
        if not self._loaded:
            raise RuntimeError("Please call load_from_json() to load mapping file first")

        cell_orders = []
        for cell in cells:
            order = self.get_order(cell)
            if order is not None:
                cell_orders.append((order, cell))

        cell_orders.sort(key=lambda x: x[0])

        return [cell for _, cell in cell_orders]

    def is_loaded(self) -> bool:
        """Check if mapping is loaded."""
        return self._loaded

    def get_metadata(self) -> Dict[str, Union[int, float, str, bool]]:
        """Return metadata from the loaded order file."""
        return dict(self._metadata)

    def get_order_source(self) -> Optional[str]:
        """Return the order file source tag."""
        value = self._metadata.get("order_source")
        return str(value) if value is not None else None

    def get_coverage_by_cell(self, cell: QuadTreeCell) -> Optional[Dict[str, Union[int, bool, List[int]]]]:
        """Get coverage information associated with the cell."""
        q_code = cell.get_quadrant_code(self._max_level) if self._max_level is not None else None
        if q_code is None:
            return None
        coverage = self._quad_code_to_coverage.get(q_code)
        return dict(coverage) if coverage is not None else None

    def get_effective_subtree_contiguous(self) -> Optional[bool]:
        value = self._metadata.get("effective_subtree_contiguous")
        return value if isinstance(value, bool) else None

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get mapping statistics information."""
        if not self._loaded:
            return {"loaded": False, "count": 0}

        return {
            "loaded": True,
            "count": len(self._tuple_to_order),
            "quad_code_count": len(self._quad_code_to_order),
        }


def create_order_mapping_loader(filepath: Union[str, Path]) -> LSFCMappingLoader:
    """Convenience function: create mapping loader from JSON file.
    
    Args:
        filepath: JSON file path
        
    Returns:
        Loaded mapping loader
        
    Raises:
        ValueError: Unsupported file format
    """
    loader = LSFCMappingLoader()
    filepath = Path(filepath)

    if filepath.suffix.lower() == '.json':
        loader.load_from_json(filepath)
    else:
        raise ValueError(
            f"Unsupported file format: {filepath.suffix}, only .json format is supported"
        )

    return loader
