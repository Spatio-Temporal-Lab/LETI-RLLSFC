"""Stable local ordering helpers shared by indexing and optimization."""

from typing import Iterable, List


UINT64_MASK = (1 << 64) - 1


def deterministic_hamming_order(shapes: Iterable[int]) -> List[int]:
    """Mirror the Java loader's unsigned, deterministic nearest-neighbour rule."""
    remaining = sorted({int(shape) & UINT64_MASK for shape in shapes})
    if not remaining:
        return []

    ordered = [remaining.pop(0)]
    while remaining:
        current = ordered[-1]
        next_index = min(
            range(len(remaining)),
            key=lambda index: (
                bin(current ^ remaining[index]).count("1"),
                remaining[index],
            ),
        )
        ordered.append(remaining.pop(next_index))
    return ordered
