"""Experience replay buffer for storing and managing reinforcement learning interaction data."""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    """Single-step interaction experience."""
    state: np.ndarray
    action: int
    reward: float
    log_prob: torch.Tensor
    value: torch.Tensor
    mask: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for storing and batch retrieving training data."""

    def __init__(self):
        self.transitions: List[Transition] = []

    def store(self, transition: Transition) -> None:
        """Store single-step experience."""
        self.transitions.append(transition)

    def get_all(self) -> List[Transition]:
        """Get all stored experiences."""
        return self.transitions

    def clear(self) -> None:
        """Clear the buffer."""
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)

    def is_empty(self) -> bool:
        return len(self.transitions) == 0
