"""Training orchestration for the traversal learning pipeline."""

__all__ = ["TraversalTrainer"]


def __getattr__(name):
    """Lazy import to avoid circular dependency"""
    if name == "TraversalTrainer":
        from .traversal_trainer import TraversalTrainer
        return TraversalTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

