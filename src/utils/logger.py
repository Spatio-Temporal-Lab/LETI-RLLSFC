import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    log_name: str, 
    verbose: bool = False,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Setup logger.
    
    Args:
        log_name: Logger name
        verbose: Whether to output verbose information
        log_file: Log file path (optional)
    """
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
