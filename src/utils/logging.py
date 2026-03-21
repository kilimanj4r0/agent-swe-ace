"""Centralized logging configuration using loguru."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(run_dir: Path | None = None, log_level: str = "INFO") -> None:
    """
    Configure loguru for experiment logging.

    Args:
        run_dir: Optional run directory for experiment.log
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    # Remove default handler
    logger.remove()

    # Console handler with simpler format
    logger.add(
        sys.stderr,
        level=log_level,
        format="<level>{level:8}</level> | {message}",
        colorize=True,
    )

    # File handler for experiment.log (if run_dir provided)
    if run_dir:
        log_file = run_dir / "experiment.log"
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {name}:{line} | {message}",
            rotation="10 MB",
        )
        logger.info(f"Logging to {log_file}")
