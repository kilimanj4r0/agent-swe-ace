"""Centralized logging configuration using loguru."""

import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

# Thread-local storage for instance context
_thread_local = threading.local()


@contextmanager
def instance_context(instance_id: str):
    """Set instance_id context for all log messages in the current thread.

    Usage:
        with instance_context("django__django-12345"):
            logger.info("Processing")  # → [django__django-12345] Processing
    """
    old = getattr(_thread_local, "instance_id", None)
    _thread_local.instance_id = instance_id
    try:
        yield
    finally:
        _thread_local.instance_id = old


def _inject_instance_tag(record):
    """Filter+patch: inject instance_tag into record extra, always return True."""
    instance_id = getattr(_thread_local, "instance_id", None)
    record["extra"]["instance_tag"] = f"[{instance_id}] " if instance_id else ""
    return True


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
        format="<level>{level:8}</level> | {extra[instance_tag]}{message}",
        colorize=True,
        filter=_inject_instance_tag,
    )

    # File handler for experiment.log (if run_dir provided)
    if run_dir:
        log_file = run_dir / "experiment.log"
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {extra[instance_tag]}{name}:{line} | {message}",
            rotation="10 MB",
            filter=_inject_instance_tag,
        )
        logger.info(f"Logging to {log_file}")
