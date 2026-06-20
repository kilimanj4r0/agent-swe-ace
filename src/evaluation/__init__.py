"""Evaluation module for patch validation."""

from pathlib import Path
from typing import Optional

from evaluation.swebench import validate_patch_docker


def validate_patch(
    instance: dict,
    patch: str,
    use_docker: bool = True,
    timeout: int = 1800,
    rm_image: bool = True,
    output_dir: Optional[Path] = None,
    namespace: Optional[str] = None,
) -> bool:
    """
    Validate a patch against an instance.

    Args:
        instance: SWE-bench instance dict
        patch: The patch string to validate
        use_docker: If True, use Docker harness; otherwise use simple validation
        timeout: Timeout in seconds for Docker evaluation
        rm_image: Remove Docker image after evaluation (default True, saves disk space)
        output_dir: Optional output directory for SWE-bench logs
        namespace: Optional Docker registry namespace prefix (e.g., "ghcr.io/epoch-research/")

    Returns:
        True if patch resolves all tests
    """
    if use_docker:
        return validate_patch_docker(
            instance=instance,
            patch=patch,
            timeout=timeout,
            rm_image=rm_image,
            output_dir=output_dir,
            namespace=namespace,
        )


__all__ = [
    "validate_patch",
    "validate_patch_docker",
]
