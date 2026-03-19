"""Docker environment setup for SWE-bench instances."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def create_docker_environment(instance: dict, timeout: int = 120):
    """
    Create Docker environment for SWE-bench instance.

    Args:
        instance: SWE-bench instance dict with instance_id, repo, etc.
        timeout: Command timeout in seconds

    Returns:
        DockerEnvironment instance
    """
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.run.extra.swebench import get_swebench_docker_image_name

    image_name = get_swebench_docker_image_name(instance)
    logger.info(f"Using Docker environment with image: {image_name}")

    return DockerEnvironment(
        image=image_name,
        cwd="/testbed",
        timeout=timeout,
    )


def create_local_environment(work_dir: Path):
    """
    Create local environment for running agent.

    Args:
        work_dir: Working directory for agent

    Returns:
        LocalEnvironment instance
    """
    from minisweagent.environments.local import LocalEnvironment

    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using local environment with work dir: {work_dir}")

    return LocalEnvironment(cwd=str(work_dir))
