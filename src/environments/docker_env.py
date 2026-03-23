"""Docker environment setup for SWE-bench instances."""

from pathlib import Path
from typing import Optional

from loguru import logger


def create_docker_environment(instance: dict, timeout: int = 120, namespace: Optional[str] = None):
    """
    Create Docker environment for SWE-bench instance.

    Args:
        instance: SWE-bench instance dict with instance_id, repo, etc.
        timeout: Command timeout in seconds
        namespace: Optional Docker registry namespace prefix (e.g., "ghcr.io/epoch-research/")

    Returns:
        DockerEnvironment instance
    """
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.run.extra.swebench import get_swebench_docker_image_name

    image_name = get_swebench_docker_image_name(instance)
    # get_swebench_docker_image_name returns "docker.io/swebench/sweb.eval.x86_64.{id}:latest"
    # We need "sweb.eval.x86_64.{id}:latest" then prepend namespace
    if image_name.startswith("docker.io/swebench/"):
        image_name = image_name[len("docker.io/swebench/"):]
    if namespace:
        # Strip trailing slash from namespace to avoid double slashes
        normalized_namespace = namespace.rstrip("/")
        image_name = f"{normalized_namespace}/{image_name}"
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
