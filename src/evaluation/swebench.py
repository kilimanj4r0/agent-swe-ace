"""SWE-bench patch validation using Docker harness."""

from typing import Dict

import docker
from loguru import logger


def validate_patch_docker(instance: dict, patch: str, timeout: int = 1800) -> bool:
    """
    Validate patch using SWE-bench Docker harness directly.

    Args:
        instance: SWE-bench instance dict
        patch: The patch string to validate
        timeout: Timeout in seconds (default 30 min)

    Returns:
        True if patch resolves all tests
    """
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.test_spec.test_spec import make_test_spec

    instance_id = instance["instance_id"]

    try:
        client = docker.from_env()
        test_spec = make_test_spec(instance)
        pred = {
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": "ace-swe-wrapper"
        }

        logger.info(f"Running Docker evaluation for {instance_id}...")
        result = run_instance(
            test_spec=test_spec,
            pred=pred,
            rm_image=False,
            force_rebuild=False,
            client=client,
            run_id="ace_eval",
            timeout=timeout
        )

        resolved = result.get("resolved", False)
        logger.info(f"Evaluation result for {instance_id}: resolved={resolved}")
        return resolved

    except Exception as e:
        logger.error(f"Docker evaluation error for {instance_id}: {e}")
        return False
