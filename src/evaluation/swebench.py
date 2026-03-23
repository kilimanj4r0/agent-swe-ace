"""SWE-bench patch validation using Docker harness."""

from pathlib import Path
from typing import Optional

import docker
from loguru import logger


def validate_patch_docker(
    instance: dict,
    patch: str,
    timeout: int = 1800,
    rm_image: bool = True,
    output_dir: Optional[Path] = None,
) -> bool:
    """
    Validate patch using SWE-bench Docker harness directly.

    Args:
        instance: SWE-bench instance dict
        patch: The patch string to validate
        timeout: Timeout in seconds (default 30 min)
        rm_image: Remove Docker image after evaluation (default True, saves disk space)
        output_dir: Optional output directory for SWE-bench logs

    Returns:
        True if patch resolves all tests
    """
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.test_spec.test_spec import make_test_spec
    import swebench.harness.constants as constants

    instance_id = instance["instance_id"]

    # Redirect SWE-bench logs to output_dir if provided
    original_log_dir = None
    if output_dir:
        original_log_dir = constants.RUN_EVALUATION_LOG_DIR
        constants.RUN_EVALUATION_LOG_DIR = Path(output_dir) / "swebench_logs"

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
            rm_image=rm_image,
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

    finally:
        # Restore original log directory
        if original_log_dir is not None:
            constants.RUN_EVALUATION_LOG_DIR = original_log_dir
