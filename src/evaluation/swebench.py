"""SWE-bench patch validation using Docker harness."""

import threading
from pathlib import Path
from typing import Optional

import docker
from loguru import logger

# Lock to serialize evaluation calls (swebench mutates a global constant)
_eval_lock = threading.Lock()


def validate_patch_docker(
    instance: dict,
    patch: str,
    timeout: int = 1800,
    rm_image: bool = True,
    output_dir: Optional[Path] = None,
    namespace: Optional[str] = None,
) -> bool:
    """
    Validate patch using SWE-bench Docker harness directly.

    Args:
        instance: SWE-bench instance dict
        patch: The patch string to validate
        timeout: Timeout in seconds (default 30 min)
        rm_image: Remove Docker image after evaluation (default True, saves disk space)
        output_dir: Optional output directory for SWE-bench logs
        namespace: Optional Docker registry namespace prefix (e.g., "ghcr.io/epoch-research/")

    Returns:
        True if patch resolves all tests
    """
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.test_spec.test_spec import make_test_spec
    import swebench.harness.constants as constants

    instance_id = instance["instance_id"]

    with _eval_lock:
        # Redirect SWE-bench logs to output_dir if provided
        original_log_dir = None
        if output_dir:
            original_log_dir = constants.RUN_EVALUATION_LOG_DIR
            constants.RUN_EVALUATION_LOG_DIR = Path(output_dir) / "swebench_logs"

        try:
            client = docker.from_env()
            # Strip trailing slash from namespace to avoid double slashes
            normalized_namespace = namespace.rstrip("/") if namespace else None
            test_spec = make_test_spec(instance, namespace=normalized_namespace)
            logger.info(f"Using image: {test_spec.instance_image_key}")

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
