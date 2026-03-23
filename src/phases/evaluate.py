# src/phases/evaluate.py
"""Phase 2: Evaluate patch using SWE-bench harness."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation import validate_patch
from loguru import logger
from data_io.writers import save_result


@dataclass
class EvaluateResult:
    """Result from evaluate phase."""

    instance_id: str
    iteration: int
    resolved: bool
    feedback: str
    metrics: Dict[str, Any]
    result_path: Optional[Path] = None


class EvaluatePhase:
    """
    Phase 2: Evaluate patch using SWE-bench Docker harness.

    This phase:
    1. Takes patch from predict phase
    2. Runs SWE-bench Docker evaluation
    3. Saves result to data/results/
    4. Returns resolved status and feedback
    """

    def __init__(
        self,
        use_docker: bool = True,
        timeout: int = 1800,
        rm_image: bool = True,
        output_dir: Optional[Path] = None,
        run_name: str = "default",
        benchmark: str = "swebench-lite",
    ):
        """
        Initialize evaluate phase.

        Args:
            use_docker: Use Docker harness (recommended)
            timeout: Evaluation timeout in seconds
            rm_image: Remove Docker image after evaluation (saves disk space)
            output_dir: Base directory for outputs
            run_name: Name of the experiment run
            benchmark: Benchmark name for output path
        """
        self.use_docker = use_docker
        self.timeout = timeout
        self.rm_image = rm_image
        self.output_dir = Path(output_dir) if output_dir else Path("data")
        self.run_name = run_name
        self.benchmark = benchmark

    def run(
        self,
        instance: Dict[str, Any],
        patch: str,
        iteration: int = 0,
    ) -> EvaluateResult:
        """
        Evaluate patch against SWE-bench test suite.

        Args:
            instance: SWE-bench instance dict
            patch: Generated patch from predict phase
            iteration: Current iteration number

        Returns:
            EvaluateResult with resolved status and feedback
        """
        instance_id = instance.get("instance_id", "unknown")
        logger.info(f"[Evaluate] Evaluating patch for {instance_id} (iter {iteration})")

        # Handle empty patch
        if not patch or not patch.strip():
            logger.warning(f"[Evaluate] Empty patch for {instance_id}")
            result = EvaluateResult(
                instance_id=instance_id,
                iteration=iteration,
                resolved=False,
                feedback="No patch submitted. Agent did not produce a valid patch.",
                metrics={"resolved": 0.0, "patch_empty": 1.0},
            )
            # Save result
            result.result_path = save_result(
                result={
                    "resolved": result.resolved,
                    "feedback": result.feedback,
                    "metrics": result.metrics,
                },
                run_dir=self.output_dir,
                benchmark=self.benchmark,
                instance_id=instance_id,
                iteration=iteration,
            )
            return result

        # Run evaluation
        try:
            resolved = validate_patch(
                instance=instance,
                patch=patch,
                use_docker=self.use_docker,
                timeout=self.timeout,
                rm_image=self.rm_image,
                output_dir=self.output_dir,
            )
        except Exception as e:
            logger.error(f"[Evaluate] Error evaluating {instance_id}: {e}")
            resolved = False

        # Build result
        if resolved:
            feedback = "Patch resolved all tests successfully!"
            logger.info(f"[Evaluate] {instance_id} RESOLVED!")
        else:
            feedback = "Patch did not resolve the issue. Tests failed or patch invalid."
            logger.info(f"[Evaluate] {instance_id} NOT resolved")

        result = EvaluateResult(
            instance_id=instance_id,
            iteration=iteration,
            resolved=resolved,
            feedback=feedback,
            metrics={
                "resolved": 1.0 if resolved else 0.0,
                "patch_length": len(patch),
            },
        )

        # Save result
        result.result_path = save_result(
            result={
                "resolved": result.resolved,
                "feedback": result.feedback,
                "metrics": result.metrics,
                "patch": patch[:1000] + "..." if len(patch) > 1000 else patch,
            },
            run_dir=self.output_dir,
            benchmark=self.benchmark,
            instance_id=instance_id,
            iteration=iteration,
        )

        return result


def run_evaluate(
    instance: Dict[str, Any],
    patch: str,
    output_dir: Path,
    run_name: str,
    benchmark: str = "swebench-lite",
    iteration: int = 0,
    use_docker: bool = True,
    timeout: int = 1800,
    rm_image: bool = True,
) -> EvaluateResult:
    """
    Convenience function to run evaluate phase.

    Args:
        instance: SWE-bench instance dict
        patch: Patch to evaluate
        output_dir: Output directory
        run_name: Run name
        benchmark: Benchmark name
        iteration: Iteration number
        use_docker: Use Docker evaluation
        timeout: Evaluation timeout
        rm_image: Remove Docker image after evaluation

    Returns:
        EvaluateResult
    """
    phase = EvaluatePhase(
        use_docker=use_docker,
        timeout=timeout,
        rm_image=rm_image,
        output_dir=output_dir,
        run_name=run_name,
        benchmark=benchmark,
    )
    return phase.run(instance=instance, patch=patch, iteration=iteration)
