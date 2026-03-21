"""SWEBenchEnvironment - TaskEnvironment for SWE-bench evaluation."""

from typing import Optional

from ace_next import TaskEnvironment, EnvironmentResult, Sample
from ace_next.core.outputs import AgentOutput
from loguru import logger


class SWEBenchEnvironment(TaskEnvironment):
    """
    Evaluates patches using SWE-bench harness.

    Implements ace_next's TaskEnvironment interface for use with ACE.
    This allows SWE-bench evaluation to be used within ace-framework's
    orchestration.

    Usage:
        >>> from ace_next import ACE
        >>> from environments.swebench_environment import SWEBenchEnvironment
        >>>
        >>> environment = SWEBenchEnvironment(use_docker=True)
        >>> ace = ACE.from_roles(agent=agent, ..., environment=environment)
        >>> results = ace.run(samples)

    Example Implementation (for reference):
        >>> class MathEnvironment(TaskEnvironment):
        ...     def evaluate(self, sample, agent_output):
        ...         predicted = extract_number(agent_output.final_answer)
        ...         correct = str(predicted) == sample.ground_truth
        ...         return EnvironmentResult(
        ...             feedback="Correct!" if correct else f"Incorrect. Expected {sample.ground_truth}",
        ...             ground_truth=sample.ground_truth,
        ...             metrics={'accuracy': 1.0 if correct else 0.0}
        ...         )
    """

    def __init__(
        self,
        use_docker: bool = True,
        timeout: int = 1800,
    ):
        """
        Initialize the environment.

        Args:
            use_docker: If True, use full Docker harness for evaluation.
                       If False, use simple heuristic validation.
            timeout: Timeout in seconds for Docker evaluation (default 30 min).
        """
        self.use_docker = use_docker
        self.timeout = timeout

    def evaluate(
        self,
        sample: Sample,
        agent_output: AgentOutput
    ) -> EnvironmentResult:
        """
        Validate patch against SWE-bench test suite.

        Args:
            sample: Sample with instance metadata containing the SWE-bench instance
            agent_output: Agent's output with final_answer containing the patch

        Returns:
            EnvironmentResult with:
            - feedback: Description of evaluation result
            - ground_truth: The expected solution (if available)
            - metrics: Dict with 'resolved' (0.0 or 1.0)
        """
        from evaluation import validate_patch

        # Get instance from metadata
        instance = sample.metadata.get('instance')
        patch = agent_output.final_answer

        # Handle missing patch
        if not patch or not patch.strip():
            logger.warning(f"No patch submitted for {sample.metadata.get('instance_id', 'unknown')}")
            return EnvironmentResult(
                feedback="No patch submitted. The agent did not produce a valid patch.",
                ground_truth=sample.ground_truth,
                metrics={"resolved": 0.0, "patch_empty": 1.0}
            )

        # Handle missing instance
        if not instance:
            logger.error("No instance in sample metadata")
            return EnvironmentResult(
                feedback="Cannot evaluate: missing instance metadata",
                ground_truth=sample.ground_truth,
                metrics={"resolved": 0.0, "error": 1.0}
            )

        # Validate patch
        instance_id = instance.get('instance_id', 'unknown')
        logger.info(f"Evaluating patch for {instance_id} (docker={self.use_docker})")

        try:
            resolved = validate_patch(
                instance,
                patch,
                use_docker=self.use_docker,
                timeout=self.timeout
            )
        except Exception as e:
            logger.error(f"Evaluation error for {instance_id}: {e}")
            return EnvironmentResult(
                feedback=f"Evaluation error: {str(e)}",
                ground_truth=sample.ground_truth,
                metrics={"resolved": 0.0, "error": 1.0}
            )

        # Build result
        if resolved:
            feedback = "Patch resolved all tests! The issue has been successfully fixed."
            logger.info(f"Patch for {instance_id} RESOLVED")
        else:
            feedback = "Patch did not resolve the issue. Some tests failed or the patch format was invalid."
            logger.info(f"Patch for {instance_id} NOT resolved")

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=sample.ground_truth,
            metrics={
                "resolved": 1.0 if resolved else 0.0,
                "patch_length": len(patch),
            }
        )
