# src/phases/predict.py
"""Phase 1: Run mini-swe-agent with skillbook injection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ace import Skillbook
from loguru import logger

from data_io.writers import save_trajectory


@dataclass
class PredictResult:
    """Result from predict phase."""

    instance_id: str
    iteration: int
    exit_status: str
    patch: str
    trajectory: list
    error: Optional[str] = None
    trajectory_path: Optional[Path] = None


class PredictPhase:
    """
    Phase 1: Run agent to generate patch with skillbook.

    This phase:
    1. Takes a SWE-bench instance and optional skillbook
    2. Runs mini-swe-agent with skillbook injected into prompt
    3. Saves trajectory to data/trajectories/
    4. Returns patch and trajectory for next phases
    """

    def __init__(
        self,
        agent,  # MiniSWEAgent instance
        output_dir: Path,
        run_name: str = "default",
        benchmark: str = "swebench-lite",
        model_name: Optional[str] = None,
    ):
        """
        Initialize predict phase.

        Args:
            agent: MiniSWEAgent instance
            output_dir: Base directory for outputs
            run_name: Name of the experiment run
            benchmark: Benchmark name for output path
            model_name: Agent LLM model name (saved in trajectory metadata)
        """
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.benchmark = benchmark
        self.model_name = model_name

    def run(
        self,
        instance: Dict[str, Any],
        skillbook: Optional[Skillbook],
        iteration: int = 0,
        phase: Optional[str] = None,
    ) -> PredictResult:
        """
        Run agent on instance with skillbook.

        Args:
            instance: SWE-bench instance dict
            skillbook: Optional skillbook for prompt injection
            iteration: Current iteration number

        Returns:
            PredictResult with patch, trajectory, and metadata
        """
        instance_id = instance.get("instance_id", "unknown")
        logger.info(f"[Predict] Running agent for {instance_id} (iter {iteration})")

        # Run agent
        result = self.agent.run(
            problem=instance.get("problem_statement", ""),
            instance=instance,
            skillbook=skillbook,
        )

        # Build trajectory dict
        trajectory = {
            "info": {
                "exit_status": result.exit_status,
                "submission": result.patch,
                "iteration": iteration,
                "instance_id": instance_id,
                "model": self.model_name,
                "message_count": len(result.trajectory),
                "assistant_message_count": sum(
                    1 for m in result.trajectory if m.get("role") == "assistant"
                ),
            },
            "messages": result.trajectory,
        }

        # Save trajectory
        trajectory_path = save_trajectory(
            trajectory=trajectory,
            run_dir=self.output_dir,
            benchmark=self.benchmark,
            instance_id=instance_id,
            iteration=iteration,
            phase=phase,
        )

        logger.info(
            f"[Predict] Agent finished: {result.exit_status}, "
            f"patch={len(result.patch)} chars, "
            f"traj={len(result.trajectory)} messages"
        )

        return PredictResult(
            instance_id=instance_id,
            iteration=iteration,
            exit_status=result.exit_status,
            patch=result.patch,
            trajectory=result.trajectory,
            error=result.error,
            trajectory_path=trajectory_path,
        )


def run_predict(
    instance: Dict[str, Any],
    skillbook: Optional[Skillbook],
    agent,
    output_dir: Path,
    run_name: str,
    benchmark: str = "swebench-lite",
    iteration: int = 0,
) -> PredictResult:
    """
    Convenience function to run predict phase.

    Args:
        instance: SWE-bench instance dict
        skillbook: Optional skillbook
        agent: MiniSWEAgent instance
        output_dir: Output directory
        run_name: Run name
        benchmark: Benchmark name
        iteration: Iteration number

    Returns:
        PredictResult
    """
    phase = PredictPhase(agent=agent, output_dir=output_dir, run_name=run_name, benchmark=benchmark)
    return phase.run(instance=instance, skillbook=skillbook, iteration=iteration)


# --- Helper functions for skillbook injection ---

_MINI_SWE_CONFIG = None


def _load_mini_swe_config() -> dict:
    """Load mini-swe-agent's swebench config from YAML file (cached)."""
    global _MINI_SWE_CONFIG
    if _MINI_SWE_CONFIG is not None:
        return _MINI_SWE_CONFIG

    import yaml
    from pathlib import Path

    # Try to find the swebench config in minisweagent package
    try:
        import minisweagent
        package_dir = Path(minisweagent.__file__).parent
        # Use swebench.yaml for proper git diff submission
        config_path = package_dir / "config" / "extra" / "swebench.yaml"
        if config_path.exists():
            with open(config_path) as f:
                _MINI_SWE_CONFIG = yaml.safe_load(f)
                logger.debug(f"Loaded mini-swe-agent config from: {config_path}")
                return _MINI_SWE_CONFIG
    except Exception as e:
        logger.debug(f"Could not load minisweagent swebench config: {e}")


def build_system_template() -> str:
    """Get mini-swe-agent's default system template."""
    config = _load_mini_swe_config()
    return config["agent"]["system_template"]


def wrap_skillbook_context(skillbook: Skillbook) -> str:
    """
    Format skillbook skills as context for the agent.

    Args:
        skillbook: Skillbook to format

    Returns:
        Formatted skillbook context string
    """
    skills = skillbook.skills()
    if not skills:
        return ""

    sections = []
    for skill in skills:
        section = f"### {skill.id}\n\n{skill.content}"
        if getattr(skill, "justification", None):
            section += f"\n\n**Why this helps:** {skill.justification}"
        sections.append(section)

    return "\n\n".join(sections)


def build_action_observation_template() -> str:
    """Get mini-swe-agent's action observation template with output truncation."""
    config = _load_mini_swe_config()
    return config["agent"]["action_observation_template"]


def build_instance_template(skillbook: Optional[Skillbook] = None) -> str:
    """
    Build instance template with skillbook context injected.

    Loads mini-swe-agent's default instance template and injects
    the skillbook context with defensive checks.

    Args:
        skillbook: Skillbook to inject into template (optional)

    Returns:
        Instance template string with skillbook section if valid skills exist
    """
    config = _load_mini_swe_config()
    default_template = config["agent"]["instance_template"]

    # Early return for None/empty skillbook
    if not skillbook:
        logger.debug("No skillbook provided - using default template")
        return default_template

    skills = skillbook.skills()
    if not skills:
        logger.debug("Skillbook has no skills - using default template")
        return default_template

    # Get formatted context
    skillbook_context = wrap_skillbook_context(skillbook)

    # Guard against empty context even when skills exist
    # (prevents injecting section with no content)
    if not skillbook_context or not skillbook_context.strip():
        logger.warning(
            f"Skillbook has {len(skills)} skill(s) but context is empty - "
            "using default template"
        )
        return default_template

    skillbook_section = f"""

## Learned Strategies (Skillbook)

These are strategies learned from previous attempts. Use them to guide your approach:

{skillbook_context}

⚠️ **CRITICAL REMINDERS:**
1. These skills describe approaches, NOT complete solutions. You MUST implement actual code changes.
2. Do NOT put multiple bash commands in one response - use ONE command per response.
3. Before submitting, verify your patch exists with: `git diff --cached`
4. If you think you've "successfully implemented" but git diff is empty, you haven't actually edited any files.
5. The source code in /testbed IS writable - do not claim you "cannot modify installed packages".

When you apply a strategy successfully, reference it with [skill-id] notation in your reasoning."""

    # Inject before <example_response> if it exists
    if "<example_response>" in default_template:
        parts = default_template.split("<example_response>", 1)
        logger.debug(f"Injected skillbook with {len(skills)} skill(s) before <example_response>")
        return parts[0] + skillbook_section + "\n\n<example_response>" + parts[1]

    # Fallback: append at end (log warning since this may not be ideal placement)
    logger.warning(
        "<example_response> tag not found in template - "
        "appending skillbook at end"
    )
    return default_template + skillbook_section
