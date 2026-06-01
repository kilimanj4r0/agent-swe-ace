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
    2. Optionally retrieves top-k relevant skills via SkillRetriever
    3. Runs mini-swe-agent with skillbook injected into prompt
    4. Saves trajectory to data/trajectories/
    5. Returns patch and trajectory for next phases
    """

    def __init__(
        self,
        agent,  # MiniSWEAgent instance
        output_dir: Path,
        run_name: str = "default",
        benchmark: str = "swebench-lite",
        model_name: Optional[str] = None,
        skill_retriever=None,  # Optional[SkillRetriever]
    ):
        """
        Initialize predict phase.

        Args:
            agent: MiniSWEAgent instance
            output_dir: Base directory for outputs
            run_name: Name of the experiment run
            benchmark: Benchmark name for output path
            model_name: Agent LLM model name (saved in trajectory metadata)
            skill_retriever: Optional SkillRetriever for top-k skill filtering.
                Applies on single-phase (phase=None) and val skillbook pass (phase="val").
        """
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.benchmark = benchmark
        self.model_name = model_name
        self.skill_retriever = skill_retriever

        # Retrieval stats accumulator for the run
        self._retrieval_run_stats = {
            "instances_retrieved": 0,
            "instances_skipped_threshold": 0,
            "total_before": 0,
            "total_after": 0,
        }

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

        # Apply skill retrieval if configured and phase matches
        retrieval_stats = None
        if self.skill_retriever and skillbook:
            # Retrieval applies on:
            #   - single-phase mode (phase=None): always
            #   - two-phase mode: only on "val" (val skillbook pass)
            # Skipped on "train" and "val_baseline" (empty skillbook there anyway)
            phase_matches = phase is None or phase == "val"
            if phase_matches:
                n_skills = len(skillbook.skills())
                if n_skills <= self.skill_retriever.skip_threshold:
                    self._retrieval_run_stats["instances_skipped_threshold"] += 1
                else:
                    skillbook, retrieval_stats = self._retrieve_skills(skillbook, instance)

        # Run agent
        result = self.agent.run(
            problem=instance.get("problem_statement", ""),
            instance=instance,
            skillbook=skillbook,
        )

        # Build trajectory dict
        info = {
            "exit_status": result.exit_status,
            "submission": result.patch,
            "iteration": iteration,
            "instance_id": instance_id,
            "model": self.model_name,
            "message_count": len(result.trajectory),
            "assistant_message_count": sum(
                1 for m in result.trajectory if m.get("role") == "assistant"
            ),
        }
        if retrieval_stats:
            info["retrieval_stats"] = retrieval_stats
        trajectory = {"info": info, "messages": result.trajectory}

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

    def _retrieve_skills(
        self, skillbook: Skillbook, instance: Dict[str, Any]
    ) -> tuple:
        """Filter skillbook to top-k relevant skills via SkillRetriever.

        Returns:
            (filtered_skillbook, retrieval_stats_dict) tuple.
            If retrieval fails or returns all skills, returns original skillbook.
        """
        original_count = len(skillbook.skills())
        selected = self.skill_retriever.retrieve(skillbook, instance)
        selected_count = len(selected)

        if selected_count == original_count:
            # Nothing filtered out
            return skillbook, None

        # Track selected skill IDs
        selected_ids = [s.id for s in selected]
        all_ids = [s.id for s in skillbook.skills()]
        dropped_ids = [sid for sid in all_ids if sid not in selected_ids]

        # Build filtered skillbook with only selected skills
        filtered_sb = Skillbook()
        for skill in selected:
            filtered_sb.add_skill(
                section=skill.section,
                content=skill.content,
                justification=getattr(skill, "justification", None),
                evidence=getattr(skill, "evidence", None),
            )

        logger.info(
            f"[Predict] Skill retrieval: {original_count} → {len(filtered_sb.skills())} skills "
            f"(selected: {selected_ids})"
        )

        # Accumulate run-level stats
        self._retrieval_run_stats["instances_retrieved"] += 1
        self._retrieval_run_stats["total_before"] += original_count
        self._retrieval_run_stats["total_after"] += len(filtered_sb.skills())

        stats = {
            "total": original_count,
            "selected": len(filtered_sb.skills()),
            "selected_ids": selected_ids,
            "dropped_ids": dropped_ids,
        }
        return filtered_sb, stats

    def get_retrieval_summary(self) -> Optional[Dict[str, Any]]:
        """Get retrieval stats accumulated over the run, or None if retrieval disabled."""
        if not self.skill_retriever:
            return None
        rs = self._retrieval_run_stats
        n = rs["instances_retrieved"]
        summary = {
            "enabled": True,
            "model": self.skill_retriever.model,
            "top_k": self.skill_retriever.top_k,
            "skip_threshold": self.skill_retriever.skip_threshold,
            "filter_target": self.skill_retriever.filter_target,
            "chunk_size": self.skill_retriever.chunk_size,
            "instances_retrieved": n,
            "instances_skipped_threshold": rs["instances_skipped_threshold"],
            "avg_skills_before": round(rs["total_before"] / n, 1) if n > 0 else 0,
            "avg_skills_after": round(rs["total_after"] / n, 1) if n > 0 else 0,
        }
        return summary


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


def _escape_jinja(text: str) -> str:
    """Wrap text containing Jinja2 delimiters in {% raw %} blocks."""
    if '{%' in text or '{{' in text or '{#' in text:
        return '{% raw %}' + text + '{% endraw %}'
    return text


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
        section = f"### {_escape_jinja(skill.id)}\n\n{_escape_jinja(skill.content)}"
        if getattr(skill, "justification", None):
            section += f"\n\n**Why this helps:** {_escape_jinja(skill.justification)}"
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
