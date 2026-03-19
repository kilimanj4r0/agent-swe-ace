# src/phases/learn.py
"""Phase 3: Update skillbook from failed attempts."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ace_next import Skillbook

from data_io.writers import save_skillbook

logger = logging.getLogger(__name__)


@dataclass
class LearnResult:
    """Result from learn phase."""

    instance_id: str
    iteration: int
    skills_added: int
    skills_updated: int
    skillbook_path: Optional[Path] = None


class LearnPhase:
    """
    Phase 3: Update skillbook from failures using ACE.

    This phase:
    1. Takes trajectory from failed attempt
    2. Uses ACE Reflector to analyze what went wrong
    3. Uses ACE SkillManager to create/update skills
    4. Saves updated skillbook
    5. Returns skillbook for next iteration
    """

    def __init__(
        self,
        reflector,  # ACE Reflector instance
        skill_manager,  # ACE SkillManager instance
        output_dir: Path,
        run_name: str = "default",
        benchmark: str = "swebench-lite",
        skillbook_mode: str = "per_instance",  # "per_instance" or "per_run"
    ):
        """
        Initialize learn phase.

        Args:
            reflector: ACE Reflector for analyzing trajectories
            skill_manager: ACE SkillManager for updating skills
            output_dir: Base directory for outputs
            run_name: Name of the experiment run
            benchmark: Benchmark name for output path
            skillbook_mode: "per_instance" or "per_run"
        """
        self.reflector = reflector
        self.skill_manager = skill_manager
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.benchmark = benchmark
        self.skillbook_mode = skillbook_mode

    def run(
        self,
        skillbook: Skillbook,
        instance: Dict[str, Any],
        trajectory: Dict,
        patch: str,
        iteration: int = 0,
        feedback: Optional[str] = None,
    ) -> LearnResult:
        """
        Learn from trajectory and update skillbook.

        Args:
            skillbook: Current skillbook to update
            instance: SWE-bench instance dict
            trajectory: Agent trajectory from predict phase
            patch: Generated patch (for context)
            iteration: Current iteration number
            feedback: Optional evaluation feedback

        Returns:
            LearnResult with updated skillbook info
        """
        instance_id = instance.get("instance_id", "unknown")
        logger.info(f"[Learn] Analyzing trajectory for {instance_id} (iter {iteration})")

        # Extract messages from trajectory
        messages = trajectory.get("messages", [])

        # Build agent output for ACE
        from ace.roles import AgentOutput
        agent_output = AgentOutput(
            reasoning="\n".join([m.get("content", "") for m in messages if m.get("role") == "assistant"]),
            final_answer=patch or "",
            skill_ids=[],
            raw={"trajectory_length": len(messages)}
        )

        # Get question (problem statement) from instance
        question = instance.get("problem_statement", "")

        # Reflect on trajectory
        try:
            reflection = self.reflector.reflect(
                question=question,
                agent_output=agent_output,
                skillbook=skillbook,
                feedback=feedback,
            )
            logger.debug(f"[Learn] Reflection: {reflection}")
        except Exception as e:
            logger.error(f"[Learn] Reflection failed: {e}")
            return LearnResult(
                instance_id=instance_id,
                iteration=iteration,
                skills_added=0,
                skills_updated=0,
            )

        # Update skills based on reflection
        try:
            # Build question context and progress for SkillManager
            question_context = f"Instance: {instance_id}\nRepo: {instance.get('repo', 'unknown')}\nProblem: {question[:500]}..."
            progress = f"Iteration {iteration}: Patch submitted but tests failed."

            update_result = self.skill_manager.update_skills(
                reflections=(reflection,),
                skillbook=skillbook,
                question_context=question_context,
                progress=progress,
            )

            # Apply the update batch to the skillbook (THIS WAS MISSING!)
            if hasattr(update_result, 'update') and update_result.update:
                skillbook.apply_update(update_result.update)

                # Count operations from the UpdateBatch
                operations = update_result.update.operations
                skills_added = sum(1 for op in operations if op.type.upper() == "ADD")
                skills_updated = sum(1 for op in operations if op.type.upper() == "UPDATE")
            else:
                skills_added = 0
                skills_updated = 0

            logger.info(f"[Learn] Added {skills_added} skills, updated {skills_updated} skills")
        except Exception as e:
            logger.error(f"[Learn] Skill update failed: {e}")
            skills_added = 0
            skills_updated = 0

        # Save skillbook
        skillbook_path = save_skillbook(
            skillbook=skillbook,
            run_dir=self.output_dir,
            benchmark=self.benchmark,
            iteration=iteration + 1,  # Save for next iteration
            instance_id=instance_id if self.skillbook_mode == "per_instance" else None,
        )

        return LearnResult(
            instance_id=instance_id,
            iteration=iteration,
            skills_added=skills_added,
            skills_updated=skills_updated,
            skillbook_path=skillbook_path,
        )


def run_learn(
    skillbook: Skillbook,
    instance: Dict[str, Any],
    trajectory: Dict,
    patch: str,
    reflector,
    skill_manager,
    output_dir: Path,
    run_name: str,
    benchmark: str = "swebench-lite",
    skillbook_mode: str = "per_instance",
    iteration: int = 0,
    feedback: Optional[str] = None,
) -> LearnResult:
    """
    Convenience function to run learn phase.

    Args:
        skillbook: Current skillbook
        instance: SWE-bench instance dict
        trajectory: Agent trajectory
        patch: Generated patch
        reflector: ACE Reflector instance
        skill_manager: ACE SkillManager instance
        output_dir: Output directory
        run_name: Run name
        benchmark: Benchmark name
        skillbook_mode: "per_instance" or "per_run"
        iteration: Iteration number
        feedback: Optional evaluation feedback

    Returns:
        LearnResult
    """
    phase = LearnPhase(
        reflector=reflector,
        skill_manager=skill_manager,
        output_dir=output_dir,
        run_name=run_name,
        benchmark=benchmark,
        skillbook_mode=skillbook_mode,
    )
    return phase.run(
        skillbook=skillbook,
        instance=instance,
        trajectory=trajectory,
        patch=patch,
        iteration=iteration,
        feedback=feedback,
    )
