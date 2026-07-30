# src/phases/learn.py
"""Phase 3: Update skillbook from failed attempts."""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ace import Skillbook
from ace.deduplication import DeduplicationManager
from ace.protocols.deduplication import DeduplicationConfig
from loguru import logger

from data_io.writers import save_skillbook

# Module-level shared SentenceTransformer model and lock.
# Concurrent threads (iterate_repos) must share one model to avoid
# "Cannot copy out of meta tensor" errors from simultaneous model loading.
_shared_model = None
_shared_model_name = None
_shared_model_lock = threading.Lock()


def _get_shared_st_model(model_name: str, device: str):
    """Get or create the shared SentenceTransformer model (thread-safe)."""
    global _shared_model, _shared_model_name
    if _shared_model is not None and _shared_model_name == model_name:
        return _shared_model
    with _shared_model_lock:
        if _shared_model is not None and _shared_model_name == model_name:
            return _shared_model
        from sentence_transformers import SentenceTransformer
        _shared_model = SentenceTransformer(model_name, device=device)
        _shared_model_name = model_name
    return _shared_model


@dataclass
class LearnResult:
    """Result from learn phase."""

    instance_id: str
    iteration: int
    skills_added: int
    skills_updated: int
    skills_removed: int = 0
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
        skillbook_mode: str = "per_instance",  # "per_instance", "per_repo", or "global"
        dedup_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize learn phase.

        Args:
            reflector: ACE Reflector for analyzing trajectories
            skill_manager: ACE SkillManager for updating skills
            output_dir: Base directory for outputs
            run_name: Name of the experiment run
            benchmark: Benchmark name for output path
            skillbook_mode: "per_instance", "per_repo", or "global"
            dedup_config: Optional deduplication config dict
        """
        self.reflector = reflector
        self.skill_manager = skill_manager
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.benchmark = benchmark
        self.skillbook_mode = skillbook_mode

        # Treat ``enabled`` as the authoritative switch and never mutate the
        # resolved experiment config shared by other phases/repo workers.
        raw_dedup_config = dict(dedup_config or {})
        dedup_enabled = bool(raw_dedup_config.pop("enabled", False))

        if dedup_enabled:
            embedding_device = raw_dedup_config.pop("embedding_device", "cpu")
            cfg = DeduplicationConfig(**raw_dedup_config)
            self.dedup_manager = DeduplicationManager(cfg)
            _detector = self.dedup_manager.detector
            model_name = cfg.local_model_name

            # Eagerly load the shared model on the correct device, then inject
            # it into the detector. This avoids concurrent SentenceTransformer
            # loading which causes "Cannot copy out of meta tensor" errors.
            model = _get_shared_st_model(model_name, embedding_device)
            with _detector._model_lock:
                _detector._model = model

            logger.info(f"[Learn] Deduplication enabled with threshold {cfg.similarity_threshold} (device={embedding_device})")
        else:
            self.dedup_manager = None

    def _consolidate(self, skillbook: Skillbook) -> int:
        """Detect similar pairs and apply deterministic consolidation operations."""
        if not self.dedup_manager:
            return 0
        self.dedup_manager.detector.ensure_embeddings(skillbook)
        pairs = self.dedup_manager.detector.detect_similar_pairs(skillbook)
        if not pairs:
            return 0
        logger.info(f"[Learn] Found {len(pairs)} similar pairs, consolidating")

        from ace.deduplication.operations import DeleteOp, KeepOp, MergeOp
        ops = []
        for skill_a, skill_b, similarity in pairs:
            score_a = skill_a.helpful - skill_a.harmful
            score_b = skill_b.helpful - skill_b.harmful
            a_validated = skill_a.helpful > 0 or skill_a.harmful > 0
            b_validated = skill_b.helpful > 0 or skill_b.harmful > 0

            keep, remove = (skill_a, skill_b) if score_a >= score_b else (skill_b, skill_a)

            if a_validated and b_validated:
                delta = abs(score_a - score_b)
                if delta > 1:
                    ops.append(MergeOp(
                        source_ids=[keep.id, remove.id],
                        keep_id=keep.id,
                        merged_content=keep.content,
                        reasoning=f"Merged ({similarity:.0%} similar), kept {keep.id} (score {score_a} vs {score_b})",
                    ))
                else:
                    ops.append(KeepOp(
                        skill_ids=[skill_a.id, skill_b.id],
                        differentiation=f"Similar scores ({score_a} vs {score_b})",
                        reasoning=f"Both validated, scores within margin ({similarity:.0%} similar)",
                    ))
            else:
                ops.append(DeleteOp(
                    skill_id=remove.id,
                    reasoning=f"Duplicate of {keep.id} ({similarity:.0%} similar), unvalidated",
                ))

        self.dedup_manager.apply_operations(ops, skillbook)
        logger.info(f"[Learn] Consolidation: applied {len(ops)} operations")
        return len(ops)

    def run(
        self,
        skillbook: Skillbook,
        instance: Dict[str, Any],
        trajectory: Dict,
        patch: str,
        iteration: int = 0,
        feedback: Optional[str] = None,
        ground_truth: Optional[str] = None,
        phase: Optional[str] = None,
        resolved: bool = False,
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
            ground_truth: Optional ground truth (test lists)
            phase: Phase identifier (train/val_baseline/val)
            resolved: Whether the patch resolved the issue

        Returns:
            LearnResult with updated skillbook info
        """
        instance_id = instance.get("instance_id", "unknown")
        logger.info(f"[Learn] Analyzing trajectory for {instance_id} (iter {iteration})")

        # Extract messages from trajectory
        messages = trajectory.get("messages", [])

        # Build agent output for ACE
        from ace.core.outputs import AgentOutput
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
            if feedback is None:
                feedback = (
                    "All tests passed. Patch resolved the issue successfully."
                    if resolved
                    else "Patch did not resolve the issue. Tests failed."
                )

            logger.debug(
                f"[Learn] Reflector inputs for {instance_id} (iter {iteration}): "
                f"resolved={resolved}, feedback={'present' if feedback else 'None'}, "
                f"ground_truth={ground_truth[:200] if ground_truth else 'None'}..."
            )

            reflection = self.reflector.reflect(
                question=question,
                agent_output=agent_output,
                skillbook=skillbook,
                feedback=feedback,
                ground_truth=ground_truth,
                resolved=resolved,
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
            if resolved:
                progress = f"Iteration {iteration}: Patch resolved the issue successfully."
            else:
                progress = f"Iteration {iteration}: Patch submitted but tests failed."

            update_result = self.skill_manager.update_skills(
                reflections=(reflection,),
                skillbook=skillbook,
                question_context=question_context,
                progress=progress,
            )

            # Apply the update batch to the skillbook (THIS WAS MISSING!)
            if hasattr(update_result, 'update') and update_result.update:
                # Stamp provenance metadata onto each operation so skills
                # track which instance/repo they originated from.
                provenance = {
                    "instance_id": instance_id,
                    "repo": instance.get("repo", "unknown"),
                }
                for op in update_result.update.operations:
                    if op.type.upper() in ("ADD", "UPDATE"):
                        op.insight_source = provenance

                skillbook.apply_update(update_result.update)

                # Count operations from the UpdateBatch
                operations = update_result.update.operations
                skills_added = sum(1 for op in operations if op.type.upper() == "ADD")
                skills_updated = sum(1 for op in operations if op.type.upper() == "UPDATE")
            else:
                skills_added = 0
                skills_updated = 0

            logger.info(f"[Learn] Added {skills_added} skills, updated {skills_updated} skills")

            # Run deduplication if enabled
            skills_removed = self._consolidate(skillbook)
        except Exception as e:
            logger.error(f"[Learn] Skill update failed: {e}")
            skills_added = 0
            skills_updated = 0
            skills_removed = 0

        # Save skillbook
        skillbook_path = save_skillbook(
            skillbook=skillbook,
            run_dir=self.output_dir,
            benchmark=self.benchmark,
            iteration=iteration + 1,  # Save for next iteration
            instance_id=instance_id if self.skillbook_mode == "per_instance" else None,
            phase=phase if self.skillbook_mode != "per_instance" else None,
        )

        return LearnResult(
            instance_id=instance_id,
            iteration=iteration,
            skills_added=skills_added,
            skills_updated=skills_updated,
            skills_removed=skills_removed,
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
        skillbook_mode: "per_instance", "per_repo", or "global"
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
