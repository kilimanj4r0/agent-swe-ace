# src/runners/main_loop.py
"""Main experiment loop: Predict → Evaluate → Learn."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ace_next import Skillbook
from loguru import logger

from data_io.writers import save_statistics, save_config


@dataclass
class IterationResult:
    """Result from a single iteration."""

    iteration: int
    predict_result: Any
    evaluate_result: Any
    learn_result: Optional[Any] = None


@dataclass
class InstanceResult:
    """Result from all iterations for an instance."""

    instance_id: str
    iterations: List[IterationResult] = field(default_factory=list)
    final_resolved: bool = False
    total_attempts: int = 0


class ExperimentLoop:
    """
    Main experiment loop: Predict → Evaluate → Learn.

    For each instance:
    1. Predict: Run agent with current skillbook
    2. Evaluate: Test patch with SWE-bench
    3. Learn (if failed): Update skillbook

    Repeat until resolved or max_attempts reached.
    """

    def __init__(
        self,
        predict_phase,  # PredictPhase instance
        evaluate_phase,  # EvaluatePhase instance
        learn_phase,  # LearnPhase instance
        output_dir: Path,
        run_name: str = "default",
        max_attempts: int = 3,
        skillbook_mode: str = "per_instance",  # per_instance, per_repo, global
    ):
        """
        Initialize experiment loop.

        Args:
            predict_phase: Phase 1 runner
            evaluate_phase: Phase 2 runner
            learn_phase: Phase 3 runner
            output_dir: Output directory
            run_name: Name of this run
            max_attempts: Maximum attempts per instance
            skillbook_mode: How to manage skillbooks
        """
        self.predict = predict_phase
        self.evaluate = evaluate_phase
        self.learn = learn_phase
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.max_attempts = max_attempts
        self.skillbook_mode = skillbook_mode

        # Global skillbook for 'global' mode
        self.global_skillbook = Skillbook()
        # Per-repo skillbooks for 'per_repo' mode
        self.repo_skillbooks: Dict[str, Skillbook] = {}

    def get_skillbook(self, repo: str) -> Skillbook:
        """Get skillbook based on mode."""
        if self.skillbook_mode == "global":
            return self.global_skillbook
        elif self.skillbook_mode == "per_repo":
            if repo not in self.repo_skillbooks:
                self.repo_skillbooks[repo] = Skillbook()
            return self.repo_skillbooks[repo]
        else:  # per_instance
            return Skillbook()

    def update_skillbook(self, repo: str, skillbook: Skillbook):
        """Update skillbook based on mode."""
        if self.skillbook_mode == "global":
            self.global_skillbook = skillbook
        elif self.skillbook_mode == "per_repo":
            self.repo_skillbooks[repo] = skillbook
        # per_instance: skillbook is not persisted

    def run_instance(
        self,
        instance: Dict[str, Any],
        initial_skillbook: Optional[Skillbook] = None,
    ) -> InstanceResult:
        """
        Run experiment loop for a single instance.

        Args:
            instance: SWE-bench instance dict
            initial_skillbook: Optional starting skillbook

        Returns:
            InstanceResult with all iteration results
        """
        instance_id = instance.get("instance_id", "unknown")
        repo = instance.get("repo", "unknown")

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting instance: {instance_id}")
        logger.info(f"Repo: {repo}")
        logger.info(f"{'='*60}")

        # Get skillbook for this instance
        skillbook = initial_skillbook or self.get_skillbook(repo)

        result = InstanceResult(instance_id=instance_id)

        for iteration in range(self.max_attempts):
            logger.info(f"\n--- Iteration {iteration + 1}/{self.max_attempts} ---")

            # Phase 1: Predict
            predict_result = self.predict.run(
                instance=instance,
                skillbook=skillbook,
                iteration=iteration,
            )

            # Phase 2: Evaluate
            evaluate_result = self.evaluate.run(
                instance=instance,
                patch=predict_result.patch,
                iteration=iteration,
            )

            # Record iteration
            iter_result = IterationResult(
                iteration=iteration,
                predict_result=predict_result,
                evaluate_result=evaluate_result,
            )
            result.iterations.append(iter_result)
            result.total_attempts = iteration + 1

            # Check if resolved
            if evaluate_result.resolved:
                logger.info(f"[{instance_id}] RESOLVED at iteration {iteration + 1}!")
                result.final_resolved = True
                break

            # Phase 3: Learn (only if not resolved and not last attempt)
            if iteration < self.max_attempts - 1:
                logger.info(f"[{instance_id}] Not resolved, learning from failure...")

                trajectory = {
                    "info": {"exit_status": predict_result.exit_status},
                    "messages": predict_result.trajectory,
                }

                learn_result = self.learn.run(
                    skillbook=skillbook,
                    instance=instance,
                    trajectory=trajectory,
                    patch=predict_result.patch,
                    iteration=iteration,
                )
                iter_result.learn_result = learn_result

                # Update skillbook for next iteration
                self.update_skillbook(repo, skillbook)

        return result

    def run(
        self,
        instances: List[Dict[str, Any]],
        config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run experiment loop on multiple instances.

        Args:
            instances: List of SWE-bench instance dicts
            config: Optional config to save

        Returns:
            Summary dict with statistics
        """
        logger.info(f"\nStarting experiment: {self.run_name}")
        logger.info(f"Instances: {len(instances)}")
        logger.info(f"Max attempts: {self.max_attempts}")
        logger.info(f"Skillbook mode: {self.skillbook_mode}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        if config:
            save_config(config=config, run_dir=self.output_dir)

        # Track results
        all_results: Dict[str, InstanceResult] = {}
        resolved_ids: List[str] = []
        unresolved_ids: List[str] = []

        # Process each instance
        for i, instance in enumerate(instances):
            instance_id = instance.get("instance_id", f"unknown-{i}")
            logger.info(f"\n[{i+1}/{len(instances)}] Processing {instance_id}")

            result = self.run_instance(instance)
            all_results[instance_id] = result

            if result.final_resolved:
                resolved_ids.append(instance_id)
            else:
                unresolved_ids.append(instance_id)

        # Calculate statistics
        total = len(instances)
        resolved_count = len(resolved_ids)
        resolution_rate = resolved_count / total if total > 0 else 0.0

        statistics = {
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
            "total_instances": total,
            "resolved_count": resolved_count,
            "unresolved_count": len(unresolved_ids),
            "resolution_rate": resolution_rate,
            "resolved_ids": resolved_ids,
            "unresolved_ids": unresolved_ids,
            "config": {
                "max_attempts": self.max_attempts,
                "skillbook_mode": self.skillbook_mode,
            },
        }

        # Save statistics
        save_statistics(statistics=statistics, run_dir=self.output_dir)

        logger.info(f"\n{'='*60}")
        logger.info("Experiment Complete!")
        logger.info(f"Resolved: {resolved_count}/{total} ({resolution_rate:.1%})")
        logger.info(f"{'='*60}")

        return statistics


def run_experiment(
    instances: List[Dict[str, Any]],
    predict_phase,
    evaluate_phase,
    learn_phase,
    output_dir: Path,
    run_name: str = "default",
    max_attempts: int = 3,
    skillbook_mode: str = "per_instance",
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run experiment.

    Args:
        instances: List of SWE-bench instance dicts
        predict_phase: PredictPhase instance
        evaluate_phase: EvaluatePhase instance
        learn_phase: LearnPhase instance
        output_dir: Output directory
        run_name: Run name
        max_attempts: Max attempts per instance
        skillbook_mode: Skillbook management mode
        config: Optional config to save

    Returns:
        Summary dict with statistics
    """
    loop = ExperimentLoop(
        predict_phase=predict_phase,
        evaluate_phase=evaluate_phase,
        learn_phase=learn_phase,
        output_dir=output_dir,
        run_name=run_name,
        max_attempts=max_attempts,
        skillbook_mode=skillbook_mode,
    )
    return loop.run(instances=instances, config=config)
