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
from utils.llm_observer import get_project_url, is_enabled as is_observability_enabled


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


@dataclass
class BaselineIter0Data:
    """Container for baseline iter_0 data."""

    trajectory: Dict[str, Any]
    result: Dict[str, Any]
    patch: str
    resolved: bool


class ExperimentLoop:
    """
    Main experiment loop: Predict → Evaluate → Learn.

    For each instance:
    1. Predict: Run agent with current skillbook
    2. Evaluate: Test patch with SWE-bench
    3. Learn (if failed): Update skillbook

    Repeat until resolved or max_attempts reached.

    When baseline_dir is provided, iter_0 predict/evaluate are skipped
    and existing baseline data is loaded instead.
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
        baseline_dir: Optional[Path] = None,
        benchmark: str = "princeton-nlp__SWE-bench_Lite",
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
            baseline_dir: Optional path to baseline run with existing iter_0 data
            benchmark: Benchmark name for finding baseline files
        """
        self.predict = predict_phase
        self.evaluate = evaluate_phase
        self.learn = learn_phase
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.max_attempts = max_attempts
        self.skillbook_mode = skillbook_mode
        self.baseline_dir = Path(baseline_dir) if baseline_dir else None
        self.benchmark = benchmark

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

    def load_baseline_iter0(self, instance_id: str) -> Optional[BaselineIter0Data]:
        """
        Load baseline iter_0 data for an instance.

        Args:
            instance_id: Instance ID to load data for

        Returns:
            BaselineIter0Data if found, None otherwise
        """
        if not self.baseline_dir:
            return None

        baseline_dir = self.baseline_dir / self.benchmark

        # Try to load trajectory
        traj_path = baseline_dir / "trajectories" / instance_id / "iter_0.json"
        if not traj_path.exists():
            # Try old format with .traj.json
            traj_path = baseline_dir / "trajectories" / instance_id / f"{instance_id}.traj.json"
            if not traj_path.exists():
                logger.debug(f"No baseline trajectory found for {instance_id}")
                return None

        # Try to load result
        result_path = baseline_dir / "results" / instance_id / "iter_0.json"
        if not result_path.exists():
            logger.debug(f"No baseline result found for {instance_id}")
            return None

        try:
            with open(traj_path) as f:
                trajectory = json.load(f)

            with open(result_path) as f:
                result = json.load(f)

            # Extract patch from trajectory or result
            patch = result.get("patch", "")
            if not patch:
                patch = trajectory.get("info", {}).get("submission", "")

            resolved = result.get("resolved", False)

            logger.info(f"[{instance_id}] Loaded baseline iter_0 data (resolved={resolved})")

            return BaselineIter0Data(
                trajectory=trajectory,
                result=result,
                patch=patch,
                resolved=resolved,
            )

        except Exception as e:
            logger.warning(f"Failed to load baseline data for {instance_id}: {e}")
            return None

    def copy_baseline_iter0_to_output(self, instance_id: str, baseline_data: BaselineIter0Data) -> None:
        """
        Copy baseline iter_0 files to output directory.

        Args:
            instance_id: Instance ID
            baseline_data: Baseline data to copy
        """
        import shutil

        # Copy trajectory
        src_traj = self.baseline_dir / self.benchmark / "trajectories" / instance_id / "iter_0.json"
        dst_traj = self.output_dir / self.benchmark / "trajectories" / instance_id / "iter_0.json"
        if src_traj.exists():
            dst_traj.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_traj, dst_traj)
            logger.debug(f"Copied baseline trajectory to {dst_traj}")

        # Copy result
        src_result = self.baseline_dir / self.benchmark / "results" / instance_id / "iter_0.json"
        dst_result = self.output_dir / self.benchmark / "results" / instance_id / "iter_0.json"
        if src_result.exists():
            dst_result.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_result, dst_result)
            logger.debug(f"Copied baseline result to {dst_result}")

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

        # Check for baseline iter_0 data
        baseline_data = self.load_baseline_iter0(instance_id)

        start_iteration = 0
        if baseline_data:
            # Copy baseline data to output directory
            self.copy_baseline_iter0_to_output(instance_id, baseline_data)

            # Create IterationResult from baseline data
            from dataclasses import dataclass as create_dataclass

            @create_dataclass
            class MockPredictResult:
                exit_status: str
                patch: str
                trajectory: list

            @create_dataclass
            class MockEvaluateResult:
                resolved: bool
                feedback: str
                result_path: str

            mock_predict = MockPredictResult(
                exit_status=baseline_data.trajectory.get("info", {}).get("exit_status", "Submitted"),
                patch=baseline_data.patch,
                trajectory=baseline_data.trajectory.get("messages", []),
            )

            mock_evaluate = MockEvaluateResult(
                resolved=baseline_data.resolved,
                feedback=baseline_data.result.get("feedback", "Baseline run"),
                result_path=str(self.output_dir / self.benchmark / "results" / instance_id / "iter_0.json"),
            )

            iter_result = IterationResult(
                iteration=0,
                predict_result=mock_predict,
                evaluate_result=mock_evaluate,
            )
            result.iterations.append(iter_result)
            result.total_attempts = 1

            if baseline_data.resolved:
                logger.info(f"[{instance_id}] RESOLVED at baseline iter_0!")
                result.final_resolved = True
                return result

            # Run learn phase for unresolved baseline (always, to update skillbook)
            logger.info(f"[{instance_id}] Baseline iter_0 not resolved, learning from failure...")

            learn_result = self.learn.run(
                skillbook=skillbook,
                instance=instance,
                trajectory=baseline_data.trajectory,
                patch=baseline_data.patch,
                iteration=0,
            )
            iter_result.learn_result = learn_result
            self.update_skillbook(repo, skillbook)

            # Continue from iter_1
            start_iteration = 1

        for iteration in range(start_iteration, self.max_attempts):
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
        if self.baseline_dir:
            logger.info(f"Baseline dir: {self.baseline_dir} (loading existing iter_0)")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        if config:
            save_config(config=config, run_dir=self.output_dir)

        # Track results
        all_results: Dict[str, InstanceResult] = {}
        resolved_ids: List[str] = []
        unresolved_ids: List[str] = []
        baseline_resolved_ids: List[str] = []
        baseline_unresolved_ids: List[str] = []

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

        # Get observability project URL if enabled
        observability_project_url = None
        if is_observability_enabled():
            observability_project_url = get_project_url()

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

        # Add baseline info if using baseline mode
        if self.baseline_dir:
            statistics["baseline_dir"] = str(self.baseline_dir)

        # Add observability project URL if available
        if observability_project_url:
            statistics["observability_project_url"] = observability_project_url

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
