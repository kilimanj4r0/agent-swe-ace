# src/runners/main_loop.py
"""Main experiment loop: Predict → Evaluate → Learn."""

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ace import Skillbook
from loguru import logger

from data_io.resume_scanner import ResumePoint, copy_instance_artifacts
from data_io.writers import save_statistics, save_config
from utils.llm_observer import get_project_url, is_enabled as is_observability_enabled
from utils.logging import instance_context


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

    When resume_state is provided, completed instances are copied from
    previous runs and partial instances resume from the last successful
    iteration.
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
        resume_state: Optional[Dict[str, ResumePoint]] = None,
        benchmark: str = "princeton-nlp__SWE-bench_Lite",
        concurrency: int = 1,
        agent_factory: Optional[Callable] = None,
    ):
        """
        Initialize experiment loop.

        Args:
            predict_phase: Phase 1 runner (used in sequential mode)
            evaluate_phase: Phase 2 runner
            learn_phase: Phase 3 runner
            output_dir: Output directory
            run_name: Name of this run
            max_attempts: Maximum attempts per instance
            skillbook_mode: How to manage skillbooks
            resume_state: Dict mapping instance_id -> ResumePoint for resuming
            benchmark: Benchmark name for finding files
            concurrency: Number of parallel instances (1 = sequential)
            agent_factory: Callable that returns a new MiniSWEAgent (for concurrent mode)
        """
        self.predict = predict_phase
        self.evaluate = evaluate_phase
        self.learn = learn_phase
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.max_attempts = max_attempts
        self.skillbook_mode = skillbook_mode
        self.resume_state = resume_state or {}
        self.benchmark = benchmark
        self.concurrency = concurrency
        self.agent_factory = agent_factory

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

    def _get_resume_start(self, instance_id: str) -> int:
        """Get the start iteration for an instance based on resume_state.

        Returns:
            -1 if instance is fully complete (should be skipped)
            0 or higher for the iteration to start from
        """
        rp = self.resume_state.get(instance_id)
        if rp is None:
            return 0
        if rp.is_fully_complete:
            return -1
        return rp.start_iteration

    def _copy_resume_artifacts(self, instance_id: str) -> None:
        """Copy artifacts from resume dir to output dir for a partial instance."""
        rp = self.resume_state.get(instance_id)
        if rp is None or rp.last_complete_iter < 0:
            return
        copy_instance_artifacts(
            source_dir=rp.resume_dir,
            dest_dir=self.output_dir,
            benchmark=self.benchmark,
            instance_id=instance_id,
            up_to_iter=rp.last_complete_iter,
        )
        logger.info(
            f"[{instance_id}] Resumed from iter_0..iter_{rp.last_complete_iter} "
            f"(source: {rp.resume_dir.name})"
        )

    def _load_resolved_status(self, instance_id: str) -> Optional[bool]:
        """Read resolved status from the last completed iteration's result file."""
        rp = self.resume_state.get(instance_id)
        if rp is None:
            return None
        results_dir = rp.resume_dir / self.benchmark / "results" / instance_id
        iter_files = sorted(results_dir.glob("iter_*.json"))
        # Find the result for last_complete_iter
        target = results_dir / f"iter_{rp.last_complete_iter}.json"
        if target.exists():
            try:
                with open(target) as f:
                    return json.load(f).get("resolved", False)
            except Exception:
                pass
        return None

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

        with instance_context(instance_id):
            return self._run_instance_inner(instance, instance_id, repo, initial_skillbook)

    def _run_instance_inner(self, instance, instance_id, repo, initial_skillbook=None):
        """Inner implementation with instance context set."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting instance: {instance_id}")
        logger.info(f"Repo: {repo}")
        logger.info(f"{'='*60}")

        # Get skillbook for this instance
        skillbook = initial_skillbook or self.get_skillbook(repo)

        result = InstanceResult(instance_id=instance_id)

        # Check resume state
        start_iteration = self._get_resume_start(instance_id)

        if start_iteration == -1:
            # Fully complete — should not reach here (filtered in run()),
            # but handle gracefully
            logger.info(f"[{instance_id}] Already complete, skipping")
            result.final_resolved = self._load_resolved_status(instance_id) or False
            return result

        if start_iteration > 0:
            # Partial resume — copy existing artifacts
            self._copy_resume_artifacts(instance_id)

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

            # Phase 3: Learn (always run on unresolved to save skillbook)
            if not evaluate_result.resolved:
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

    def _run_instance_concurrent(self, instance: Dict[str, Any]) -> InstanceResult:
        """Run a single instance with its own agent (for concurrent execution)."""
        instance_id = instance.get("instance_id", "unknown")
        with instance_context(instance_id):
            return self._run_instance_concurrent_inner(instance)

    def _run_instance_concurrent_inner(self, instance: Dict[str, Any]) -> InstanceResult:
        """Inner implementation of concurrent instance run."""
        instance_id = instance.get("instance_id", "unknown")
        repo = instance.get("repo", "unknown")

        # Create a fresh agent + predict phase for this worker
        agent = self.agent_factory()
        from phases.predict import PredictPhase
        worker_predict = PredictPhase(
            agent=agent,
            output_dir=self.predict.output_dir,
            run_name=self.predict.run_name,
            benchmark=self.predict.benchmark,
            model_name=self.predict.model_name,
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"[{instance_id}] Starting (concurrent)")
        logger.info(f"{'='*60}")

        skillbook = self.get_skillbook(repo)
        result = InstanceResult(instance_id=instance_id)

        # Check resume state
        start_iteration = self._get_resume_start(instance_id)

        if start_iteration == -1:
            logger.info(f"[{instance_id}] Already complete, skipping")
            result.final_resolved = self._load_resolved_status(instance_id) or False
            return result

        if start_iteration > 0:
            self._copy_resume_artifacts(instance_id)

        for iteration in range(start_iteration, self.max_attempts):
            logger.info(f"[{instance_id}] Iteration {iteration + 1}/{self.max_attempts}")

            # Phase 1: Predict (use worker's own predict phase)
            predict_result = worker_predict.run(
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

            iter_result = IterationResult(
                iteration=iteration,
                predict_result=predict_result,
                evaluate_result=evaluate_result,
            )
            result.iterations.append(iter_result)
            result.total_attempts = iteration + 1

            if evaluate_result.resolved:
                logger.info(f"[{instance_id}] RESOLVED at iteration {iteration + 1}!")
                result.final_resolved = True
                break

            # Phase 3: Learn (always run on unresolved to save skillbook)
            if not evaluate_result.resolved:
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
        logger.info(f"Concurrency: {self.concurrency}")
        if self.resume_state:
            complete = sum(1 for rp in self.resume_state.values() if rp.is_fully_complete)
            partial = sum(1 for rp in self.resume_state.values() if not rp.is_fully_complete and rp.last_complete_iter >= 0)
            logger.info(f"Resume: {complete} complete, {partial} partial")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifacts for fully complete instances (they won't be in instances list)
        # and partial instances (they will be in instances list)
        for instance_id, rp in self.resume_state.items():
            if rp.is_fully_complete:
                copy_instance_artifacts(
                    source_dir=rp.resume_dir,
                    dest_dir=self.output_dir,
                    benchmark=self.benchmark,
                    instance_id=instance_id,
                    up_to_iter=rp.last_complete_iter,
                )
            # Partial instances get their artifacts copied inside _run_instance_inner

        # Save config
        if config:
            save_config(config=config, run_dir=self.output_dir)

        # Track results
        all_results: Dict[str, InstanceResult] = {}
        resolved_ids: List[str] = []
        unresolved_ids: List[str] = []
        error_info: Optional[str] = None

        try:
            if self.concurrency <= 1:
                # Sequential mode
                for i, instance in enumerate(instances):
                    instance_id = instance.get("instance_id", f"unknown-{i}")
                    logger.info(f"\n[{i+1}/{len(instances)}] Processing {instance_id}")

                    result = self.run_instance(instance)
                    all_results[instance_id] = result

                    if result.final_resolved:
                        resolved_ids.append(instance_id)
                    else:
                        unresolved_ids.append(instance_id)
            else:
                # Concurrent mode
                results_lock = threading.Lock()

                def _worker(instance):
                    instance_id = instance.get("instance_id", "unknown")
                    try:
                        result = self._run_instance_concurrent(instance)
                        with results_lock:
                            all_results[instance_id] = result
                            if result.final_resolved:
                                resolved_ids.append(instance_id)
                            else:
                                unresolved_ids.append(instance_id)
                        return result
                    except Exception as e:
                        logger.error(f"[{instance_id}] Worker failed: {e}")
                        with results_lock:
                            unresolved_ids.append(instance_id)
                            all_results[instance_id] = InstanceResult(
                                instance_id=instance_id,
                                final_resolved=False,
                                total_attempts=0,
                            )
                        return None

                logger.info(f"Launching {len(instances)} instances with concurrency={self.concurrency}")
                with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                    futures = {
                        executor.submit(_worker, inst): inst
                        for inst in instances
                    }
                    done_count = 0
                    for future in as_completed(futures):
                        done_count += 1
                        inst = futures[future]
                        instance_id = inst.get("instance_id", "unknown")
                        logger.info(f"[{done_count}/{len(instances)}] Completed {instance_id}")

        except Exception as e:
            import traceback
            error_info = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"Experiment interrupted: {e}")
            logger.error(traceback.format_exc())

        finally:
            # Add fully-complete resumed instances to statistics
            resumed_resolved = []
            resumed_unresolved = []
            for instance_id, rp in self.resume_state.items():
                if rp.is_fully_complete:
                    resolved = self._load_resolved_status(instance_id) or False
                    if resolved:
                        resumed_resolved.append(instance_id)
                    else:
                        resumed_unresolved.append(instance_id)

            resolved_ids = resumed_resolved + resolved_ids
            unresolved_ids = resumed_unresolved + unresolved_ids

            # Always save statistics, even if interrupted
            total_processed = len(all_results) + len(resumed_resolved) + len(resumed_unresolved)
            total_planned = len(instances) + len(resumed_resolved) + len(resumed_unresolved)
            resolved_count = len(resolved_ids)
            resolution_rate = resolved_count / total_processed if total_processed > 0 else 0.0

            # Get observability project URL if enabled
            observability_project_url = None
            if is_observability_enabled():
                observability_project_url = get_project_url()

            statistics = {
                "run_name": self.run_name,
                "timestamp": datetime.now().isoformat(),
                "total_instances": total_planned,
                "processed_instances": total_processed,
                "resolved_count": resolved_count,
                "unresolved_count": len(unresolved_ids),
                "resolution_rate": resolution_rate,
                "resolved_ids": resolved_ids,
                "unresolved_ids": unresolved_ids,
                "config": {
                    "max_attempts": self.max_attempts,
                    "skillbook_mode": self.skillbook_mode,
                    "concurrency": self.concurrency,
                },
            }

            # Add error info if interrupted
            if error_info:
                statistics["status"] = "interrupted"
                statistics["error"] = error_info
            else:
                statistics["status"] = "completed"

            # Add resume info
            if self.resume_state:
                resume_dirs = list(set(str(rp.resume_dir) for rp in self.resume_state.values()))
                statistics["resume_dirs"] = resume_dirs
                statistics["resumed_complete_count"] = len(resumed_resolved) + len(resumed_unresolved)

            # Compute skillbook-assisted resolution stats
            skillbook_assisted_ids = []
            skillbook_by_iteration = {}
            for instance_id, result in all_results.items():
                if result.final_resolved and result.iterations:
                    resolving_iter = result.iterations[-1].iteration
                    if resolving_iter > 0:
                        skillbook_assisted_ids.append(instance_id)
                        iter_key = str(resolving_iter)
                        skillbook_by_iteration.setdefault(iter_key, []).append(instance_id)

            statistics["skillbook_assisted"] = {
                "count": len(skillbook_assisted_ids),
                "ids": skillbook_assisted_ids,
                "by_iteration": skillbook_by_iteration,
            }

            # Add observability project URL if available
            if observability_project_url:
                statistics["observability_project_url"] = observability_project_url

            # Save statistics
            save_statistics(statistics=statistics, run_dir=self.output_dir)

            logger.info(f"\n{'='*60}")
            if error_info:
                logger.info("Experiment Interrupted!")
            else:
                logger.info("Experiment Complete!")
            logger.info(f"Resolved: {resolved_count}/{total_processed} ({resolution_rate:.1%})")
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
    concurrency: int = 1,
    agent_factory: Optional[Callable] = None,
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
        concurrency: Number of parallel instances (1 = sequential)
        agent_factory: Callable that returns a new MiniSWEAgent (for concurrent mode)

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
        concurrency=concurrency,
        agent_factory=agent_factory,
    )
    return loop.run(instances=instances, config=config)
