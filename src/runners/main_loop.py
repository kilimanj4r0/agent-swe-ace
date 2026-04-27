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
        force_learn: bool = True,
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
        self.force_learn = force_learn

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
        frozen_skillbook: bool = False,
        force_learn: bool = False,
        max_attempts_override: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> InstanceResult:
        """
        Run experiment loop for a single instance.

        Args:
            instance: SWE-bench instance dict
            initial_skillbook: Optional starting skillbook
            frozen_skillbook: If True, never run learn phase
            force_learn: If True, run learn even when resolved
            max_attempts_override: Override self.max_attempts for this call
            phase: Phase subdirectory for output ("train", "val_baseline", "val")

        Returns:
            InstanceResult with all iteration results
        """
        instance_id = instance.get("instance_id", "unknown")
        repo = instance.get("repo", "unknown")

        with instance_context(instance_id):
            return self._run_instance_inner(
                instance, instance_id, repo, initial_skillbook,
                frozen_skillbook=frozen_skillbook,
                force_learn=force_learn,
                max_attempts_override=max_attempts_override,
                phase=phase,
            )

    def _run_instance_inner(self, instance, instance_id, repo, initial_skillbook=None,
                            frozen_skillbook=False, force_learn=False,
                            max_attempts_override=None, phase=None):
        """Inner implementation with instance context set."""
        effective_max = max_attempts_override if max_attempts_override is not None else self.max_attempts

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting instance: {instance_id}")
        logger.info(f"Repo: {repo}")
        if frozen_skillbook:
            logger.info("Mode: frozen skillbook (no learning)")
        if force_learn:
            logger.info("Mode: force learn (learn even if resolved)")
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

        for iteration in range(start_iteration, effective_max):
            logger.info(f"\n--- Iteration {iteration + 1}/{effective_max} ---")

            # Phase 1: Predict
            predict_result = self.predict.run(
                instance=instance,
                skillbook=skillbook,
                iteration=iteration,
                phase=phase,
            )

            # Phase 2: Evaluate
            evaluate_result = self.evaluate.run(
                instance=instance,
                patch=predict_result.patch,
                iteration=iteration,
                phase=phase,
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

            # Phase 3: Learn (conditionally)
            # Skip learn when: single attempt AND config force_learn=False AND not overridden by runtime force_learn
            skip_learn = effective_max <= 1 and not self.force_learn and not force_learn
            should_learn = not frozen_skillbook and not skip_learn and (force_learn or not evaluate_result.resolved)
            if should_learn:
                if evaluate_result.resolved:
                    logger.info(f"[{instance_id}] Learning from success (force_learn=True)...")
                else:
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
                    phase=phase,
                )
                iter_result.learn_result = learn_result

                # Update skillbook for next iteration
                self.update_skillbook(repo, skillbook)

            if evaluate_result.resolved:
                break

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

            # Phase 3: Learn (run on unresolved unless single-attempt with force_learn=False)
            skip_learn = self.max_attempts <= 1 and not self.force_learn
            if not evaluate_result.resolved and not skip_learn:
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
        val_instances: Optional[List[Dict[str, Any]]] = None,
        baseline_run_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run experiment loop on multiple instances.

        When val_instances is provided, runs in two-phase mode:
        1. Train phase: run instances with force_learn=True, max_attempts=1
        2. Val baseline: run val instances with empty skillbook
        3. Val skillbook: run val instances with learned skillbook

        Args:
            instances: List of SWE-bench instance dicts (train instances in two-phase mode)
            config: Optional config to save
            val_instances: Optional list of val instances for two-phase mode
            baseline_run_dir: Optional path to previous run with baseline results

        Returns:
            Summary dict with statistics
        """
        two_phase = val_instances is not None and len(val_instances) > 0

        # Validate incompatible settings
        if two_phase and self.skillbook_mode == "per_instance":
            raise ValueError(
                "skillbook.mode='per_instance' is incompatible with two-phase experiments "
                "(train/val split). Use 'per_repo' or 'global' so skills accumulate across "
                "training instances."
            )

        logger.info(f"\nStarting experiment: {self.run_name}")
        logger.info(f"Instances: {len(instances)}")
        logger.info(f"Max attempts: {self.max_attempts}")
        logger.info(f"Skillbook mode: {self.skillbook_mode}")
        logger.info(f"Concurrency: {self.concurrency}")
        if two_phase:
            logger.info(f"Two-phase mode: {len(instances)} train, {len(val_instances)} val")
        if self.resume_state:
            complete = sum(1 for rp in self.resume_state.values() if rp.is_fully_complete)
            partial = sum(1 for rp in self.resume_state.values() if not rp.is_fully_complete and rp.last_complete_iter >= 0)
            logger.info(f"Resume: {complete} complete, {partial} partial")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifacts for fully complete instances
        for instance_id, rp in self.resume_state.items():
            if rp.is_fully_complete:
                copy_instance_artifacts(
                    source_dir=rp.resume_dir,
                    dest_dir=self.output_dir,
                    benchmark=self.benchmark,
                    instance_id=instance_id,
                    up_to_iter=rp.last_complete_iter,
                )

        # Save config
        if config:
            save_config(config=config, run_dir=self.output_dir)

        # Track results
        all_results: Dict[str, InstanceResult] = {}
        resolved_ids: List[str] = []
        unresolved_ids: List[str] = []
        error_info: Optional[str] = None
        skill_count = 0  # For two-phase stats
        reused_from_baseline = 0  # Count of train instances reused from baseline
        baseline_resolved_count = 0
        baseline_unresolved_count = 0

        # Train phase parameters
        train_force_learn = two_phase  # Force learn in two-phase mode
        train_max_attempts = 1 if two_phase else self.max_attempts
        train_phase = "train" if two_phase else None

        # Timing
        start_time = datetime.now()
        instance_durations: List[float] = []

        try:
            if self.concurrency <= 1:
                # Sequential mode
                for i, instance in enumerate(instances):
                    instance_id = instance.get("instance_id", f"unknown-{i}")
                    logger.info(f"\n[TRAIN {i+1}/{len(instances)}] Processing {instance_id}")

                    inst_start = datetime.now()
                    if two_phase and baseline_run_dir:
                        result = self._run_train_instance_reusing_baseline(
                            instance, baseline_run_dir, phase="train",
                        )
                        # Check if reused (no predict_result means it was reused)
                        if result.iterations and result.iterations[0].predict_result is None:
                            reused_from_baseline += 1
                            if result.final_resolved:
                                baseline_resolved_count += 1
                            else:
                                baseline_unresolved_count += 1
                    else:
                        result = self.run_instance(
                            instance,
                            force_learn=train_force_learn,
                            max_attempts_override=train_max_attempts,
                            phase=train_phase,
                        )
                    all_results[instance_id] = result
                    instance_durations.append((datetime.now() - inst_start).total_seconds())

                    if result.final_resolved:
                        resolved_ids.append(instance_id)
                    else:
                        unresolved_ids.append(instance_id)
            else:
                # Concurrent mode (only for non-two-phase or train phase)
                results_lock = threading.Lock()

                def _worker(instance):
                    instance_id = instance.get("instance_id", "unknown")
                    inst_start = datetime.now()
                    try:
                        result = self._run_instance_concurrent(instance)
                        with results_lock:
                            all_results[instance_id] = result
                            instance_durations.append((datetime.now() - inst_start).total_seconds())
                            if result.final_resolved:
                                resolved_ids.append(instance_id)
                            else:
                                unresolved_ids.append(instance_id)
                        return result
                    except Exception as e:
                        logger.error(f"[{instance_id}] Worker failed: {e}")
                        with results_lock:
                            instance_durations.append((datetime.now() - inst_start).total_seconds())
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

            # === Two-phase: Val passes ===
            val_baseline_stats = None
            val_skillbook_stats = None

            if two_phase:
                # Get the final learned skillbook
                if self.skillbook_mode == "global":
                    final_skillbook = self.global_skillbook
                elif self.skillbook_mode == "per_repo":
                    # Use the first (should be only) repo's skillbook
                    final_skillbook = next(iter(self.repo_skillbooks.values()), Skillbook())
                else:
                    final_skillbook = Skillbook()

                skill_count = len(final_skillbook.skills())
                logger.info(f"\n{'='*60}")
                logger.info(f"TRAIN PHASE COMPLETE")
                logger.info(f"Skills learned: {skill_count}")
                logger.info(f"{'='*60}")

                # Save final skillbook snapshot
                from data_io.writers import save_skillbook
                save_skillbook(
                    skillbook=final_skillbook,
                    run_dir=self.output_dir,
                    benchmark=self.benchmark,
                    iteration=0,
                    phase=None,  # Save to skillbooks/ root as final_skillbook.json
                )
                # Also save as final_skillbook.json
                import shutil as sh
                skillbooks_dir = self.output_dir / self.benchmark / "skillbooks"
                if skillbooks_dir.exists():
                    latest = sorted(skillbooks_dir.glob("iter_*.json"))
                    if latest:
                        sh.copy2(latest[-1], skillbooks_dir / "final_skillbook.json")

                # Val baseline pass (empty skillbook)
                val_baseline_stats = self._run_val_pass(
                    val_instances=val_instances,
                    skillbook=Skillbook(),  # Empty skillbook
                    phase="val_baseline",
                    baseline_run_dir=baseline_run_dir,
                )

                # Val skillbook pass (learned skillbook)
                val_skillbook_stats = self._run_val_pass(
                    val_instances=val_instances,
                    skillbook=final_skillbook,
                    phase="val",
                )

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

            end_time = datetime.now()
            experiment_time = (end_time - start_time).total_seconds()
            avg_instance_time = (
                sum(instance_durations) / len(instance_durations)
                if instance_durations
                else 0.0
            )

            statistics = {
                "run_name": self.run_name,
                "timestamp": end_time.isoformat(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "experiment_time_seconds": round(experiment_time, 1),
                "avg_instance_time_seconds": round(avg_instance_time, 1),
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

            # Two-phase statistics
            if two_phase:
                train_resolved = [iid for iid in resolved_ids if iid not in resumed_resolved]
                train_unresolved = [iid for iid in unresolved_ids if iid not in resumed_unresolved]
                train_total = len(train_resolved) + len(train_unresolved)

                statistics["train_phase"] = {
                    "total_instances": train_total,
                    "resolved_count": len(train_resolved),
                    "unresolved_count": len(train_unresolved),
                    "resolution_rate": len(train_resolved) / train_total if train_total > 0 else 0.0,
                    "resolved_ids": train_resolved,
                    "unresolved_ids": train_unresolved,
                    "total_skills_learned": skill_count if two_phase else 0,
                    "reused_from_baseline": reused_from_baseline,
                    "freshly_run": train_total - reused_from_baseline,
                    "baseline_resolved_count": baseline_resolved_count,
                    "baseline_unresolved_count": baseline_unresolved_count,
                    "baseline_resolution_rate": baseline_resolved_count / reused_from_baseline if reused_from_baseline > 0 else 0.0,
                }

                if val_baseline_stats:
                    statistics["val_baseline_phase"] = val_baseline_stats

                if val_skillbook_stats:
                    statistics["val_skillbook_phase"] = val_skillbook_stats

                # Summary comparison
                baseline_rate = val_baseline_stats["resolution_rate"] if val_baseline_stats else 0.0
                skillbook_rate = val_skillbook_stats["resolution_rate"] if val_skillbook_stats else 0.0
                improvement = skillbook_rate - baseline_rate

                # Compute per-instance deltas
                baseline_resolved = set(val_baseline_stats.get("resolved_ids", [])) if val_baseline_stats else set()
                skillbook_resolved = set(val_skillbook_stats.get("resolved_ids", [])) if val_skillbook_stats else set()
                newly_resolved = sorted(skillbook_resolved - baseline_resolved)
                lost = sorted(baseline_resolved - skillbook_resolved)

                statistics["summary"] = {
                    "train_resolution_rate": len(train_resolved) / train_total if train_total > 0 else 0.0,
                    "val_baseline_resolution_rate": baseline_rate,
                    "val_skillbook_resolution_rate": skillbook_rate,
                    "skillbook_improvement": f"{improvement:+.3f}",
                    "skillbook_improvement_pct": f"{(improvement / baseline_rate * 100) if baseline_rate > 0 else 0:+.1f}%",
                    "newly_resolved_by_skillbook": newly_resolved,
                    "lost_by_skillbook": lost,
                }
            else:
                # Single-phase: compute skillbook-assisted resolution stats
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
            logger.info(f"Experiment time: {experiment_time:.1f}s ({experiment_time/60:.1f}min)")
            if instance_durations:
                logger.info(f"Avg per instance: {avg_instance_time:.1f}s ({avg_instance_time/60:.1f}min)")
            if two_phase and val_skillbook_stats:
                logger.info(f"Val baseline: {val_baseline_stats['resolution_rate']:.1%}")
                logger.info(f"Val skillbook: {val_skillbook_stats['resolution_rate']:.1%}")
                logger.info(f"Skillbook improvement: {improvement:+.3f}")
                if reused_from_baseline > 0:
                    bl_rate = baseline_resolved_count / reused_from_baseline
                    logger.info(f"Baseline resolved: {baseline_resolved_count}/{reused_from_baseline} ({bl_rate:.1%})")
            logger.info(f"{'='*60}")

        return statistics

    def _run_train_instance_reusing_baseline(
        self, instance: Dict[str, Any], baseline_dir: Path, phase: str = "train",
    ) -> InstanceResult:
        """Run a single train instance, reusing existing trajectory from baseline_dir.

        If the baseline has a valid trajectory (exit_status Submitted/LimitsExceeded)
        and a result for this instance, skip predict→eval and only run learn.
        Otherwise fall back to full predict→eval→learn.
        """
        instance_id = instance.get("instance_id", "unknown")
        repo = instance.get("repo", "unknown")

        traj_path = baseline_dir / self.benchmark / "trajectories" / instance_id / "iter_0.json"
        result_path = baseline_dir / self.benchmark / "results" / instance_id / "iter_0.json"

        if traj_path.exists() and result_path.exists():
            try:
                with open(traj_path) as f:
                    traj_data = json.load(f)
                with open(result_path) as f:
                    result_data = json.load(f)

                exit_status = traj_data.get("info", {}).get("exit_status", "")

                # Only reuse if trajectory has a valid exit status
                if exit_status not in ("Submitted", "LimitsExceeded"):
                    logger.info(
                        f"[TRAIN] {instance_id}: baseline trajectory has invalid "
                        f"exit_status='{exit_status}', running full predict→eval→learn"
                    )
                    return self.run_instance(
                        instance, force_learn=True, max_attempts_override=1, phase=phase,
                    )

                # Reuse existing data → only learn
                resolved = result_data.get("resolved", False)
                patch = traj_data.get("info", {}).get("submission", "")
                logger.info(
                    f"[TRAIN] {instance_id}: reusing existing trajectory from baseline "
                    f"(exit_status={exit_status}, resolved={resolved})"
                )

                # Copy artifacts to train/ subdirs
                dest_traj = self.output_dir / self.benchmark / "trajectories" / phase / instance_id
                dest_result = self.output_dir / self.benchmark / "results" / phase / instance_id
                dest_traj.mkdir(parents=True, exist_ok=True)
                dest_result.mkdir(parents=True, exist_ok=True)
                shutil.copy2(traj_path, dest_traj / "iter_0.json")
                shutil.copy2(result_path, dest_result / "iter_0.json")

                skillbook = self.get_skillbook(repo)

                # Run learn phase only
                trajectory = {
                    "info": traj_data.get("info", {}),
                    "messages": traj_data.get("messages", []),
                }
                learn_result = self.learn.run(
                    skillbook=skillbook,
                    instance=instance,
                    trajectory=trajectory,
                    patch=patch,
                    iteration=0,
                    phase=phase,
                )
                self.update_skillbook(repo, skillbook)

                # Build result
                result = InstanceResult(instance_id=instance_id)
                result.final_resolved = resolved
                result.total_attempts = 1
                result.iterations.append(IterationResult(
                    iteration=0,
                    predict_result=None,
                    evaluate_result=None,
                    learn_result=learn_result,
                ))
                return result

            except Exception as e:
                logger.warning(
                    f"[TRAIN] {instance_id}: failed to load baseline data ({e}), "
                    f"running full predict→eval→learn"
                )
                return self.run_instance(
                    instance, force_learn=True, max_attempts_override=1, phase=phase,
                )
        else:
            # No existing data → run full predict→eval→learn
            return self.run_instance(
                instance, force_learn=True, max_attempts_override=1, phase=phase,
            )

    def _run_val_pass(
        self,
        val_instances: List[Dict[str, Any]],
        skillbook: Skillbook,
        phase: str,
        baseline_run_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run validation pass on instances with a given skillbook.

        Args:
            val_instances: List of val instance dicts
            skillbook: Skillbook to use (empty for baseline, learned for skillbook pass)
            phase: Phase name ("val_baseline" or "val")
            baseline_run_dir: Optional previous run dir to load baseline results from

        Returns:
            Dict with resolution statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"VAL {phase.upper()} PASS")
        logger.info(f"Instances: {len(val_instances)}, Skills: {len(skillbook.skills())}")
        logger.info(f"{'='*60}")

        resolved_ids = []
        unresolved_ids = []
        results = {}

        # For val_baseline with baseline_run_dir, try loading existing results first
        loaded_ids = {}
        instances_to_run = list(val_instances)

        if phase == "val_baseline" and baseline_run_dir:
            loaded_ids, missing = self._load_baseline_results(baseline_run_dir, val_instances)
            instances_to_run = missing
            logger.info(
                f"Loaded {len(loaded_ids)} baseline results from {baseline_run_dir}, "
                f"{len(missing)} need re-execution"
            )

        # Run instances
        for i, instance in enumerate(instances_to_run):
            instance_id = instance.get("instance_id", f"unknown-{i}")
            logger.info(f"\n[{phase.upper()} {i+1}/{len(instances_to_run)}] Processing {instance_id}")

            result = self.run_instance(
                instance,
                initial_skillbook=skillbook,
                frozen_skillbook=True,
                max_attempts_override=1,
                phase=phase,
            )
            results[instance_id] = result

            if result.final_resolved:
                resolved_ids.append(instance_id)
            else:
                unresolved_ids.append(instance_id)

        # Merge loaded results
        for iid, was_resolved in loaded_ids.items():
            if was_resolved:
                resolved_ids.append(iid)
            else:
                unresolved_ids.append(iid)

        total = len(val_instances)
        resolved_count = len(resolved_ids)

        return {
            "total_instances": total,
            "resolved_count": resolved_count,
            "unresolved_count": len(unresolved_ids),
            "resolution_rate": resolved_count / total if total > 0 else 0.0,
            "resolved_ids": resolved_ids,
            "unresolved_ids": unresolved_ids,
            "skillbook_skills": len(skillbook.skills()),
        }

    def _load_baseline_results(
        self, baseline_dir: Path, val_instances: list
    ) -> tuple:
        """Load baseline results from a previous run.

        Returns:
            (loaded_results, missing_instances) where loaded_results maps
            instance_id -> resolved status and missing_instances are instances
            that need to be re-executed.
        """
        loaded = {}
        missing = []
        baseline_dir = Path(baseline_dir)

        for inst in val_instances:
            iid = inst["instance_id"]
            traj_path = baseline_dir / self.benchmark / "trajectories" / iid / "iter_0.json"
            result_path = baseline_dir / self.benchmark / "results" / iid / "iter_0.json"

            if traj_path.exists() and result_path.exists():
                try:
                    with open(traj_path) as f:
                        traj_data = json.load(f)
                    with open(result_path) as f:
                        result_data = json.load(f)

                    exit_status = traj_data.get("info", {}).get("exit_status", "")
                    if exit_status not in ("Submitted", "LimitsExceeded"):
                        logger.info(
                            f"[VAL_BASELINE] {iid}: baseline trajectory has invalid "
                            f"exit_status='{exit_status}', will re-run"
                        )
                        missing.append(inst)
                        continue

                    # Copy artifacts to current run's val_baseline/ dirs
                    dest_traj_dir = self.output_dir / self.benchmark / "trajectories" / "val_baseline" / iid
                    dest_result_dir = self.output_dir / self.benchmark / "results" / "val_baseline" / iid
                    dest_traj_dir.mkdir(parents=True, exist_ok=True)
                    dest_result_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(traj_path, dest_traj_dir / "iter_0.json")
                    shutil.copy2(result_path, dest_result_dir / "iter_0.json")

                    loaded[iid] = result_data.get("resolved", False)
                except Exception as e:
                    logger.warning(f"Failed to load baseline for {iid}: {e}")
                    missing.append(inst)
            else:
                missing.append(inst)

        return loaded, missing


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
