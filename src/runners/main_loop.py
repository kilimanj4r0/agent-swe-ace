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

from phases.evaluate import EvaluateResult
from phases.predict import PredictResult

from data_io.resume_scanner import ResumePoint, copy_instance_artifacts
from data_io.writers import save_statistics, save_config


def _build_ground_truth(instance: Dict[str, Any], max_chars: int = 512) -> str:
    """Build ground truth from SWE-bench test lists (not gold patch).

    Args:
        instance: SWE-bench instance dict with FAIL_TO_PASS / PASS_TO_PASS.
        max_chars: Maximum characters for the ground truth string. Some instances
            have 100K+ chars of test lists which blows up the prompt.
    """
    parts = []
    fail_to_pass = instance.get("FAIL_TO_PASS", "")
    pass_to_pass = instance.get("PASS_TO_PASS", "")
    if fail_to_pass:
        parts.append(f"Tests to fix (FAIL_TO_PASS): {fail_to_pass}")
    if pass_to_pass:
        parts.append(f"Tests to preserve (PASS_TO_PASS): {pass_to_pass}")
    result = "\n".join(parts) if parts else "(none)"
    if len(result) > max_chars:
        result = result[:max_chars] + f"\n... (truncated, {len(result)} total chars)"
    return result
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
    status: str = "completed"
    infrastructure_error: Optional[str] = None


def _is_infrastructure_error(predict_result: Any) -> bool:
    """Return whether predict failed before producing a task-level outcome."""
    return (
        getattr(predict_result, "error_kind", None) == "infrastructure"
        or getattr(predict_result, "exit_status", None) == "error"
    )


def _record_infrastructure_error(
    result: InstanceResult,
    iteration: int,
    predict_result: Any,
) -> None:
    """Record a terminal infrastructure failure for one instance."""
    result.iterations.append(
        IterationResult(
            iteration=iteration,
            predict_result=predict_result,
            evaluate_result=None,
        )
    )
    result.total_attempts = iteration + 1
    result.status = "infrastructure_error"
    result.infrastructure_error = (
        getattr(predict_result, "error", None)
        or "predict phase infrastructure error"
    )


def _append_outcome(
    result: InstanceResult,
    instance_id: str,
    resolved_ids: List[str],
    unresolved_ids: List[str],
    infrastructure_error_ids: List[str],
) -> None:
    """Classify an instance into exactly one experiment outcome bucket."""
    if result.status == "infrastructure_error":
        infrastructure_error_ids.append(instance_id)
    elif result.final_resolved:
        resolved_ids.append(instance_id)
    else:
        unresolved_ids.append(instance_id)


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
        skip_learn: bool = False,
        val_resume_state: Optional[Dict[str, ResumePoint]] = None,
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
            val_resume_state: Dict mapping instance_id -> ResumePoint for val phase resume
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
        self.skip_learn = skip_learn
        self.val_resume_state = val_resume_state or {}
        self._baseline_run_dir = None

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

    def _make_worker_predict(self):
        """Build a fresh PredictPhase with its own agent for a concurrent worker.

        Each worker needs its own agent (n_calls/cost counters are not thread-safe)
        and its own PredictPhase (so _retrieval_run_stats is not shared). Mirrors the
        construction previously open-coded in _run_instance_concurrent_inner.
        """
        from phases.predict import PredictPhase
        return PredictPhase(
            agent=self.agent_factory(),
            output_dir=self.predict.output_dir,
            run_name=self.predict.run_name,
            benchmark=self.predict.benchmark,
            model_name=self.predict.model_name,
            # Forward the shared retriever. Retrievers are designed to be shared
            # across worker threads (BM25/embedding guard state with locks). Omitting
            # it made every concurrency>1 run silently skip retrieval, feeding the
            # full skillbook (instances_retrieved=0). Regression from b77f430.
            skill_retriever=self.predict.skill_retriever,
        )

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

    def _copy_resume_artifacts(self, instance_id: str, phase: Optional[str] = None) -> None:
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
            source_phase=rp.phase,
            dest_phase=phase,
        )
        logger.info(
            f"[{instance_id}] Resumed from iter_0..iter_{rp.last_complete_iter} "
            f"(source: {rp.resume_dir.name})"
        )

    def _load_resolved_status(self, instance_id: str) -> Optional[bool]:
        """Read resolved status from the last completed iteration's result file.

        Checks the current run dir first (artifacts copied from resume source),
        then falls back to the resume source dir.
        """
        rp = self.resume_state.get(instance_id)
        if rp is None:
            return None
        iter_idx = rp.last_complete_iter

        search_dirs = [
            self.output_dir / self.benchmark / "results" / instance_id,
        ]
        if rp is not None:
            search_dirs.append(rp.resume_dir / self.benchmark / "results" / instance_id)

        for results_dir in search_dirs:
            target = results_dir / f"iter_{iter_idx}.json"
            if target.exists():
                try:
                    with open(target) as f:
                        return json.load(f).get("resolved", False)
                except Exception:
                    pass
        return None

    def _load_resolving_iteration(self, instance_id: str) -> Optional[int]:
        """Find which iteration resolved the instance.

        Checks the current run dir first (artifacts copied from resume source),
        then falls back to the resume source dir.

        Returns the iteration number that first shows resolved=True,
        or None if not found.
        """
        rp = self.resume_state.get(instance_id)
        if rp is None:
            return None

        # Prefer current run dir — artifacts were copied there at start,
        # so this works even if the resume source was renamed/removed.
        search_dirs = [
            self.output_dir / self.benchmark / "results" / instance_id,
        ]
        if rp is not None:
            search_dirs.append(rp.resume_dir / self.benchmark / "results" / instance_id)

        for results_dir in search_dirs:
            if not results_dir.is_dir():
                continue
            # Scan all available iteration files
            for k in range(10):
                result_file = results_dir / f"iter_{k}.json"
                if result_file.exists():
                    try:
                        with open(result_file) as f:
                            if json.load(f).get("resolved", False):
                                return k
                    except Exception:
                        continue
        return None

    def run_instance(
        self,
        instance: Dict[str, Any],
        initial_skillbook: Optional[Skillbook] = None,
        frozen_skillbook: bool = False,
        force_learn: bool = False,
        max_attempts_override: Optional[int] = None,
        phase: Optional[str] = None,
        skip_baseline_reuse: bool = False,
        predict_phase=None,
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
            skip_baseline_reuse: If True, ignore self._baseline_run_dir

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
                skip_baseline_reuse=skip_baseline_reuse,
                predict_phase=predict_phase,
            )

    def _run_instance_inner(self, instance, instance_id, repo, initial_skillbook=None,
                            frozen_skillbook=False, force_learn=False,
                            max_attempts_override=None, phase=None,
                            skip_baseline_reuse=False, predict_phase=None):
        """Inner implementation with instance context set."""
        predict = predict_phase or self.predict
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

        # Train solves WITHOUT a skillbook (distillation): every train attempt is
        # made unaided, and the SkillManager learns from that raw attempt into the
        # accumulated book (global/per_repo). Val phases solve with the real
        # (optionally retrieval-narrowed) book, so they keep `skillbook` as-is.
        # `not initial_skillbook` is defensive: train never receives one, but if a
        # caller preloads a book we respect it.
        if phase == "train" and not initial_skillbook:
            solve_skillbook = Skillbook()
        else:
            solve_skillbook = skillbook

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
            self._copy_resume_artifacts(instance_id, phase=phase)

        # Frozen/val pass: narrow the skillbook ONCE per instance so all k attempts
        # share the same retrieved skills (true pass@k of a fixed retrieval), instead
        # of retrieving per attempt. No-op without a retriever or when the phase /
        # skillbook doesn't warrant retrieval.
        retrieval_stats = None
        if frozen_skillbook:
            solve_skillbook, retrieval_stats = predict.prepare_skillbook(
                instance, solve_skillbook, phase
            )

        for iteration in range(start_iteration, effective_max):
            logger.info(f"\n--- Iteration {iteration + 1}/{effective_max} ---")

            # Try baseline reuse at iter_0 (single-phase per_instance mode)
            baseline_reused = False
            if iteration == 0 and start_iteration == 0 and self._baseline_run_dir and not skip_baseline_reuse:
                baseline_pred, baseline_eval = self._try_load_baseline_iter0(
                    instance_id, phase=phase
                )
                if baseline_pred and baseline_eval:
                    predict_result = baseline_pred
                    evaluate_result = baseline_eval
                    baseline_reused = True

            if not baseline_reused:
                # Phase 1: Predict
                predict_result = predict.run(
                    instance=instance,
                    skillbook=solve_skillbook,
                    iteration=iteration,
                    phase=phase,
                    retrieval_stats=retrieval_stats,
                    skillbook_prepared=frozen_skillbook,
                )

                if _is_infrastructure_error(predict_result):
                    _record_infrastructure_error(result, iteration, predict_result)
                    logger.error(
                        f"[{instance_id}] Infrastructure error: "
                        f"{result.infrastructure_error}"
                    )
                    break

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
            skip_learn = (effective_max <= 1 and not self.force_learn and not force_learn) or self.skip_learn
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
                    feedback=evaluate_result.feedback,
                    ground_truth=_build_ground_truth(instance),
                    resolved=evaluate_result.resolved,
                )
                iter_result.learn_result = learn_result

                # Update skillbook for next iteration
                self.update_skillbook(repo, skillbook)

            if evaluate_result.resolved:
                # When max_attempts > 1 was explicitly set, run all attempts
                # (val_pass_k > 1 means "always K attempts for measurement")
                # Exception: skip_learn mode — no point re-running resolved instances
                if effective_max <= 1 or self.skip_learn:
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
        worker_predict = self._make_worker_predict()

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
            self._copy_resume_artifacts(instance_id, phase=phase)

        for iteration in range(start_iteration, self.max_attempts):
            logger.info(f"[{instance_id}] Iteration {iteration + 1}/{self.max_attempts}")

            # Try baseline reuse at iter_0 (single-phase per_instance mode)
            baseline_reused = False
            if iteration == 0 and start_iteration == 0 and self._baseline_run_dir:
                baseline_pred, baseline_eval = self._try_load_baseline_iter0(
                    instance_id
                )
                if baseline_pred and baseline_eval:
                    predict_result = baseline_pred
                    evaluate_result = baseline_eval
                    baseline_reused = True

            if not baseline_reused:
                # Phase 1: Predict (use worker's own predict phase)
                predict_result = worker_predict.run(
                    instance=instance,
                    skillbook=skillbook,
                    iteration=iteration,
                )

                if _is_infrastructure_error(predict_result):
                    _record_infrastructure_error(result, iteration, predict_result)
                    logger.error(
                        f"[{instance_id}] Infrastructure error: "
                        f"{result.infrastructure_error}"
                    )
                    break

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
            skip_learn = (self.max_attempts <= 1 and not self.force_learn) or self.skip_learn
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
                    feedback=evaluate_result.feedback,
                    ground_truth=_build_ground_truth(instance),
                    resolved=evaluate_result.resolved,
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
        preloaded_skillbook: Optional[Skillbook] = None,
        val_pass_k: int = 1,
        train_trajs_dir: Optional[str] = None,
        eval_on_train: bool = False,
        eval_on_train_pass_k: int = 1,
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
            preloaded_skillbook: If provided, skip training and use this skillbook for val passes
            val_pass_k: Number of attempts per val instance (default: 1)
            train_trajs_dir: Optional path to teacher trajectories dir for distillation

        Returns:
            Summary dict with statistics
        """
        two_phase = val_instances is not None and len(val_instances) > 0

        # Validate baseline_run_dir exists
        if baseline_run_dir:
            baseline_run_dir = Path(baseline_run_dir)
            if not baseline_run_dir.exists():
                logger.warning(
                    f"Baseline run dir does not exist: {baseline_run_dir} — "
                    f"baseline reuse disabled, all instances will run from scratch"
                )
                baseline_run_dir = None

        # Store baseline_run_dir for single-phase reuse in run_instance()
        self._baseline_run_dir = baseline_run_dir
        self._train_trajs_dir = Path(train_trajs_dir) if train_trajs_dir else None

        if self._train_trajs_dir and not two_phase:
            logger.warning(
                "train_trajs_dir is set but no val_ratio configured — "
                "train_trajs_dir only works in two-phase mode. Ignoring."
            )
            self._train_trajs_dir = None

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
        if baseline_run_dir:
            logger.info(f"Baseline reuse: {baseline_run_dir}")
        if two_phase:
            logger.info(f"Two-phase mode: {len(instances)} train, {len(val_instances)} val")
        if self.resume_state:
            complete = sum(1 for rp in self.resume_state.values() if rp.is_fully_complete)
            partial = sum(1 for rp in self.resume_state.values() if not rp.is_fully_complete and rp.last_complete_iter >= 0)
            logger.info(f"Resume: {complete} complete, {partial} partial")
        if self.val_resume_state:
            val_complete = sum(1 for rp in self.val_resume_state.values() if rp.is_fully_complete)
            logger.info(f"Val resume: {val_complete} complete")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifacts for fully complete train instances
        dest_train_phase = "train" if two_phase else None
        for instance_id, rp in self.resume_state.items():
            if rp.is_fully_complete:
                copy_instance_artifacts(
                    source_dir=rp.resume_dir,
                    dest_dir=self.output_dir,
                    benchmark=self.benchmark,
                    instance_id=instance_id,
                    up_to_iter=rp.last_complete_iter,
                    source_phase=rp.phase,
                    dest_phase=dest_train_phase,
                )

        # Copy artifacts for fully complete val instances
        for instance_id, rp in self.val_resume_state.items():
            if rp.is_fully_complete:
                copy_instance_artifacts(
                    source_dir=rp.resume_dir,
                    dest_dir=self.output_dir,
                    benchmark=self.benchmark,
                    instance_id=instance_id,
                    up_to_iter=rp.last_complete_iter,
                    source_phase=rp.phase,
                    dest_phase=rp.phase,
                )

        # Save config
        if config:
            save_config(config=config, run_dir=self.output_dir)

        # Track results
        all_results: Dict[str, InstanceResult] = {}
        resolved_ids: List[str] = []
        unresolved_ids: List[str] = []
        infrastructure_error_ids: List[str] = []
        error_info: Optional[str] = None
        skill_count = 0  # For two-phase stats
        reused_from_baseline = 0  # Count of train instances reused from baseline
        baseline_resolved_count = 0
        baseline_unresolved_count = 0
        teacher_trajs_found = 0
        teacher_trajs_skipped = 0
        teacher_trajs_resolved = 0

        # Train phase parameters
        train_force_learn = two_phase  # Force learn in two-phase mode
        train_max_attempts = 1 if two_phase else self.max_attempts
        train_phase = "train" if two_phase else None

        # Timing
        start_time = datetime.now()
        instance_durations: List[float] = []
        val_baseline_stats = None
        val_skillbook_stats = None
        train_eval_stats = None
        train_eval_baseline_stats = None

        try:
            # Pre-check baseline skillbook compatibility (once, before train loop)
            baseline_sb_compat = None
            if two_phase and baseline_run_dir and self.skillbook_mode in ("global", "per_repo"):
                baseline_sb_compat = self._check_baseline_skillbook_compat(baseline_run_dir)

            if preloaded_skillbook is None:
                if two_phase or self.concurrency <= 1:
                    # Sequential train (two-phase train is always sequential; per_instance
                    # only goes concurrent when concurrency > 1, handled in the else below).
                    for i, instance in enumerate(instances):
                        instance_id = instance.get("instance_id", f"unknown-{i}")
                        logger.info(f"\n[TRAIN {i+1}/{len(instances)}] Processing {instance_id}")

                        inst_start = datetime.now()
                        if two_phase and self._train_trajs_dir is not None:
                            # Teacher trajectory distillation (priority for train)
                            teacher_result = self._run_train_instance_from_teacher(
                                instance, self._train_trajs_dir, phase="train",
                            )
                            if teacher_result is not None:
                                teacher_trajs_found += 1
                                if teacher_result.final_resolved:
                                    teacher_trajs_resolved += 1
                                result = teacher_result
                            else:
                                teacher_trajs_skipped += 1
                                logger.warning(f"[TRAIN] {instance_id}: no teacher trajectory, skipping")
                                continue
                        elif two_phase and baseline_run_dir:
                            result = self._run_train_instance_reusing_baseline(
                                instance, baseline_run_dir, phase="train",
                                allow_sb_merge=baseline_sb_compat is not False,
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

                        _append_outcome(
                            result,
                            instance_id,
                            resolved_ids,
                            unresolved_ids,
                            infrastructure_error_ids,
                        )
                else:
                    # Concurrent mode — per_instance only (two-phase train is always
                    # sequential; see the branch above). Each per_instance instance gets
                    # its own ephemeral skillbook, so concurrent skillbook mutation is safe.
                    results_lock = threading.Lock()

                    def _worker(instance):
                        instance_id = instance.get("instance_id", "unknown")
                        inst_start = datetime.now()
                        try:
                            result = self._run_instance_concurrent(instance)
                            with results_lock:
                                all_results[instance_id] = result
                                instance_durations.append((datetime.now() - inst_start).total_seconds())
                                _append_outcome(
                                    result,
                                    instance_id,
                                    resolved_ids,
                                    unresolved_ids,
                                    infrastructure_error_ids,
                                )
                            return result
                        except Exception as e:
                            logger.error(f"[{instance_id}] Worker failed: {e}")
                            with results_lock:
                                instance_durations.append((datetime.now() - inst_start).total_seconds())
                                infrastructure_error_ids.append(instance_id)
                                all_results[instance_id] = InstanceResult(
                                    instance_id=instance_id,
                                    final_resolved=False,
                                    total_attempts=0,
                                    status="infrastructure_error",
                                    infrastructure_error=str(e),
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

            if two_phase:
                if preloaded_skillbook is not None:
                    # Validation-only mode: skip training, use preloaded skillbook
                    final_skillbook = preloaded_skillbook
                    skill_count = len(final_skillbook.skills())
                    logger.info(f"\n{'='*60}")
                    logger.info(f"VALIDATION-ONLY MODE (training skipped)")
                    logger.info(f"Loaded skillbook: {skill_count} skills")
                    logger.info(f"{'='*60}")
                else:
                    # Normal training: get the final learned skillbook
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

                    # Post-train dedup sweep
                    if self.learn.dedup_manager is not None:
                        dedup_ops = self.learn._consolidate(final_skillbook)
                        if isinstance(dedup_ops, int) and dedup_ops > 0:
                            skill_count = len(final_skillbook.skills())
                            logger.info(f"Post-train dedup: applied {dedup_ops} operations, {skill_count} skills remain")

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

                if eval_on_train:
                    # TrainBL: empty skillbook on the TRAIN split. Runs a SINGLE
                    # attempt (pass@1) — it is a cheap raw-ability reference, not a
                    # pass@k pass. TrainSB below carries the pass@k measurement.
                    train_eval_baseline_stats = self._run_val_pass(
                        val_instances=instances,
                        skillbook=Skillbook(),
                        phase="train_eval_baseline",
                        baseline_run_dir=baseline_run_dir,
                        max_attempts=1,
                    )
                    # TrainSB: learned skillbook on the TRAIN split at pass_k
                    # (retrieval fires per config — gated in predict.py).
                    train_eval_stats = self._run_val_pass(
                        val_instances=instances,
                        skillbook=final_skillbook,
                        phase="train_eval",
                        max_attempts=eval_on_train_pass_k,
                    )
                else:
                    # Val baseline pass (empty skillbook)
                    val_baseline_stats = self._run_val_pass(
                        val_instances=val_instances,
                        skillbook=Skillbook(),  # Empty skillbook
                        phase="val_baseline",
                        baseline_run_dir=baseline_run_dir,
                        max_attempts=val_pass_k,
                    )

                    # Val skillbook pass (learned skillbook)
                    val_skillbook_stats = self._run_val_pass(
                        val_instances=val_instances,
                        skillbook=final_skillbook,
                        phase="val",
                        max_attempts=val_pass_k,
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
                "infrastructure_error_count": len(infrastructure_error_ids),
                "resolution_rate": resolution_rate,
                "resolved_ids": resolved_ids,
                "unresolved_ids": unresolved_ids,
                "infrastructure_error_ids": infrastructure_error_ids,
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
            elif infrastructure_error_ids:
                statistics["status"] = "degraded"
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
                train_infrastructure_errors = list(infrastructure_error_ids)
                train_total = (
                    len(train_resolved)
                    + len(train_unresolved)
                    + len(train_infrastructure_errors)
                )

                statistics["train_phase"] = {
                    "total_instances": train_total,
                    "resolved_count": len(train_resolved),
                    "unresolved_count": len(train_unresolved),
                    "infrastructure_error_count": len(train_infrastructure_errors),
                    "resolution_rate": len(train_resolved) / train_total if train_total > 0 else 0.0,
                    "resolved_ids": train_resolved,
                    "unresolved_ids": train_unresolved,
                    "infrastructure_error_ids": train_infrastructure_errors,
                    "total_skills_learned": skill_count if two_phase else 0,
                    "reused_from_baseline": reused_from_baseline,
                    "freshly_run": train_total - reused_from_baseline,
                    "baseline_resolved_count": baseline_resolved_count,
                    "baseline_unresolved_count": baseline_unresolved_count,
                    "baseline_resolution_rate": baseline_resolved_count / reused_from_baseline if reused_from_baseline > 0 else 0.0,
                    "teacher_trajs_dir": str(self._train_trajs_dir) if self._train_trajs_dir else None,
                    "teacher_trajs_found": teacher_trajs_found,
                    "teacher_trajs_skipped": teacher_trajs_skipped,
                    "teacher_trajs_resolved": teacher_trajs_resolved,
                }

                if val_baseline_stats:
                    statistics["val_baseline_phase"] = val_baseline_stats

                if val_skillbook_stats:
                    statistics["val_skillbook_phase"] = val_skillbook_stats

                if eval_on_train:
                    if train_eval_baseline_stats:
                        statistics["train_eval_baseline_phase"] = train_eval_baseline_stats
                    if train_eval_stats:
                        statistics["train_eval_phase"] = train_eval_stats

                # Summary comparison
                if eval_on_train:
                    teb = train_eval_baseline_stats or {}
                    te = train_eval_stats or {}
                    # TrainBL runs a single attempt -> its resolution_rate IS pass@1.
                    teb_p1 = teb.get("resolution_rate", 0.0)
                    # TrainSB metrics: pass@1 (iter0), pass@k (cumulative resolution_rate,
                    # the memorization ceiling), and avg@k (mean of per-iteration rates —
                    # the statistically-sound per-attempt metric). pass_at_k / per_attempt_rate
                    # are only populated when max_attempts > 1; fall back to resolution_rate.
                    te_pass_at = te.get("pass_at_k", {}) or {}
                    te_per = te.get("per_attempt_rate", {}) or {}
                    te_p1 = te_pass_at.get("pass@1", {}).get(
                        "rate", te.get("resolution_rate", 0.0)
                    )
                    te_pk = te.get("resolution_rate", 0.0)  # cumulative pass@k (ceiling)
                    te_avgk = (
                        sum(v.get("rate", 0.0) for v in te_per.values()) / len(te_per)
                        if te_per
                        else te.get("resolution_rate", 0.0)
                    )
                    # Skillbook effect at two attempt-matched levels: pass@1 (clean,
                    # single-attempt) and avg@k (per-attempt average; TrainBL contributes
                    # its lone iter0 rate).
                    delta_p1 = te_p1 - teb_p1
                    delta_avgk = te_avgk - teb_p1
                    # Resolved-set diff (informational; TrainBL is pass@1, TrainSB is pass@k).
                    teb_resolved = set(teb.get("resolved_ids", []))
                    te_resolved = set(te.get("resolved_ids", []))
                    statistics["summary"] = {
                        "train_eval_pass1_rate": te_p1,
                        "train_eval_resolution_rate": te_pk,        # pass@k (cumulative ceiling)
                        "train_eval_avg_rate": te_avgk,             # mean per-attempt rate
                        "train_eval_baseline_resolution_rate": teb_p1,  # pass@1 (1 attempt)
                        "train_skillbook_improvement": f"{delta_p1:+.3f}",          # Δ@pass@1
                        "train_skillbook_improvement_avg": f"{delta_avgk:+.3f}",    # Δ@avg@k
                        "train_skillbook_improvement_pct": (
                            f"{(delta_p1 / teb_p1 * 100) if teb_p1 > 0 else 0:+.1f}%"
                        ),
                        "newly_resolved_by_skillbook": sorted(te_resolved - teb_resolved),
                        "lost_by_skillbook": sorted(teb_resolved - te_resolved),
                    }
                    # Top-level mirrors the TrainSB pass (headline metric = pass@k ceiling).
                    if te:
                        statistics["total_instances"] = te.get("total_instances", 0)
                        statistics["resolved_count"] = te.get("resolved_count", 0)
                        statistics["unresolved_count"] = te.get("unresolved_count", 0)
                        statistics["resolution_rate"] = te.get("resolution_rate", 0.0)
                        statistics["resolved_ids"] = te.get("resolved_ids", [])
                        statistics["unresolved_ids"] = te.get("unresolved_ids", [])
                else:
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

                # Include resumed-complete resolved instances that resolved at iter > 0
                for iid in resumed_resolved:
                    resolving_iter = self._load_resolving_iteration(iid)
                    if resolving_iter is not None and resolving_iter > 0:
                        skillbook_assisted_ids.append(iid)
                        iter_key = str(resolving_iter)
                        skillbook_by_iteration.setdefault(iter_key, []).append(iid)

                statistics["skillbook_assisted"] = {
                    "count": len(skillbook_assisted_ids),
                    "ids": skillbook_assisted_ids,
                    "by_iteration": skillbook_by_iteration,
                }

            phase_infrastructure_ids = set(statistics["infrastructure_error_ids"])
            for phase_key in (
                "train_phase",
                "val_baseline_phase",
                "val_skillbook_phase",
                "train_eval_baseline_phase",
                "train_eval_phase",
            ):
                phase_infrastructure_ids.update(
                    statistics.get(phase_key, {}).get(
                        "infrastructure_error_ids", []
                    )
                )

            # Keep the top-level outcome collections mutually exclusive even when
            # the same instance appears in more than one two-phase pass.
            statistics["infrastructure_error_ids"] = sorted(
                phase_infrastructure_ids
            )
            statistics["infrastructure_error_count"] = len(
                phase_infrastructure_ids
            )
            statistics["resolved_ids"] = [
                iid
                for iid in statistics["resolved_ids"]
                if iid not in phase_infrastructure_ids
            ]
            statistics["unresolved_ids"] = [
                iid
                for iid in statistics["unresolved_ids"]
                if iid not in phase_infrastructure_ids
            ]
            statistics["resolved_count"] = len(statistics["resolved_ids"])
            statistics["unresolved_count"] = len(statistics["unresolved_ids"])
            accounted = (
                statistics["resolved_count"]
                + statistics["unresolved_count"]
                + statistics["infrastructure_error_count"]
            )
            statistics["resolution_rate"] = (
                statistics["resolved_count"] / accounted if accounted else 0.0
            )
            resolved_count = statistics["resolved_count"]
            resolution_rate = statistics["resolution_rate"]
            if not error_info and phase_infrastructure_ids:
                statistics["status"] = "degraded"

            # Add observability project URL if available
            if observability_project_url:
                statistics["observability_project_url"] = observability_project_url

            # Add retrieval stats if available
            retrieval_summary = self.predict.get_retrieval_summary()
            if retrieval_summary:
                statistics["retrieval"] = retrieval_summary

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

    def _try_load_baseline_iter0(self, instance_id: str, phase: str | None = None):
        """Try to load iter_0 trajectory+result from baseline dir for single-phase reuse.

        If found and valid, copies artifacts to output dir and returns
        (PredictResult, EvaluateResult). Returns (None, None) otherwise.
        """
        baseline_dir = self._baseline_run_dir
        if not baseline_dir:
            return None, None

        traj_path = baseline_dir / self.benchmark / "trajectories" / instance_id / "iter_0.json"
        result_path = baseline_dir / self.benchmark / "results" / instance_id / "iter_0.json"

        if not traj_path.exists() or not result_path.exists():
            return None, None

        try:
            with open(traj_path) as f:
                traj_data = json.load(f)
            with open(result_path) as f:
                result_data = json.load(f)

            exit_status = traj_data.get("info", {}).get("exit_status", "")
            if exit_status not in ("Submitted", "LimitsExceeded", "ContextWindowExceeded"):
                return None, None

            # Copy artifacts to output dir
            if phase:
                dest_traj = self.output_dir / self.benchmark / "trajectories" / phase / instance_id
                dest_result = self.output_dir / self.benchmark / "results" / phase / instance_id
            else:
                dest_traj = self.output_dir / self.benchmark / "trajectories" / instance_id
                dest_result = self.output_dir / self.benchmark / "results" / instance_id

            dest_traj.mkdir(parents=True, exist_ok=True)
            dest_result.mkdir(parents=True, exist_ok=True)
            shutil.copy2(traj_path, dest_traj / "iter_0.json")
            shutil.copy2(result_path, dest_result / "iter_0.json")

            patch = traj_data.get("info", {}).get("submission", "")
            resolved = result_data.get("resolved", False)
            feedback = result_data.get("feedback", "")

            predict_result = PredictResult(
                instance_id=instance_id,
                iteration=0,
                exit_status=exit_status,
                patch=patch,
                trajectory=traj_data.get("messages", []),
                trajectory_path=dest_traj / "iter_0.json",
            )
            evaluate_result = EvaluateResult(
                instance_id=instance_id,
                iteration=0,
                resolved=resolved,
                feedback=feedback,
                metrics=result_data.get("metrics", {}),
                result_path=dest_result / "iter_0.json",
            )

            logger.info(
                f"[{instance_id}] Reusing baseline iter_0 "
                f"(exit={exit_status}, resolved={resolved})"
            )
            return predict_result, evaluate_result

        except Exception as e:
            logger.debug(f"[{instance_id}] Baseline iter_0 load failed: {e}")
            return None, None

    def _check_baseline_skillbook_compat(self, baseline_dir: Path) -> bool | None:
        """Check if baseline skillbooks are compatible with current config.

        Compares skillbook settings between baseline and current run.
        Returns True if compatible, False if not, None if cannot determine.
        Logs the result.
        """
        baseline_cfg_path = baseline_dir / "config.json"
        current_cfg_path = self.output_dir / "config.json"
        if not baseline_cfg_path.exists() or not current_cfg_path.exists():
            logger.info("[TRAIN] Cannot check baseline skillbook compatibility (missing config)")
            return None
        try:
            b_cfg = json.loads(baseline_cfg_path.read_text())
            c_cfg = json.loads(current_cfg_path.read_text())
            b_exp = b_cfg.get("experiment", {})
            c_exp = c_cfg.get("experiment", {})
            b_sb = b_exp.get("skillbook", {})
            c_sb = c_exp.get("skillbook", {})

            mismatches = []
            for key in ("custom_swe_learn",):
                b_val = b_sb.get(key, b_exp.get(key, False))
                c_val = c_sb.get(key, c_exp.get(key, False))
                if b_val != c_val:
                    mismatches.append(f"{key}: baseline={b_val} vs current={c_val}")

            if mismatches:
                logger.info(
                    f"[TRAIN] Baseline skillbook incompatible ({'; '.join(mismatches)}), "
                    f"will relearn all skills"
                )
                return False

            logger.info(
                f"[TRAIN] Baseline skillbook compatible "
                f"(custom_swe_learn={c_val}, "
                f"mode: baseline={b_sb.get('mode','?')} -> current={c_sb.get('mode','?')})"
            )
            return True
        except Exception as e:
            logger.warning(f"[TRAIN] Failed to check baseline compatibility: {e}")
            return None

    def _run_train_instance_reusing_baseline(
        self, instance: Dict[str, Any], baseline_dir: Path, phase: str = "train",
        allow_sb_merge: bool = True,
    ) -> InstanceResult:
        """Run a single train instance, reusing existing trajectory from baseline_dir.

        If the baseline has a valid trajectory (exit_status Submitted/LimitsExceeded)
        and a result for this instance, skip predict→eval and only run learn.
        Otherwise fall back to full predict→eval→learn.
        """
        instance_id = instance.get("instance_id", "unknown")
        repo = instance.get("repo", "unknown")

        # Search for baseline trajectory in both phase-based (train/) and flat layouts
        traj_path = None
        result_path = None
        for prefix in ["train", None]:
            if prefix:
                tp = baseline_dir / self.benchmark / "trajectories" / prefix / instance_id / "iter_0.json"
                rp = baseline_dir / self.benchmark / "results" / prefix / instance_id / "iter_0.json"
            else:
                tp = baseline_dir / self.benchmark / "trajectories" / instance_id / "iter_0.json"
                rp = baseline_dir / self.benchmark / "results" / instance_id / "iter_0.json"
            if tp.exists() and rp.exists():
                traj_path = tp
                result_path = rp
                break

        if traj_path and result_path:
            try:
                with open(traj_path) as f:
                    traj_data = json.load(f)
                with open(result_path) as f:
                    result_data = json.load(f)

                exit_status = traj_data.get("info", {}).get("exit_status", "")

                # Only reuse if trajectory has a valid exit status
                if exit_status not in ("Submitted", "LimitsExceeded", "ContextWindowExceeded"):
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

                # Check if baseline has a skillbook for this instance (global/per_repo only)
                baseline_sb_path = (
                    baseline_dir / self.benchmark / "skillbooks" / instance_id
                )
                baseline_sb_file = None
                if baseline_sb_path.exists() and self.skillbook_mode in ("global", "per_repo"):
                    iters = sorted(baseline_sb_path.glob("iter_*.json"))
                    if iters:
                        baseline_sb_file = iters[-1]
                        if not allow_sb_merge:
                            baseline_sb_file = None

                if baseline_sb_file is not None:
                    # Merge baseline skills into current skillbook, skip relearn
                    from data_io.readers import load_skillbook as _load_sb
                    baseline_sb = _load_sb(baseline_sb_file)
                    existing_contents = {
                        s.content for s in skillbook.skills()
                    }
                    merged = 0
                    for skill in baseline_sb.skills():
                        if skill.content not in existing_contents:
                            skillbook.add_skill(
                                section=skill.section,
                                content=skill.content,
                                justification=skill.justification,
                                evidence=skill.evidence,
                            )
                            merged += 1
                    self.update_skillbook(repo, skillbook)
                    logger.info(
                        f"[TRAIN] {instance_id}: merged {merged} baseline skills "
                        f"(skipped relearn, {len(skillbook.skills())} total in skillbook)"
                    )

                    # Consolidate after merge to deduplicate similar skills
                    if self.learn.dedup_manager is not None and merged > 0:
                        dedup_ops = self.learn._consolidate(skillbook)
                        if isinstance(dedup_ops, int) and dedup_ops > 0:
                            self.update_skillbook(repo, skillbook)

                    learn_result = None
                else:
                    # No baseline skillbook or per_instance mode → run learn
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
                        feedback=result_data.get("feedback"),
                        ground_truth=_build_ground_truth(instance),
                        resolved=resolved,
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

    def _run_train_instance_from_teacher(
        self, instance: Dict[str, Any], train_trajs_dir: Path, phase: str = "train",
    ) -> Optional[InstanceResult]:
        """Run learn phase using a teacher trajectory.

        Returns None if no teacher trajectory found for this instance.
        """
        from data_io.readers import load_teacher_trajectory
        from data_io.writers import save_trajectory

        instance_id = instance.get("instance_id", "unknown")
        repo = instance.get("repo", "unknown")

        traj_data = load_teacher_trajectory(train_trajs_dir, instance_id)
        if traj_data is None:
            return None

        patch = traj_data["info"].get("submission", "")
        resolved = traj_data["info"].get("resolved", False)
        skillbook = self.get_skillbook(repo)

        logger.info(
            f"[TRAIN] {instance_id}: using teacher trajectory "
            f"(resolved={resolved}, {len(traj_data['messages'])} messages)"
        )

        # Run learn phase (always, since force_learn=True in train phase)
        learn_result = None
        try:
            learn_result = self.learn.run(
                skillbook=skillbook,
                instance=instance,
                trajectory=traj_data,
                patch=patch,
                iteration=0,
                phase=phase,
                ground_truth=_build_ground_truth(instance),
                resolved=resolved,
            )
            self.update_skillbook(repo, skillbook)
        except Exception as e:
            logger.error(f"[TRAIN] {instance_id}: teacher learn failed: {e}")

        # Save teacher trajectory to output for reproducibility
        save_trajectory(traj_data, self.output_dir, self.benchmark,
                        instance_id, iteration=0, phase=phase)

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

    def _run_val_pass(
        self,
        val_instances: List[Dict[str, Any]],
        skillbook: Skillbook,
        phase: str,
        baseline_run_dir: Optional[Path] = None,
        max_attempts: int = 1,
    ) -> Dict[str, Any]:
        """Run validation pass on instances with a given skillbook.

        Args:
            val_instances: List of val instance dicts
            skillbook: Skillbook to use (empty for baseline, learned for skillbook pass)
            phase: Phase name ("val_baseline" or "val")
            baseline_run_dir: Optional previous run dir to load baseline results from
            max_attempts: Number of attempts per val instance (default: 1)

        Returns:
            Dict with resolution statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"VAL {phase.upper()} PASS")
        logger.info(f"Instances: {len(val_instances)}, Skills: {len(skillbook.skills())}, Attempts: {max_attempts}")
        logger.info(f"{'='*60}")

        resolved_ids = []
        unresolved_ids = []
        infrastructure_error_ids = []
        results = {}

        # For val_baseline with baseline_run_dir, try loading existing results first
        loaded_ids = {}
        loaded_iter_details = {}  # iid -> {iter: resolved} for pass@k stats
        instances_to_run = list(val_instances)

        _BASELINE_PHASES = {"val_baseline": "val_baseline", "train_eval_baseline": "train"}
        if phase in _BASELINE_PHASES and baseline_run_dir:
            loaded_ids, loaded_iter_details, instances_to_run = self._load_baseline_results_multi(
                baseline_run_dir, val_instances, max_attempts=max_attempts,
                source_phase=_BASELINE_PHASES[phase], dest_phase=phase,
            )
            logger.info(
                f"Loaded {len(loaded_ids)} baseline results from {baseline_run_dir}, "
                f"{len(instances_to_run)} need re-execution"
            )

        # Skip val instances that are already complete from val_resume_state
        resume_complete_ids = set()
        for iid, rp in self.val_resume_state.items():
            if rp.phase == phase and rp.is_fully_complete:
                resume_complete_ids.add(iid)
        if resume_complete_ids:
            # Load resolved status from resume source
            for iid in resume_complete_ids:
                rp = self.val_resume_state[iid]
                resolved = False
                # Read resolved from result file in source
                if rp.phase:
                    rpath = rp.resume_dir / self.benchmark / "results" / rp.phase / iid / f"iter_{rp.last_complete_iter}.json"
                else:
                    rpath = rp.resume_dir / self.benchmark / "results" / iid / f"iter_{rp.last_complete_iter}.json"
                if rpath.exists():
                    try:
                        with open(rpath) as f:
                            resolved = json.load(f).get("resolved", False)
                    except Exception:
                        pass
                loaded_ids[iid] = resolved
            # Remove complete instances from run list
            before = len(instances_to_run)
            instances_to_run = [
                inst for inst in instances_to_run
                if inst["instance_id"] not in resume_complete_ids
            ]
            logger.info(
                f"[{phase}] Resumed {len(resume_complete_ids)} complete instances, "
                f"{len(instances_to_run)} to run"
            )

        # Run instances (concurrent when concurrency > 1; the skillbook is frozen/
        # read-only, evaluation is serialized by the global _eval_lock, and each
        # worker gets its own agent + PredictPhase, so this is safe).
        if self.concurrency > 1 and len(instances_to_run) > 1:
            results_lock = threading.Lock()

            def _val_worker(instance, idx):
                instance_id = instance.get("instance_id", f"unknown-{idx}")
                try:
                    with instance_context(instance_id):
                        logger.info(f"\n[{phase.upper()}] Processing {instance_id}")
                        worker_predict = self._make_worker_predict()
                        result = self.run_instance(
                            instance,
                            initial_skillbook=skillbook,
                            frozen_skillbook=True,
                            max_attempts_override=max_attempts,
                            phase=phase,
                            skip_baseline_reuse=True,
                            predict_phase=worker_predict,
                        )
                    with results_lock:
                        results[instance_id] = result
                        _append_outcome(
                            result,
                            instance_id,
                            resolved_ids,
                            unresolved_ids,
                            infrastructure_error_ids,
                        )
                    return result
                except Exception as e:
                    logger.error(f"[{phase.upper()}] {instance_id} worker failed: {e}")
                    with results_lock:
                        infrastructure_error_ids.append(instance_id)
                        results[instance_id] = InstanceResult(
                            instance_id=instance_id,
                            final_resolved=False,
                            total_attempts=0,
                            status="infrastructure_error",
                            infrastructure_error=str(e),
                        )
                    return None

            logger.info(
                f"[{phase.upper()}] running {len(instances_to_run)} instances "
                f"with concurrency={self.concurrency}"
            )
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {
                    executor.submit(_val_worker, inst, i)
                    for i, inst in enumerate(instances_to_run)
                }
                done_count = 0
                for future in as_completed(futures):
                    done_count += 1
                    logger.info(f"[{phase.upper()} {done_count}/{len(instances_to_run)}] completed")
        else:
            for i, instance in enumerate(instances_to_run):
                instance_id = instance.get("instance_id", f"unknown-{i}")
                logger.info(f"\n[{phase.upper()} {i+1}/{len(instances_to_run)}] Processing {instance_id}")

                result = self.run_instance(
                    instance,
                    initial_skillbook=skillbook,
                    frozen_skillbook=True,
                    max_attempts_override=max_attempts,
                    phase=phase,
                    skip_baseline_reuse=True,
                )
                results[instance_id] = result

                _append_outcome(
                    result,
                    instance_id,
                    resolved_ids,
                    unresolved_ids,
                    infrastructure_error_ids,
                )

        # Merge loaded results
        for iid, was_resolved in loaded_ids.items():
            if was_resolved:
                resolved_ids.append(iid)
            else:
                unresolved_ids.append(iid)

        total = len(val_instances)

        # Ensure all val instances are accounted for (crashed/errored instances
        # that never reached resolved/unresolved classification go to unresolved)
        classified = (
            set(resolved_ids)
            | set(unresolved_ids)
            | set(infrastructure_error_ids)
        )
        val_id_set = {inst["instance_id"] for inst in val_instances}
        missing = sorted(val_id_set - classified)
        if missing:
            logger.warning(
                f"[{phase}] {len(missing)} val instances not classified, "
                f"treating as unresolved: {missing}"
            )
            unresolved_ids.extend(missing)

        resolved_count = len(resolved_ids)

        stats = {
            "total_instances": total,
            "resolved_count": resolved_count,
            "unresolved_count": len(unresolved_ids),
            "infrastructure_error_count": len(infrastructure_error_ids),
            "resolution_rate": resolved_count / total if total > 0 else 0.0,
            "resolved_ids": resolved_ids,
            "unresolved_ids": unresolved_ids,
            "infrastructure_error_ids": infrastructure_error_ids,
            "status": "degraded" if infrastructure_error_ids else "completed",
            "skillbook_skills": len(skillbook.skills()),
        }

        # Per-attempt breakdown for val_pass_k > 1
        if max_attempts > 1:
            pass_at = {}
            all_attempts = {}  # iid -> [resolved_per_iter]

            for iid, result in results.items():
                attempts = []
                for ir in result.iterations:
                    r = ir.evaluate_result.resolved if ir.evaluate_result else False
                    attempts.append(r)
                all_attempts[iid] = attempts

            # Include loaded baseline instances (multi-iteration aware)
            for iid, iter_resolved in loaded_iter_details.items():
                all_attempts[iid] = [iter_resolved[n] for n in sorted(iter_resolved)]

            for n in range(1, max_attempts + 1):
                count = sum(
                    1 for a in all_attempts.values() if any(a[:n])
                )
                pass_at[f"pass@{n}"] = {
                    "count": count,
                    "total": total,
                    "rate": count / total if total else 0.0,
                }

            # Per-attempt (per-iteration) resolution rates: how many instances
            # resolved at each individual attempt, independent of other attempts.
            # Unlike cumulative pass@k, averaging these yields a correct per-attempt
            # mean. compare_runs.py reads this for the "avg" metric.
            per_attempt = {}
            for i in range(max_attempts):
                resolved_at_i = sum(
                    1 for a in all_attempts.values() if i < len(a) and a[i]
                )
                per_attempt[f"iter_{i}"] = {
                    "resolved": resolved_at_i,
                    "total": total,
                    "rate": resolved_at_i / total if total else 0.0,
                }

            stats["max_attempts"] = max_attempts
            stats["pass_at_k"] = pass_at
            stats["per_attempt_rate"] = per_attempt

        return stats

    def _load_baseline_results_multi(
        self, baseline_dir: Path, val_instances: list, max_attempts: int = 1,
        source_phase: str = "val_baseline", dest_phase: str = "val_baseline",
    ) -> tuple:
        """Load baseline val results, supporting multi-iteration reuse.

        For each val instance, finds existing iterations in baseline_run_dir.
        Only fully reuses an instance if >= max_attempts iterations exist.
        Otherwise falls back to fresh execution.

        Returns:
            (loaded_ids, loaded_iter_details, missing_instances) where:
            - loaded_ids: dict of iid -> resolved (bool), fully reused instances
            - loaded_iter_details: dict of iid -> {iter_num: resolved}, for pass@k stats
            - missing_instances: list of instance dicts that need fresh execution
        """
        loaded = {}
        loaded_iter_details = {}
        missing = []
        baseline_dir = Path(baseline_dir)

        for inst in val_instances:
            iid = inst["instance_id"]

            # Find existing iterations across phase-based and flat layouts
            iters_found = []  # list of (traj_path, result_path, resolved)
            for prefix in [source_phase, None]:
                if prefix:
                    traj_dir = baseline_dir / self.benchmark / "trajectories" / prefix / iid
                    result_dir = baseline_dir / self.benchmark / "results" / prefix / iid
                else:
                    traj_dir = baseline_dir / self.benchmark / "trajectories" / iid
                    result_dir = baseline_dir / self.benchmark / "results" / iid

                if traj_dir.exists() and result_dir.exists():
                    for n in range(max_attempts):
                        tp = traj_dir / f"iter_{n}.json"
                        rp = result_dir / f"iter_{n}.json"
                        if tp.exists() and rp.exists():
                            try:
                                with open(tp) as f:
                                    traj_data = json.load(f)
                                with open(rp) as f:
                                    result_data = json.load(f)

                                exit_status = traj_data.get("info", {}).get("exit_status", "")
                                if exit_status not in ("Submitted", "LimitsExceeded", "ContextWindowExceeded"):
                                    break  # Stop at invalid iteration

                                iters_found.append((tp, rp, result_data.get("resolved", False)))
                            except Exception:
                                break  # Stop at broken iteration
                    break  # Found data in this prefix, don't check others

            if len(iters_found) >= max_attempts:
                # Fully covered — copy all iterations to output
                iter_resolved = {}
                for n, (tp, rp, resolved) in enumerate(iters_found[:max_attempts]):
                    dest_traj_dir = self.output_dir / self.benchmark / "trajectories" / dest_phase / iid
                    dest_result_dir = self.output_dir / self.benchmark / "results" / dest_phase / iid
                    dest_traj_dir.mkdir(parents=True, exist_ok=True)
                    dest_result_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tp, dest_traj_dir / f"iter_{n}.json")
                    shutil.copy2(rp, dest_result_dir / f"iter_{n}.json")
                    iter_resolved[n] = resolved

                loaded[iid] = any(iter_resolved.values())
                loaded_iter_details[iid] = iter_resolved
                logger.debug(
                    f"[VAL_BASELINE] {iid}: reused {len(iters_found)} iterations from baseline"
                )
            else:
                # Not enough iterations — needs fresh run
                if iters_found:
                    logger.debug(
                        f"[VAL_BASELINE] {iid}: only {len(iters_found)}/{max_attempts} "
                        f"iterations in baseline, will re-run"
                    )
                missing.append(inst)

        return loaded, loaded_iter_details, missing


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
