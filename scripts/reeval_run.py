#!/usr/bin/env python3
"""Re-evaluate existing run results to fix cached/invalid evaluation outcomes.

Supports both flat runs and split-mode runs (train/val/val_baseline phases).

Usage:
    uv run python scripts/reeval_run.py data/run_test_baseline
    uv run python scripts/reeval_run.py data/run_test_baseline --dry-run
    uv run python scripts/reeval_run.py data/run_test_baseline --skip-docker
    uv run python scripts/reeval_run.py data/run_test_baseline --filter django__django-14752
    uv run python scripts/reeval_run.py data/run_test_baseline --workers 4
    uv run python scripts/reeval_run.py data/run_20260429_111748_completed_global_split_default
    uv run python scripts/reeval_run.py data/run_20260429_131817_completed_django_split_default --skip-docker
"""

import argparse
import concurrent.futures
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

from loguru import logger

# Ensure src/ is on the path for evaluation imports
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, _SRC_DIR)


def _find_benchmark_dir(run_dir: Path) -> Path:
    for sub in run_dir.iterdir():
        if sub.is_dir() and "__" in sub.name:
            return sub
    raise FileNotFoundError(f"No benchmark dir found in {run_dir}")


def _is_valid_patch(patch: str) -> bool:
    stripped = patch.lstrip()
    if stripped.startswith("diff --git"):
        return True
    if (
        re.search(r"^--- ", stripped, re.MULTILINE)
        and re.search(r"^\+\+\+ ", stripped, re.MULTILINE)
        and re.search(r"^@@ ", stripped, re.MULTILINE)
    ):
        return True
    return False


def _load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def _load_project_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_eval_config(project_config: dict) -> dict:
    """Read eval settings from config.yaml (current machine setup)."""
    return {
        "timeout": project_config.get("evaluation", {}).get("timeout", 1800),
        "namespace": project_config.get("environment", {}).get("namespace"),
        "rm_image": project_config.get("evaluation", {}).get("rm_image", True),
    }


def _discover_phases(bench_dir: Path) -> list[str | None]:
    """Discover phase subdirectories under results/.

    Returns [None] for flat runs, or ['train', 'val', 'val_baseline', ...] for split runs.
    """
    results_dir = bench_dir / "results"
    if not results_dir.exists():
        return [None]

    # Check if results/ contains phase subdirs (split mode)
    known_phases = {"train", "val", "val_baseline"}
    found = []
    for sub in sorted(results_dir.iterdir()):
        if sub.is_dir() and sub.name in known_phases:
            found.append(sub.name)

    if found:
        return found
    return [None]


def _find_iterations(bench_dir: Path, kind: str, instance_id: str, phase: str | None = None) -> list[int]:
    if phase:
        inst_dir = bench_dir / kind / phase / instance_id
    else:
        inst_dir = bench_dir / kind / instance_id
    if not inst_dir.exists():
        return []
    return sorted(
        int(m.group(1))
        for f in inst_dir.iterdir()
        if (m := re.match(r"iter_(\d+).json", f.name))
    )


def load_patch(bench_dir: Path, instance_id: str, iteration: int, phase: str | None = None) -> str | None:
    if phase:
        traj_path = bench_dir / "trajectories" / phase / instance_id / f"iter_{iteration}.json"
    else:
        traj_path = bench_dir / "trajectories" / instance_id / f"iter_{iteration}.json"
    if not traj_path.exists():
        return None
    with open(traj_path) as f:
        traj = json.load(f)
    return traj.get("info", {}).get("submission", "")


def load_result(bench_dir: Path, instance_id: str, iteration: int, phase: str | None = None) -> dict | None:
    if phase:
        result_path = bench_dir / "results" / phase / instance_id / f"iter_{iteration}.json"
    else:
        result_path = bench_dir / "results" / instance_id / f"iter_{iteration}.json"
    if not result_path.exists():
        return None
    with open(result_path) as f:
        return json.load(f)


def write_result(result_path: Path, result: dict):
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def cleanup_higher_iterations(
    bench_dir: Path, instance_id: str, resolved_iter: int,
    phase: str | None = None, dry_run: bool = False,
):
    for kind in ("trajectories", "results", "skillbooks"):
        if phase:
            inst_dir = bench_dir / kind / phase / instance_id
        else:
            inst_dir = bench_dir / kind / instance_id
        if not inst_dir.exists():
            continue
        for f in sorted(inst_dir.iterdir()):
            m = re.match(r"iter_(\d+).json", f.name)
            if m and int(m.group(1)) > resolved_iter:
                if dry_run:
                    logger.info(f"  [dry-run] Would delete {f}")
                else:
                    logger.info(f"  Deleting {f} (iter > {resolved_iter} resolved)")
                    f.unlink()


def _count_results_in_dir(results_dir: Path) -> tuple[list[str], list[str]]:
    """Count resolved/unresolved instances from a results directory."""
    resolved = []
    unresolved = []
    if not results_dir.exists():
        return resolved, unresolved
    for inst_dir in sorted(results_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        iter_files = sorted(inst_dir.glob("iter_*.json"))
        if not iter_files:
            continue
        with open(iter_files[-1]) as f:
            r = json.load(f)
        iid = inst_dir.name
        if r.get("resolved"):
            resolved.append(iid)
        else:
            unresolved.append(iid)
    return resolved, unresolved


def update_statistics(run_dir: Path, bench_dir: Path):
    stats_path = run_dir / "statistics.json"
    if not stats_path.exists():
        logger.warning("No statistics.json found, skipping update")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    phases = _discover_phases(bench_dir)

    if phases == [None]:
        # Flat run — update top-level counts
        resolved_ids, unresolved_ids = _count_results_in_dir(bench_dir / "results")
        stats["resolved_count"] = len(resolved_ids)
        stats["unresolved_count"] = len(unresolved_ids)
        total = len(resolved_ids) + len(unresolved_ids)
        stats["total_instances"] = total
        stats["processed_instances"] = total
        stats["resolution_rate"] = len(resolved_ids) / total if total else 0.0
        stats["resolved_ids"] = resolved_ids
        stats["unresolved_ids"] = unresolved_ids
        logger.info(f"Updated statistics.json: {len(resolved_ids)} resolved, {len(unresolved_ids)} unresolved")
    else:
        # Split run — update per-phase counts
        for phase in phases:
            results_dir = bench_dir / "results" / phase
            resolved_ids, unresolved_ids = _count_results_in_dir(results_dir)
            total = len(resolved_ids) + len(unresolved_ids)
            rate = len(resolved_ids) / total if total else 0.0

            phase_key = {
                "train": "train_phase",
                "val": "val_skillbook_phase",
                "val_baseline": "val_baseline_phase",
            }.get(phase, f"{phase}_phase")

            if phase_key in stats:
                stats[phase_key]["resolved_count"] = len(resolved_ids)
                stats[phase_key]["unresolved_count"] = len(unresolved_ids)
                stats[phase_key]["total_instances"] = total
                stats[phase_key]["resolution_rate"] = rate
                stats[phase_key]["resolved_ids"] = resolved_ids
                stats[phase_key]["unresolved_ids"] = unresolved_ids
                logger.info(f"  {phase}: {len(resolved_ids)} resolved, {len(unresolved_ids)} unresolved (rate={rate:.3f})")

            # Also update top-level from train phase
            if phase == "train":
                stats["resolved_count"] = len(resolved_ids)
                stats["unresolved_count"] = len(unresolved_ids)
                stats["total_instances"] = total
                stats["processed_instances"] = total
                stats["resolution_rate"] = rate
                stats["resolved_ids"] = resolved_ids
                stats["unresolved_ids"] = unresolved_ids

        # Recalculate summary if present
        if "summary" in stats:
            train_rate = stats.get("train_phase", {}).get("resolution_rate", 0)
            val_bl_rate = stats.get("val_baseline_phase", {}).get("resolution_rate", 0)
            val_sb_rate = stats.get("val_skillbook_phase", {}).get("resolution_rate", 0)
            improvement = val_sb_rate - val_bl_rate
            stats["summary"]["train_resolution_rate"] = train_rate
            stats["summary"]["val_baseline_resolution_rate"] = val_bl_rate
            stats["summary"]["val_skillbook_resolution_rate"] = val_sb_rate
            stats["summary"]["skillbook_improvement"] = f"{improvement:+.3f}"
            if val_bl_rate > 0:
                pct = improvement / val_bl_rate * 100
                stats["summary"]["skillbook_improvement_pct"] = f"{pct:+.1f}%"
            else:
                stats["summary"]["skillbook_improvement_pct"] = "N/A"

            # Recalculate newly_resolved_by_skillbook and lost_by_skillbook
            val_bl_resolved = set(stats.get("val_baseline_phase", {}).get("resolved_ids", []))
            val_sb_resolved = set(stats.get("val_skillbook_phase", {}).get("resolved_ids", []))
            stats["summary"]["newly_resolved_by_skillbook"] = sorted(val_sb_resolved - val_bl_resolved)
            stats["summary"]["lost_by_skillbook"] = sorted(val_bl_resolved - val_sb_resolved)

        logger.info(f"Updated statistics.json (split run, phases: {phases})")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)


def audit_run(bench_dir: Path) -> dict:
    """Classify all instances across all phases. Returns dict with lists per category."""
    categories = {
        "immediate_fix": [],    # resolved + invalid patch
        "needs_docker": [],     # valid patch (resolved or unresolved)
        "correct_skip": [],     # invalid/empty + unresolved
    }

    phases = _discover_phases(bench_dir)

    for phase in phases:
        results_dir = bench_dir / "results"
        if phase:
            results_dir = results_dir / phase

        if not results_dir.exists():
            continue

        for inst_dir in sorted(results_dir.iterdir()):
            if not inst_dir.is_dir():
                continue
            instance_id = inst_dir.name
            iters = _find_iterations(bench_dir, "results", instance_id, phase=phase)
            if not iters:
                continue
            iter_0 = iters[0]

            result = load_result(bench_dir, instance_id, iter_0, phase=phase)
            if result is None:
                continue
            resolved = result.get("resolved", False)

            patch = load_patch(bench_dir, instance_id, iter_0, phase=phase) or ""
            phase_label = phase or "default"
            if not patch.strip():
                categories["correct_skip"].append((instance_id, iter_0, "empty_patch", phase))
            elif not _is_valid_patch(patch):
                if resolved:
                    categories["immediate_fix"].append((instance_id, iter_0, patch[:80], phase))
                else:
                    categories["correct_skip"].append((instance_id, iter_0, "invalid_patch", phase))
            else:
                categories["needs_docker"].append((instance_id, iter_0, resolved, phase))

    return categories


def print_audit(categories: dict):
    print("\n" + "=" * 70)
    print("AUDIT RESULTS")
    print("=" * 70)
    print(f"  Immediate fixes (resolved + invalid patch): {len(categories['immediate_fix'])}")
    for iid, it, patch_preview, phase in categories["immediate_fix"]:
        phase_tag = f" [{phase}]" if phase else ""
        print(f"    {iid}{phase_tag} (iter_{it}): {repr(patch_preview)}")
    print(f"  Needs Docker re-eval (valid patch):         {len(categories['needs_docker'])}")
    resolved_docker = sum(1 for _, _, r, _ in categories["needs_docker"] if r)
    unresolved_docker = sum(1 for _, _, r, _ in categories["needs_docker"] if not r)
    print(f"    Currently resolved: {resolved_docker}, unresolved: {unresolved_docker}")
    # Per-phase breakdown
    phases_seen = sorted(set(p for _, _, _, p in categories["needs_docker"] if p))
    if phases_seen:
        for ph in phases_seen:
            ph_items = [(iid, it, r) for iid, it, r, p in categories["needs_docker"] if p == ph]
            ph_res = sum(1 for _, _, r in ph_items if r)
            ph_unres = sum(1 for _, _, r in ph_items if not r)
            print(f"    {ph}: {len(ph_items)} instances ({ph_res} resolved, {ph_unres} unresolved)")
    print(f"  Already correct (skip):                     {len(categories['correct_skip'])}")
    print()


def _write_eval_result(
    bench_dir: Path, instance_id: str, iteration: int,
    resolved: bool, patch: str, phase: str | None = None, dry_run: bool = False,
):
    """Write Docker eval result to disk and cleanup higher iterations."""
    if phase:
        result_path = bench_dir / "results" / phase / instance_id / f"iter_{iteration}.json"
    else:
        result_path = bench_dir / "results" / instance_id / f"iter_{iteration}.json"
    if resolved:
        feedback = "Patch resolved all tests successfully! Re-evaluated from SWE-bench cache fix."
    else:
        feedback = "Patch did not resolve the issue. Re-evaluated from SWE-bench cache fix."

    new_result = {
        "resolved": resolved,
        "feedback": feedback,
        "metrics": {"resolved": 1.0 if resolved else 0.0, "patch_length": len(patch)},
        "patch": patch[:1000] + "..." if len(patch) > 1000 else patch,
        "reevaluated": True,
        "instance_id": instance_id,
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
    }

    if dry_run:
        logger.info(f"  [dry-run] Would rewrite {result_path}")
    else:
        write_result(result_path, new_result)

    if resolved:
        cleanup_higher_iterations(bench_dir, instance_id, iteration, phase=phase, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Multiprocessing worker — runs in a separate process, bypasses the
# threading.Lock in swebench.py (each process gets its own copy).
# ---------------------------------------------------------------------------

def _eval_worker(args_tuple):
    """Worker for parallel Docker evaluation. Returns (instance_id, resolved, error_str)."""
    instance_id, instance_dict, patch, timeout, rm_image, output_dir_str, namespace = args_tuple
    import sys as _sys
    from pathlib import Path as _Path
    if _SRC_DIR not in _sys.path:
        _sys.path.insert(0, _SRC_DIR)
    from evaluation.swebench import validate_patch_docker
    try:
        resolved = validate_patch_docker(
            instance=instance_dict,
            patch=patch,
            timeout=timeout,
            rm_image=rm_image,
            output_dir=_Path(output_dir_str),
            namespace=namespace,
        )
        return instance_id, resolved, None
    except Exception as e:
        return instance_id, False, str(e)


def _run_docker_sequential(work_items, bench_dir, instance_map, eval_config, run_dir, dry_run):
    """Sequential Docker evaluation (workers <= 1)."""
    total = len(work_items)
    for idx, (instance_id, iteration, was_resolved, phase) in enumerate(work_items):
        patch = load_patch(bench_dir, instance_id, iteration, phase=phase) or ""
        label = f"[{idx + 1}/{total}]"
        phase_tag = f" [{phase}]" if phase else ""
        prev = "resolved" if was_resolved else "unresolved"
        logger.info(f"{label} {instance_id}{phase_tag} (iter_{iteration}, was {prev})")

        if dry_run:
            logger.info(f"  [dry-run] Would Docker-eval patch ({len(patch)} chars)")
            continue

        if instance_id not in instance_map:
            logger.warning(f"  Instance {instance_id} not in dataset, skipping")
            continue

        try:
            from evaluation.swebench import validate_patch_docker
            resolved = validate_patch_docker(
                instance=instance_map[instance_id],
                patch=patch,
                timeout=eval_config["timeout"],
                rm_image=eval_config["rm_image"],
                output_dir=run_dir,
                namespace=eval_config["namespace"],
            )
        except Exception as e:
            logger.error(f"  Docker eval failed for {instance_id}: {e}")
            resolved = False

        _write_eval_result(bench_dir, instance_id, iteration, resolved, patch, phase=phase, dry_run=dry_run)
        logger.info(f"  Result: {'RESOLVED' if resolved else 'not resolved'}")


def _run_docker_parallel(work_items, bench_dir, instance_map, eval_config, run_dir, workers, dry_run):
    """Parallel Docker evaluation using ProcessPoolExecutor."""
    total = len(work_items)
    logger.info(f"Using {workers} parallel workers for Docker evaluation")

    # Pre-load patches in main process (avoid file I/O in workers)
    patches = {}
    for instance_id, iteration, _, phase in work_items:
        patches[instance_id] = load_patch(bench_dir, instance_id, iteration, phase=phase) or ""

    # Build work args for each instance
    worker_args = [
        (
            instance_id,
            instance_map[instance_id],
            patches[instance_id],
            eval_config["timeout"],
            eval_config["rm_image"],
            str(run_dir),
            eval_config["namespace"],
        )
        for instance_id, iteration, was_resolved, phase in work_items
        if instance_id in instance_map
    ]

    # Map future -> (instance_id, iteration, was_resolved, phase)
    meta = {iid: (it, wr, ph) for iid, it, wr, ph in work_items}

    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_eval_worker, args): args[0] for args in worker_args}
        try:
            for future in concurrent.futures.as_completed(futures):
                instance_id, resolved, error = future.result()
                iteration, was_resolved, phase = meta[instance_id]
                done += 1
                phase_tag = f" [{phase}]" if phase else ""
                tag = f"[{done}/{total}]"

                if error:
                    logger.error(f"{tag} {instance_id}{phase_tag}: ERROR - {error}")
                else:
                    prev = "resolved" if was_resolved else "unresolved"
                    status = "RESOLVED" if resolved else "not resolved"
                    logger.info(f"{tag} {instance_id}{phase_tag} (was {prev}): {status}")

                if not dry_run:
                    _write_eval_result(
                        bench_dir, instance_id, iteration, resolved,
                        patches[instance_id], phase=phase, dry_run=False,
                    )
        except KeyboardInterrupt:
            logger.warning("Interrupted, shutting down workers...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise


def _process_run(
    run_dir: Path, project_config: dict, eval_config: dict, workers: int,
    filter_ids: list[str] | None, dry_run: bool, skip_docker: bool,
    instance_map: dict | None,
):
    """Process a single run directory."""
    bench_dir = _find_benchmark_dir(run_dir)
    benchmark_name = bench_dir.name
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Benchmark: {benchmark_name}")

    phases = _discover_phases(bench_dir)
    if phases != [None]:
        logger.info(f"Split-mode run detected, phases: {phases}")

    run_config = _load_run_config(run_dir)

    # Phase 1: Audit
    categories = audit_run(bench_dir)
    print_audit(categories)

    # Apply filter
    if filter_ids:
        filter_set = set(filter_ids)
        categories["immediate_fix"] = [
            (iid, it, p, ph) for iid, it, p, ph in categories["immediate_fix"] if iid in filter_set
        ]
        categories["needs_docker"] = [
            (iid, it, r, ph) for iid, it, r, ph in categories["needs_docker"] if iid in filter_set
        ]
        logger.info(f"Filtered to {len(filter_ids)} instance(s)")

    if not categories["immediate_fix"] and not categories["needs_docker"]:
        print("Nothing to do.")
        return

    # Phase 2: Fix immediate wins
    if categories["immediate_fix"]:
        print(f"\n{'[dry-run] ' if dry_run else ''}Fixing {len(categories['immediate_fix'])} immediate wins...")
        for instance_id, iteration, patch_preview, phase in categories["immediate_fix"]:
            if phase:
                result_path = bench_dir / "results" / phase / instance_id / f"iter_{iteration}.json"
            else:
                result_path = bench_dir / "results" / instance_id / f"iter_{iteration}.json"
            new_result = {
                "resolved": False,
                "feedback": "Patch is not a valid diff format (expected diff --git or unified diff with hunks). Re-evaluated: was false positive from SWE-bench cache.",
                "metrics": {"resolved": 0.0, "patch_invalid_format": 1.0},
                "reevaluated": True,
                "instance_id": instance_id,
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
            }
            if dry_run:
                logger.info(f"  [dry-run] Would rewrite {result_path}")
            else:
                logger.info(f"  Fixing {instance_id} (iter_{iteration}): resolved=true -> false")
                write_result(result_path, new_result)
            cleanup_higher_iterations(bench_dir, instance_id, iteration, phase=phase, dry_run=dry_run)

    # Phase 3: Docker re-evaluation
    if categories["needs_docker"] and not skip_docker:
        print(f"\n{'[dry-run] ' if dry_run else ''}Docker re-evaluating {len(categories['needs_docker'])} instances...")

        if not dry_run and instance_map is None:
            from datasets import load_dataset

            logger.info("Loading SWE-bench dataset...")
            dataset = load_dataset(
                run_config.get("benchmark", {}).get("dataset", "princeton-nlp/SWE-bench_Lite"),
                split=run_config.get("benchmark", {}).get("split", "test"),
            )
            instance_map = {inst["instance_id"]: dict(inst) for inst in dataset}
            logger.info(f"Loaded {len(instance_map)} instances from dataset")

        work_items = [(iid, it, wr, ph) for iid, it, wr, ph in categories["needs_docker"]]

        if dry_run:
            for idx, (instance_id, iteration, was_resolved, phase) in enumerate(work_items):
                patch = load_patch(bench_dir, instance_id, iteration, phase=phase) or ""
                label = f"[{idx + 1}/{len(work_items)}]"
                phase_tag = f" [{phase}]" if phase else ""
                prev = "resolved" if was_resolved else "unresolved"
                logger.info(f"{label} {instance_id}{phase_tag} (iter_{iteration}, was {prev})")
                logger.info(f"  [dry-run] Would Docker-eval patch ({len(patch)} chars)")
        elif workers <= 1:
            _run_docker_sequential(work_items, bench_dir, instance_map, eval_config, run_dir, dry_run=False)
        else:
            _run_docker_parallel(work_items, bench_dir, instance_map, eval_config, run_dir, workers, dry_run=False)

    elif categories["needs_docker"] and skip_docker:
        print(f"\nSkipping Docker re-eval ({len(categories['needs_docker'])} instances). Use without --skip-docker to re-evaluate.")

    # Phase 5: Update statistics
    if not dry_run:
        logger.info("Updating statistics.json...")
        update_statistics(run_dir, bench_dir)
    else:
        logger.info("[dry-run] Would update statistics.json")


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate existing run results")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Path(s) to run directory")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--skip-docker", action="store_true", help="Only fix immediate wins, skip Docker eval")
    parser.add_argument("--filter", action="append", help="Only re-evaluate specific instance(s)")
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"),
        help="Project config file for eval/env settings (default: config.yaml)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel Docker eval workers (default: from config.yaml experiment.concurrency, or 1)",
    )
    args = parser.parse_args()

    for rd in args.run_dirs:
        if not rd.exists():
            print(f"Run dir not found: {rd}", file=sys.stderr)
            sys.exit(1)

    project_config = _load_project_config(args.config.resolve())
    eval_config = _get_eval_config(project_config)

    workers = args.workers
    if workers is None:
        workers = project_config.get("experiment", {}).get("concurrency", 1)
    logger.info(f"Eval config: timeout={eval_config['timeout']}s, namespace={eval_config['namespace']}, rm_image={eval_config['rm_image']}, workers={workers}")

    # Load dataset once for all runs (if Docker eval needed and not dry-run)
    instance_map = None
    if not args.dry_run and not args.skip_docker:
        from datasets import load_dataset
        # Load using config.yaml's benchmark settings
        dataset_name = project_config.get("benchmark", {}).get("dataset", "princeton-nlp/SWE-bench_Lite")
        dataset_split = project_config.get("benchmark", {}).get("split", "test")
        logger.info(f"Loading dataset: {dataset_name} ({dataset_split})...")
        dataset = load_dataset(dataset_name, split=dataset_split)
        instance_map = {inst["instance_id"]: dict(inst) for inst in dataset}
        logger.info(f"Loaded {len(instance_map)} instances from dataset")

    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        print(f"\n{'=' * 70}")
        print(f"Processing: {run_dir.name}")
        print(f"{'=' * 70}")
        _process_run(
            run_dir=run_dir,
            project_config=project_config,
            eval_config=eval_config,
            workers=workers,
            filter_ids=args.filter,
            dry_run=args.dry_run,
            skip_docker=args.skip_docker,
            instance_map=instance_map,
        )

    print("\nAll runs done.")


if __name__ == "__main__":
    main()
