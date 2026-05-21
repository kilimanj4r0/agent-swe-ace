"""Resume directory scanning for experiment resumption.

Scans previous run directories to determine which instances are complete
and which need re-running, then supports resuming from the last successful
iteration.

Usage:
    uv run python -m src.data_io.resume_scanner -c configs/<override>.yaml
    uv run python -m src.data_io.resume_scanner -c config.yaml --resume-dir data/run_xxx
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class ResumePoint:
    """Resume state for a single instance."""

    resume_dir: Path
    """Source run directory with existing artifacts."""

    last_complete_iter: int
    """Index of the last fully complete iteration (-1 = nothing complete)."""

    is_fully_complete: bool
    """True if the instance is fully done (resolved or all attempts exhausted)."""

    break_reason: Optional[str] = None
    """Why the iteration chain broke (only set for partial instances)."""

    break_iteration: Optional[int] = None
    """Which iteration the chain broke at (only set for partial instances)."""

    @property
    def start_iteration(self) -> int:
        """Iteration to start from (0 if nothing complete)."""
        return self.last_complete_iter + 1 if not self.is_fully_complete else -1


# exit_status values that indicate the agent ran successfully
_GOOD_EXIT_STATUSES = {"Submitted", "LimitsExceeded"}


def _diagnose_break(
    resume_dir: Path, benchmark: str, instance_id: str, break_at: int
) -> Optional[str]:
    """Determine why the iteration chain broke at the given iteration."""
    traj_file = resume_dir / benchmark / "trajectories" / instance_id / f"iter_{break_at}.json"
    if not traj_file.exists():
        return "missing_trajectory"

    try:
        with open(traj_file) as f:
            traj_data = json.load(f)
    except Exception:
        return "load_error"

    exit_status = traj_data.get("info", {}).get("exit_status", "")
    if exit_status not in _GOOD_EXIT_STATUSES:
        return f"bad_exit_status: {exit_status}"

    result_file = resume_dir / benchmark / "results" / instance_id / f"iter_{break_at}.json"
    if not result_file.exists():
        return "missing_result"

    # Iteration itself was OK but skillbook for next iteration is missing
    return "missing_skillbook"


def scan_resume_state(
    resume_dir: Path,
    benchmark: str,
    instance_id: str,
    max_attempts: int,
    skip_learn: bool = False,
) -> Optional[ResumePoint]:
    """Check resume state for a single instance in a resume directory.

    Walks iterations in order (iter_0, iter_1, ...) looking for the longest
    chain of successful iterations from the start. The chain breaks at the
    first incomplete iteration.

    Returns None if the instance has no data in this resume directory.
    """
    traj_dir = resume_dir / benchmark / "trajectories" / instance_id
    if not traj_dir.is_dir():
        return None

    # Collect existing iteration files
    iter_files = sorted(traj_dir.glob("iter_*.json"))
    if not iter_files:
        return None

    last_complete = -1

    for k in range(max_attempts):
        traj_file = traj_dir / f"iter_{k}.json"
        if not traj_file.exists():
            break

        # Check trajectory exit_status
        try:
            with open(traj_file) as f:
                traj_data = json.load(f)
        except Exception:
            break

        exit_status = traj_data.get("info", {}).get("exit_status", "")
        if exit_status not in _GOOD_EXIT_STATUSES:
            break

        # Check result file
        result_file = resume_dir / benchmark / "results" / instance_id / f"iter_{k}.json"
        if not result_file.exists():
            break

        # Read result to check resolved
        try:
            with open(result_file) as f:
                result_data = json.load(f)
        except Exception:
            break

        resolved = result_data.get("resolved", False)

        if resolved:
            return ResumePoint(
                resume_dir=resume_dir,
                last_complete_iter=k,
                is_fully_complete=True,
            )

        # Not resolved — need skillbook for next iteration (unless skip_learn)
        if k < max_attempts - 1 and not skip_learn:
            skillbook_file = (
                resume_dir / benchmark / "skillbooks" / instance_id / f"iter_{k + 1}.json"
            )
            if not skillbook_file.exists():
                break

        last_complete = k

    # Chain walk ended. Before marking as partial/broken, check if any
    # result beyond the chain break shows resolved=True. The experiment
    # may have continued past a broken iteration and succeeded later.
    result_dir = resume_dir / benchmark / "results" / instance_id
    if result_dir.is_dir():
        for k in range(max_attempts):
            result_file = result_dir / f"iter_{k}.json"
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        if json.load(f).get("resolved", False):
                            return ResumePoint(
                                resume_dir=resume_dir,
                                last_complete_iter=k,
                                is_fully_complete=True,
                            )
                except Exception:
                    continue

    # Determine break reason from where the chain stopped
    break_iteration = last_complete + 1 if last_complete < max_attempts - 1 else None
    break_reason = None
    if break_iteration is not None:
        break_reason = _diagnose_break(resume_dir, benchmark, instance_id, break_iteration)

    # If we completed all max_attempts iterations without resolving
    is_complete = (last_complete == max_attempts - 1)

    return ResumePoint(
        resume_dir=resume_dir,
        last_complete_iter=last_complete,
        is_fully_complete=is_complete,
        break_reason=break_reason,
        break_iteration=break_iteration if not is_complete else None,
    )


def scan_resume_dirs(
    resume_dirs: List[Path],
    benchmark: str,
    instance_ids: List[str],
    max_attempts: int,
    skip_learn: bool = False,
) -> Dict[str, ResumePoint]:
    """Scan multiple resume directories and find the best resume point per instance.

    If an instance appears in multiple directories, the one with the highest
    last_complete_iter wins.
    """
    best: Dict[str, ResumePoint] = {}

    for resume_dir in resume_dirs:
        if not resume_dir.is_dir():
            logger.warning(f"Resume directory not found: {resume_dir}")
            continue

        for instance_id in instance_ids:
            rp = scan_resume_state(resume_dir, benchmark, instance_id, max_attempts, skip_learn=skip_learn)
            if rp is None:
                continue

            existing = best.get(instance_id)
            if existing is None or rp.last_complete_iter > existing.last_complete_iter:
                best[instance_id] = rp

    # Log summary
    complete = sum(1 for rp in best.values() if rp.is_fully_complete)
    partial = sum(1 for rp in best.values() if not rp.is_fully_complete and rp.last_complete_iter >= 0)
    broken = sum(1 for rp in best.values() if not rp.is_fully_complete and rp.last_complete_iter < 0)
    new_count = len(instance_ids) - len(best)
    logger.info(
        f"Resume scan: {len(best)} instances found in {len(resume_dirs)} dir(s) — "
        f"{complete} complete, {partial} partial, {broken} broken, "
        f"{new_count} new"
    )

    # Log broken instances (data exists but no iteration succeeded)
    broken_items = [(iid, rp) for iid, rp in best.items() if not rp.is_fully_complete and rp.last_complete_iter < 0]
    if broken_items:
        logger.info(f"Broken instances ({len(broken_items)}):")
        for iid, rp in sorted(broken_items):
            broke = f"iter_{rp.break_iteration}" if rp.break_iteration is not None else "?"
            logger.info(f"  {iid}: no successful iteration, broke at={broke} ({rp.break_reason})")

    # Log partial instance details
    partial_items = [(iid, rp) for iid, rp in best.items() if not rp.is_fully_complete and rp.last_complete_iter >= 0]
    if partial_items:
        logger.info(f"Partial instances ({len(partial_items)}):")
        for iid, rp in sorted(partial_items):
            last = f"iter_{rp.last_complete_iter}" if rp.last_complete_iter >= 0 else "none"
            broke = f"iter_{rp.break_iteration}" if rp.break_iteration is not None else "?"
            logger.info(f"  {iid}: last complete={last}, broke at={broke} ({rp.break_reason})")

    return best


def copy_instance_artifacts(
    source_dir: Path,
    dest_dir: Path,
    benchmark: str,
    instance_id: str,
    up_to_iter: int,
) -> None:
    """Copy trajectory, result, and skillbook files for iter_0..iter_{up_to_iter}.

    Also copies skillbook iter_{up_to_iter+1} if it exists (produced by learn
    phase after the last complete iteration).
    """
    for subdir in ("trajectories", "results", "skillbooks"):
        src_instance_dir = source_dir / benchmark / subdir / instance_id
        if not src_instance_dir.is_dir():
            continue

        dst_instance_dir = dest_dir / benchmark / subdir / instance_id
        dst_instance_dir.mkdir(parents=True, exist_ok=True)

        for f in src_instance_dir.iterdir():
            if not f.is_file() or f.suffix != ".json":
                continue

            # Check if this file is within the range we want to copy
            # iter_N.json where N <= up_to_iter for trajectories/results
            # iter_N.json where N <= up_to_iter+1 for skillbooks
            stem = f.stem  # e.g. "iter_0"
            if not stem.startswith("iter_"):
                continue

            try:
                iter_num = int(stem.split("_", 1)[1])
            except (ValueError, IndexError):
                continue

            max_iter = up_to_iter if subdir != "skillbooks" else up_to_iter + 1
            if iter_num <= max_iter:
                shutil.copy2(f, dst_instance_dir / f.name)

    logger.debug(f"[resume] Copied iter_0..iter_{up_to_iter} for {instance_id}")


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base (returns new dict)."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


if __name__ == "__main__":
    import argparse
    import sys

    import yaml

    parser = argparse.ArgumentParser(
        description="Scan resume state for experiment instances",
        usage="uv run python -m src.data_io.resume_scanner -c <config> [--resume-dir <path>]",
    )
    parser.add_argument("-c", "--config", required=True, help="Override config path (deep-merged on top of config.yaml)")
    parser.add_argument("--resume-dir", help="Resume directory (overrides config's resume_dirs)")
    args = parser.parse_args()

    # Load base config
    base_config_path = Path("config.yaml")
    if base_config_path.exists():
        with open(base_config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Deep-merge override config
    override_path = Path(args.config)
    if override_path.exists():
        with open(override_path) as f:
            override = yaml.safe_load(f) or {}
        config = _deep_merge(config, override)
    else:
        print(f"Config not found: {override_path}", file=sys.stderr)
        sys.exit(1)

    # Extract settings
    benchmark_dataset = config.get("benchmark", {}).get("dataset", "princeton-nlp/SWE-bench_Lite")
    benchmark_split = config.get("benchmark", {}).get("split", "test")
    max_attempts = config.get("experiment", {}).get("max_attempts", 4)
    exclude = set(config.get("benchmark", {}).get("exclude_instances") or [])

    # Resume dirs
    if args.resume_dir:
        resume_dirs = [Path(args.resume_dir)]
    else:
        resume_dirs_cfg = config.get("experiment", {}).get("resume_dirs", [])
        resume_dirs = [Path(d) for d in resume_dirs_cfg]

    if not resume_dirs:
        print("No resume_dirs in config and --resume-dir not provided", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    from datasets import load_dataset

    ds = load_dataset(benchmark_dataset, split=benchmark_split)
    instance_ids = sorted([x["instance_id"] for x in ds if x["instance_id"] not in exclude])

    # Resolve benchmark subdir name
    benchmark_subdir = benchmark_dataset.replace("/", "__")

    # Scan
    result = scan_resume_dirs(resume_dirs, benchmark_subdir, instance_ids, max_attempts)

    # Print summary table
    complete = {iid: rp for iid, rp in result.items() if rp.is_fully_complete}
    partial = {iid: rp for iid, rp in result.items() if not rp.is_fully_complete and rp.last_complete_iter >= 0}
    broken = {iid: rp for iid, rp in result.items() if not rp.is_fully_complete and rp.last_complete_iter < 0}
    no_data = [iid for iid in instance_ids if iid not in result]

    print(f"\n{'='*80}")
    print(f"Resume state: {benchmark_dataset} | max_attempts={max_attempts}")
    print(f"Dirs: {', '.join(str(d) for d in resume_dirs)}")
    print(f"{'='*80}")
    print(f"  Total: {len(instance_ids)} | Complete: {len(complete)} | Partial: {len(partial)} | Broken: {len(broken)} | New: {len(no_data)}")

    if broken:
        print(f"\n{'─'*80}")
        print(f"Broken instances ({len(broken)}): data exists but no iteration succeeded")
        print(f"{'─'*80}")
        print(f"  {'Instance':<55} {'Broke at':>9} {'Reason'}")
        print(f"  {'─'*53} {'─':>9} {'─'*20}")
        for iid in sorted(broken):
            rp = broken[iid]
            broke = f"iter_{rp.break_iteration}" if rp.break_iteration is not None else "?"
            print(f"  {iid:<55} {broke:>9} {rp.break_reason}")

    if partial:
        print(f"\n{'─'*80}")
        print(f"Partial instances ({len(partial)}):")
        print(f"{'─'*80}")
        print(f"  {'Instance':<45} {'Last OK':>8} {'Broke at':>9} {'Reason'}")
        print(f"  {'─'*43} {'─':>8} {'─':>9} {'─'*20}")
        for iid in sorted(partial):
            rp = partial[iid]
            last = f"iter_{rp.last_complete_iter}" if rp.last_complete_iter >= 0 else "-"
            broke = f"iter_{rp.break_iteration}" if rp.break_iteration is not None else "?"
            print(f"  {iid:<45} {last:>8} {broke:>9} {rp.break_reason}")

    if no_data:
        print(f"\n{'─'*80}")
        print(f"New instances ({len(no_data)}): no data found in resume dirs")

    print()
