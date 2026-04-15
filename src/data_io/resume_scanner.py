"""Resume directory scanning for experiment resumption.

Scans previous run directories to determine which instances are complete
and which need re-running, then supports resuming from the last successful
iteration.
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

    @property
    def start_iteration(self) -> int:
        """Iteration to start from (0 if nothing complete)."""
        return self.last_complete_iter + 1 if not self.is_fully_complete else -1


# exit_status values that indicate the agent ran successfully
_GOOD_EXIT_STATUSES = {"Submitted", "LimitsExceeded"}


def scan_resume_state(
    resume_dir: Path,
    benchmark: str,
    instance_id: str,
    max_attempts: int,
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
            # Instance is fully done — no more iterations needed
            last_complete = k
            return ResumePoint(
                resume_dir=resume_dir,
                last_complete_iter=last_complete,
                is_fully_complete=True,
            )

        # Not resolved — need skillbook for next iteration
        if k < max_attempts - 1:
            skillbook_file = (
                resume_dir / benchmark / "skillbooks" / instance_id / f"iter_{k + 1}.json"
            )
            if not skillbook_file.exists():
                break

        last_complete = k

    # If we completed all max_attempts iterations without resolving
    is_complete = (last_complete == max_attempts - 1)

    return ResumePoint(
        resume_dir=resume_dir,
        last_complete_iter=last_complete,
        is_fully_complete=is_complete,
    )


def scan_resume_dirs(
    resume_dirs: List[Path],
    benchmark: str,
    instance_ids: List[str],
    max_attempts: int,
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
            rp = scan_resume_state(resume_dir, benchmark, instance_id, max_attempts)
            if rp is None:
                continue

            existing = best.get(instance_id)
            if existing is None or rp.last_complete_iter > existing.last_complete_iter:
                best[instance_id] = rp

    # Log summary
    complete = sum(1 for rp in best.values() if rp.is_fully_complete)
    partial = sum(1 for rp in best.values() if not rp.is_fully_complete and rp.last_complete_iter >= 0)
    logger.info(
        f"Resume scan: {len(best)} instances found in {len(resume_dirs)} dir(s) — "
        f"{complete} complete, {partial} partial, "
        f"{len(instance_ids) - len(best)} new"
    )

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
