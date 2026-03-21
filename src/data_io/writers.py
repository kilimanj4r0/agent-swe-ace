# src/io/writers.py
"""Data saving functions."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


def extract_benchmark_name(dataset: str) -> str:
    """
    Extract benchmark name from dataset string.

    "princeton-nlp/SWE-bench_Lite" -> "princeton-nlp__SWE-bench_Lite"
    """
    name = dataset.replace("/", "__")
    return name


def get_run_dir(base_dir: Path, timestamp: Optional[str] = None) -> Path:
    """
    Get run directory path with timestamp.

    Args:
        base_dir: Base data directory
        timestamp: Optional timestamp string (default: now)

    Returns:
        Path to run directory
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"run_{timestamp}"


def save_trajectory(
    trajectory: Dict,
    run_dir: Path,
    benchmark: str,
    instance_id: str,
    iteration: int,
) -> Path:
    """
    Save an agent trajectory to JSON file.

    Args:
        trajectory: Trajectory dict with 'info' and 'messages'
        run_dir: Run directory (e.g., data/run_20260319_143052)
        benchmark: Benchmark name (e.g., "swebench-lite")
        instance_id: SWE-bench instance ID
        iteration: Iteration number (0-indexed)

    Returns:
        Path to saved file
    """
    output_dir = run_dir / benchmark / "trajectories" / instance_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"iter_{iteration}.json"

    with open(output_path, "w") as f:
        json.dump(trajectory, f, indent=2, default=str)

    logger.debug(f"Saved trajectory to {output_path}")
    return output_path


def save_skillbook(
    skillbook: "Skillbook",
    run_dir: Path,
    benchmark: str,
    iteration: int,
    instance_id: Optional[str] = None,
) -> Path:
    """
    Save a skillbook to JSON file.

    Args:
        skillbook: Skillbook instance
        run_dir: Run directory
        benchmark: Benchmark name
        iteration: Iteration number (0-indexed)
        instance_id: Optional instance ID for per-instance mode

    Returns:
        Path to saved file
    """
    if instance_id:
        # Per-instance mode
        output_dir = run_dir / benchmark / "skillbooks" / instance_id
    else:
        # Per-run mode
        output_dir = run_dir / benchmark / "skillbooks"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"iter_{iteration}.json"

    # Convert skillbook to dict
    skills_list = skillbook.skills()
    data = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "instance_id": instance_id,
        "skill_count": len(skills_list),
        "skills": {},
    }

    for skill in skills_list:
        data["skills"][skill.id] = {
            "id": skill.id,
            "section": getattr(skill, "section", "general"),
            "justification": getattr(skill, "justification", None),
            "evidence": getattr(skill, "evidence", None),
            "content": getattr(skill, "content", ""),
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.debug(f"Saved skillbook ({len(skillbook.skills())} skills) to {output_path}")
    return output_path


def save_result(
    result: Dict[str, Any],
    run_dir: Path,
    benchmark: str,
    instance_id: str,
    iteration: int,
) -> Path:
    """
    Save an evaluation result to JSON file.

    Args:
        result: Result dict with resolved, feedback, metrics, etc.
        run_dir: Run directory
        benchmark: Benchmark name
        instance_id: SWE-bench instance ID
        iteration: Iteration number (0-indexed)

    Returns:
        Path to saved file
    """
    output_dir = run_dir / benchmark / "results" / instance_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"iter_{iteration}.json"

    # Add metadata
    result["instance_id"] = instance_id
    result["iteration"] = iteration
    result["timestamp"] = datetime.now().isoformat()

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.debug(f"Saved result to {output_path}")
    return output_path


def save_config(config: Dict, run_dir: Path) -> Path:
    """
    Save config for the run.

    Args:
        config: Configuration dict
        run_dir: Run directory

    Returns:
        Path to saved file
    """
    output_path = run_dir / "config.json"
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    logger.debug(f"Saved config to {output_path}")
    return output_path


def save_statistics(
    statistics: Dict,
    run_dir: Path,
) -> Path:
    """
    Save statistics for the run.

    Args:
        statistics: Statistics dict (see format below)
        run_dir: Run directory

    Returns:
        Path to saved file

    Statistics format:
    {
        "run_name": "run_20260319_143052",
        "benchmark": "swebench-lite",
        "total_instances": 300,
        "resolved_count": 45,
        "unresolved_count": 255,
        "resolution_rate": 0.15,
        "resolved_ids": [...],
        "unresolved_ids": [...],
        "per_iteration": {
            "0": {"resolved": 30, "avg_trajectory_length": 45.2, "skills_count": 0},
            "1": {"resolved": 15, "avg_trajectory_length": 38.7, "skills_count": 25}
        },
        "total_skills_learned": 25,
        "skill_ids": [...]
    }
    """
    output_path = run_dir / "statistics.json"
    with open(output_path, "w") as f:
        json.dump(statistics, f, indent=2, default=str)
    logger.info(f"Saved statistics to {output_path}")
    return output_path
