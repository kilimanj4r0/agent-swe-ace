# src/io/readers.py
"""Data loading functions."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger


def extract_benchmark_name(dataset: str) -> str:
    """
    Extract benchmark name from dataset string.

    "princeton-nlp/SWE-bench_Lite" -> "princeton-nlp__SWE-bench_Lite"

    Args:
        dataset: Full dataset name from config

    Returns:
        Normalized benchmark name
    """
    name = dataset.replace("/", "__")
    return name


def load_instance(source: Union[Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load a SWE-bench instance from file or dict.

    Args:
        source: Path to JSON file or instance dict

    Returns:
        Instance dictionary with instance_id, repo, problem_statement, etc.
    """
    if isinstance(source, dict):
        return source

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Instance file not found: {source}")

    with open(source) as f:
        instance = json.load(f)

    logger.debug(f"Loaded instance: {instance.get('instance_id', 'unknown')}")
    return instance


def load_skillbook(source: Optional[Union[Path, str, Dict]]) -> "Skillbook":
    """
    Load a skillbook from file, dict, or create empty.

    Args:
        source: Path to JSON file, skillbook dict, or None for empty

    Returns:
        Skillbook instance
    """
    from ace_next import Skillbook, Skill

    skillbook = Skillbook()

    if source is None:
        logger.debug("Created empty skillbook")
        return skillbook

    if isinstance(source, Skillbook):
        return source

    # Load from dict or file
    if isinstance(source, dict):
        data = source
    else:
        source = Path(source)
        if not source.exists():
            logger.warning(f"Skillbook file not found: {source}, using empty")
            return skillbook
        with open(source) as f:
            data = json.load(f)

    # Populate skillbook from data
    for skill_id, skill_data in data.get("skills", {}).items():
        skill = Skill(
            id=skill_data["id"],
            section=skill_data.get("section", "general"),
            content=skill_data.get("content", ""),
            justification=skill_data.get("justification"),
            evidence=skill_data.get("evidence"),
        )
        skillbook._skills[skill_id] = skill

    logger.debug(f"Loaded skillbook with {len(skillbook.skills())} skills")
    return skillbook


def load_trajectory(source: Union[Path, Dict]) -> Dict:
    """
    Load an agent trajectory from file or dict.

    Args:
        source: Path to JSON file or trajectory dict

    Returns:
        Trajectory dict with 'info' and 'messages' keys
    """
    if isinstance(source, dict):
        return source

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Trajectory file not found: {source}")

    with open(source) as f:
        trajectory = json.load(f)

    logger.debug(f"Loaded trajectory with {len(trajectory.get('messages', []))} messages")
    return trajectory


def load_results(run_dir: Path, benchmark: str) -> Dict[str, Dict]:
    """
    Load all results for a run.

    Args:
        run_dir: Path to run directory
        benchmark: Benchmark name

    Returns:
        Dict mapping instance_id to result dict (latest iteration)
    """
    results = {}
    results_dir = run_dir / benchmark / "results"

    if not results_dir.exists():
        return results

    for instance_dir in results_dir.iterdir():
        if not instance_dir.is_dir():
            continue
        # Get latest iteration
        iter_files = sorted(instance_dir.glob("iter_*.json"))
        if iter_files:
            with open(iter_files[-1]) as f:
                result = json.load(f)
            instance_id = instance_dir.name
            results[instance_id] = result

    logger.debug(f"Loaded {len(results)} results from {run_dir}")
    return results


def load_statistics(run_dir: Path) -> Optional[Dict]:
    """
    Load statistics for a run.

    Args:
        run_dir: Path to run directory

    Returns:
        Statistics dict or None
    """
    stats_file = run_dir / "statistics.json"
    if not stats_file.exists():
        return None

    with open(stats_file) as f:
        return json.load(f)
