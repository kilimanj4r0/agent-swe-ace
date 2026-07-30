#!/usr/bin/env python3
"""
Transform baseline trajectory data to run format.

This script transforms the baseline data from:
  trajectories/{instance_id}/{instance_id}.traj.json
  results/resolved_trajectories_.../{instance_id}__{instance_id}.traj.json

To the expected run format:
  trajectories/{instance_id}/iter_0.json
  results/{instance_id}/iter_0.json
  statistics.json
"""

import json
from datetime import datetime
from pathlib import Path


def get_instance_id_from_path(filename: str) -> str:
    """Extract instance_id from trajectory filename.

    Example: django__django-11039__django__django-11039.traj.json -> django__django-11039
    """
    # Remove .traj.json suffix
    name = filename.replace(".traj.json", "")
    # Split by __ and take first two parts (repo__issue-id)
    parts = name.split("__")
    if len(parts) >= 2:
        return f"{parts[0]}__{parts[1]}"
    return name


def transform_trajectory(input_path: Path, output_path: Path) -> dict:
    """Transform a trajectory file to the expected format.

    Returns the trajectory info for statistics.
    """
    with open(input_path) as f:
        data = json.load(f)

    # The baseline format already has 'info' and 'messages' keys
    # We just need to add iteration info if not present
    if "info" not in data:
        data["info"] = {}

    data["info"]["iteration"] = 0

    # Ensure the info has the expected fields
    info = data.get("info", {})
    if "exit_status" not in info:
        info["exit_status"] = "Submitted"
    if "submission" not in info:
        info["submission"] = ""

    # Write to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return info


def create_result_file(
    output_path: Path,
    instance_id: str,
    resolved: bool,
    patch: str = "",
    feedback: str = "",
) -> None:
    """Create a result file for an instance."""
    result = {
        "instance_id": instance_id,
        "iteration": 0,
        "timestamp": datetime.now().isoformat(),
        "resolved": resolved,
        "patch": patch,
        "feedback": feedback,
        "test_results": {
            "resolved": resolved,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def transform_baseline_to_run_format(
    baseline_dir: Path,
    output_dir: Path,
    run_name: str = "baseline_qwen3coder",
    benchmark: str = "princeton-nlp__SWE-bench_Lite",
) -> dict:
    """
    Transform baseline data to run format.

    Args:
        baseline_dir: Path to baseline data directory
        output_dir: Path to output run directory
        run_name: Name for the run
        benchmark: Benchmark name

    Returns:
        Statistics dict
    """
    benchmark_dir = baseline_dir / benchmark

    # Find resolved instances
    resolved_dir = benchmark_dir / "results"
    resolved_instances = set()

    # Find the resolved trajectories folder
    for folder in resolved_dir.iterdir():
        if folder.is_dir() and "resolved_trajectories" in folder.name:
            for traj_file in folder.glob("*.traj.json"):
                instance_id = get_instance_id_from_path(traj_file.name)
                resolved_instances.add(instance_id)

    print(f"Found {len(resolved_instances)} resolved instances")

    # Transform all trajectories
    trajectories_dir = benchmark_dir / "trajectories"
    all_instances = []
    resolved_ids = []
    unresolved_ids = []

    for instance_folder in sorted(trajectories_dir.iterdir()):
        if not instance_folder.is_dir():
            continue

        instance_id = instance_folder.name
        all_instances.append(instance_id)

        # Find the trajectory file
        traj_files = list(instance_folder.glob("*.traj.json"))
        if not traj_files:
            print(f"Warning: No trajectory file for {instance_id}")
            continue

        traj_file = traj_files[0]

        # Transform trajectory
        output_traj_path = output_dir / benchmark / "trajectories" / instance_id / "iter_0.json"
        info = transform_trajectory(traj_file, output_traj_path)

        # Determine if resolved
        is_resolved = instance_id in resolved_instances

        # Create result file
        output_result_path = output_dir / benchmark / "results" / instance_id / "iter_0.json"
        create_result_file(
            output_result_path,
            instance_id,
            resolved=is_resolved,
            patch=info.get("submission", ""),
            feedback="Baseline run" if is_resolved else "Baseline run - not resolved",
        )

        if is_resolved:
            resolved_ids.append(instance_id)
        else:
            unresolved_ids.append(instance_id)

        print(f"  {'✓' if is_resolved else '✗'} {instance_id}")

    # Calculate statistics
    total = len(all_instances)
    resolved_count = len(resolved_ids)
    resolution_rate = resolved_count / total if total > 0 else 0.0

    statistics = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "total_instances": total,
        "resolved_count": resolved_count,
        "unresolved_count": len(unresolved_ids),
        "resolution_rate": resolution_rate,
        "resolved_ids": sorted(resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "config": {
            "max_attempts": 1,
            "skillbook_mode": "per_instance",
            "baseline": True,
        },
    }

    # Save statistics
    stats_path = output_dir / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(statistics, f, indent=2, default=str)
    print(f"\nSaved statistics to {stats_path}")

    # Save config
    config = {
        "experiment": {
            "name": run_name,
            "description": "Baseline Qwen3-Coder-30B run (transformed)",
            "max_attempts": 1,
            "skillbook_mode": "per_instance",
        },
        "benchmark": {
            "dataset": "princeton-nlp/SWE-bench_Lite",
        },
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Saved config to {config_path}")

    return statistics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Transform baseline data to run format")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("data/run_baseline_qwen3coder"),
        help="Path to baseline data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/run_baseline_qwen3coder_formatted"),
        help="Path to output run directory",
    )
    parser.add_argument(
        "--run-name",
        default="baseline_qwen3coder_iter0",
        help="Name for the run",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Transform in place (modify baseline_dir directly)",
    )

    args = parser.parse_args()

    if args.in_place:
        output_dir = args.baseline_dir
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Transforming baseline data from: {args.baseline_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Run name: {args.run_name}")
    print()

    stats = transform_baseline_to_run_format(
        baseline_dir=args.baseline_dir,
        output_dir=output_dir,
        run_name=args.run_name,
    )

    print("\nTransformation complete!")
    print(f"Total: {stats['total_instances']}")
    print(f"Resolved: {stats['resolved_count']} ({stats['resolution_rate']:.1%})")
    print(f"Unresolved: {stats['unresolved_count']}")


if __name__ == "__main__":
    main()
