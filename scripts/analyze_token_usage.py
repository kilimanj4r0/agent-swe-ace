#!/usr/bin/env python3
"""Analyze token usage across SWE-bench trajectories.

This script analyzes per-instance token usage from trajectory files,
including prompt tokens, completion tokens, context growth patterns,
and differences between resolved and unresolved instances.
"""

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TokenMetrics:
    """Token usage metrics for a single instance."""
    instance_id: str
    exit_status: str
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    max_context_size: int = 0
    api_calls: int = 0
    context_sizes: list[int] = field(default_factory=list)
    prompt_tokens_per_step: list[int] = field(default_factory=list)
    completion_tokens_per_step: list[int] = field(default_factory=list)


def extract_token_metrics(trajectory_path: Path) -> Optional[TokenMetrics]:
    """Extract token metrics from a trajectory file.

    Args:
        trajectory_path: Path to the trajectory JSON file

    Returns:
        TokenMetrics object or None if parsing fails
    """
    try:
        with open(trajectory_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not parse {trajectory_path}: {e}")
        return None

    # Extract instance ID from filename or data
    instance_id = data.get("instance_id", trajectory_path.stem)

    # Get exit status
    exit_status = data.get("info", {}).get("exit_status", "Unknown")

    metrics = TokenMetrics(
        instance_id=instance_id,
        exit_status=exit_status,
    )

    # Process messages to extract token usage
    messages = data.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            extra = msg.get("extra", {})
            response = extra.get("response", {})
            usage = response.get("usage", {})

            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                metrics.total_prompt_tokens += prompt_tokens
                metrics.total_completion_tokens += completion_tokens
                metrics.total_tokens += total_tokens
                metrics.api_calls += 1
                metrics.max_context_size = max(metrics.max_context_size, prompt_tokens)

                metrics.context_sizes.append(prompt_tokens)
                metrics.prompt_tokens_per_step.append(prompt_tokens)
                metrics.completion_tokens_per_step.append(completion_tokens)

    # Fallback to model_stats if no message-level data found
    if metrics.api_calls == 0:
        model_stats = data.get("info", {}).get("model_stats", {})
        metrics.api_calls = model_stats.get("api_calls", 0)

    return metrics


def find_trajectory_files(data_dir: Path) -> list[Path]:
    """Find all trajectory files in the data directory.

    Args:
        data_dir: Base data directory

    Returns:
        List of trajectory file paths
    """
    trajectories = []

    # Pattern 1: *.traj.json files (resolved trajectories)
    for traj_file in data_dir.rglob("*.traj.json"):
        trajectories.append(traj_file)

    # Pattern 2: iter_0.json files in results directories
    for iter_file in data_dir.rglob("**/results/**/iter_0.json"):
        if iter_file not in trajectories:
            trajectories.append(iter_file)

    return trajectories


def analyze_trajectories(metrics_list: list[TokenMetrics]) -> dict:
    """Analyze token metrics across all trajectories.

    Args:
        metrics_list: List of TokenMetrics objects

    Returns:
        Dictionary with analysis results
    """
    if not metrics_list:
        return {}

    # Separate by exit status
    by_status: dict[str, list[TokenMetrics]] = {}
    for m in metrics_list:
        status = m.exit_status
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(m)

    # Calculate aggregate statistics
    all_prompt = [m.total_prompt_tokens for m in metrics_list]
    all_completion = [m.total_completion_tokens for m in metrics_list]
    all_total = [m.total_tokens for m in metrics_list]
    all_max_context = [m.max_context_size for m in metrics_list]
    all_api_calls = [m.api_calls for m in metrics_list]

    # Calculate context growth (ratio of final to initial context)
    context_growth_factors = []
    for m in metrics_list:
        if len(m.context_sizes) >= 2 and m.context_sizes[0] > 0:
            growth = m.context_sizes[-1] / m.context_sizes[0]
            context_growth_factors.append(growth)

    # Final context sizes
    final_context_sizes = [m.context_sizes[-1] for m in metrics_list if m.context_sizes]

    results = {
        "total_instances": len(metrics_list),
        "aggregate": {
            "total_prompt_tokens": sum(all_prompt),
            "total_completion_tokens": sum(all_completion),
            "total_tokens": sum(all_total),
            "avg_prompt_tokens": statistics.mean(all_prompt) if all_prompt else 0,
            "avg_completion_tokens": statistics.mean(all_completion) if all_completion else 0,
            "avg_total_tokens": statistics.mean(all_total) if all_total else 0,
            "avg_max_context": statistics.mean(all_max_context) if all_max_context else 0,
            "avg_api_calls": statistics.mean(all_api_calls) if all_api_calls else 0,
            "median_prompt_tokens": statistics.median(all_prompt) if all_prompt else 0,
            "median_completion_tokens": statistics.median(all_completion) if all_completion else 0,
            "std_prompt_tokens": statistics.stdev(all_prompt) if len(all_prompt) > 1 else 0,
            "std_completion_tokens": statistics.stdev(all_completion) if len(all_completion) > 1 else 0,
        },
        "context_growth": {
            "avg_final_context": statistics.mean(final_context_sizes) if final_context_sizes else 0,
            "median_final_context": statistics.median(final_context_sizes) if final_context_sizes else 0,
            "avg_growth_factor": statistics.mean(context_growth_factors) if context_growth_factors else 0,
            "median_growth_factor": statistics.median(context_growth_factors) if context_growth_factors else 0,
        },
        "by_exit_status": {},
    }

    # Calculate per-status statistics
    for status, status_metrics in by_status.items():
        status_prompt = [m.total_prompt_tokens for m in status_metrics]
        status_completion = [m.total_completion_tokens for m in status_metrics]
        status_max_context = [m.max_context_size for m in status_metrics]
        status_api_calls = [m.api_calls for m in status_metrics]

        results["by_exit_status"][status] = {
            "count": len(status_metrics),
            "avg_prompt_tokens": statistics.mean(status_prompt) if status_prompt else 0,
            "avg_completion_tokens": statistics.mean(status_completion) if status_completion else 0,
            "avg_max_context": statistics.mean(status_max_context) if status_max_context else 0,
            "avg_api_calls": statistics.mean(status_api_calls) if status_api_calls else 0,
            "median_prompt_tokens": statistics.median(status_prompt) if status_prompt else 0,
            "median_max_context": statistics.median(status_max_context) if status_max_context else 0,
        }

    return results


def format_number(n: int | float) -> str:
    """Format number with thousands separator."""
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


def print_analysis(results: dict) -> None:
    """Print analysis results in a formatted way.

    Args:
        results: Analysis results dictionary
    """
    print("=" * 60)
    print("Token Usage Analysis")
    print("=" * 60)
    print(f"\nTotal instances: {results['total_instances']}")

    agg = results["aggregate"]
    print("\n--- Aggregate Statistics ---")
    print(f"  Total prompt tokens:     {format_number(agg['total_prompt_tokens'])}")
    print(f"  Total completion tokens: {format_number(agg['total_completion_tokens'])}")
    print(f"  Total tokens:            {format_number(agg['total_tokens'])}")
    print(f"\n  Avg prompt tokens:       {format_number(agg['avg_prompt_tokens'])}")
    print(f"  Avg completion tokens:   {format_number(agg['avg_completion_tokens'])}")
    print(f"  Avg total tokens:        {format_number(agg['avg_total_tokens'])}")
    print(f"  Avg max context size:    {format_number(agg['avg_max_context'])}")
    print(f"  Avg API calls:           {format_number(agg['avg_api_calls'])}")
    print(f"\n  Median prompt tokens:    {format_number(agg['median_prompt_tokens'])}")
    print(f"  Median completion:       {format_number(agg['median_completion_tokens'])}")
    print(f"  Std dev prompt:          {format_number(agg['std_prompt_tokens'])}")
    print(f"  Std dev completion:      {format_number(agg['std_completion_tokens'])}")

    growth = results["context_growth"]
    print("\n--- Context Growth ---")
    print(f"  Avg final context size:  {format_number(growth['avg_final_context'])}")
    print(f"  Median final context:    {format_number(growth['median_final_context'])}")
    print(f"  Avg growth factor:       {growth['avg_growth_factor']:.2f}x")
    print(f"  Median growth factor:    {growth['median_growth_factor']:.2f}x")

    print("\n--- By Exit Status ---")
    for status, stats in results["by_exit_status"].items():
        print(f"\n  {status} ({stats['count']} instances):")
        print(f"    Avg prompt tokens:   {format_number(stats['avg_prompt_tokens'])}")
        print(f"    Avg completion:      {format_number(stats['avg_completion_tokens'])}")
        print(f"    Avg max context:     {format_number(stats['avg_max_context'])}")
        print(f"    Avg API calls:       {format_number(stats['avg_api_calls'])}")
        print(f"    Median prompt:       {format_number(stats['median_prompt_tokens'])}")
        print(f"    Median max context:  {format_number(stats['median_max_context'])}")

    print("\n" + "=" * 60)


def export_to_csv(metrics_list: list[TokenMetrics], output_path: Path) -> None:
    """Export metrics to CSV file.

    Args:
        metrics_list: List of TokenMetrics objects
        output_path: Path to output CSV file
    """
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance_id",
            "exit_status",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
            "max_context_size",
            "api_calls",
            "initial_context",
            "final_context",
            "context_growth_factor",
        ])

        for m in metrics_list:
            initial_context = m.context_sizes[0] if m.context_sizes else 0
            final_context = m.context_sizes[-1] if m.context_sizes else 0
            growth_factor = final_context / initial_context if initial_context > 0 else 0

            writer.writerow([
                m.instance_id,
                m.exit_status,
                m.total_prompt_tokens,
                m.total_completion_tokens,
                m.total_tokens,
                m.max_context_size,
                m.api_calls,
                initial_context,
                final_context,
                f"{growth_factor:.2f}",
            ])

    print(f"\nExported {len(metrics_list)} instances to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token usage in SWE-bench trajectories"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/run_baseline_qwen3coder"),
        help="Data directory containing trajectory files",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Export per-instance metrics to CSV",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        # Resolve relative to script location
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / data_dir

    print(f"Searching for trajectories in: {data_dir}")

    # Find all trajectory files
    trajectory_files = find_trajectory_files(data_dir)
    print(f"Found {len(trajectory_files)} trajectory files")

    if not trajectory_files:
        print("No trajectory files found. Exiting.")
        return

    # Extract metrics from each trajectory
    metrics_list = []
    for traj_path in trajectory_files:
        metrics = extract_token_metrics(traj_path)
        if metrics and metrics.api_calls > 0:
            metrics_list.append(metrics)

    print(f"Successfully parsed {len(metrics_list)} trajectories with token data")

    # Analyze metrics
    results = analyze_trajectories(metrics_list)

    # Print analysis
    print_analysis(results)

    # Export to CSV if requested
    if args.output_csv:
        export_to_csv(metrics_list, args.output_csv)


if __name__ == "__main__":
    main()
