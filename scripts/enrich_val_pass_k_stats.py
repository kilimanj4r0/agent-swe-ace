#!/usr/bin/env python3
"""Enrich statistics.json of a completed iterate_repos run with val_pass_k stats.

Scans val_baseline/ and val/ result directories for per-attempt resolution data,
computes pass@1..K and per-attempt resolved counts, and writes them back into
statistics.json and statistics_per_repo/<repo>.json.

Usage:
    python scripts/enrich_val_pass_k_stats.py data/run_20260521_034008
    python scripts/enrich_val_pass_k_stats.py data/run_20260521_034008 --dry-run
"""

import argparse
import json
import sys
from pathlib import Path


def collect_per_attempt_stats(results_dir: Path, repo_prefix: str | None = None) -> dict:
    """Scan val_baseline or val result dir for per-attempt resolution data.

    Args:
        results_dir: Path to val_baseline/ or val/ results directory
        repo_prefix: If set, only include instances starting with this prefix

    Returns dict with:
        total_instances, resolved_count, resolution_rate,
        max_attempts, pass_at_k
    """
    if not results_dir.exists():
        return {}

    instances = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        if repo_prefix and not d.name.startswith(repo_prefix):
            continue
        iid = d.name
        attempts = []
        for i in range(10):
            f = d / f"iter_{i}.json"
            if f.exists():
                with open(f) as fh:
                    attempts.append(json.load(fh).get("resolved", False))
            else:
                break
        if attempts:
            instances.append((iid, attempts))

    if not instances:
        return {}

    total = len(instances)
    k = max(len(a) for _, a in instances)

    pass_at = {}
    for n in range(1, k + 1):
        count = sum(1 for _, a in instances if any(a[:n]))
        pass_at[f"pass@{n}"] = {
            "count": count,
            "total": total,
            "rate": count / total if total else 0.0,
        }

    resolved_count = sum(1 for _, a in instances if any(a))

    return {
        "total_instances": total,
        "resolved_count": resolved_count,
        "resolution_rate": resolved_count / total if total else 0.0,
        "max_attempts": k,
        "pass_at_k": pass_at,
    }


def enrich_run(run_dir: Path, dry_run: bool = False):
    """Enrich statistics.json and per-repo stats with val_pass_k data."""
    stats_path = run_dir / "statistics.json"
    config_path = run_dir / "config.json"

    if not stats_path.exists() or not config_path.exists():
        print(f"Missing statistics.json or config.json in {run_dir}")
        sys.exit(1)

    with open(stats_path) as f:
        stats = json.load(f)
    with open(config_path) as f:
        cfg = json.load(f)

    dataset = cfg.get("benchmark", {}).get("dataset", "")
    bench_dir = dataset.replace("/", "__")
    results_base = run_dir / bench_dir / "results"

    if not results_base.exists():
        results_base = run_dir / "results"

    val_pass_k = cfg.get("experiment", {}).get("val_pass_k", 1)
    iterate_repos = cfg.get("benchmark", {}).get("iterate_repos", [])
    is_iterate = bool(iterate_repos)
    is_two_phase = stats.get("val_baseline_phase") is not None or (results_base / "val_baseline").exists()

    if not is_two_phase:
        print("Not a two-phase run, nothing to enrich.")
        return

    print(f"Run: {run_dir.name}")
    print(f"Dataset: {dataset}, val_pass_k: {val_pass_k}")
    print(f"iterate_repos: {is_iterate} ({len(iterate_repos)} repos)")
    print()

    changes_made = False

    # Enrich global statistics
    for phase_key in ("val_baseline_phase", "val_skillbook_phase"):
        phase_dir_name = "val_baseline" if phase_key == "val_baseline_phase" else "val"
        phase_dir = results_base / phase_dir_name
        if not phase_dir.exists():
            continue

        enriched = collect_per_attempt_stats(phase_dir)
        if not enriched:
            continue

        existing = stats.get(phase_key, {})
        stats[phase_key] = {**existing, **enriched}
        changes_made = True

        print(f"  {phase_key}:")
        print(f"    max_attempts: {enriched['max_attempts']}")
        for pk, pv in enriched.get("pass_at_k", {}).items():
            print(f"    {pk}: {pv['count']}/{pv['total']} ({pv['rate']:.1%})")
        print()

    # Enrich per-repo statistics
    if is_iterate:
        per_repo_dir = run_dir / "statistics_per_repo"
        for repo in iterate_repos:
            repo_file = per_repo_dir / f"{repo.replace('/', '__')}.json"

            # Load existing per-repo stats or create minimal structure
            if repo_file.exists():
                with open(repo_file) as f:
                    repo_stats = json.load(f)
            else:
                print(f"  {repo}: no per-repo stats file, creating from filesystem")
                repo_stats = {}

            repo_prefix = repo.split("/")[0] + "__" + repo.split("/")[1] + "-"
            repo_changed = False

            for phase_key in ("val_baseline_phase", "val_skillbook_phase"):
                phase_dir_name = "val_baseline" if phase_key == "val_baseline_phase" else "val"
                phase_dir = results_base / phase_dir_name
                enriched = collect_per_attempt_stats(phase_dir, repo_prefix)
                if not enriched:
                    continue

                existing = repo_stats.get(phase_key, {})
                repo_stats[phase_key] = {**existing, **enriched}
                repo_changed = True

            # Also enrich train phase if missing
            if "train_phase" not in repo_stats:
                train_dir = results_base / "train"
                enriched_train = collect_per_attempt_stats(train_dir, repo_prefix)
                if enriched_train:
                    repo_stats["train_phase"] = {
                        "total_instances": enriched_train["total_instances"],
                        "resolved_count": enriched_train["resolved_count"],
                        "resolution_rate": enriched_train["resolution_rate"],
                    }
                    repo_changed = True

            if repo_changed:
                print(f"  {repo}: enriched")
                if not dry_run:
                    per_repo_dir.mkdir(parents=True, exist_ok=True)
                    with open(repo_file, "w") as f:
                        json.dump(repo_stats, f, indent=2)
                changes_made = True

    if changes_made:
        if dry_run:
            print("\n[DRY RUN] Would write enriched statistics.")
        else:
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\nWritten enriched statistics to {stats_path}")
    else:
        print("\nNo changes needed.")


def main():
    parser = argparse.ArgumentParser(description="Enrich statistics with val_pass_k stats")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()

    enrich_run(args.run_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
