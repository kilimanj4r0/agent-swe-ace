#!/usr/bin/env python3
"""Analyze [skill-id] references in trajectory files across experiment runs.

Searches assistant message content in trajectory JSON files for skill references
and reports which skills were used, by which instances, with counts.

Usage:
    uv run python scripts/analyze_skill_references.py data/run_20260415_020540
    uv run python scripts/analyze_skill_references.py data/run_*_completed
    uv run python scripts/analyze_skill_references.py data/run_a data/run_b --json
    uv run python scripts/analyze_skill_references.py data/run_a --csv skills.csv
    uv run python scripts/analyze_skill_references.py data/run_a --by-instance
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Matches actual skill references, NOT the bare [skill-id] template instruction.
# Captures four formats found in trajectories:
#   [skill-00001]                         — numeric-only
#   [skill-django_orm-00003]              — topic-prefixed
#   [skill-id: django_orm-00003]          — colon variant
#   [skill-id django_orm-00003]           — space variant
SKILL_REF_RE = re.compile(
    r"\[skill-(?:[a-zA-Z_][\w]*-)?\d+\]|\[skill-id(?::|\s)[^\]]+\]"
)


@dataclass
class SkillHit:
    """A single skill reference found in a trajectory."""
    skill_id: str
    instance_id: str
    iteration: int
    run_dir: str
    message_index: int
    context: str  # ~100 chars around the reference


@dataclass
class RunSkillReport:
    """Aggregated skill reference report for one run."""
    run_dir: str
    total_trajectories: int = 0
    trajectories_with_refs: int = 0
    total_refs: int = 0
    hits: list[SkillHit] = field(default_factory=list)


def find_trajectory_files(run_dir: Path) -> list[Path]:
    """Find all iter_*.json trajectory files under a run directory."""
    trajectories = []

    # Canonical layout: {benchmark}/trajectories/{instance}/iter_N.json
    # Skip iter_0 — no skillbook is injected on the first attempt.
    _iter_re = re.compile(r"iter_(\d+)")
    for traj_file in run_dir.rglob("trajectories"):
        if traj_file.is_dir():
            for iter_file in traj_file.rglob("iter_*.json"):
                m = _iter_re.match(iter_file.stem)
                if m and int(m.group(1)) >= 1:
                    trajectories.append(iter_file)

    # Also catch *.traj.json (baseline format)
    for traj_file in run_dir.rglob("*.traj.json"):
        if traj_file not in trajectories:
            trajectories.append(traj_file)

    return sorted(trajectories)


def extract_instance_id(path: Path) -> str:
    """Extract instance_id from trajectory file path."""
    # Path: .../trajectories/[phase/]<instance_id>/iter_N.json
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "trajectories" and i + 2 < len(parts):
            # Check if next is a phase dir (train, val, val_baseline) or instance dir
            candidate = parts[i + 1]
            if candidate in ("train", "val", "val_baseline"):
                return parts[i + 2]
            return candidate
    return path.stem


def extract_iteration(path: Path) -> int:
    """Extract iteration number from filename like iter_3.json."""
    name = path.stem  # iter_3
    m = re.match(r"iter_(\d+)", name)
    return int(m.group(1)) if m else 0


def scan_trajectory(traj_path: Path, run_name: str) -> list[SkillHit]:
    """Scan a single trajectory file for skill references in assistant messages."""
    try:
        with open(traj_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    instance_id = data.get("info", {}).get("instance_id", extract_instance_id(traj_path))
    iteration = data.get("info", {}).get("iteration", extract_iteration(traj_path))

    hits = []
    for idx, msg in enumerate(data.get("messages", [])):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not content:
            continue

        for m in SKILL_REF_RE.finditer(content):
            skill_id = m.group(0)[1:-1]  # strip brackets
            # Normalize colon/space variants: "skill-id: foo-001" -> "skill-foo-001"
            if skill_id.startswith("skill-id"):
                skill_id = skill_id.replace("skill-id:", "").replace("skill-id", "").strip()
                skill_id = f"skill-{skill_id}"

            # Extract ~100 chars of context around the reference
            start = max(0, m.start() - 50)
            end = min(len(content), m.end() + 50)
            context = content[start:end].replace("\n", " ")

            hits.append(SkillHit(
                skill_id=skill_id,
                instance_id=instance_id,
                iteration=iteration,
                run_dir=run_name,
                message_index=idx,
                context=context,
            ))

    return hits


def analyze_runs(run_dirs: list[Path]) -> list[RunSkillReport]:
    """Analyze skill references across one or more run directories."""
    reports = []
    for run_dir in run_dirs:
        run_name = run_dir.name
        traj_files = find_trajectory_files(run_dir)
        if not traj_files:
            continue

        report = RunSkillReport(run_dir=run_name, total_trajectories=len(traj_files))
        instances_with_refs = set()

        for traj_path in traj_files:
            hits = scan_trajectory(traj_path, run_name)
            if hits:
                instances_with_refs.add(hits[0].instance_id)
                report.hits.extend(hits)

        report.trajectories_with_refs = len(instances_with_refs)
        report.total_refs = len(report.hits)
        reports.append(report)

    return reports


def print_table(reports: list[RunSkillReport]) -> None:
    """Print a summary table of skill references per run."""
    print("=" * 70)
    print("Skill Reference Analysis")
    print("=" * 70)

    for report in reports:
        print(f"\n  Run: {report.run_dir}")
        print(f"  Trajectories: {report.total_trajectories} total, "
              f"{report.trajectories_with_refs} with skill refs, "
              f"{report.total_refs} total references")

        if not report.hits:
            print("  No skill references found.")
            continue

        # Aggregate by skill_id
        by_skill: dict[str, list[SkillHit]] = defaultdict(list)
        for hit in report.hits:
            by_skill[hit.skill_id].append(hit)

        # Sort by frequency
        sorted_skills = sorted(by_skill.items(), key=lambda x: -len(x[1]))

        print(f"\n  {'Skill ID':<40} {'Count':>5}  {'Instances':>9}")
        print(f"  {'-' * 40} {'-' * 5}  {'-' * 9}")
        for skill_id, hits in sorted_skills:
            instances = set(h.instance_id for h in hits)
            short_instances = [iid.split("__")[-1][:30] for iid in sorted(instances)]
            inst_str = ", ".join(short_instances) if len(short_instances) <= 3 else f"{len(instances)} unique"
            print(f"  {skill_id:<40} {len(hits):>5}  {inst_str}")

    print("\n" + "=" * 70)


def print_by_instance(reports: list[RunSkillReport]) -> None:
    """Print per-instance breakdown of skill references."""
    print("=" * 70)
    print("Skill References by Instance")
    print("=" * 70)

    for report in reports:
        if not report.hits:
            continue

        by_instance: dict[str, list[SkillHit]] = defaultdict(list)
        for hit in report.hits:
            by_instance[hit.instance_id].append(hit)

        print(f"\n  Run: {report.run_dir}")
        for instance_id in sorted(by_instance):
            hits = by_instance[instance_id]
            skills = sorted(set(h.skill_id for h in hits))
            print(f"\n  {instance_id}  ({len(hits)} refs, {len(skills)} unique skills)")
            for s in skills:
                count = sum(1 for h in hits if h.skill_id == s)
                print(f"    {s}  x{count}")

    print("\n" + "=" * 70)


def print_json(reports: list[RunSkillReport], save_path: str | None = None) -> None:
    """Output results as JSON."""
    data = []
    for report in reports:
        by_skill: dict[str, dict] = defaultdict(lambda: {"count": 0, "instances": set(), "iterations": set()})
        for hit in report.hits:
            entry = by_skill[hit.skill_id]
            entry["count"] += 1
            entry["instances"].add(hit.instance_id)
            entry["iterations"].add(hit.iteration)

        data.append({
            "run_dir": report.run_dir,
            "total_trajectories": report.total_trajectories,
            "trajectories_with_refs": report.trajectories_with_refs,
            "total_refs": report.total_refs,
            "skills": {
                sid: {
                    "count": e["count"],
                    "instances": sorted(e["instances"]),
                    "iterations": sorted(e["iterations"]),
                }
                for sid, e in sorted(by_skill.items(), key=lambda x: -x[1]["count"])
            },
        })

    text = json.dumps(data, indent=2)
    if save_path:
        Path(save_path).write_text(text)
        print(f"Saved JSON to {save_path}")
    else:
        print(text)


def export_csv(reports: list[RunSkillReport], output_path: Path) -> None:
    """Export all hits to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_dir", "instance_id", "iteration", "skill_id", "message_index", "context"])
        for report in reports:
            for hit in report.hits:
                writer.writerow([
                    hit.run_dir,
                    hit.instance_id,
                    hit.iteration,
                    hit.skill_id,
                    hit.message_index,
                    hit.context,
                ])
    total = sum(r.total_refs for r in reports)
    print(f"Exported {total} skill references to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze [skill-id] references in experiment trajectories"
    )
    parser.add_argument(
        "runs", nargs="+", metavar="RUN_DIR",
        help="Run directory(ies) to analyze",
    )
    parser.add_argument(
        "--json", nargs="?", const=True, default=False,
        help="Output as JSON. Optionally specify a file path to save.",
    )
    parser.add_argument(
        "--csv", type=Path, metavar="PATH",
        help="Export all references to CSV",
    )
    parser.add_argument(
        "--by-instance", action="store_true",
        help="Show per-instance breakdown",
    )
    args = parser.parse_args()

    # Validate run directories
    run_dirs = []
    for p in args.runs:
        path = Path(p)
        if not path.exists():
            print(f"Path not found: {path}", file=__import__("sys").stderr)
            continue
        run_dirs.append(path)

    if not run_dirs:
        print("No valid run directories found.", file=__import__("sys").stderr)
        __import__("sys").exit(1)

    reports = analyze_runs(run_dirs)

    if args.csv:
        export_csv(reports, args.csv)
        print()

    if args.json is not False:
        save_path = args.json if isinstance(args.json, str) else None
        print_json(reports, save_path=save_path)
    elif args.by_instance:
        print_by_instance(reports)
    else:
        print_table(reports)


if __name__ == "__main__":
    main()
