#!/usr/bin/env python3
"""Analyze skill references and injections in trajectory files across experiment runs.

Supports both single-iteration and two-phase (train/val/val_baseline) split experiments.

In multi-iteration mode: scans assistant messages for [skill-id] references.
In split mode: extracts skills injected into user prompts (### skill-id headers)
and checks if the agent explicitly references them in responses.

Usage:
    # Single run
    uv run python scripts/analyze_skill_references.py data/run_20260415_020540

    # Multiple runs
    uv run python scripts/analyze_skill_references.py data/run_*_completed

    # JSON output
    uv run python scripts/analyze_skill_references.py data/run_a --json

    # CSV export
    uv run python scripts/analyze_skill_references.py data/run_a --csv skills.csv

    # Per-instance breakdown
    uv run python scripts/analyze_skill_references.py data/run_a --by-instance
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Matches [skill-id] references in assistant messages (multi-iteration mode)
SKILL_REF_RE = re.compile(
    r"\[skill-(?:[a-zA-Z_][\w]*-)?\d+\]|\[skill-id(?::|\s)[^\]]+\]"
)

# Matches ### skill-id headers injected into user prompts (split mode)
SKILL_HEADER_RE = re.compile(r"^### (`?)([a-zA-Z_][\w]*-\d+)\1\s*$", re.MULTILINE)

SPLIT_PHASES = {"train", "val", "val_baseline"}


@dataclass
class SkillHit:
    """A single skill reference found in a trajectory."""
    skill_id: str
    instance_id: str
    iteration: int
    phase: str | None  # None for non-split experiments
    run_dir: str
    message_index: int
    context: str  # ~100 chars around the reference


@dataclass
class InjectedSkill:
    """A skill injected into a trajectory's user prompt."""
    skill_id: str
    instance_id: str
    iteration: int
    phase: str | None
    run_dir: str


@dataclass
class RunSkillReport:
    """Aggregated skill reference report for one run."""
    run_dir: str
    is_split: bool = False
    total_trajectories: int = 0
    trajectories_with_refs: int = 0
    total_refs: int = 0
    hits: list[SkillHit] = field(default_factory=list)
    injected: list[InjectedSkill] = field(default_factory=list)


def detect_phase(path: Path) -> str | None:
    """Detect the split phase (train/val/val_baseline) from trajectory path."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "trajectories" and i + 1 < len(parts):
            candidate = parts[i + 1]
            if candidate in SPLIT_PHASES:
                return candidate
    return None


def find_trajectory_files(run_dir: Path, *, include_iter0: bool = False) -> list[Path]:
    """Find all iter_*.json trajectory files under a run directory."""
    trajectories = []
    _iter_re = re.compile(r"iter_(\d+)")

    for traj_file in run_dir.rglob("trajectories"):
        if traj_file.is_dir():
            for iter_file in traj_file.rglob("iter_*.json"):
                m = _iter_re.match(iter_file.stem)
                if m:
                    iter_num = int(m.group(1))
                    # Skip iter_0 in non-split mode (no skillbook injected on first attempt)
                    # but include iter_0 in split mode (val phase gets skillbook from train)
                    if include_iter0 or iter_num >= 1:
                        trajectories.append(iter_file)

    # Also catch *.traj.json (baseline format)
    for traj_file in run_dir.rglob("*.traj.json"):
        if traj_file not in trajectories:
            trajectories.append(traj_file)

    return sorted(trajectories)


def extract_instance_id(path: Path) -> str:
    """Extract instance_id from trajectory file path."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "trajectories" and i + 2 < len(parts):
            candidate = parts[i + 1]
            if candidate in SPLIT_PHASES:
                return parts[i + 2]
            return candidate
    return path.stem


def extract_iteration(path: Path) -> int:
    """Extract iteration number from filename like iter_3.json."""
    m = re.match(r"iter_(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def scan_trajectory(traj_path: Path, run_name: str) -> tuple[list[SkillHit], list[InjectedSkill]]:
    """Scan a trajectory for skill references and injected skills."""
    try:
        with open(traj_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return [], []

    instance_id = data.get("info", {}).get("instance_id", extract_instance_id(traj_path))
    iteration = data.get("info", {}).get("iteration", extract_iteration(traj_path))
    phase = detect_phase(traj_path)

    hits = []
    injected = []

    for idx, msg in enumerate(data.get("messages", [])):
        content = msg.get("content", "")
        if not content:
            continue

        # Check for injected skills in user messages (### skill-id headers)
        if msg.get("role") == "user":
            for m in SKILL_HEADER_RE.finditer(content):
                skill_id = m.group(2)
                injected.append(InjectedSkill(
                    skill_id=skill_id,
                    instance_id=instance_id,
                    iteration=iteration,
                    phase=phase,
                    run_dir=run_name,
                ))

        # Check for [skill-id] references in assistant messages
        if msg.get("role") == "assistant":
            for m in SKILL_REF_RE.finditer(content):
                skill_id = m.group(0)[1:-1]  # strip brackets
                if skill_id.startswith("skill-id"):
                    skill_id = skill_id.replace("skill-id:", "").replace("skill-id", "").strip()
                    skill_id = f"skill-{skill_id}"

                start = max(0, m.start() - 50)
                end = min(len(content), m.end() + 50)
                context = content[start:end].replace("\n", " ")

                hits.append(SkillHit(
                    skill_id=skill_id,
                    instance_id=instance_id,
                    iteration=iteration,
                    phase=phase,
                    run_dir=run_name,
                    message_index=idx,
                    context=context,
                ))

    return hits, injected


def is_split_run(run_dir: Path) -> bool:
    """Check if a run directory contains split experiment data."""
    for traj_dir in run_dir.rglob("trajectories"):
        if traj_dir.is_dir():
            for subdir in traj_dir.iterdir():
                if subdir.is_dir() and subdir.name in SPLIT_PHASES:
                    return True
    return False


def analyze_runs(run_dirs: list[Path]) -> list[RunSkillReport]:
    """Analyze skill references across one or more run directories."""
    reports = []
    for run_dir in run_dirs:
        run_name = run_dir.name
        split = is_split_run(run_dir)
        traj_files = find_trajectory_files(run_dir, include_iter0=split)
        if not traj_files:
            continue

        report = RunSkillReport(run_dir=run_name, is_split=split, total_trajectories=len(traj_files))
        instances_with_refs = set()

        for traj_path in traj_files:
            hits, injected = scan_trajectory(traj_path, run_name)
            if hits:
                instances_with_refs.add(hits[0].instance_id)
                report.hits.extend(hits)
            if injected:
                instances_with_refs.add(injected[0].instance_id)
                report.injected.extend(injected)

        report.trajectories_with_refs = len(instances_with_refs)
        report.total_refs = len(report.hits) + len(report.injected)
        reports.append(report)

    return reports


def print_table(reports: list[RunSkillReport]) -> None:
    """Print a summary table of skill references per run."""
    print("=" * 70)
    print("Skill Reference Analysis")
    print("=" * 70)

    for report in reports:
        print(f"\n  Run: {report.run_dir}")
        mode_label = "split (two-phase)" if report.is_split else "multi-iteration"
        print(f"  Mode: {mode_label}")
        print(f"  Trajectories: {report.total_trajectories} total, "
              f"{report.trajectories_with_refs} with skill activity, "
              f"{report.total_refs} total (refs: {len(report.hits)}, injected: {len(report.injected)})")

        if report.injected:
            _print_injected_summary(report)
        if report.hits:
            _print_ref_summary(report)
        if not report.injected and not report.hits:
            print("  No skill references or injections found.")

    print("\n" + "=" * 70)


def _print_injected_summary(report: RunSkillReport) -> None:
    """Print summary of skills injected into user prompts, grouped by phase."""
    by_phase: dict[str | None, list[InjectedSkill]] = defaultdict(list)
    for inj in report.injected:
        by_phase[inj.phase].append(inj)

    for phase in sorted(by_phase, key=lambda p: (p is None, p or "")):
        injections = by_phase[phase]
        phase_label = phase or "unknown"
        instances = set(i.instance_id for i in injections)
        unique_skills = sorted(set(i.skill_id for i in injections))

        print(f"\n  Phase: {phase_label} ({len(injections)} injections across {len(instances)} instances)")
        print(f"  Unique skills: {len(unique_skills)}")

        # Count how often each skill appears
        skill_counts: dict[str, int] = defaultdict(int)
        for inj in injections:
            skill_counts[inj.skill_id] += 1
        sorted_skills = sorted(skill_counts.items(), key=lambda x: -x[1])

        print(f"  {'Skill ID':<45} {'Count':>5}  {'Instances':>9}")
        print(f"  {'-' * 45} {'-' * 5}  {'-' * 9}")
        for skill_id, count in sorted_skills[:30]:
            inst_count = len(set(i.instance_id for i in injections if i.skill_id == skill_id))
            print(f"  {skill_id:<45} {count:>5}  {inst_count:>9}")
        if len(sorted_skills) > 30:
            print(f"  ... and {len(sorted_skills) - 30} more")


def _print_ref_summary(report: RunSkillReport) -> None:
    """Print summary for explicit [skill-id] references."""
    by_skill: dict[str, list[SkillHit]] = defaultdict(list)
    for hit in report.hits:
        by_skill[hit.skill_id].append(hit)

    sorted_skills = sorted(by_skill.items(), key=lambda x: -len(x[1]))

    print(f"\n  {'Skill ID':<40} {'Count':>5}  {'Instances':>9}")
    print(f"  {'-' * 40} {'-' * 5}  {'-' * 9}")
    for skill_id, hits in sorted_skills:
        instances = set(h.instance_id for h in hits)
        short_instances = [iid.split("__")[-1][:30] for iid in sorted(instances)]
        inst_str = ", ".join(short_instances) if len(short_instances) <= 3 else f"{len(instances)} unique"
        print(f"  {skill_id:<40} {len(hits):>5}  {inst_str}")


def print_by_instance(reports: list[RunSkillReport]) -> None:
    """Print per-instance breakdown of skill references."""
    print("=" * 70)
    print("Skill References by Instance")
    print("=" * 70)

    for report in reports:
        if not report.injected and not report.hits:
            continue

        by_instance: dict[str, dict] = defaultdict(lambda: {"hits": [], "injected": []})
        for hit in report.hits:
            by_instance[hit.instance_id]["hits"].append(hit)
        for inj in report.injected:
            by_instance[inj.instance_id]["injected"].append(inj)

        print(f"\n  Run: {report.run_dir}")
        for instance_id in sorted(by_instance):
            entry = by_instance[instance_id]
            injected_ids = sorted(set(i.skill_id for i in entry["injected"]))
            ref_ids = sorted(set(h.skill_id for h in entry["hits"]))
            phase = (entry["injected"] or entry["hits"])[0].phase or "?"

            parts = []
            if injected_ids:
                parts.append(f"{len(injected_ids)} skills injected")
            if ref_ids:
                parts.append(f"{len(ref_ids)} skills referenced")
            print(f"\n  [{phase}] {instance_id}  ({', '.join(parts)})")

            for s in injected_ids[:10]:
                print(f"    injected: {s}")
            if len(injected_ids) > 10:
                print(f"    ... and {len(injected_ids) - 10} more")
            for s in ref_ids:
                count = sum(1 for h in entry["hits"] if h.skill_id == s)
                print(f"    ref: {s}  x{count}")

    print("\n" + "=" * 70)


def print_json(reports: list[RunSkillReport], save_path: str | None = None) -> None:
    """Output results as JSON."""
    data = []
    for report in reports:
        entry: dict = {
            "run_dir": report.run_dir,
            "is_split": report.is_split,
            "total_trajectories": report.total_trajectories,
            "trajectories_with_refs": report.trajectories_with_refs,
            "total_refs": report.total_refs,
            "explicit_refs": len(report.hits),
            "injected_skills": len(report.injected),
        }

        # Injected skills by phase
        if report.injected:
            by_phase: dict[str | None, dict] = defaultdict(lambda: {"skills": {}, "instances": set()})
            for inj in report.injected:
                pe = by_phase[inj.phase]
                if inj.skill_id not in pe["skills"]:
                    pe["skills"][inj.skill_id] = {"count": 0, "instances": set()}
                pe["skills"][inj.skill_id]["count"] += 1
                pe["skills"][inj.skill_id]["instances"].add(inj.instance_id)
                pe["instances"].add(inj.instance_id)

            entry["phases"] = {}
            for phase in sorted(by_phase, key=lambda p: (p is None, p or "")):
                pe = by_phase[phase]
                entry["phases"][phase or "unknown"] = {
                    "instances": len(pe["instances"]),
                    "unique_skills": len(pe["skills"]),
                    "skills": {
                        sid: {"count": se["count"], "instances": sorted(se["instances"])}
                        for sid, se in sorted(pe["skills"].items(), key=lambda x: -x[1]["count"])
                    },
                }

        # Explicit references
        if report.hits:
            by_skill: dict[str, dict] = defaultdict(lambda: {"count": 0, "instances": set(), "iterations": set()})
            for hit in report.hits:
                se = by_skill[hit.skill_id]
                se["count"] += 1
                se["instances"].add(hit.instance_id)
                se["iterations"].add(hit.iteration)
            entry["explicit_skill_refs"] = {
                sid: {
                    "count": se["count"],
                    "instances": sorted(se["instances"]),
                    "iterations": sorted(se["iterations"]),
                }
                for sid, se in sorted(by_skill.items(), key=lambda x: -x[1]["count"])
            }

        data.append(entry)

    text = json.dumps(data, indent=2)
    if save_path:
        Path(save_path).write_text(text)
        print(f"Saved JSON to {save_path}")
    else:
        print(text)


def export_csv(reports: list[RunSkillReport], output_path: Path) -> None:
    """Export all hits and injections to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_dir", "instance_id", "iteration", "phase", "type", "skill_id", "message_index", "context"])
        for report in reports:
            for hit in report.hits:
                writer.writerow([
                    hit.run_dir,
                    hit.instance_id,
                    hit.iteration,
                    hit.phase or "",
                    "ref",
                    hit.skill_id,
                    hit.message_index,
                    hit.context,
                ])
            for inj in report.injected:
                writer.writerow([
                    inj.run_dir,
                    inj.instance_id,
                    inj.iteration,
                    inj.phase or "",
                    "injected",
                    inj.skill_id,
                    "",
                    "",
                ])
    total = sum(r.total_refs for r in reports)
    print(f"Exported {total} skill records to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze skill references and injections in experiment trajectories"
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
