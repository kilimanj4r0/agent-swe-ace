#!/usr/bin/env python3
"""Analyze skill references and injections in trajectory files across experiment runs.

Supports both single-iteration and two-phase (train/val/val_baseline) split experiments.

In multi-iteration mode: scans assistant messages for [skill-id] references.
In split mode: extracts skills injected into user prompts (### skill-id headers)
and checks if the agent explicitly references them in responses.

Prose mode (on by default, --no-prose to disable): also detects NON-bracket
engagement with the skillbook two ways:
  1. "skill"/"skills"/"skillbook" word mentions in assistant messages (engagement).
  2. Every skill-ID token referenced in prose (backtick or bare form, e.g.
     `debugging-00190`), classified against the run's skillbook as:
       presented  = injected into this trajectory AND referenced
       skillbook  = a real skill in the skillbook but NOT injected here
       unknown    = not in the skillbook (hallucinated or non-skill token)
This catches models (e.g. qwen3next) that reference skills by word/ID rather than
via [skill-id] tokens — so referenced skills are never silently dropped.

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

    # Disable prose detection (bracket/header mode only, original behavior)
    uv run python scripts/analyze_skill_references.py data/run_a --no-prose
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Matches [skill-id] references in assistant messages (multi-iteration mode)
SKILL_REF_RE = re.compile(
    r"\[skill-(?:[a-zA-Z_][\w]*-)?\d+\]|\[skill-id(?::|\s)[^\]]+\]"
)

# Matches ### skill-id headers injected into user prompts (split mode)
SKILL_HEADER_RE = re.compile(r"^### (`?)([a-zA-Z_][\w]*-\d+)\1\s*$", re.MULTILINE)

# Matches prose mentions of the skillbook/skills in assistant messages (non-bracket).
# Word-bounded "skill"/"skills"/"skillbook" — excludes "skillful"/"skillset".
SKILL_PROSE_RE = re.compile(r"\bskills?(?:book)?\b", re.IGNORECASE)

# Matches a referenced skill ID anywhere in assistant prose: <section>-<5 digits>,
# in backtick or bare form (e.g. `debugging-00190`, query_handling-00199). The
# negative lookbehind prevents matching instance IDs like django__django-12345
# (the embedded "django-12345" is preceded by an underscore/letter, no boundary).
SKILL_ID_TOKEN_RE = re.compile(r"(?<![A-Za-z_])([a-z]+(?:_[a-z]+)*-[0-9]{5})", re.IGNORECASE)

SPLIT_PHASES = {"train", "val", "val_baseline"}
PROSE_REF_CLASSES = ("presented", "skillbook", "unknown", "lookalike")
# Classes that name genuine skill references (shown in detail tables). "lookalike"
# tokens (non-skill sections, e.g. bpo-43882) are reported as a count only.
SKILL_REF_CLASSES = ("presented", "skillbook", "unknown")


@dataclass
class SkillHit:
    """A single bracket [skill-id] reference found in a trajectory."""
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
class ProseMention:
    """A non-bracket prose mention of 'skill'/'skills'/'skillbook' (engagement)."""
    instance_id: str
    iteration: int
    phase: str | None
    run_dir: str
    message_index: int
    context: str


@dataclass
class ProseRef:
    """A skill ID referenced in assistant prose (non-bracket), classified.

    classification is one of: 'presented' (injected into this trajectory),
    'skillbook' (real skill in the run's skillbook but not injected here),
    'unknown' (not found in the skillbook).
    """
    skill_id: str
    classification: str
    instance_id: str
    iteration: int
    phase: str | None
    run_dir: str
    message_index: int
    context: str


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
    # Prose-mode additions
    prose: list[ProseMention] = field(default_factory=list)
    trajectories_with_prose: int = 0
    total_prose: int = 0
    prose_refs: list[ProseRef] = field(default_factory=list)


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


def load_skillbook_ids(run_dir: Path) -> set[str]:
    """Return the union of all skill IDs across the run's final skillbooks.

    Scans every final_skillbook.json under the run dir (global +
    per_repo/<repo>/) so per-repo split runs are covered. These are the
    authoritative 'real skill' IDs used to classify prose references.
    """
    ids: set[str] = set()
    for sb in run_dir.rglob("final_skillbook.json"):
        try:
            d = json.loads(sb.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        skills = d.get("skills") if isinstance(d, dict) else d
        if isinstance(skills, dict):
            skills = skills.values()
        for s in skills or []:
            sid = s.get("id") if isinstance(s, dict) else None
            if sid:
                ids.add(sid)
    return ids


def _classify_ref(sid: str, presented: set[str], skillbook: set[str], sections: set[str]) -> str:
    """Classify a prose-referenced skill ID.

    Returns one of:
      presented  = injected into this trajectory (in `presented`)
      skillbook  = a real skill in the run's skillbook but not injected here
      unknown    = valid skill section but no such skill (mis-cited/hallucinated)
      lookalike  = prefix is not a known skill section (e.g. bpo-43882 bug ID)
    """
    if sid.rsplit("-", 1)[0] not in sections:
        return "lookalike"
    if sid in presented:
        return "presented"
    if sid in skillbook:
        return "skillbook"
    return "unknown"


def scan_trajectory(traj_path: Path, run_name: str, *,
                   include_prose: bool = True, skillbook_ids: set[str] | None = None
                   ) -> tuple[list[SkillHit], list[InjectedSkill], list[ProseMention], list[ProseRef]]:
    """Scan a trajectory for skill references, injections, and prose mentions."""
    try:
        with open(traj_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return [], [], [], []

    instance_id = data.get("info", {}).get("instance_id", extract_instance_id(traj_path))
    iteration = data.get("info", {}).get("iteration", extract_iteration(traj_path))
    phase = detect_phase(traj_path)
    skillbook_ids = skillbook_ids or set()
    # Valid section prefixes (e.g. debugging, testing, query_handling) derived from
    # the skillbook. A referenced token is only a skill if its prefix is one of
    # these — this rejects look-alikes from prose such as bpo-43882 (a bug ID).
    skillbook_sections = {sid.rsplit("-", 1)[0] for sid in skillbook_ids}

    messages = data.get("messages", [])

    # Pass 1: collect presented skill IDs from user-message injection headers.
    presented_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                presented_ids.update(m.group(2) for m in SKILL_HEADER_RE.finditer(content))

    hits: list[SkillHit] = []
    injected: list[InjectedSkill] = []
    prose: list[ProseMention] = []
    prose_refs: list[ProseRef] = []

    for idx, msg in enumerate(messages):
        content = msg.get("content", "")
        if not content or not isinstance(content, str):
            continue

        # Injected skills in user messages (### skill-id headers)
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

        if msg.get("role") != "assistant":
            continue

        # Bracket references (existing behavior)
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

        if not include_prose:
            continue

        # Mask bracket tokens so their inner IDs aren't double-counted below.
        prose_content = SKILL_REF_RE.sub(" ", content)

        # (1) Engagement: word "skill(s)/skillbook" mentions.
        for m in SKILL_PROSE_RE.finditer(prose_content):
            start = max(0, m.start() - 60)
            end = min(len(prose_content), m.end() + 60)
            prose.append(ProseMention(
                instance_id=instance_id,
                iteration=iteration,
                phase=phase,
                run_dir=run_name,
                message_index=idx,
                context=prose_content[start:end].replace("\n", " "),
            ))

        # (2) Every skill-ID token referenced in prose, classified. This is what
        # ensures referenced skills are never silently dropped: a skill cited in
        # prose (e.g. debugging-00190) is captured even if it wasn't injected.
        # Tokens are classified into presented/skillbook/unknown/lookalike (see
        # _classify_ref); nothing is dropped, look-alikes are just bucketed apart.
        for m in SKILL_ID_TOKEN_RE.finditer(prose_content):
            sid = m.group(1)
            start = max(0, m.start() - 60)
            end = min(len(prose_content), m.end() + 60)
            prose_refs.append(ProseRef(
                skill_id=sid,
                classification=_classify_ref(sid, presented_ids, skillbook_ids, skillbook_sections),
                instance_id=instance_id,
                iteration=iteration,
                phase=phase,
                run_dir=run_name,
                message_index=idx,
                context=prose_content[start:end].replace("\n", " "),
            ))

    return hits, injected, prose, prose_refs


def is_split_run(run_dir: Path) -> bool:
    """Check if a run directory contains split experiment data."""
    for traj_dir in run_dir.rglob("trajectories"):
        if traj_dir.is_dir():
            for subdir in traj_dir.iterdir():
                if subdir.is_dir() and subdir.name in SPLIT_PHASES:
                    return True
    return False


def analyze_runs(run_dirs: list[Path], *, include_prose: bool = True) -> list[RunSkillReport]:
    """Analyze skill references across one or more run directories."""
    reports = []
    for run_dir in run_dirs:
        run_name = run_dir.name
        split = is_split_run(run_dir)
        traj_files = find_trajectory_files(run_dir, include_iter0=split)
        if not traj_files:
            continue

        # Run-level skillbook universe, used to classify prose references.
        skillbook_ids = load_skillbook_ids(run_dir) if include_prose else set()

        report = RunSkillReport(run_dir=run_name, is_split=split, total_trajectories=len(traj_files))
        instances_with_refs = set()
        prose_traj = 0

        for traj_path in traj_files:
            hits, injected, prose, prose_refs = scan_trajectory(
                traj_path, run_name, include_prose=include_prose, skillbook_ids=skillbook_ids)
            if hits:
                instances_with_refs.add(hits[0].instance_id)
                report.hits.extend(hits)
            if injected:
                instances_with_refs.add(injected[0].instance_id)
                report.injected.extend(injected)
            if prose:
                prose_traj += 1
            report.prose.extend(prose)
            report.prose_refs.extend(prose_refs)

        report.trajectories_with_refs = len(instances_with_refs)
        report.total_refs = len(report.hits) + len(report.injected)
        report.trajectories_with_prose = prose_traj
        report.total_prose = len(report.prose)
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

        if report.total_prose or report.prose_refs:
            prose_instances = len({p.instance_id for p in report.prose})
            print(f"  Prose engagement: {report.total_prose} 'skill' word mentions "
                  f"across {report.trajectories_with_prose} trajectories ({prose_instances} instances)")
            if report.prose_refs:
                by_cls = Counter(r.classification for r in report.prose_refs)
                breakdown = ", ".join(f"{by_cls[c]} {c}" for c in PROSE_REF_CLASSES if by_cls[c])
                ref_inst = len({r.instance_id for r in report.prose_refs})
                print(f"  Prose skill-ID references: {len(report.prose_refs)} "
                      f"({ref_inst} instances) — {breakdown}")

        if report.injected:
            _print_injected_summary(report)
        if report.hits:
            _print_ref_summary(report)
        if report.prose or report.prose_refs:
            _print_prose_summary(report)
        if not report.injected and not report.hits and not report.prose and not report.prose_refs:
            print("  No skill references, injections, or prose mentions found.")

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


def _print_prose_summary(report: RunSkillReport) -> None:
    """Print summary of prose engagement and classified skill-ID references."""
    by_phase: dict[str | None, list[ProseRef]] = defaultdict(list)
    for r in report.prose_refs:
        by_phase[r.phase].append(r)

    for phase in sorted(by_phase, key=lambda p: (p is None, p or "")):
        refs = by_phase[phase]
        phase_label = phase or "unknown"
        by_cls = Counter(r.classification for r in refs)
        instances = {r.instance_id for r in refs}

        print(f"\n  Phase: {phase_label} — prose skill-ID references: {len(refs)} "
              f"({len(instances)} instances)")
        print("  by class: " + ", ".join(f"{by_cls[c]} {c}" for c in PROSE_REF_CLASSES if by_cls[c]))

        # Detail tables for genuine skill references only.
        labels = {"presented": "Presented (injected + referenced)",
                  "skillbook": "In skillbook, NOT injected",
                  "unknown": "Valid section, no such skill (mis-cited?)"}
        for cls in SKILL_REF_CLASSES:
            cls_refs = [r for r in refs if r.classification == cls]
            if not cls_refs:
                continue
            counts = Counter(r.skill_id for r in cls_refs)
            sorted_ids = sorted(counts.items(), key=lambda x: -x[1])
            print(f"  {labels[cls]}:")
            print(f"    {'Skill ID':<40} {'Refs':>5}  {'Instances':>9}")
            print(f"    {'-' * 40} {'-' * 5}  {'-' * 9}")
            for sid, c in sorted_ids[:20]:
                inst_count = len({r.instance_id for r in cls_refs if r.skill_id == sid})
                print(f"    {sid:<40} {c:>5}  {inst_count:>9}")
            if len(sorted_ids) > 20:
                print(f"    ... and {len(sorted_ids) - 20} more")

        # Look-alikes (non-skill sections, e.g. bpo-43882): count + top few only.
        lookalikes = [r for r in refs if r.classification == "lookalike"]
        if lookalikes:
            counts = Counter(r.skill_id for r in lookalikes)
            top = ", ".join(f"{sid} x{c}" for sid, c in counts.most_common(5))
            more = f" (+{len(counts) - 5} more)" if len(counts) > 5 else ""
            print(f"  Look-alike tokens (non-skill sections, ignored): "
                  f"{len(lookalikes)} refs{more} — {top}")

        # Sample contexts from genuine skill references.
        samples = [r for r in refs if r.classification != "lookalike"][:6]
        if samples:
            print("  Sample contexts:")
            for r in samples:
                snippet = r.context if len(r.context) <= 140 else r.context[:140] + "..."
                print(f"    iter {r.iteration} [{r.classification}]: ...{snippet}...")


def print_by_instance(reports: list[RunSkillReport]) -> None:
    """Print per-instance breakdown of skill references."""
    print("=" * 70)
    print("Skill References by Instance")
    print("=" * 70)

    for report in reports:
        if not report.injected and not report.hits and not report.prose and not report.prose_refs:
            continue

        by_instance: dict[str, dict] = defaultdict(
            lambda: {"hits": [], "injected": [], "prose": [], "refs": []})
        for hit in report.hits:
            by_instance[hit.instance_id]["hits"].append(hit)
        for inj in report.injected:
            by_instance[inj.instance_id]["injected"].append(inj)
        for p in report.prose:
            by_instance[p.instance_id]["prose"].append(p)
        for r in report.prose_refs:
            by_instance[r.instance_id]["refs"].append(r)

        print(f"\n  Run: {report.run_dir}")
        for instance_id in sorted(by_instance):
            entry = by_instance[instance_id]
            injected_ids = sorted(set(i.skill_id for i in entry["injected"]))
            ref_ids = sorted(set(h.skill_id for h in entry["hits"]))
            phase = (entry["injected"] or entry["hits"] or entry["prose"] or entry["refs"])[0].phase or "?"

            parts = []
            if injected_ids:
                parts.append(f"{len(injected_ids)} skills injected")
            if ref_ids:
                parts.append(f"{len(ref_ids)} bracket-refs")
            if entry["prose"]:
                parts.append(f"{len(entry['prose'])} prose mentions")
            if entry["refs"]:
                parts.append(f"{len(entry['refs'])} prose skill-refs")
            print(f"\n  [{phase}] {instance_id}  ({', '.join(parts)})")

            for s in injected_ids[:10]:
                print(f"    injected: {s}")
            if len(injected_ids) > 10:
                print(f"    ... and {len(injected_ids) - 10} more")
            for s in ref_ids:
                count = sum(1 for h in entry["hits"] if h.skill_id == s)
                print(f"    bracket-ref: {s}  x{count}")
            for r in entry["refs"][:10]:
                snippet = r.context if len(r.context) <= 100 else r.context[:100] + "..."
                print(f"    prose-ref [{r.classification}] {r.skill_id}: ...{snippet}...")
            if len(entry["refs"]) > 10:
                print(f"    ... and {len(entry['refs']) - 10} more prose skill-refs")

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
            "prose_mentions": report.total_prose,
            "trajectories_with_prose": report.trajectories_with_prose,
            "prose_skill_refs": len(report.prose_refs),
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

        # Explicit bracket references
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

        # Prose skill-ID references, grouped by phase and classification
        if report.prose_refs:
            entry["prose_skill_refs_by_phase"] = {}
            by_phase_refs: dict[str | None, list[ProseRef]] = defaultdict(list)
            for r in report.prose_refs:
                by_phase_refs[r.phase].append(r)
            for phase in sorted(by_phase_refs, key=lambda p: (p is None, p or "")):
                refs = by_phase_refs[phase]
                cls_map: dict[str, dict] = defaultdict(
                    lambda: {"count": 0, "skills": defaultdict(lambda: {"count": 0, "instances": set()})})
                for r in refs:
                    cm = cls_map[r.classification]
                    cm["count"] += 1
                    cm["skills"][r.skill_id]["count"] += 1
                    cm["skills"][r.skill_id]["instances"].add(r.instance_id)
                entry["prose_skill_refs_by_phase"][phase or "unknown"] = {
                    "total": len(refs),
                    "instances": len({r.instance_id for r in refs}),
                    "by_class": {
                        cls: {
                            "count": cm["count"],
                            "skills": {
                                sid: {"count": se["count"], "instances": sorted(se["instances"])}
                                for sid, se in sorted(cm["skills"].items(), key=lambda x: -x[1]["count"])
                            },
                        }
                        for cls, cm in cls_map.items()
                    },
                }

        data.append(entry)

    text = json.dumps(data, indent=2)
    if save_path:
        Path(save_path).write_text(text)
        print(f"Saved JSON to {save_path}")
    else:
        print(text)


def export_csv(reports: list[RunSkillReport], output_path: Path) -> None:
    """Export all hits, injections, prose mentions, and prose skill-refs to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_dir", "instance_id", "iteration", "phase", "type",
                         "skill_id", "classification", "message_index", "context"])
        for report in reports:
            for hit in report.hits:
                writer.writerow([hit.run_dir, hit.instance_id, hit.iteration, hit.phase or "",
                                 "ref", hit.skill_id, "", hit.message_index, hit.context])
            for inj in report.injected:
                writer.writerow([inj.run_dir, inj.instance_id, inj.iteration, inj.phase or "",
                                 "injected", inj.skill_id, "", "", ""])
            for p in report.prose:
                writer.writerow([p.run_dir, p.instance_id, p.iteration, p.phase or "",
                                 "prose", "", "", p.message_index, p.context])
            for r in report.prose_refs:
                writer.writerow([r.run_dir, r.instance_id, r.iteration, r.phase or "",
                                 "prose_ref", r.skill_id, r.classification, r.message_index, r.context])
    total = sum(r.total_refs + r.total_prose + len(r.prose_refs) for r in reports)
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
    parser.add_argument(
        "--prose/--no-prose", dest="prose", default=True,
        help="Also detect non-bracket prose 'skill' mentions and skill-ID references "
             "(default: on).",
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

    reports = analyze_runs(run_dirs, include_prose=args.prose)

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
