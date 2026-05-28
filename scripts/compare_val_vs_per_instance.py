#!/usr/bin/env python3
"""Compare val instances from repos-split runs against per-instance (flat) runs.

Collects all val instance IDs from repos-split runs (via statistics_per_repo/*.json),
then checks resolution status across repos-split runs (val_baseline/val_skillbook phases)
and per-instance runs. Outputs a markdown report with summary and per-instance detail tables.

Usage:
    python scripts/compare_val_vs_per_instance.py \
      --repos-split-dirs data/run_repos_split_default data/run_repos_split_swe \
      --per-instance-dirs data/run_baseline data/run_qwen3_4a_swe data/run_qwen3_4a_default \
      --output docs/val_vs_per_instance_comparison.md
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers reused / adapted from compare_runs.py
# ---------------------------------------------------------------------------

def _find_benchmark_dir(run_dir: Path) -> Path | None:
    """Find the benchmark subdirectory (e.g. princeton-nlp__SWE-bench_Lite)."""
    for sub in run_dir.iterdir():
        if sub.is_dir() and "__" in sub.name:
            return sub
    return None


def _extract_phase_data(stats: dict, phase_key: str) -> dict:
    """Extract per-phase data from a statistics dict."""
    ps = stats.get(phase_key, {})
    return {
        "total": ps.get("total_instances", 0),
        "resolved": ps.get("resolved_count", 0),
        "rate": ps.get("resolution_rate", 0.0),
        "resolved_ids": set(ps.get("resolved_ids", [])),
        "unresolved_ids": set(ps.get("unresolved_ids", [])),
    }


def _load_per_repo_stats(run_dir: Path) -> dict[str, dict]:
    """Load all per-repo statistics from statistics_per_repo/*.json.

    Returns dict mapping repo name (e.g. "django/django") to stats dict.
    """
    per_repo_dir = run_dir / "statistics_per_repo"
    if not per_repo_dir.exists():
        return {}

    result = {}
    for path in sorted(per_repo_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        # Convert "django__django.json" -> "django/django"
        repo_name = path.stem.replace("__", "/", 1)
        result[repo_name] = data
    return result


def _instance_repo(instance_id: str) -> str:
    """Extract repo from instance_id (e.g. 'django__django-12345' -> 'django/django')."""
    # Instance IDs are like "owner__repo-number"
    parts = instance_id.rsplit("-", 1)
    if len(parts) == 2:
        return parts[0].replace("__", "/", 1)
    return instance_id


def _check_resolved_on_disk(run_dir: Path, instance_id: str, phases: list[str] | None = None) -> bool | None:
    """Check if an instance is resolved by reading result files on disk.

    For flat runs: looks in <run_dir>/<benchmark>/results/<instance>/iter_*.json
    For phase runs: looks in <run_dir>/<benchmark>/results/<phase>/<instance>/iter_*.json

    Returns True if any iteration resolved, False if found but not resolved, None if not found.
    """
    bench_dir = _find_benchmark_dir(run_dir)
    if bench_dir is None:
        return None

    results_dir = bench_dir / "results"
    if not results_dir.exists():
        return None

    found_any = False

    if phases:
        # Split-mode: check specific phase subdirs
        for phase in phases:
            inst_dir = results_dir / phase / instance_id
            if not inst_dir.exists():
                continue
            for iter_file in sorted(inst_dir.glob("iter_*.json")):
                try:
                    with open(iter_file) as f:
                        r = json.load(f)
                    found_any = True
                    if r.get("resolved"):
                        return True
                except (json.JSONDecodeError, OSError):
                    continue
    else:
        # Flat mode: check direct instance dir
        inst_dir = results_dir / instance_id
        if not inst_dir.exists():
            return None
        for iter_file in sorted(inst_dir.glob("iter_*.json")):
            try:
                with open(iter_file) as f:
                    r = json.load(f)
                found_any = True
                if r.get("resolved"):
                    return True
            except (json.JSONDecodeError, OSError):
                continue

    return False if found_any else None


def _load_run_label(run_dir: Path) -> str:
    """Load a short label from config.json for a run."""
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            name = config.get("experiment", {}).get("name", "")
            if name:
                return name
        except (json.JSONDecodeError, OSError):
            pass
    return run_dir.name


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_val_instances(repos_split_dirs: list[Path]) -> set[str]:
    """Collect the union of all val instance IDs across repos-split runs."""
    all_val_ids: set[str] = set()

    for run_dir in repos_split_dirs:
        per_repo = _load_per_repo_stats(run_dir)
        for _repo, stats in per_repo.items():
            vb = _extract_phase_data(stats, "val_baseline_phase")
            vs = _extract_phase_data(stats, "val_skillbook_phase")
            all_val_ids |= vb["resolved_ids"] | vb["unresolved_ids"]
            all_val_ids |= vs["resolved_ids"] | vs["unresolved_ids"]

    return all_val_ids


def get_repos_split_resolution(run_dir: Path) -> dict[str, dict[str, bool | None]]:
    """Get resolution status for each val instance in a repos-split run.

    Returns: {instance_id: {"val_baseline": bool|None, "val_skillbook": bool|None}}
    """
    result: dict[str, dict[str, bool | None]] = {}
    per_repo = _load_per_repo_stats(run_dir)

    for _repo, stats in per_repo.items():
        vb = _extract_phase_data(stats, "val_baseline_phase")
        vs = _extract_phase_data(stats, "val_skillbook_phase")

        all_ids = vb["resolved_ids"] | vb["unresolved_ids"] | vs["resolved_ids"] | vs["unresolved_ids"]

        for inst_id in all_ids:
            entry = result.get(inst_id, {})
            # val_baseline: resolved if in resolved_ids
            if inst_id in vb["resolved_ids"]:
                entry["val_baseline"] = True
            elif inst_id in vb["unresolved_ids"]:
                entry["val_baseline"] = False
            else:
                entry["val_baseline"] = None
            # val_skillbook: resolved if in resolved_ids
            if inst_id in vs["resolved_ids"]:
                entry["val_skillbook"] = True
            elif inst_id in vs["unresolved_ids"]:
                entry["val_skillbook"] = False
            else:
                entry["val_skillbook"] = None

            result[inst_id] = entry

    # Fallback: check disk for any instance still None
    for inst_id, entry in result.items():
        if entry.get("val_baseline") is None:
            entry["val_baseline"] = _check_resolved_on_disk(run_dir, inst_id, phases=["val_baseline"])
        if entry.get("val_skillbook") is None:
            entry["val_skillbook"] = _check_resolved_on_disk(run_dir, inst_id, phases=["val"])

    return result


def get_per_instance_resolution(run_dir: Path, val_ids: set[str]) -> dict[str, bool | None]:
    """Get resolution status for val instances in a flat per-instance run.

    Returns: {instance_id: bool|None}
    """
    # First try statistics.json
    stats_path = run_dir / "statistics.json"
    resolved_from_stats: set[str] = set()
    all_from_stats: set[str] = set()

    if stats_path.exists():
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            resolved_from_stats = set(stats.get("resolved_ids", []))
            all_from_stats = resolved_from_stats | set(stats.get("unresolved_ids", []))
        except (json.JSONDecodeError, OSError):
            pass

    result: dict[str, bool | None] = {}
    for inst_id in val_ids:
        if inst_id in resolved_from_stats:
            result[inst_id] = True
        elif inst_id in all_from_stats:
            result[inst_id] = False
        else:
            # Not in stats -- try disk fallback
            result[inst_id] = _check_resolved_on_disk(run_dir, inst_id, phases=None)

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _bool_to_str(val: bool | None, true_str: str = "yes", false_str: str = "no", none_str: str = "-") -> str:
    if val is True:
        return true_str
    elif val is False:
        return false_str
    return none_str


def generate_report(
    repos_split_dirs: list[Path],
    per_instance_dirs: list[Path],
    val_ids: set[str],
    repos_split_data: list[dict[str, dict[str, bool | None]]],
    per_instance_data: list[dict[str, bool | None]],
) -> str:
    """Generate the full markdown report."""
    lines: list[str] = []

    # Assign short labels
    rs_labels = [_load_run_label(d) for d in repos_split_dirs]
    rs_short = [f"split-{i+1}" for i in range(len(repos_split_dirs))]
    pi_labels = [_load_run_label(d) for d in per_instance_dirs]
    pi_short = [f"per-inst-{i+1}" for i in range(len(per_instance_dirs))]

    sorted_val_ids = sorted(val_ids, key=lambda x: (_instance_repo(x), x))

    # --- Header ---
    lines.append("# Val vs Per-Instance Comparison Report")
    lines.append("")
    lines.append(f"Total val instances: {len(val_ids)}")
    lines.append("")

    # --- Run legend ---
    lines.append("## Run Legend")
    lines.append("")
    lines.append("| Short ID | Label | Directory |")
    lines.append("|----------|-------|-----------|")
    for short, label, d in zip(rs_short, rs_labels, repos_split_dirs):
        lines.append(f"| {short} | {label} | `{d.name}` |")
    for short, label, d in zip(pi_short, pi_labels, per_instance_dirs):
        lines.append(f"| {short} | {label} | `{d.name}` |")
    lines.append("")

    # --- Summary table ---
    lines.append("## Summary Table")
    lines.append("")
    lines.append("How many of the val instances were resolved in each run.")
    lines.append("")

    # Count repos
    repos_with_val = sorted({_instance_repo(i) for i in val_ids})

    summary_header = "| Run | Type | Val Resolved | Val Total | Rate |"
    summary_sep = "|-----|------|-------------|-----------|------|"
    lines.append(summary_header)
    lines.append(summary_sep)

    for idx, data in enumerate(repos_split_data):
        # For repos-split, combine val_baseline and val_skillbook counts
        # Show val_skillbook as the primary metric
        vb_resolved = sum(1 for v in data.values() if v.get("val_baseline") is True)
        vs_resolved = sum(1 for v in data.values() if v.get("val_skillbook") is True)
        total = len(val_ids)
        lines.append(
            f"| {rs_short[idx]} | split (val_bl) | {vb_resolved} | {total} | "
            f"{vb_resolved/total*100:.1f}% |"
        )
        lines.append(
            f"| {rs_short[idx]} | split (val_sb) | {vs_resolved} | {total} | "
            f"{vs_resolved/total*100:.1f}% |"
        )

    for idx, data in enumerate(per_instance_data):
        resolved = sum(1 for v in data.values() if v is True)
        total = len(val_ids)
        lines.append(
            f"| {pi_short[idx]} | per-instance | {resolved} | {total} | "
            f"{resolved/total*100:.1f}% |"
        )

    lines.append("")

    # --- Per-repo summary ---
    lines.append("## Per-Repo Summary")
    lines.append("")

    repo_header = "| Repo | Val Count "
    for s in rs_short:
        repo_header += f"| {s} val_bl | {s} val_sb "
    for s in pi_short:
        repo_header += f"| {s} "
    repo_header += "|"

    repo_sep = "|------|-----------"
    for _ in rs_short:
        repo_sep += "|---------|---------"
    for _ in pi_short:
        repo_sep += "|---------"
    repo_sep += "|"

    lines.append(repo_header)
    lines.append(repo_sep)

    for repo in repos_with_val:
        repo_ids = {i for i in val_ids if _instance_repo(i) == repo}
        row = f"| {repo} | {len(repo_ids)} "

        for data in repos_split_data:
            vb_r = sum(1 for iid in repo_ids if data.get(iid, {}).get("val_baseline") is True)
            vs_r = sum(1 for iid in repo_ids if data.get(iid, {}).get("val_skillbook") is True)
            row += f"| {vb_r}/{len(repo_ids)} | {vs_r}/{len(repo_ids)} "

        for data in per_instance_data:
            r = sum(1 for iid in repo_ids if data.get(iid) is True)
            row += f"| {r}/{len(repo_ids)} "

        row += "|"
        lines.append(row)

    lines.append("")

    # --- Per-instance detail table ---
    lines.append("## Per-Instance Detail Table")
    lines.append("")

    detail_header = "| Instance ID | Repo "
    for s in rs_short:
        detail_header += f"| {s} val_bl | {s} val_sb "
    for s in pi_short:
        detail_header += f"| {s} "
    detail_header += "|"

    detail_sep = "|-------------|------"
    for _ in rs_short:
        detail_sep += "|---------|---------"
    for _ in pi_short:
        detail_sep += "|---------"
    detail_sep += "|"

    lines.append(detail_header)
    lines.append(detail_sep)

    for inst_id in sorted_val_ids:
        repo = _instance_repo(inst_id)
        row = f"| {inst_id} | {repo} "

        for data in repos_split_data:
            entry = data.get(inst_id, {})
            vb = _bool_to_str(entry.get("val_baseline"), "R", ".", "-")
            vs = _bool_to_str(entry.get("val_skillbook"), "R", ".", "-")
            row += f"| {vb} | {vs} "

        for data in per_instance_data:
            v = _bool_to_str(data.get(inst_id), "R", ".", "-")
            row += f"| {v} "

        row += "|"
        lines.append(row)

    lines.append("")
    lines.append("Legend: R = resolved, . = not resolved, - = not found")
    lines.append("")

    # --- Highlight interesting instances ---
    lines.append("## Analysis: Interesting Instances")
    lines.append("")

    # Instances resolved in ANY per-instance run but NOT in any repos-split val_baseline
    pi_resolved_any: set[str] = set()
    for data in per_instance_data:
        for iid, v in data.items():
            if v is True:
                pi_resolved_any.add(iid)

    rs_vb_resolved_any: set[str] = set()
    rs_vs_resolved_any: set[str] = set()
    for data in repos_split_data:
        for iid, entry in data.items():
            if entry.get("val_baseline") is True:
                rs_vb_resolved_any.add(iid)
            if entry.get("val_skillbook") is True:
                rs_vs_resolved_any.add(iid)

    rs_val_resolved_any = rs_vb_resolved_any | rs_vs_resolved_any

    # Per-instance resolved, but NOT in any repos-split val
    pi_only = sorted(pi_resolved_any - rs_val_resolved_any)
    lines.append(f"### Resolved in per-instance but NOT in repos-split val ({len(pi_only)} instances)")
    lines.append("")
    if pi_only:
        lines.append("| Instance ID | Repo "
                     + "".join(f"| {s} " for s in pi_short) + "|")
        lines.append("|-------------|------"
                     + "".join("|---------" for _ in pi_short) + "|")
        for iid in pi_only:
            repo = _instance_repo(iid)
            row = f"| {iid} | {repo} "
            for data in per_instance_data:
                v = _bool_to_str(data.get(iid), "R", ".", "-")
                row += f"| {v} "
            row += "|"
            lines.append(row)
    else:
        lines.append("(none)")
    lines.append("")

    # Repos-split val resolved, but NOT in any per-instance
    rs_only = sorted(rs_val_resolved_any - pi_resolved_any)
    lines.append(f"### Resolved in repos-split val but NOT in per-instance ({len(rs_only)} instances)")
    lines.append("")
    if rs_only:
        lines.append("| Instance ID | Repo "
                     + "".join(f"| {s} val_bl | {s} val_sb " for s in rs_short) + "|")
        lines.append("|-------------|------"
                     + "".join("|---------|---------" for _ in rs_short) + "|")
        for iid in rs_only:
            repo = _instance_repo(iid)
            row = f"| {iid} | {repo} "
            for data in repos_split_data:
                entry = data.get(iid, {})
                vb = _bool_to_str(entry.get("val_baseline"), "R", ".", "-")
                vs = _bool_to_str(entry.get("val_skillbook"), "R", ".", "-")
                row += f"| {vb} | {vs} "
            row += "|"
            lines.append(row)
    else:
        lines.append("(none)")
    lines.append("")

    # Instances resolved in val_skillbook but NOT in val_baseline (skillbook helped)
    sb_helped: dict[str, list[str]] = {}  # instance -> list of run labels where it helped
    for idx, data in enumerate(repos_split_data):
        for iid, entry in data.items():
            if entry.get("val_baseline") is not True and entry.get("val_skillbook") is True:
                sb_helped.setdefault(iid, []).append(rs_short[idx])

    lines.append(f"### Resolved by skillbook (val_sb but NOT val_bl) ({len(sb_helped)} instances)")
    lines.append("")
    if sb_helped:
        lines.append("| Instance ID | Repo | In runs |")
        lines.append("|-------------|------|---------|")
        for iid in sorted(sb_helped):
            repo = _instance_repo(iid)
            runs_str = ", ".join(sb_helped[iid])
            lines.append(f"| {iid} | {repo} | {runs_str} |")
    else:
        lines.append("(none)")
    lines.append("")

    # Instances lost by skillbook (val_bl resolved, val_sb not)
    sb_lost: dict[str, list[str]] = {}
    for idx, data in enumerate(repos_split_data):
        for iid, entry in data.items():
            if entry.get("val_baseline") is True and entry.get("val_skillbook") is not True:
                sb_lost.setdefault(iid, []).append(rs_short[idx])

    lines.append(f"### Lost by skillbook (val_bl resolved, val_sb NOT) ({len(sb_lost)} instances)")
    lines.append("")
    if sb_lost:
        lines.append("| Instance ID | Repo | In runs |")
        lines.append("|-------------|------|---------|")
        for iid in sorted(sb_lost):
            repo = _instance_repo(iid)
            runs_str = ", ".join(sb_lost[iid])
            lines.append(f"| {iid} | {repo} | {runs_str} |")
    else:
        lines.append("(none)")
    lines.append("")

    # --- Overlap matrix ---
    lines.append("## Overlap Summary")
    lines.append("")

    # Compare each pair of runs
    all_runs_labels = []
    all_runs_resolved_sets = []
    for idx, data in enumerate(repos_split_data):
        all_runs_labels.append(f"{rs_short[idx]} (val_bl)")
        all_runs_resolved_sets.append({iid for iid, e in data.items() if e.get("val_baseline") is True})
        all_runs_labels.append(f"{rs_short[idx]} (val_sb)")
        all_runs_resolved_sets.append({iid for iid, e in data.items() if e.get("val_skillbook") is True})
    for idx, data in enumerate(per_instance_data):
        all_runs_labels.append(pi_short[idx])
        all_runs_resolved_sets.append({iid for iid, v in data.items() if v is True})

    n = len(all_runs_labels)
    lines.append(f"Number of resolved val instances per run and pairwise overlaps:")
    lines.append("")
    for i in range(n):
        s_i = all_runs_resolved_sets[i]
        overlaps = []
        for j in range(n):
            if i == j:
                overlaps.append(str(len(s_i)))
            else:
                overlap = len(s_i & all_runs_resolved_sets[j])
                overlaps.append(str(overlap))
        lines.append(f"  {all_runs_labels[i]}: [{', '.join(overlaps)}]")

    lines.append("")
    lines.append("(Matrix is symmetric; diagonal = total resolved per run; off-diagonal = overlap count)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare val instances from repos-split runs vs per-instance runs"
    )
    parser.add_argument(
        "--repos-split-dirs", nargs="+", required=True, metavar="DIR",
        help="Directories for repos-split (iterate_repos) completed runs",
    )
    parser.add_argument(
        "--per-instance-dirs", nargs="+", required=True, metavar="DIR",
        help="Directories for per-instance (flat) completed runs",
    )
    parser.add_argument(
        "--output", "-o", default=None, metavar="FILE",
        help="Output file path (markdown). Prints to stdout if not specified.",
    )
    args = parser.parse_args()

    # Validate directories
    repos_split_dirs = []
    for p in args.repos_split_dirs:
        path = Path(p)
        if not path.exists():
            print(f"Error: repos-split dir not found: {path}", file=sys.stderr)
            sys.exit(1)
        repos_split_dirs.append(path)

    per_instance_dirs = []
    for p in args.per_instance_dirs:
        path = Path(p)
        if not path.exists():
            print(f"Error: per-instance dir not found: {path}", file=sys.stderr)
            sys.exit(1)
        per_instance_dirs.append(path)

    # Collect val instance IDs from repos-split runs
    val_ids = collect_val_instances(repos_split_dirs)
    if not val_ids:
        print("Error: no val instances found in repos-split runs", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(val_ids)} val instances across {len(repos_split_dirs)} repos-split run(s)",
          file=sys.stderr)

    # Repo breakdown
    repos = sorted({_instance_repo(i) for i in val_ids})
    for repo in repos:
        count = sum(1 for i in val_ids if _instance_repo(i) == repo)
        print(f"  {repo}: {count}", file=sys.stderr)

    # Get resolution data for repos-split runs
    repos_split_data: list[dict[str, dict[str, bool | None]]] = []
    for run_dir in repos_split_dirs:
        print(f"Loading repos-split run: {run_dir.name}", file=sys.stderr)
        data = get_repos_split_resolution(run_dir)
        repos_split_data.append(data)
        vb_count = sum(1 for v in data.values() if v.get("val_baseline") is True)
        vs_count = sum(1 for v in data.values() if v.get("val_skillbook") is True)
        print(f"  val_baseline resolved: {vb_count}/{len(val_ids)}, "
              f"val_skillbook resolved: {vs_count}/{len(val_ids)}", file=sys.stderr)

    # Get resolution data for per-instance runs
    per_instance_data: list[dict[str, bool | None]] = []
    for run_dir in per_instance_dirs:
        print(f"Loading per-instance run: {run_dir.name}", file=sys.stderr)
        data = get_per_instance_resolution(run_dir, val_ids)
        per_instance_data.append(data)
        res_count = sum(1 for v in data.values() if v is True)
        print(f"  resolved: {res_count}/{len(val_ids)}", file=sys.stderr)

    # Generate report
    report = generate_report(
        repos_split_dirs=repos_split_dirs,
        per_instance_dirs=per_instance_dirs,
        val_ids=val_ids,
        repos_split_data=repos_split_data,
        per_instance_data=per_instance_data,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n")
        print(f"Report written to {output_path}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
