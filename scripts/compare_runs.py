#!/usr/bin/env python3
"""Compare completed experiment runs and print a summary table.

Usage:
    uv run python scripts/compare_runs.py data/run_20260415_020540 data/run_20260416_103210
    uv run python scripts/compare_runs.py data/run_*_completed
    uv run python scripts/compare_runs.py data/run_a data/run_b --json
    uv run python scripts/compare_runs.py data/run_a data/run_b --json out.json
    uv run python scripts/compare_runs.py data/run_a data/run_b --diff
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


def load_run(run_dir: Path) -> dict | None:
    stats_path = run_dir / "statistics.json"
    config_path = run_dir / "config.json"
    if not stats_path.exists() or not config_path.exists():
        return None

    with open(stats_path) as f:
        stats = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    exp = config.get("experiment", {})
    llm = config.get("llm", {})
    agent_llm = llm.get("agent", {}).get("model", "N/A")
    ace_llm = llm.get("ace", {}).get("model", "N/A")

    baseline_dir = stats.get("baseline_dir", None)
    has_baseline = baseline_dir is not None

    # Detect if this is a baseline run itself
    is_baseline = stats.get("config", {}).get("baseline", False)

    custom_swe = exp.get("custom_swe_learn", False)
    learn_phase = "custom_swe" if custom_swe else "default"
    if is_baseline:
        learn_phase = "baseline"

    skillbook_assisted = stats.get("skillbook_assisted", {"count": 0, "ids": [], "by_iteration": {}})

    # Compute duration: start from dir name (run_YYYYMMDD_HHMMSS), end from statistics.timestamp
    duration_h = None
    m = re.match(r"run_(\d{8}_\d{6})", run_dir.name)
    if m:
        try:
            start = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            end = datetime.fromisoformat(stats["timestamp"])
            duration_h = round((end - start).total_seconds() / 3600, 1)
        except (KeyError, ValueError):
            pass

    return {
        "run_dir": run_dir.name,
        "benchmark": config.get("benchmark", {}).get("dataset", "N/A"),
        "total_instances": stats.get("total_instances", "N/A"),
        "processed_instances": stats.get("processed_instances", stats.get("total_instances", "N/A")),
        "resolved_count": stats.get("resolved_count", 0),
        "resolved_ids": stats.get("resolved_ids", []),
        "unresolved_count": stats.get("unresolved_count", 0),
        "unresolved_ids": stats.get("unresolved_ids", []),
        "resolution_rate": stats.get("resolution_rate", 0.0),
        "agent_llm": agent_llm,
        "ace_llm": ace_llm,
        "max_attempts": exp.get("max_attempts", "N/A"),
        "learn_phase": learn_phase,
        "has_baseline_dir": has_baseline,
        "baseline_dir": baseline_dir,
        "skillbook_mode": exp.get("skillbook_mode", "N/A"),
        "skillbook_assisted": skillbook_assisted,
        "concurrency": exp.get("concurrency", 1),
        "is_baseline": is_baseline,
        "duration_h": duration_h,
    }


def load_runs_from_args(run_paths: list[str]) -> list[dict]:
    runs = []
    for p in run_paths:
        path = Path(p)
        if not path.exists():
            print(f"Path not found: {path}", file=sys.stderr)
            sys.exit(1)
        run = load_run(path)
        if run is None:
            print(f"Skipping (missing statistics.json or config.json): {path}", file=sys.stderr)
            continue
        runs.append(run)
    return runs


def format_assisted(sa: dict) -> str:
    """Compact skillbook assisted: total count + per-iteration breakdown."""
    count = sa.get("count", 0)
    by_iter = sa.get("by_iteration", {})
    if count == 0:
        return "0"
    iter_counts = ",".join(f"i{k}:{len(v)}" for k, v in sorted(by_iter.items(), key=lambda x: int(x[0])))
    return f"{count} ({iter_counts})"


def print_table(runs: list[dict]):
    # Sort: baselines first, then by run dir name
    runs.sort(key=lambda r: (0 if r["is_baseline"] else 1, r["run_dir"]))

    # Assign short IDs
    run_id_map = {}
    for idx, r in enumerate(runs):
        tag = f"#{idx:03d}"
        run_id_map[r["run_dir"]] = tag

    def model_short(m):
        return m.split("/")[-1].replace("-Instruct", "").replace("-A3B", "") if m != "N/A" else "-"

    def llm_col(r):
        a, b = model_short(r["agent_llm"]), model_short(r["ace_llm"])
        return a if a == b else f"{a}/{b}"

    rate_str = lambda r: f"{r['resolution_rate'] * 100:.1f}%"
    baseline_str = lambda r: "T" if r["has_baseline_dir"] else ("BL" if r["is_baseline"] else "F")

    rows = []
    for r in runs:
        rows.append({
            "ID": run_id_map[r["run_dir"]],
            "Proc": str(r["processed_instances"]),
            "Resolv": str(r["resolved_count"]),
            "Unres": str(r["unresolved_count"]),
            "Rate": rate_str(r),
            "LLM": llm_col(r),
            "Att": str(r["max_attempts"]),
            "Time": f"{r['duration_h']}h" if r["duration_h"] is not None else "-",
            "Learn": r["learn_phase"],
            "BL": baseline_str(r),
            "SB Assist": format_assisted(r["skillbook_assisted"]),
        })

    if not rows:
        print("No runs found.")
        return

    # Compute column widths
    headers = list(rows[0].keys())
    col_widths = {}
    for h in headers:
        col_widths[h] = max(len(h), *(len(row[h]) for row in rows))

    # Print header
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)
    print(header_line)
    print(sep_line)

    # Print rows
    for row in rows:
        line = " | ".join(row[h].ljust(col_widths[h]) for h in headers)
        print(line)

    # Print ID -> run dir legend
    print()
    for r in runs:
        print(f"  {run_id_map[r['run_dir']]}  {r['run_dir']}")


def print_json(runs: list[dict], save_path: str | None = None):
    runs.sort(key=lambda r: (0 if r["is_baseline"] else 1, r["run_dir"]))
    text = json.dumps(runs, indent=2)
    if save_path:
        Path(save_path).write_text(text)
        print(f"Saved JSON to {save_path}")
    else:
        print(text)


def print_diff(runs: list[dict]):
    if len(runs) != 2:
        print(f"--diff requires exactly 2 runs, got {len(runs)}", file=sys.stderr)
        sys.exit(1)

    a, b = runs[0], runs[1]
    a_label, b_label = a["run_dir"], b["run_dir"]

    def _set(val):
        return set(val) if isinstance(val, list) else set()

    a_res = _set(a["resolved_ids"])
    b_res = _set(b["resolved_ids"])
    a_unr = _set(a["unresolved_ids"])
    b_unr = _set(b["unresolved_ids"])
    a_sb = _set(a["skillbook_assisted"].get("ids", []))
    b_sb = _set(b["skillbook_assisted"].get("ids", []))

    def _section(label, a_ids, b_ids):
        only_a = sorted(a_ids - b_ids)
        only_b = sorted(b_ids - a_ids)
        both = sorted(a_ids & b_ids)
        print(f"  {label}  {a_label}: {len(a_ids)}  {b_label}: {len(b_ids)}  "
              f"overlap: {len(both)}  only {a_label}: {len(only_a)}  only {b_label}: {len(only_b)}")
        if both:
            print(f"    both ({len(both)}): {both}")
        if only_a:
            print(f"    only {a_label} ({len(only_a)}): {only_a}")
        if only_b:
            print(f"    only {b_label} ({len(only_b)}): {only_b}")

    print("\n=== same-category ===")
    _section("resolved", a_res, b_res)
    _section("unresolved", a_unr, b_unr)
    _section("skillbook_assisted", a_sb, b_sb)

    print("\n=== cross-category ===")
    _section(f"{a_label} resolved vs {b_label} unresolved", a_res, b_unr)
    _section(f"{a_label} unresolved vs {b_label} resolved", a_unr, b_res)


def main():
    parser = argparse.ArgumentParser(description="Compare completed experiment runs")
    parser.add_argument("runs", nargs="+", metavar="RUN_DIR",
                        help="Run directories to compare")
    parser.add_argument("--json", nargs="?", const=True, default=False,
                        help="Output as JSON. Optionally specify a file path to save.")
    parser.add_argument("--diff", action="store_true",
                        help="Compare two runs: show overlapping/non-overlapping IDs")
    args = parser.parse_args()

    runs = load_runs_from_args(args.runs)
    if not runs:
        print("No valid runs found.", file=sys.stderr)
        sys.exit(1)

    if args.diff:
        print_diff(runs)
    elif args.json is not False:
        save_path = args.json if isinstance(args.json, str) else None
        print_json(runs, save_path=save_path)
    else:
        print_table(runs)


if __name__ == "__main__":
    main()
