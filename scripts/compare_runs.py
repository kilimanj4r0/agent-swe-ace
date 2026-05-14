#!/usr/bin/env python3
"""Compare completed experiment runs and print a summary table.

Supports both flat runs and split-mode runs (train/val/val_baseline phases).

Usage:
    uv run python scripts/compare_runs.py data/run_20260415_020540 data/run_20260416_103210
    uv run python scripts/compare_runs.py data/run_*_completed
    uv run python scripts/compare_runs.py data/run_a data/run_b --json
    uv run python scripts/compare_runs.py data/run_a data/run_b --json out.json
    uv run python scripts/compare_runs.py data/run_a data/run_b --diff
    uv run python scripts/compare_runs.py data/run_a data/run_b --iter 0
    uv run python scripts/compare_runs.py data/run_a data/run_b --iter 0 --diff
    uv run python scripts/compare_runs.py data/run_*_split_*              # Split-mode table
    uv run python scripts/compare_runs.py data/run_split_a --phase val    # Only val_skillbook phase
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


def _find_benchmark_dir(run_dir: Path) -> Path | None:
    """Find the benchmark subdirectory (e.g. princeton-nlp__SWE-bench_Lite)."""
    for sub in run_dir.iterdir():
        if sub.is_dir() and "__" in sub.name:
            return sub
    return None


def _extract_phase_data(stats: dict, phase_key: str) -> dict:
    """Extract per-phase data from statistics.json."""
    ps = stats.get(phase_key, {})
    return {
        "total": ps.get("total_instances", 0),
        "resolved": ps.get("resolved_count", 0),
        "rate": ps.get("resolution_rate", 0.0),
        "resolved_ids": ps.get("resolved_ids", []),
        "unresolved_ids": ps.get("unresolved_ids", []),
    }


def _count_iter0_resolved(run_dir: Path) -> tuple[int, int]:
    """Count resolved instances at iter_0 from result files."""
    bench_dir = _find_benchmark_dir(run_dir)
    if bench_dir is None:
        return 0, 0

    results_dir = bench_dir / "results"
    if not results_dir.exists():
        return 0, 0

    resolved = 0
    total = 0

    # Check for phase subdirs (split mode)
    known_phases = {"train", "val", "val_baseline"}
    phase_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name in known_phases]

    scan_dirs = []
    if phase_dirs:
        for pd in phase_dirs:
            scan_dirs.extend(d for d in pd.iterdir() if d.is_dir())
    else:
        scan_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

    for inst_dir in scan_dirs:
        iter_file = inst_dir / "iter_0.json"
        if not iter_file.exists():
            continue
        total += 1
        with open(iter_file) as f:
            r = json.load(f)
        if r.get("resolved"):
            resolved += 1

    return resolved, total


def load_run(run_dir: Path, iteration: int | None = None, phase: str | None = None) -> dict | None:
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

    is_baseline = stats.get("config", {}).get("baseline", False)

    custom_swe = exp.get("skillbook", {}).get("custom_swe_learn", exp.get("custom_swe_learn", False))
    learn_phase = "custom_swe" if custom_swe else "default"
    if is_baseline:
        learn_phase = "baseline"

    skillbook_assisted = stats.get("skillbook_assisted", {"count": 0, "ids": [], "by_iteration": {}})

    # Detect split mode
    is_split = "val_skillbook_phase" in stats
    split_data = {}
    if is_split:
        split_data = {
            "train": _extract_phase_data(stats, "train_phase"),
            "val_baseline": _extract_phase_data(stats, "val_baseline_phase"),
            "val_skillbook": _extract_phase_data(stats, "val_skillbook_phase"),
            "skillbook_improvement": stats.get("summary", {}).get("skillbook_improvement", "N/A"),
            "skillbook_improvement_pct": stats.get("summary", {}).get("skillbook_improvement_pct", "N/A"),
            "newly_resolved": stats.get("summary", {}).get("newly_resolved_by_skillbook", []),
            "lost": stats.get("summary", {}).get("lost_by_skillbook", []),
        }

    # Compute duration
    duration_h = None
    m = re.match(r"run_(\d{8}_\d{6})", run_dir.name)
    if m:
        try:
            start = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            end = datetime.fromisoformat(stats["timestamp"])
            duration_h = round((end - start).total_seconds() / 3600, 1)
        except (KeyError, ValueError):
            pass

    filter_repos = config.get("benchmark", {}).get("filter_repos")
    experiment_name = exp.get("name", "")

    # Count iter_0 resolved (for comparing baseline vs skillbook-assisted)
    iter0_resolved, iter0_total = _count_iter0_resolved(run_dir)

    result = {
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
        "skillbook_mode": exp.get("skillbook", {}).get("mode", exp.get("skillbook_mode", "N/A")),
        "skillbook_assisted": skillbook_assisted,
        "concurrency": exp.get("concurrency", 1),
        "is_baseline": is_baseline,
        "duration_h": duration_h,
        "is_split": is_split,
        "split": split_data,
        "filter_repos": filter_repos,
        "experiment_name": experiment_name,
        "iter0_resolved": iter0_resolved,
        "iter0_total": iter0_total,
    }

    # --phase override: replace top-level data with specific phase
    if phase and is_split:
        phase_map = {
            "train": "train",
            "val_baseline": "val_baseline",
            "val": "val_skillbook",
        }
        phase_key = phase_map.get(phase)
        if phase_key and phase_key in split_data:
            pd = split_data[phase_key]
            result["total_instances"] = pd["total"]
            result["processed_instances"] = pd["total"]
            result["resolved_count"] = pd["resolved"]
            result["resolved_ids"] = pd["resolved_ids"]
            result["unresolved_count"] = pd["total"] - pd["resolved"]
            result["unresolved_ids"] = pd["unresolved_ids"]
            result["resolution_rate"] = pd["rate"]
            # Show in flat table when viewing a specific phase
            result["is_split"] = False

    # Override with per-iteration data if requested
    if iteration is not None:
        iter_data = _load_iteration_data(run_dir, iteration)
        if iter_data is not None:
            result.update(iter_data)

    return result


def _load_iteration_data(run_dir: Path, iteration: int) -> dict | None:
    """Load per-iteration resolved/unresolved data from result files."""
    bench_dir = _find_benchmark_dir(run_dir)
    if bench_dir is None:
        return None

    results_dir = bench_dir / "results"
    if not results_dir.exists():
        return None

    resolved_ids = []
    unresolved_ids = []
    processed = 0

    # Check for phase subdirs (split mode)
    known_phases = {"train", "val", "val_baseline"}
    phase_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name in known_phases]

    if phase_dirs:
        for phase_dir in phase_dirs:
            for inst_dir in sorted(phase_dir.iterdir()):
                if not inst_dir.is_dir():
                    continue
                iter_file = inst_dir / f"iter_{iteration}.json"
                if not iter_file.exists():
                    continue
                with open(iter_file) as f:
                    r = json.load(f)
                processed += 1
                if r.get("resolved"):
                    resolved_ids.append(inst_dir.name)
                else:
                    unresolved_ids.append(inst_dir.name)
    else:
        for inst_dir in sorted(results_dir.iterdir()):
            if not inst_dir.is_dir():
                continue
            iter_file = inst_dir / f"iter_{iteration}.json"
            if not iter_file.exists():
                continue
            with open(iter_file) as f:
                r = json.load(f)
            processed += 1
            if r.get("resolved"):
                resolved_ids.append(inst_dir.name)
            else:
                unresolved_ids.append(inst_dir.name)

    if processed == 0:
        return None

    return {
        "total_instances": processed,
        "processed_instances": processed,
        "resolved_count": len(resolved_ids),
        "resolved_ids": sorted(resolved_ids),
        "unresolved_count": len(unresolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "resolution_rate": len(resolved_ids) / processed if processed else 0.0,
    }


def load_runs_from_args(run_paths: list[str], iteration: int | None = None, phase: str | None = None) -> list[dict]:
    runs = []
    for p in run_paths:
        path = Path(p)
        if not path.exists():
            print(f"Path not found: {path}", file=sys.stderr)
            sys.exit(1)
        run = load_run(path, iteration=iteration, phase=phase)
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


def _format_sb_assist(r: dict) -> str:
    """SB Assist with delta: count (per-iter) +Δrate%pp"""
    sa = r["skillbook_assisted"]
    count = sa.get("count", 0)
    by_iter = sa.get("by_iteration", {})
    if count == 0:
        return "0"

    i0_r, i0_t = r["iter0_resolved"], r["iter0_total"]
    if i0_t > 0:
        delta = r["resolution_rate"] - (i0_r / i0_t)
        delta_str = f" {delta*100:+.1f}pp"
    else:
        delta_str = ""

    iter_counts = ",".join(f"i{k}:{len(v)}" for k, v in sorted(by_iter.items(), key=lambda x: int(x[0])))
    return f"{count} ({iter_counts}){delta_str}"


def _fmt_phase(pd: dict) -> str:
    """Format a phase dict as 'resolved/total rate%'."""
    r, t = pd["resolved"], pd["total"]
    pct = f"{pd['rate'] * 100:.1f}%"
    return f"{r}/{t} {pct}"


def _fmt_delta_pp(delta) -> str:
    """Format a rate delta as '+N.Npp' / '-N.Npp'."""
    if delta is None or delta == "N/A":
        return "-"
    return f"{float(delta) * 100:+.1f}pp"


def _print_table_rows(headers: list[str], rows: list[dict]):
    """Print a formatted table with auto-width columns."""
    col_widths = {}
    for h in headers:
        col_widths[h] = max(len(h), *(len(row.get(h, "")) for row in rows))

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)
    print(header_line)
    print(sep_line)
    for row in rows:
        line = " | ".join(row.get(h, "").ljust(col_widths[h]) for h in headers)
        print(line)


def print_table(runs: list[dict], iteration: int | None = None):
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

    # Partition into flat and split runs
    flat_runs = [r for r in runs if not r["is_split"]]
    split_runs = [r for r in runs if r["is_split"]]

    # --- Flat runs table ---
    if flat_runs:
        if split_runs:
            print("per_instance runs:")
        flat_headers = ["ID", "Proc", "Unres", "Res", "Rate", "i0 Rate", "LLM", "Att", "Time", "Learn", "SB Assist"]
        flat_rows = []
        for r in flat_runs:
            i0_r, i0_t = r["iter0_resolved"], r["iter0_total"]
            i0_str = f"{i0_r} {i0_r/i0_t*100:.1f}%" if i0_t > 0 else "-"
            flat_rows.append({
                "ID": run_id_map[r["run_dir"]],
                "Proc": str(r["processed_instances"]),
                "Unres": str(r["unresolved_count"]),
                "Res": str(r["resolved_count"]),
                "Rate": rate_str(r),
                "i0 Rate": i0_str,
                "LLM": llm_col(r),
                "Att": str(r["max_attempts"]),
                "Time": f"{r['duration_h']}h" if r["duration_h"] is not None else "-",
                "Learn": r["learn_phase"],
                "SB Assist": _format_sb_assist(r),
            })
        _print_table_rows(flat_headers, flat_rows)
        print()

    # --- Split runs table ---
    if split_runs:
        if flat_runs:
            print("Split-mode runs (per_repo/global):")
        split_headers = ["ID", "Repo", "Train", "ValBL", "ValSB", "SB Δ", "New/Lost", "LLM", "Time", "Learn"]
        split_rows = []
        for r in split_runs:
            s = r["split"]
            repos = r["filter_repos"]
            repo_str = ",".join(repos) if repos else "all"
            split_rows.append({
                "ID": run_id_map[r["run_dir"]],
                "Repo": repo_str,
                "Train": _fmt_phase(s["train"]),
                "ValBL": _fmt_phase(s["val_baseline"]),
                "ValSB": _fmt_phase(s["val_skillbook"]),
                "SB Δ": _fmt_delta_pp(s["skillbook_improvement"]),
                "New/Lost": f"{len(s['newly_resolved'])}/{len(s['lost'])}",
                "LLM": llm_col(r),
                "Time": f"{r['duration_h']}h" if r["duration_h"] is not None else "-",
                "Learn": r["learn_phase"],
            })
        _print_table_rows(split_headers, split_rows)

        # Print details for newly resolved / lost
        has_details = any(r["split"]["newly_resolved"] or r["split"]["lost"] for r in split_runs)
        if has_details:
            print()
            for r in split_runs:
                s = r["split"]
                tag = run_id_map[r["run_dir"]]
                if s["newly_resolved"]:
                    print(f"  {tag} newly resolved by skillbook: {s['newly_resolved']}")
                if s["lost"]:
                    print(f"  {tag} lost by skillbook: {s['lost']}")
        print()

    # Print ID -> run dir legend with experiment name
    if iteration is not None:
        print(f"  (Showing iter_{iteration} results)")
    for r in runs:
        name = r.get("experiment_name", "")
        name_tag = f"  ({name})" if name else ""
        print(f"  {run_id_map[r['run_dir']]}  {r['run_dir']}{name_tag}")


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

    # Split-mode comparison
    if a["is_split"] and b["is_split"]:
        print("\n=== split-mode val comparison ===")
        a_vs = a["split"]["val_skillbook"]
        b_vs = b["split"]["val_skillbook"]
        _section("val_skillbook resolved", _set(a_vs["resolved_ids"]), _set(b_vs["resolved_ids"]))

        print(f"\n  skillbook improvement:  {a_label}: {a['split']['skillbook_improvement_pct']}  "
              f"{b_label}: {b['split']['skillbook_improvement_pct']}")
        _section("newly resolved by skillbook",
                 _set(a["split"]["newly_resolved"]), _set(b["split"]["newly_resolved"]))
        _section("lost by skillbook",
                 _set(a["split"]["lost"]), _set(b["split"]["lost"]))
    elif a["is_split"] or b["is_split"]:
        flat_label = b_label if a["is_split"] else a_label
        print(f"\n  Note: {flat_label} is not a split run — showing train-phase data only")


def main():
    parser = argparse.ArgumentParser(description="Compare completed experiment runs")
    parser.add_argument("runs", nargs="+", metavar="RUN_DIR",
                        help="Run directories to compare")
    parser.add_argument("--json", nargs="?", const=True, default=False,
                        help="Output as JSON. Optionally specify a file path to save.")
    parser.add_argument("--diff", action="store_true",
                        help="Compare two runs: show overlapping/non-overlapping IDs")
    parser.add_argument("--iter", type=int, metavar="N", default=None,
                        help="Compare specific iteration N results instead of overall statistics")
    parser.add_argument("--phase", choices=["train", "val_baseline", "val"], default=None,
                        help="For split runs: show only this phase in the main table")
    args = parser.parse_args()

    runs = load_runs_from_args(args.runs, iteration=args.iter, phase=args.phase)
    if not runs:
        print("No valid runs found.", file=sys.stderr)
        sys.exit(1)

    if args.diff:
        print_diff(runs)
    elif args.json is not False:
        save_path = args.json if isinstance(args.json, str) else None
        print_json(runs, save_path=save_path)
    else:
        print_table(runs, iteration=args.iter)


if __name__ == "__main__":
    main()
