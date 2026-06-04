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
        "pass_at_k": ps.get("pass_at_k", {}),
        "per_attempt_rate": ps.get("per_attempt_rate", {}),
    }


def _extract_exit_status(path: Path) -> str | None:
    """Extract info.exit_status from a trajectory file using ijson-like streaming.

    Avoids loading the full JSON (trajectories contain large message arrays).
    """
    import re
    # Read only enough of the file to find "info":{"exit_status":"..."}
    # The info object is typically near the top of the file
    with open(path) as f:
        chunk = f.read(4096)
    # Try to find exit_status in the first chunk
    m = re.search(r'"exit_status"\s*:\s*"([^"]*)"', chunk)
    if m:
        return m[1]
    # Fallback: null or missing
    if '"exit_status": null' in chunk or '"exit_status":null' in chunk:
        return None
    return "Unknown"


def _count_exit_statuses(run_dir: Path, instance_filter: set[str] | None = None) -> dict[str, dict[int, int]]:
    """Count exit statuses per iteration from trajectory files.

    Returns {exit_status: {iteration: count, ...}, ...}.
    """
    bench_dir = _find_benchmark_dir(run_dir)
    if bench_dir is None:
        return {}

    trajs_dir = bench_dir / "trajectories"
    if not trajs_dir.exists():
        return {}

    # Check for phase subdirs (split mode)
    known_phases = {"train", "val", "val_baseline"}
    phase_dirs = [d for d in trajs_dir.iterdir() if d.is_dir() and d.name in known_phases]

    result: dict[str, dict[int, int]] = {}
    scan_dirs = []
    if phase_dirs:
        for pd in phase_dirs:
            scan_dirs.extend(d for d in pd.iterdir() if d.is_dir())
    else:
        scan_dirs = [d for d in trajs_dir.iterdir() if d.is_dir()]

    for inst_dir in scan_dirs:
        if instance_filter is not None and inst_dir.name not in instance_filter:
            continue
        for fname in inst_dir.iterdir():
            if not fname.name.endswith(".json") or not fname.name.startswith("iter_"):
                continue
            it = int(fname.name.replace("iter_", "").replace(".json", ""))
            es = _extract_exit_status(fname)
            if es is None:
                continue
            result.setdefault(es, {}).setdefault(it, 0)
            result[es][it] += 1

    return result


def _total_exit_status_counts(es_data: dict[str, dict[int, int]]) -> dict[str, int]:
    """Sum exit status counts across all iterations: {status: total_count}."""
    return {status: sum(it_counts.values()) for status, it_counts in es_data.items()}


def _collect_all_statuses(rows_data: list[dict]) -> list[str]:
    """Collect sorted unique exit status names from all rows' exit_statuses."""
    statuses = set()
    for r in rows_data:
        statuses.update(r.get("exit_statuses", {}).keys())
    # Canonical order: Submitted first, then alphabetical
    priority = {"Submitted": 0}
    return sorted(statuses, key=lambda s: (priority.get(s, 1), s))


def _collect_all_statuses_from_es(es_list: list[dict]) -> list[str]:
    """Collect sorted unique exit status names from raw es_data dicts."""
    statuses = set()
    for es in es_list:
        statuses.update(es.keys())
    priority = {"Submitted": 0}
    return sorted(statuses, key=lambda s: (priority.get(s, 1), s))


def _fmt_exit_status_header(statuses: list[str]) -> str:
    """Format header: 'Submitted/LimitsExceeded/error'."""
    return "/".join(statuses) if statuses else "Exit Status"


def _fmt_exit_status_row(es_data: dict[str, dict[int, int]], statuses: list[str]) -> str:
    """Format row: '217/15/6' matching the header order."""
    totals = _total_exit_status_counts(es_data)
    parts = [str(totals.get(s, 0)) for s in statuses]
    return "/".join(parts) if parts else "-"


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

    skip_learn = exp.get("skip_learn", False)
    custom_swe = exp.get("skillbook", {}).get("custom_swe_learn", exp.get("custom_swe_learn", False))
    learn_phase = "custom_swe" if custom_swe else "default"
    if skip_learn:
        learn_phase = "no skillbook"
    elif is_baseline:
        learn_phase = "baseline"

    skillbook_assisted = stats.get("skillbook_assisted", {"count": 0, "ids": [], "by_iteration": {}})

    # Detect split mode
    is_split = "val_skillbook_phase" in stats
    split_data = {}
    if is_split:
        _nr = stats.get("summary", {}).get("newly_resolved_by_skillbook")
        _lt = stats.get("summary", {}).get("lost_by_skillbook")
        split_data = {
            "train": _extract_phase_data(stats, "train_phase"),
            "val_baseline": _extract_phase_data(stats, "val_baseline_phase"),
            "val_skillbook": _extract_phase_data(stats, "val_skillbook_phase"),
            "skillbook_improvement": stats.get("summary", {}).get("skillbook_improvement", "N/A"),
            "skillbook_improvement_pct": stats.get("summary", {}).get("skillbook_improvement_pct", "N/A"),
            "newly_resolved": _nr if _nr is not None else [],
            "lost": _lt if _lt is not None else [],
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

    train_trajs_dir = exp.get("train_trajs_dir")
    filter_repos = config.get("benchmark", {}).get("filter_repos")
    experiment_name = exp.get("name", "")

    # Count iter_0 resolved (for comparing baseline vs skillbook-assisted)
    iter0_resolved, iter0_total = _count_iter0_resolved(run_dir)

    # Exit status counts per iteration
    exit_statuses = _count_exit_statuses(run_dir)

    # Retrieval info
    retrieval = stats.get("retrieval", {})
    if not retrieval:
        retrieval_cfg = exp.get("skillbook", {}).get("retrieval", {})
        retrieval = {"enabled": retrieval_cfg.get("enabled", False), "top_k": retrieval_cfg.get("top_k")}

    # Step limit from agent config
    step_limit = config.get("agent", {}).get("step_limit", "N/A")

    # Detect iterate_repos mode
    is_iterate_repos = stats.get("mode") == "iterate_repos" or bool(
        config.get("benchmark", {}).get("iterate_repos")
    )
    repos = stats.get("repos", None)

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
        "is_iterate_repos": is_iterate_repos,
        "repos": repos,
        "train_trajs_dir": train_trajs_dir,
        "step_limit": step_limit,
        "exit_statuses": exit_statuses,
        "retrieval_enabled": retrieval.get("enabled", False),
        "retrieval_top_k": retrieval.get("top_k"),
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
    seen = set()
    for p in run_paths:
        path = Path(p).resolve()
        if str(path) in seen:
            continue
        seen.add(str(path))
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
    """SB Assist: count +Δpp [i0:N i1:N ...]"""
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

    iter_parts = [f"i{k}:{len(v)}" for k, v in sorted(by_iter.items(), key=lambda x: int(x[0]))]
    if iter_parts:
        iter_str = "\n[" + " ".join(iter_parts) + "]"
    else:
        iter_str = ""

    return f"{count}{delta_str}{iter_str}"


def _fmt_phase(pd: dict, distil: bool = False) -> str:
    """Format a phase dict as 'resolved/total rate% [p@1:N p@2:M ...] avg:N.N%'.
    If distil=True, prefix with 'distil'."""
    r, t = pd["resolved"], pd["total"]
    pct = f"{pd['rate'] * 100:.1f}%"
    base = f"{r}/{t} {pct}"
    if distil:
        base = f"distil {base}"

    pak = pd.get("pass_at_k", {})
    par = pd.get("per_attempt_rate", {})
    extra_lines = []

    if len(pak) > 1:
        # Show per-pass@k resolved counts (skip last since it equals overall resolved)
        parts = []
        for k_label in sorted(pak, key=lambda x: int(x.split("@")[1])):
            n = int(k_label.split("@")[1])
            info = pak[k_label]
            # Skip pass@k that matches the overall resolved count
            if n < len(pak):
                short_label = k_label.replace("pass@", "p@")
                parts.append(f"{short_label}:{info['count']}")
        if parts:
            extra_lines.append("[{0}]".format(", ".join(parts)))

    # Show average per-attempt resolution rate when multiple attempts exist
    if len(par) > 1:
        avg_rate = sum(v["rate"] for v in par.values()) / len(par)
        extra_lines.append(f"avg:{avg_rate * 100:.1f}%")

    if extra_lines:
        return base + "\n" + "\n".join(extra_lines)
    return base


def _fmt_delta_pp(delta) -> str:
    """Format a rate delta as '+N.Npp' / '-N.Npp'."""
    if delta is None or delta == "N/A":
        return "-"
    return f"{float(delta) * 100:+.1f}pp"


def _fmt_learn(r: dict) -> str:
    """Format Learn column: phase name + retrieval info on next line."""
    learn = r["learn_phase"]
    if r.get("retrieval_enabled") and r.get("retrieval_top_k") is not None:
        learn += f"\nret k={r['retrieval_top_k']}"
    return learn


def _print_table_rows(headers: list[str], rows: list[dict]):
    """Print a formatted table with auto-width columns, supporting multi-line cells."""
    # Compute widths: for multi-line cells, take max line width
    col_widths = {}
    for h in headers:
        max_w = len(h)
        for row in rows:
            val = row.get(h, "")
            for line in val.split("\n"):
                max_w = max(max_w, len(line))
        col_widths[h] = max_w

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)
    print(header_line)
    print(sep_line)
    for row in rows:
        # Split each cell into lines
        cell_lines = {}
        max_lines = 1
        for h in headers:
            lines = row.get(h, "").split("\n")
            cell_lines[h] = lines
            max_lines = max(max_lines, len(lines))
        for i in range(max_lines):
            parts = []
            for h in headers:
                line = cell_lines[h][i] if i < len(cell_lines[h]) else ""
                parts.append(line.ljust(col_widths[h]))
            print(" | ".join(parts))


_DATASET_ALIASES = {
    "SWE-bench_Lite": "lite",
    "SWE-bench_Verified": "verified",
}


def _shorten_dataset(dataset: str) -> str:
    """Short dataset alias (e.g. 'princeton-nlp/SWE-bench_Lite' → 'lite')."""
    short = dataset.rsplit("/", 1)[-1] if "/" in dataset else dataset
    return _DATASET_ALIASES.get(short, short)


def _load_per_repo_stats(run_dir: Path, repo: str) -> dict | None:
    """Load per-repo statistics from statistics_per_repo/<owner>__<repo>.json."""
    repo_filename = repo.replace("/", "__") + ".json"
    path = run_dir / "statistics_per_repo" / repo_filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def print_table(runs: list[dict], iteration: int | None = None, run_paths: list[str] | None = None):
    # Sort: baselines first, then by run dir name
    runs.sort(key=lambda r: (0 if r["is_baseline"] else 1, r["run_dir"]))

    # Assign short IDs
    run_id_map = {}
    for idx, r in enumerate(runs):
        tag = f"#{idx:03d}"
        run_id_map[r["run_dir"]] = tag

    # Map run_dir name -> full Path for loading per-repo files
    run_dir_paths: dict[str, Path] = {}
    if run_paths:
        for p in run_paths:
            path = Path(p)
            if path.exists():
                run_dir_paths[path.name] = path.resolve()

    _MODEL_ALIASES = {
        "Qwen3-Coder-30B": "qwen3coder",
        "Qwen3-Coder-Next-FP8": "qwen3coder-next",
        "glm-4.5-flash": "glm45-flash",
    }

    def model_short(m):
        base = m.split("/")[-1].replace("-Instruct", "").replace("-A3B", "") if m != "N/A" else "-"
        return _MODEL_ALIASES.get(base, base)

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
        flat_headers = ["ID", "Dataset", "Proc", "Unres", "Res", "Rate", "i0 Rate", "LLM", "Att", "Steps", "Learn", "SB Assist", "Traj Exit Status"]
        flat_rows = []
        flat_statuses = _collect_all_statuses(flat_runs)
        flat_headers = ["ID", "Dataset", "Proc", "Unres", "Res", "Rate", "i0 Rate", "LLM", "Att", "Steps", "Learn", "SB Assist", _fmt_exit_status_header(flat_statuses)]
        flat_rows = []
        for r in flat_runs:
            i0_r, i0_t = r["iter0_resolved"], r["iter0_total"]
            i0_str = f"{i0_r} {i0_r/i0_t*100:.1f}%" if i0_t > 0 else "-"
            flat_rows.append({
                "ID": run_id_map[r["run_dir"]],
                "Dataset": _shorten_dataset(r["benchmark"]),
                "Proc": str(r["processed_instances"]),
                "Unres": str(r["unresolved_count"]),
                "Res": str(r["resolved_count"]),
                "Rate": rate_str(r),
                "i0 Rate": i0_str,
                "LLM": llm_col(r),
                "Att": str(r["max_attempts"]),
                "Steps": str(r["step_limit"]),
                "Learn": _fmt_learn(r),
                "SB Assist": _format_sb_assist(r),
                _fmt_exit_status_header(flat_statuses): _fmt_exit_status_row(r["exit_statuses"], flat_statuses),
            })
        _print_table_rows(flat_headers, flat_rows)
        print()

    # --- Split runs: separate per_repo and global tables ---
    if split_runs:
        # per_repo = iterate_repos OR single-repo split (filter_repos set)
        # global = no filter_repos, all repos together
        per_repo_runs = [r for r in split_runs if r["is_iterate_repos"] or r.get("filter_repos")]
        global_runs = [r for r in split_runs if not r["is_iterate_repos"] and not r.get("filter_repos")]
        global_headers = ["ID", "Dataset", "Train", "ValBL", "ValSB", "SB Δ", "New/Lost", "LLM", "Learn", "Traj Exit Status"]
        all_details: list[tuple[str, dict]] = []

        # --- Per-repo table ---
        if per_repo_runs:
            print("Split-mode runs (per_repo):")
            per_repo_rows = []
            # First pass: collect all exit status data and build rows
            row_es_data: list[dict] = []  # parallel to per_repo_rows
            for r in per_repo_runs:
                parent_tag = run_id_map[r["run_dir"]]
                full_path = run_dir_paths.get(r["run_dir"])
                agg = r["split"]
                is_distil = bool(r.get("train_trajs_dir"))

                if r["is_iterate_repos"] and r.get("repos"):
                    # Aggregate row (from top-level statistics)
                    n_repos = len(r["repos"])

                    # When top-level stats lack newly_resolved/lost (e.g. validation-only
                    # or retrieval runs), aggregate from per-repo statistics files.
                    agg_nr = agg["newly_resolved"]
                    agg_lost = agg["lost"]
                    if not agg_nr and not agg_lost and full_path:
                        for repo in r["repos"]:
                            prd = _load_per_repo_stats(full_path, repo)
                            if prd:
                                agg_nr = agg_nr + prd.get("summary", {}).get("newly_resolved_by_skillbook", [])
                                agg_lost = agg_lost + prd.get("summary", {}).get("lost_by_skillbook", [])

                    per_repo_rows.append({
                        "ID": parent_tag,
                        "Dataset": _shorten_dataset(r["benchmark"]),
                        "Repo": f"{n_repos} repos",
                        "Train": _fmt_phase(agg["train"], distil=is_distil),
                        "ValBL": _fmt_phase(agg["val_baseline"]),
                        "ValSB": _fmt_phase(agg["val_skillbook"]),
                        "SB Δ": _fmt_delta_pp(agg["skillbook_improvement"]),
                        "New/Lost": f"{len(agg_nr)}/{len(agg_lost)}",
                        "LLM": llm_col(r),
                        "Learn": _fmt_learn(r),
                    })
                    row_es_data.append(r["exit_statuses"])
                    all_details.append((parent_tag, agg))

                    # Per-repo detail rows (no ID)
                    for repo in r["repos"]:
                        per_repo_data = None
                        if full_path:
                            per_repo_data = _load_per_repo_stats(full_path, repo)

                        if per_repo_data:
                            train_phase_raw = per_repo_data.get("train_phase", {})
                            repo_distil = bool(train_phase_raw.get("teacher_trajs_dir")) or is_distil
                            s = {
                                "train": _extract_phase_data(per_repo_data, "train_phase"),
                                "val_baseline": _extract_phase_data(per_repo_data, "val_baseline_phase"),
                                "val_skillbook": _extract_phase_data(per_repo_data, "val_skillbook_phase"),
                                "skillbook_improvement": per_repo_data.get("summary", {}).get("skillbook_improvement", "N/A"),
                                "skillbook_improvement_pct": per_repo_data.get("summary", {}).get("skillbook_improvement_pct", "N/A"),
                                "newly_resolved": per_repo_data.get("summary", {}).get("newly_resolved_by_skillbook", []),
                                "lost": per_repo_data.get("summary", {}).get("lost_by_skillbook", []),
                            }

                            # Compute per-repo exit status from trajectory files
                            repo_ids: set[str] = set()
                            for pk in ["train_phase", "val_baseline_phase", "val_skillbook_phase"]:
                                pd = per_repo_data.get(pk, {})
                                repo_ids.update(pd.get("resolved_ids", []))
                                repo_ids.update(pd.get("unresolved_ids", []))
                            repo_exit_statuses = _count_exit_statuses(full_path, instance_filter=repo_ids) if full_path and repo_ids else {}
                        else:
                            repo_distil = is_distil
                            s = {
                                "train": {"resolved": 0, "total": 0, "rate": 0.0, "resolved_ids": [], "unresolved_ids": []},
                                "val_baseline": {"resolved": 0, "total": 0, "rate": 0.0, "resolved_ids": [], "unresolved_ids": []},
                                "val_skillbook": {"resolved": 0, "total": 0, "rate": 0.0, "resolved_ids": [], "unresolved_ids": []},
                                "skillbook_improvement": "N/A",
                                "newly_resolved": [],
                                "lost": [],
                            }
                            repo_exit_statuses = {}

                        per_repo_rows.append({
                            "ID": "",
                            "Dataset": "",
                            "Repo": repo,
                            "Train": _fmt_phase(s["train"], distil=repo_distil),
                            "ValBL": _fmt_phase(s["val_baseline"]),
                            "ValSB": _fmt_phase(s["val_skillbook"]),
                            "SB Δ": _fmt_delta_pp(s["skillbook_improvement"]),
                            "New/Lost": f"{len(s['newly_resolved'])}/{len(s['lost'])}",
                            "LLM": "",
                            "Learn": "",
                        })
                        row_es_data.append(repo_exit_statuses)
                        all_details.append(("", s))
                else:
                    # Single-repo split run (filter_repos set but not iterate_repos)
                    repo = r["filter_repos"][0] if r.get("filter_repos") else "all"
                    per_repo_rows.append({
                        "ID": parent_tag,
                        "Dataset": _shorten_dataset(r["benchmark"]),
                        "Repo": repo,
                        "Train": _fmt_phase(agg["train"], distil=is_distil),
                        "ValBL": _fmt_phase(agg["val_baseline"]),
                        "ValSB": _fmt_phase(agg["val_skillbook"]),
                        "SB Δ": _fmt_delta_pp(agg["skillbook_improvement"]),
                        "New/Lost": f"{len(agg['newly_resolved'])}/{len(agg['lost'])}",
                        "LLM": llm_col(r),
                        "Learn": _fmt_learn(r),
                    })
                    row_es_data.append(r["exit_statuses"])
                    all_details.append((parent_tag, agg))

            # Second pass: collect all statuses and fill exit status column
            pr_statuses = _collect_all_statuses_from_es(row_es_data)
            es_header = _fmt_exit_status_header(pr_statuses)
            for row, es in zip(per_repo_rows, row_es_data):
                row[es_header] = _fmt_exit_status_row(es, pr_statuses)
            per_repo_headers = ["ID", "Dataset", "Repo", "Train", "ValBL", "ValSB", "SB Δ", "New/Lost", "LLM", "Learn", es_header]

            _print_table_rows(per_repo_headers, per_repo_rows)
            print()

        # --- Global table ---
        if global_runs:
            print("Split-mode runs (global):")
            global_statuses = _collect_all_statuses(global_runs)
            es_header = _fmt_exit_status_header(global_statuses)
            global_headers = ["ID", "Dataset", "Train", "ValBL", "ValSB", "SB Δ", "New/Lost", "LLM", "Learn", es_header]
            global_rows = []
            for r in global_runs:
                parent_tag = run_id_map[r["run_dir"]]
                s = r["split"]
                is_distil = bool(r.get("train_trajs_dir"))
                global_rows.append({
                    "ID": parent_tag,
                    "Dataset": _shorten_dataset(r["benchmark"]),
                    "Train": _fmt_phase(s["train"], distil=is_distil),
                    "ValBL": _fmt_phase(s["val_baseline"]),
                    "ValSB": _fmt_phase(s["val_skillbook"]),
                    "SB Δ": _fmt_delta_pp(s["skillbook_improvement"]),
                    "New/Lost": f"{len(s['newly_resolved'])}/{len(s['lost'])}",
                    "LLM": llm_col(r),
                    "Learn": _fmt_learn(r),
                    es_header: _fmt_exit_status_row(r["exit_statuses"], global_statuses),
                })
                all_details.append((parent_tag, s))

            _print_table_rows(global_headers, global_rows)
            print()

        # Print details for newly resolved / lost
        has_details = any(s.get("newly_resolved") or s.get("lost") for _, s in all_details)
        if has_details:
            for tag, s in all_details:
                if s.get("newly_resolved"):
                    print(f"  {tag} newly resolved by skillbook: {s['newly_resolved']}")
                if s.get("lost"):
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
        print_table(runs, iteration=args.iter, run_paths=args.runs)


if __name__ == "__main__":
    main()
