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

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Aggregated val-baseline reference for qwen3 split025 experiments. Built across
# 12 runs x 5 attempts = 60 attempts per instance (113 val instances) under a
# uniform empty-skillbook / no-learning condition. For matching runs we override
# the ValBL avg with this low-noise reference instead of a single run's noisy
# 5-attempt val_baseline, so the SB Delta compares against a stable baseline.
AGGREGATED_VAL_BASELINE_DIR = _REPO_ROOT / "data" / "val_baseline_aggregated_split025_vpk5_qwen3"
_aggregated_val_baseline_cache: dict | None | bool = None  # None=unloaded, False=missing


def _load_aggregated_val_baseline() -> dict | None:
    """Load (cached) aggregated val-baseline stats. Returns None if unavailable."""
    global _aggregated_val_baseline_cache
    if _aggregated_val_baseline_cache is not None:
        return _aggregated_val_baseline_cache if _aggregated_val_baseline_cache else None
    overall_path = AGGREGATED_VAL_BASELINE_DIR / "stats" / "overall.json"
    per_repo_path = AGGREGATED_VAL_BASELINE_DIR / "stats" / "per_repo.json"
    if not overall_path.exists() or not per_repo_path.exists():
        _aggregated_val_baseline_cache = False
        return None
    with open(overall_path) as f:
        overall = json.load(f)
    with open(per_repo_path) as f:
        per_repo = json.load(f)
    _aggregated_val_baseline_cache = {"overall": overall, "per_repo": per_repo}
    return _aggregated_val_baseline_cache


def _is_qwen3_coder(agent_llm: str) -> bool:
    """True for the Qwen3-Coder-30B agent model (excludes the Qwen3-Coder-Next variant)."""
    m = (agent_llm or "").lower()
    return "qwen3-coder" in m and "next" not in m


def _is_qwen3_split025(run: dict) -> bool:
    """True for qwen3 split025 experiments that share the aggregated baseline's val set."""
    return "split025" in run.get("run_dir", "") and _is_qwen3_coder(run.get("agent_llm", ""))


def _apply_aggregated_val_baseline(pd: dict, repo: str | None) -> None:
    """Override a val_baseline phase dict's avg with the aggregated reference.

    Sets pd['aggregated_avg'] = {'avg': rate, 'att_per_inst': N}. repo selects the
    per-repo value (iterate_repos / single-repo rows); None uses the overall value.
    No-op if the aggregated baseline is unavailable or the repo is missing.
    """
    agg = _load_aggregated_val_baseline()
    if agg is None:
        return
    if repo:
        entry = agg["per_repo"].get(repo.replace("/", "__"))
        if not entry:
            return
        avg = entry.get("avg")
        n_att = entry.get("n_attempts")
        n_inst = entry.get("n_instances")
    else:
        ov = agg["overall"]
        avg = ov.get("avg")
        n_att = ov.get("total_attempts")
        n_inst = ov.get("n_instances")
    if avg is None:
        return
    att_per_inst = round(n_att / n_inst) if n_inst else n_att
    pd["aggregated_avg"] = {"avg": avg, "att_per_inst": att_per_inst}


def _find_benchmark_dir(run_dir: Path) -> Path | None:
    """Find the benchmark subdirectory (e.g. princeton-nlp__SWE-bench_Lite)."""
    for sub in run_dir.iterdir():
        if sub.is_dir() and "__" in sub.name:
            return sub
    return None


def _resolve_run_dir(dir_ref: str | None) -> Path | None:
    """Resolve a run-dir reference to an existing path.

    Configs often store a shorthand prefix (e.g. ``data/run_20260605_111733``)
    while the actual dir is ``run_20260605_111733_completed_...``; fall back to a
    prefix glob when the exact path doesn't exist.
    """
    if not dir_ref:
        return None
    p = Path(dir_ref)
    if not p.is_absolute():
        p = _REPO_ROOT / dir_ref
    if p.exists() and p.is_dir():
        return p
    for cand in sorted(p.parent.glob(p.name + "*")):
        if cand.is_dir():
            return cand
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


def _merge_exit_statuses(*es_dicts: dict | None) -> dict:
    """Merge per-phase exit_statuses dicts into one combined {status: {iter: count}}.

    Reads the backfilled per-phase ``exit_statuses`` (e.g. train/val_baseline/
    val_skillbook) from a per-repo statistics file so we can show a per-repo
    Traj-Exit-Status cell without re-scanning every trajectory file. Tolerates
    string or int iteration keys (statistics.json round-trips them to strings).
    """
    merged: dict[str, dict] = {}
    for es in es_dicts:
        if not es:
            continue
        for status, it_counts in es.items():
            tgt = merged.setdefault(status, {})
            for it, c in it_counts.items():
                tgt[it] = tgt.get(it, 0) + c
    return merged


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


def _shorten_status(status: str) -> str:
    """Shorten exit status names for compact column headers."""
    _ALIASES = {
        "Submitted": "Sub",
        "LimitsExceeded": "Lim",
        "ContextWindowExceeded": "Ctx",
        "ModelRetryError": "Retry",
    }
    return _ALIASES.get(status, status)


def _fmt_exit_status_header(statuses: list[str]) -> str:
    """Format header: 'Sub/Lim/Ctx'."""
    return "/".join(_shorten_status(s) for s in statuses) if statuses else "Exit"


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


def _phase_results_dir(run_dir: Path) -> Path | None:
    """Return the <bench>/results dir for a run, or None if not found."""
    bench_dir = _find_benchmark_dir(run_dir)
    if bench_dir is None:
        return None
    results_dir = bench_dir / "results"
    return results_dir if results_dir.exists() else None


def _compute_per_attempt_rate(results_dir: Path, repo_prefix: str | None = None) -> dict:
    """Scan instance result dirs for per-iteration resolution.

    Returns a per_attempt_rate dict in the same schema main_loop writes:
    ``{"iter_0": {"resolved": N, "total": T, "rate": r}, ...}``.

    This is the per-attempt (per-iteration) rate -- how many instances resolved at
    each individual attempt -- as opposed to cumulative pass@k. Averaging these
    rates is the correct "avg per attempt"; averaging cumulative pass@k rates is
    inflated and wrong.

    Args:
        results_dir: a phase results dir (e.g. ``.../results/val_baseline``) whose
            children are per-instance dirs containing ``iter_N.json``.
        repo_prefix: if set (iterate_repos per-repo rows), only count instances
            whose id starts with this prefix (e.g. ``"django__django-"``).
    """
    if not results_dir.exists():
        return {}

    per_iter_counts: dict[int, int] = {}
    total = 0
    max_iter = -1
    for inst_dir in sorted(results_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        if repo_prefix and not inst_dir.name.startswith(repo_prefix):
            continue
        has_any = False
        for i in range(10):  # iterations are contiguous from iter_0
            f = inst_dir / f"iter_{i}.json"
            if not f.exists():
                break
            has_any = True
            max_iter = max(max_iter, i)
            with open(f) as fh:
                if json.load(fh).get("resolved"):
                    per_iter_counts[i] = per_iter_counts.get(i, 0) + 1
        if has_any:
            total += 1

    if total == 0 or max_iter < 0:
        return {}

    out = {}
    for i in range(max_iter + 1):
        resolved = per_iter_counts.get(i, 0)
        out[f"iter_{i}"] = {
            "resolved": resolved,
            "total": total,
            "rate": resolved / total,
        }
    return out


def _backfill_per_attempt_rate(pd: dict, results_dir: Path, repo_prefix: str | None = None) -> None:
    """Set ``pd['per_attempt_rate']`` from result files if not already present.

    Lets old runs (whose statistics.json only stored cumulative pass@k) show a
    correct per-attempt avg without re-running the experiment or the enrich script.
    """
    if pd.get("per_attempt_rate"):
        return
    par = _compute_per_attempt_rate(results_dir, repo_prefix=repo_prefix)
    if par:
        pd["per_attempt_rate"] = par


def _per_attempt_resolved_sets(results_dir: Path, repo_prefix: str | None = None) -> dict[int, set[str]]:
    """Per-iteration resolved instance-id sets: ``{iter_index: {instance_id, ...}}``.

    Sibling of ``_compute_per_attempt_rate`` that returns id *sets* (needed to
    split new vs lost per attempt) instead of counts. The stored
    ``per_attempt_rate`` only has counts, so it cannot separate gains from
    losses.

    Mirrors ``_compute_per_attempt_rate`` exactly: an instance is added to set[i]
    for EVERY iteration whose ``iter_i.json`` is resolved (breaking only on a
    missing file). Val instances run all K attempts rather than stopping on first
    resolve, so an instance CAN appear in several iterations' sets. This keeps
    the per-attempt identity ``avg_new - avg_lost == (avg_sb - avg_bl) * total``
    intact.
    """
    sets: dict[int, set[str]] = {}
    if not results_dir.exists():
        return sets
    for inst_dir in sorted(results_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        if repo_prefix and not inst_dir.name.startswith(repo_prefix):
            continue
        for i in range(10):  # iterations are contiguous from iter_0
            f = inst_dir / f"iter_{i}.json"
            if not f.exists():
                break
            with open(f) as fh:
                if json.load(fh).get("resolved"):
                    sets.setdefault(i, set()).add(inst_dir.name)
    return sets


def _avg_new_lost(bl_sets: dict[int, set[str]], sb_sets: dict[int, set[str]]) -> tuple[float, float]:
    """Average per-attempt new/lost counts across attempts.

    For each attempt index i: ``new_i = |SB_i \\ BL_i|``, ``lost_i = |BL_i \\ SB_i|``.
    Returns the mean of each over the union of attempt indices. This is the gross
    split; when both phases use the same baseline, ``avg_new - avg_lost`` equals
    the avg-rate delta applied to the totals (the SB Δ (avg) count line).
    """
    iters = sorted(set(bl_sets) | set(sb_sets))
    if not iters:
        return 0.0, 0.0
    n = len(iters)
    new = sum(len(sb_sets.get(i, set()) - bl_sets.get(i, set())) for i in iters) / n
    lost = sum(len(bl_sets.get(i, set()) - sb_sets.get(i, set())) for i in iters) / n
    return new, lost


def _fmt_new_lost(newly_resolved: list, lost: list, full_path: Path | None,
                  vpk: int, repo_prefix: str | None = None,
                  per_attempt_new_lost: dict | None = None) -> str:
    """New/Lost cell: line 1 accumulated set diff, optional line 2 avg new/lost.

    Line 1: ``{new}/{lost}`` over the accumulated (any-of-K) resolved sets.
    Line 2: ``avg N.N/M.N`` -- per-attempt set-diff averaged across attempts,
    shown only when ``vpk > 1`` and the run's own val_baseline per-iteration
    results exist. Uses each run's OWN val_baseline (the aggregated reference has
    no per-iteration sets), so ``avg_new - avg_lost`` need not match the
    SB Δ (avg) count when that count is built on the aggregated baseline.

    ``per_attempt_new_lost`` (``{avg_new, avg_lost}``) is read from the backfilled
    statistics.json summary when available; otherwise we fall back to computing it
    from the per-iteration result files. The avg line is shown whenever EITHER
    phase has per-iteration resolved data -- so a repo whose val_baseline resolved
    nothing still reports the skillbook's new resolves (the fallback guard matches
    the backfill's condition, keeping both paths identical).
    """
    cell = f"{len(newly_resolved)}/{len(lost)}"
    if vpk <= 1 or full_path is None:
        return cell

    avg_new = avg_lost = None
    if per_attempt_new_lost:
        avg_new = per_attempt_new_lost.get("avg_new")
        avg_lost = per_attempt_new_lost.get("avg_lost")
    if avg_new is None or avg_lost is None:
        # Fallback: derive from per-iteration result files.
        results_root = _phase_results_dir(full_path)
        if results_root is None:
            return cell
        bl_sets = _per_attempt_resolved_sets(results_root / "val_baseline", repo_prefix=repo_prefix)
        sb_sets = _per_attempt_resolved_sets(results_root / "val", repo_prefix=repo_prefix)
        if not bl_sets and not sb_sets:
            return cell  # no per-iteration data at all -> can't define new/lost
        avg_new, avg_lost = _avg_new_lost(bl_sets, sb_sets)
    return f"{cell}\navg {avg_new:.1f}/{avg_lost:.1f}"


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
            "per_attempt_new_lost": stats.get("summary", {}).get("per_attempt_new_lost"),
        }
        # Backfill per_attempt_rate from result files so the per-attempt avg is
        # correct. statistics.json stores only cumulative pass@k (whose mean is not
        # a valid per-attempt average); per-iteration rates are derived from files.
        # val_skillbook results live under the "val" subdir.
        _results_root = _phase_results_dir(run_dir)
        if _results_root is not None:
            _backfill_per_attempt_rate(split_data["val_baseline"], _results_root / "val_baseline")
            _backfill_per_attempt_rate(split_data["val_skillbook"], _results_root / "val")

        # Validation-only runs (skillbook_source_dir set) skip training, so their
        # own train_phase is empty (0/0). Inherit train stats from the source run
        # so the Train column reflects the skillbook's actual training results.
        _tr = split_data["train"]
        if _tr.get("total", 0) == 0:
            _src = exp.get("skillbook_source_dir")
            _src_path = _resolve_run_dir(_src) if _src else None
            _src_stats_path = _src_path / "statistics.json" if _src_path else None
            if _src_stats_path and _src_stats_path.exists():
                with open(_src_stats_path) as _f:
                    _sp = json.load(_f).get("train_phase", {})
                if _sp.get("total_instances", 0) > 0:
                    _tr["resolved"] = _sp.get("resolved_count", 0)
                    _tr["total"] = _sp.get("total_instances", 0)
                    _tr["rate"] = _sp.get("resolution_rate", 0.0)
                    _tr["resolved_ids"] = _sp.get("resolved_ids", [])
                    _tr["unresolved_ids"] = _sp.get("unresolved_ids", [])

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

    # Count iter_0 resolved — prefer the backfilled `iter0` in statistics.json,
    # fall back to scanning result files.
    _i0 = stats.get("iter0")
    if _i0 and _i0.get("total"):
        iter0_resolved, iter0_total = _i0["resolved"], _i0["total"]
    else:
        iter0_resolved, iter0_total = _count_iter0_resolved(run_dir)

    # Exit status counts per iteration — prefer the backfilled `exit_statuses` in
    # statistics.json, fall back to scanning trajectory files.
    exit_statuses = stats.get("exit_statuses") or _count_exit_statuses(run_dir)

    # Retrieval info. Prefer the config keyword (clean: llm/embedding/bm25/random)
    # over statistics.json, which stores retriever class names or omits type.
    retrieval = stats.get("retrieval", {})
    retrieval_cfg = exp.get("skillbook", {}).get("retrieval", {})
    if not retrieval:
        retrieval = {"enabled": retrieval_cfg.get("enabled", False), "top_k": retrieval_cfg.get("top_k")}
    retrieval_type_raw = retrieval_cfg.get("type") or retrieval.get("type")

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
        "retrieval_type": retrieval_type_raw,
        "val_pass_k": exp.get("val_pass_k", 1),
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


def _fmt_phase(pd: dict, distil: bool = False, val_pass_k: int = 1) -> str:
    """Format a phase dict. Line layout (each shown only when present):
      line 1: avg:N.N% [(agg Natt)]   -- per-attempt avg, shown first when vpk > 1
      line 2: resolved/total rate%    -- 'distil '-prefixed for distillation train
      line 3: [p@1:N, p@2:M, ...]     -- cumulative pass@k counts
    """
    r, t = pd["resolved"], pd["total"]
    pct = f"{pd['rate'] * 100:.1f}%"
    base = f"{r}/{t} {pct}"
    if distil:
        base = f"distil {base}"

    pak = pd.get("pass_at_k", {})
    par = pd.get("per_attempt_rate", {})
    lines = []

    # Per-attempt avg FIRST (the primary metric), only when vpk > 1. Sourced from
    # per_attempt_rate (per-iteration rates), never cumulative pass@k -- the mean
    # of monotonic pass@k rates is inflated, not a per-attempt average. An
    # aggregated_avg override (qwen3 split025 reference) takes precedence.
    if val_pass_k > 1:
        aavg = pd.get("aggregated_avg")
        if aavg:
            lines.append(f"avg:{aavg['avg'] * 100:.1f}% (agg {aavg['att_per_inst']}att)")
        elif len(par) > 1:
            avg_rate = sum(v["rate"] for v in par.values()) / len(par)
            lines.append(f"avg:{avg_rate * 100:.1f}%")

    lines.append(base)

    if len(pak) > 1:
        # Cumulative per-pass@k resolved counts
        parts = []
        for k_label in sorted(pak, key=lambda x: int(x.split("@")[1])):
            info = pak[k_label]
            short_label = k_label.replace("pass@", "p@")
            parts.append(f"{short_label}:{info['count']}")
        if parts:
            lines.append("[{0}]".format(", ".join(parts)))

    return "\n".join(lines)


def _fmt_delta_pp(delta) -> str:
    """Format a rate delta as '+N.Npp' / '-N.Npp'."""
    if delta is None or delta == "N/A":
        return "-"
    return f"{float(delta) * 100:+.1f}pp"


def _fmt_sb_delta(delta_rate, total: int) -> str:
    """SB Δ as two lines: '±N.Npp' then the same delta as a resolved-count diff.

    Line 1 is the rate delta in percentage points; line 2 is that delta expressed
    as an absolute resolved-instance difference (delta_rate * total, rounded).
    """
    if delta_rate is None or delta_rate == "N/A":
        return "-"
    dr = float(delta_rate)
    pp = _fmt_delta_pp(dr)
    count = round(dr * total) if total else 0
    cnt = f"{count:+d}" if count != 0 else "0"
    return f"{pp}\n{cnt}"


def _compute_avg_rate(pd: dict) -> float | None:
    """Average per-attempt resolution rate. Returns None if unavailable.

    An aggregated_avg override (qwen3 split025 reference baseline) takes
    precedence. Otherwise uses per_attempt_rate (per-iteration rates). Cumulative
    pass@k is intentionally NOT used: its rates are monotonic, so averaging them
    is inflated and does not represent a per-attempt average. per_attempt_rate is
    written by the runner (main_loop) or backfilled from result files in load_run.
    """
    if pd.get("aggregated_avg"):
        return pd["aggregated_avg"]["avg"]
    par = pd.get("per_attempt_rate", {})
    if len(par) > 1:
        return sum(v["rate"] for v in par.values()) / len(par)
    return None


def _aggregate_pass_at_k(per_repo_stats: list[dict], phase_key: str) -> dict[str, dict]:
    """Aggregate pass_at_k across repos by summing counts and totals."""
    agg: dict[str, dict] = {}
    for prd in per_repo_stats:
        pak = prd.get(phase_key, {}).get("pass_at_k", {})
        for k_label, info in pak.items():
            if k_label not in agg:
                agg[k_label] = {"count": 0, "total": 0, "rate": 0.0}
            agg[k_label]["count"] += info.get("count", 0)
            agg[k_label]["total"] += info.get("total", 0)
    # Recompute rates from aggregated counts
    for k_label in agg:
        t = agg[k_label]["total"]
        agg[k_label]["rate"] = agg[k_label]["count"] / t if t > 0 else 0.0
    return agg


def _retrieval_code(raw) -> str:
    """Map a retrieval type to a short code: llm / emb / bm25 / rand.

    Accepts either the config keyword (llm/embedding/bm25/random) or the
    retriever class name stored in statistics.json. Absent => llm (default).
    """
    if not raw:
        return "llm"
    s = str(raw).lower()
    if "embed" in s:
        return "emb"
    if "bm25" in s:
        return "bm25"
    if "random" in s:
        return "rand"
    return "llm"


def _fmt_learn(r: dict) -> str:
    """Learn column: skillbook learn phase, with retrieval type/k on the (Ret) line."""
    learn = r["learn_phase"]
    if r.get("retrieval_enabled"):
        code = _retrieval_code(r.get("retrieval_type"))
        k = r.get("retrieval_top_k")
        learn += f"\n{code} k{k}" if k is not None else f"\n{code}"
    else:
        learn += "\nno ret"
    return learn


def _print_table_rows(headers: list[str], rows: list[dict]):
    """Print a formatted table with auto-width columns, supporting multi-line
    headers and cells (newline-separated lines)."""
    # Compute widths: take the max line width across the header and all cells.
    col_widths = {}
    for h in headers:
        max_w = max((len(line) for line in h.split("\n")), default=0)
        for row in rows:
            for line in row.get(h, "").split("\n"):
                max_w = max(max_w, len(line))
        col_widths[h] = max_w

    def _join(cells):
        return " | ".join(cell.ljust(col_widths[h]) for h, cell in zip(headers, cells))

    # Header (may span multiple lines)
    header_lines = [h.split("\n") for h in headers]
    n_header_lines = max(len(lines) for lines in header_lines)
    for i in range(n_header_lines):
        print(_join(lines[i] if i < len(lines) else "" for lines in header_lines))
    print("-+-".join("-" * col_widths[h] for h in headers))

    for row in rows:
        # Split each cell into lines
        cell_lines = {}
        max_lines = 1
        for h in headers:
            lines = row.get(h, "").split("\n")
            cell_lines[h] = lines
            max_lines = max(max_lines, len(lines))
        for i in range(max_lines):
            print(_join(cell_lines[h][i] if i < len(cell_lines[h]) else "" for h in headers))


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
        flat_headers = ["ID", "Dataset", "Proc", "Unres", "Res", "Rate", "i0 Rate", "LLM", "Att", "Steps", "Learn\n(Ret)", "SB Assist", _fmt_exit_status_header(flat_statuses)]
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
                "Learn\n(Ret)": _fmt_learn(r),
                "SB Assist": _format_sb_assist(r),
                _fmt_exit_status_header(flat_statuses): _fmt_exit_status_row(r["exit_statuses"], flat_statuses),
            })
        _print_table_rows(flat_headers, flat_rows)
        print()

    # --- Split runs: separate per_repo and global tables ---
    if split_runs:
        # per_repo = iterate_repos OR skillbook mode per_repo (single-repo or multi-repo)
        # global = skillbook mode global (may have filter_repos for subset selection)
        per_repo_runs = [r for r in split_runs if r["is_iterate_repos"] or r.get("skillbook_mode") == "per_repo"]
        global_runs = [r for r in split_runs if not r["is_iterate_repos"] and r.get("skillbook_mode") != "per_repo"]
        global_headers = ["ID", "Dataset", "Train", "ValBL", "ValSB", "SB Δ (avg)", "New/Lost", "LLM", "Learn", "Traj Exit Status"]

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
                    vpk = r.get("val_pass_k", 1)

                    # When top-level stats lack newly_resolved/lost (e.g. validation-only
                    # or retrieval runs), aggregate from per-repo statistics files.
                    agg_nr = agg["newly_resolved"]
                    agg_lost = agg["lost"]
                    all_prd: list[dict] = []
                    if full_path:
                        for repo in r["repos"]:
                            prd = _load_per_repo_stats(full_path, repo)
                            if prd:
                                all_prd.append(prd)
                                if not agg_nr and not agg_lost:
                                    agg_nr = agg_nr + prd.get("summary", {}).get("newly_resolved_by_skillbook", [])
                                    agg_lost = agg_lost + prd.get("summary", {}).get("lost_by_skillbook", [])

                    # Build aggregate phase dicts with aggregated pass_at_k
                    agg_train = dict(agg["train"])
                    agg_valbl = dict(agg["val_baseline"])
                    agg_valsb = dict(agg["val_skillbook"])
                    if all_prd:
                        agg_train["pass_at_k"] = _aggregate_pass_at_k(all_prd, "train_phase")
                        agg_valbl["pass_at_k"] = _aggregate_pass_at_k(all_prd, "val_baseline_phase")
                        agg_valsb["pass_at_k"] = _aggregate_pass_at_k(all_prd, "val_skillbook_phase")

                    # Backfill combined per_attempt_rate from result files (scan all
                    # repos; instance ids are already namespaced) for a correct avg.
                    if full_path:
                        _results_root = _phase_results_dir(full_path)
                        if _results_root is not None:
                            _backfill_per_attempt_rate(agg_valbl, _results_root / "val_baseline")
                            _backfill_per_attempt_rate(agg_valsb, _results_root / "val")

                    # Override ValBL avg with the aggregated reference baseline
                    # (qwen3 split025) so SB Δ uses a low-noise baseline.
                    if _is_qwen3_split025(r):
                        _apply_aggregated_val_baseline(agg_valbl, repo=None)

                    # Compute SB Δ from avg rates when vpk > 1
                    if vpk > 1:
                        avg_bl = _compute_avg_rate(agg_valbl)
                        avg_sb = _compute_avg_rate(agg_valsb)
                        if avg_bl is not None and avg_sb is not None:
                            sb_delta = avg_sb - avg_bl
                        else:
                            sb_delta = agg["skillbook_improvement"]
                    else:
                        sb_delta = agg["skillbook_improvement"]

                    per_repo_rows.append({
                        "ID": parent_tag,
                        "Dataset": _shorten_dataset(r["benchmark"]),
                        "Repo": f"{n_repos} repos",
                        "Train": _fmt_phase(agg_train, distil=is_distil, val_pass_k=vpk),
                        "ValBL": _fmt_phase(agg_valbl, val_pass_k=vpk),
                        "ValSB": _fmt_phase(agg_valsb, val_pass_k=vpk),
                        "SB Δ (avg)": _fmt_sb_delta(sb_delta, agg_valbl.get("total", 0)),
                        "New/Lost": _fmt_new_lost(agg_nr, agg_lost, full_path, vpk,
                                                  per_attempt_new_lost=agg.get("per_attempt_new_lost")),
                        "LLM": llm_col(r),
                        "Learn\n(Ret)": _fmt_learn(r),
                    })
                    row_es_data.append(r["exit_statuses"])

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
                                "per_attempt_new_lost": per_repo_data.get("summary", {}).get("per_attempt_new_lost"),
                            }

                            # Backfill per_attempt_rate for this repo from result files
                            # (instance ids are namespaced by repo, so filter by prefix).
                            if full_path:
                                _results_root = _phase_results_dir(full_path)
                                if _results_root is not None:
                                    _prefix = repo.replace("/", "__") + "-"
                                    _backfill_per_attempt_rate(s["val_baseline"], _results_root / "val_baseline", repo_prefix=_prefix)
                                    _backfill_per_attempt_rate(s["val_skillbook"], _results_root / "val", repo_prefix=_prefix)

                            # Override per-repo ValBL avg with aggregated reference (qwen3 split025)
                            if _is_qwen3_split025(r):
                                _apply_aggregated_val_baseline(s["val_baseline"], repo=repo)

                            # Per-repo exit status — prefer the backfilled per-phase
                            # exit_statuses merged from the per-repo stats file; fall
                            # back to scanning trajectory files filtered by repo ids.
                            repo_exit_statuses = _merge_exit_statuses(
                                per_repo_data.get("train_phase", {}).get("exit_statuses"),
                                per_repo_data.get("val_baseline_phase", {}).get("exit_statuses"),
                                per_repo_data.get("val_skillbook_phase", {}).get("exit_statuses"),
                            )
                            if not repo_exit_statuses:
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

                        # SB Δ from per-attempt avg (falls back to skillbook_improvement
                        # when avg is unavailable, e.g. vpk=1).
                        _bl = _compute_avg_rate(s["val_baseline"])
                        _sb = _compute_avg_rate(s["val_skillbook"])
                        _repo_delta = (_sb - _bl) if (_bl is not None and _sb is not None) else s["skillbook_improvement"]

                        per_repo_rows.append({
                            "ID": "",
                            "Dataset": "",
                            "Repo": repo,
                            "Train": _fmt_phase(s["train"], distil=repo_distil, val_pass_k=r.get("val_pass_k", 1)),
                            "ValBL": _fmt_phase(s["val_baseline"], val_pass_k=r.get("val_pass_k", 1)),
                            "ValSB": _fmt_phase(s["val_skillbook"], val_pass_k=r.get("val_pass_k", 1)),
                            "SB Δ (avg)": _fmt_sb_delta(_repo_delta, s["val_baseline"].get("total", 0)),
                            "New/Lost": _fmt_new_lost(
                                s["newly_resolved"], s["lost"], full_path,
                                r.get("val_pass_k", 1),
                                repo_prefix=repo.replace("/", "__") + "-",
                                per_attempt_new_lost=s.get("per_attempt_new_lost"),
                            ),
                            "LLM": "",
                            "Learn\n(Ret)": "",
                        })
                        row_es_data.append(repo_exit_statuses)
                else:
                    # Single-repo split run (filter_repos set but not iterate_repos)
                    repo = r["filter_repos"][0] if r.get("filter_repos") else "all"
                    # Override ValBL avg with aggregated reference for this repo (qwen3 split025)
                    if _is_qwen3_split025(r):
                        _apply_aggregated_val_baseline(agg["val_baseline"], repo=repo if repo != "all" else None)
                    # SB Δ from per-attempt avg (falls back to skillbook_improvement when
                    # avg is unavailable, e.g. vpk=1).
                    _bl = _compute_avg_rate(agg["val_baseline"])
                    _sb = _compute_avg_rate(agg["val_skillbook"])
                    _sr_delta = (_sb - _bl) if (_bl is not None and _sb is not None) else agg["skillbook_improvement"]
                    per_repo_rows.append({
                        "ID": parent_tag,
                        "Dataset": _shorten_dataset(r["benchmark"]),
                        "Repo": repo,
                        "Train": _fmt_phase(agg["train"], distil=is_distil, val_pass_k=r.get("val_pass_k", 1)),
                        "ValBL": _fmt_phase(agg["val_baseline"], val_pass_k=r.get("val_pass_k", 1)),
                        "ValSB": _fmt_phase(agg["val_skillbook"], val_pass_k=r.get("val_pass_k", 1)),
                        "SB Δ (avg)": _fmt_sb_delta(_sr_delta, agg["val_baseline"].get("total", 0)),
                        "New/Lost": _fmt_new_lost(
                            agg["newly_resolved"], agg["lost"], full_path, r.get("val_pass_k", 1)
                        ),
                        "LLM": llm_col(r),
                        "Learn\n(Ret)": _fmt_learn(r),
                    })
                    row_es_data.append(r["exit_statuses"])

            # Second pass: collect all statuses and fill exit status column
            pr_statuses = _collect_all_statuses_from_es(row_es_data)
            es_header = _fmt_exit_status_header(pr_statuses)
            for row, es in zip(per_repo_rows, row_es_data):
                row[es_header] = _fmt_exit_status_row(es, pr_statuses)
            per_repo_headers = ["ID", "Dataset", "Repo", "Train", "ValBL", "ValSB", "SB Δ (avg)", "New/Lost", "LLM", "Learn\n(Ret)", es_header]

            _print_table_rows(per_repo_headers, per_repo_rows)
            print()

        # --- Global table ---
        if global_runs:
            print("Split-mode runs (global):")
            global_statuses = _collect_all_statuses(global_runs)
            es_header = _fmt_exit_status_header(global_statuses)
            global_headers = ["ID", "Dataset", "Train", "ValBL", "ValSB", "SB Δ (avg)", "New/Lost", "LLM", "Learn\n(Ret)", es_header]
            global_rows = []
            for r in global_runs:
                parent_tag = run_id_map[r["run_dir"]]
                s = r["split"]
                is_distil = bool(r.get("train_trajs_dir"))
                vpk = r.get("val_pass_k", 1)
                full_path = run_dir_paths.get(r["run_dir"])
                # Override ValBL avg with the aggregated reference baseline (qwen3
                # split025) so SB Δ compares against a low-noise 60-attempt baseline.
                if _is_qwen3_split025(r):
                    _apply_aggregated_val_baseline(s["val_baseline"], repo=None)
                # Compute SB Δ from avg rates when vpk > 1
                if vpk > 1:
                    avg_bl = _compute_avg_rate(s["val_baseline"])
                    avg_sb = _compute_avg_rate(s["val_skillbook"])
                    if avg_bl is not None and avg_sb is not None:
                        sb_delta = avg_sb - avg_bl
                    else:
                        sb_delta = s["skillbook_improvement"]
                else:
                    sb_delta = s["skillbook_improvement"]
                global_rows.append({
                    "ID": parent_tag,
                    "Dataset": _shorten_dataset(r["benchmark"]),
                    "Train": _fmt_phase(s["train"], distil=is_distil, val_pass_k=vpk),
                    "ValBL": _fmt_phase(s["val_baseline"], val_pass_k=vpk),
                    "ValSB": _fmt_phase(s["val_skillbook"], val_pass_k=vpk),
                    "SB Δ (avg)": _fmt_sb_delta(sb_delta, s["val_baseline"].get("total", 0)),
                    "New/Lost": _fmt_new_lost(s["newly_resolved"], s["lost"], full_path, vpk,
                                              per_attempt_new_lost=s.get("per_attempt_new_lost")),
                    "LLM": llm_col(r),
                    "Learn\n(Ret)": _fmt_learn(r),
                    es_header: _fmt_exit_status_row(r["exit_statuses"], global_statuses),
                })

            _print_table_rows(global_headers, global_rows)
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
