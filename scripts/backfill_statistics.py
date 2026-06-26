#!/usr/bin/env python3
"""Backfill statistics.json with metrics compare_runs.py derives from raw files.

Why: statistics.json (written by main_loop / save_statistics) faithfully stores
resolution outcomes but omits the *process* signals that compare_runs.py computes
on the fly from the trajectories/ + results/ tree:

  1. exit-status breakdown      (Submitted/LimitsExceeded/... per iteration)
  2. iter_0 resolution count/rate
  3. per-attempt resolved-id sets (→ per-attempt new/lost dynamics)
  4. per_attempt_rate / pass_at_k  (missing from iterate_repos top-level + old runs)

This script reuses compare_runs.py's scanning functions so the appended values
match exactly what the table shows, then writes them back into statistics.json
(and each statistics_per_repo/<repo>.json for iterate_repos runs).

Safety:
  - DRY-RUN by default. Pass --apply to write.
  - Additive + idempotent: a key is (re)written only if its value would change.
  - A `_backfill` provenance marker is added so derived fields are distinguishable
    from runner-written ones. `data/` is gitignored, so there is no VCS safety net.

Usage:
    uv run python scripts/backfill_statistics.py                 # dry-run, all runs
    uv run python scripts/backfill_statistics.py --apply         # write all
    uv run python scripts/backfill_statistics.py data/run_XXX    # specific run(s)
    uv run python scripts/backfill_statistics.py --apply --verbose
"""

import argparse
import json
import sys
from pathlib import Path

# Import the scanner logic from compare_runs (same dir) so derived values match.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_runs as cr  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

# (disk subdir under trajectories|results) -> (statistics.json phase key)
PHASES = [
    ("train", "train_phase"),
    ("val_baseline", "val_baseline_phase"),
    ("val", "val_skillbook_phase"),
]
# Phases that carry multi-attempt data worth a per_attempt_rate / per_attempt_resolved.
VAL_PHASES = {"val_baseline", "val"}

SCRIPT_NAME = "backfill_statistics.py"


def _norm_es(es: dict) -> dict:
    """Stringify inner iteration keys so the in-memory form matches the JSON
    round-trip (JSON object keys are always strings). Without this, {0: n} !=
    {"0": n} after save→reload and every run would look like a change.
    """
    return {status: {str(it): c for it, c in it_counts.items()}
            for status, it_counts in es.items()}


def _set(d: dict, key: str, value, changes: list, label: str) -> bool:
    """Set d[key]=value, recording label in `changes` only if it actually changed."""
    if d.get(key) != value:
        d[key] = value
        changes.append(label)
        return True
    return False


def _phase_exit_statuses(bench_dir: Path, phase: str, prefix: str | None = None) -> dict:
    """Exit-status counts for one phase subdir: {status: {iter: count}}.

    Reuses cr._extract_exit_status (regex over the first 4KB). `prefix` filters
    instance ids by repo namespace (iterate_repos per-repo rows).
    """
    traj_phase = bench_dir / "trajectories" / phase
    result: dict[str, dict[int, int]] = {}
    if not traj_phase.exists():
        return result
    for inst_dir in traj_phase.iterdir():
        if not inst_dir.is_dir():
            continue
        if prefix and not inst_dir.name.startswith(prefix):
            continue
        for fname in inst_dir.iterdir():
            if not (fname.name.startswith("iter_") and fname.name.endswith(".json")):
                continue
            it = int(fname.name.replace("iter_", "").replace(".json", ""))
            es = cr._extract_exit_status(fname)
            if es is None:
                continue
            result.setdefault(es, {}).setdefault(it, 0)
            result[es][it] += 1
    return result


def _phase_iter0(bench_dir: Path, phase: str, prefix: str | None = None) -> dict:
    """iter_0 resolution for one phase subdir: {resolved, total, rate}."""
    res_phase = bench_dir / "results" / phase
    resolved = total = 0
    if not res_phase.exists():
        return {"resolved": 0, "total": 0, "rate": 0.0}
    for inst_dir in res_phase.iterdir():
        if not inst_dir.is_dir():
            continue
        if prefix and not inst_dir.name.startswith(prefix):
            continue
        f = inst_dir / "iter_0.json"
        if not f.exists():
            continue
        total += 1
        with open(f) as fh:
            if json.load(fh).get("resolved"):
                resolved += 1
    return {"resolved": resolved, "total": total, "rate": (resolved / total) if total else 0.0}


def _sets_to_lists(sets: dict[int, set[str]]) -> dict[str, list[str]]:
    """{iter: set(ids)} -> {"iter_N": sorted ids} (JSON-friendly)."""
    return {f"iter_{i}": sorted(s) for i, s in sorted(sets.items())}


def _compute_pass_at_k(res_dir: Path, repo_prefix: str | None = None) -> dict:
    """Cumulative pass@k from result files: {pass@n: {count, total, rate}}.

    Mirrors main_loop's pass@k (any-of-first-n resolved) but derived purely from
    the per-iteration resolved-id sets. Used where the runner never stored it
    (iterate_repos top-level, older per-repo files). Schema matches the runner so
    compare_runs reads it unchanged.
    """
    sets = cr._per_attempt_resolved_sets(res_dir, repo_prefix=repo_prefix)
    if not sets:
        return {}
    max_k = max(sets) + 1
    total = 0
    if res_dir.exists():
        for inst_dir in res_dir.iterdir():
            if not inst_dir.is_dir():
                continue
            if repo_prefix and not inst_dir.name.startswith(repo_prefix):
                continue
            if any((inst_dir / f"iter_{i}.json").exists() for i in range(max_k + 1)):
                total += 1
    if total == 0:
        return {}
    pak: dict[str, dict] = {}
    cumulative: set[str] = set()
    for n in range(1, max_k + 1):
        cumulative |= sets.get(n - 1, set())
        pak[f"pass@{n}"] = {"count": len(cumulative), "total": total, "rate": len(cumulative) / total}
    return pak


def _backfill_phase(bench_dir: Path, phase: str, stats_key: str, st: dict,
                    prefix: str | None, changes: list) -> None:
    """Append exit/iter0/per_attempt fields to one phase dict in `st`."""
    pd = st.get(stats_key)
    if not pd or not bench_dir:
        return

    pes = _norm_es(_phase_exit_statuses(bench_dir, phase, prefix=prefix))
    if pes:
        _set(pd, "exit_statuses", pes, changes, f"exit_statuses[{phase}]")
        _set(pd, "exit_status_totals", cr._total_exit_status_counts(pes), changes, f"exit_totals[{phase}]")

    pi = _phase_iter0(bench_dir, phase, prefix=prefix)
    if pi["total"] > 0:
        _set(pd, "iter0", pi, changes, f"iter0[{phase}]")

    # Per-attempt resolved-id sets (val phases only). Captures per-attempt
    # new/lost dynamics that the stored cumulative newly_resolved/lost cannot.
    if phase in VAL_PHASES:
        res_dir = bench_dir / "results" / phase
        sets = cr._per_attempt_resolved_sets(res_dir, repo_prefix=prefix)
        if sets:
            _set(pd, "per_attempt_resolved", _sets_to_lists(sets), changes, f"per_attempt_resolved[{phase}]")
        # per_attempt_rate where missing (old runs / iterate_repos top-level).
        if not pd.get("per_attempt_rate"):
            par = cr._compute_per_attempt_rate(res_dir, repo_prefix=prefix)
            if par:
                _set(pd, "per_attempt_rate", par, changes, f"per_attempt_rate[{phase}]")
        # pass_at_k where missing (same cause); computed from the same files.
        if not pd.get("pass_at_k"):
            pak = _compute_pass_at_k(res_dir, repo_prefix=prefix)
            if pak:
                _set(pd, "pass_at_k", pak, changes, f"pass_at_k[{phase}]")


def _backfill_new_lost(bench_dir: Path, st: dict, prefix: str | None, changes: list) -> None:
    """summary.per_attempt_new_lost from val_baseline vs val resolved-id sets."""
    if not bench_dir:
        return
    bl_sets = cr._per_attempt_resolved_sets(bench_dir / "results" / "val_baseline", repo_prefix=prefix)
    sb_sets = cr._per_attempt_resolved_sets(bench_dir / "results" / "val", repo_prefix=prefix)
    if not bl_sets and not sb_sets:
        return
    avg_new, avg_lost = cr._avg_new_lost(bl_sets, sb_sets)
    summary = st.setdefault("summary", {})
    _set(summary, "per_attempt_new_lost",
         {"avg_new": round(avg_new, 3), "avg_lost": round(avg_lost, 3)},
         changes, "per_attempt_new_lost")


def _backfill_iterate(run_dir: Path, st: dict, cfg: dict, bench_dir: Path | None,
                      changes: list) -> None:
    """iterate_repos: backfill top-level aggregates + each statistics_per_repo file."""
    repos = st.get("repos") or cfg.get("benchmark", {}).get("iterate_repos") or []
    all_prd: list[dict] = []
    per_repo_dir = run_dir / "statistics_per_repo"

    for repo in repos:
        prd = cr._load_per_repo_stats(run_dir, repo)
        if not prd:
            continue
        all_prd.append(prd)
        prefix = repo.replace("/", "__") + "-"

        # Backfill per-repo phase fields (filtered by repo prefix).
        repo_changes: list = []
        for disk_phase, stats_key in PHASES:
            _backfill_phase(bench_dir, disk_phase, stats_key, prd, prefix, repo_changes)
        _backfill_new_lost(bench_dir, prd, prefix, repo_changes)

        if repo_changes:
            prd["_backfill"] = {"script": SCRIPT_NAME, "derived": sorted(set(repo_changes))}
            repo_file = per_repo_dir / (repo.replace("/", "__") + ".json")
            with open(repo_file, "w") as f:
                json.dump(prd, f, indent=2, default=str)
            changes.append(f"per_repo[{repo}]")

    # Top-level aggregates (the top-level phase dicts are minimal: 3 keys each).
    # Compute per_attempt_rate + pass_at_k directly from result files (instance ids
    # are namespaced by repo, so the combined dir is the correct whole-run view).
    if bench_dir:
        for disk_phase, stats_key in PHASES:
            if disk_phase not in VAL_PHASES:
                continue
            pd = st.setdefault(stats_key, {})
            res_dir = bench_dir / "results" / disk_phase
            if not pd.get("per_attempt_rate"):
                par = cr._compute_per_attempt_rate(res_dir)
                if par:
                    _set(pd, "per_attempt_rate", par, changes, f"per_attempt_rate[{stats_key}]")
            if not pd.get("pass_at_k"):
                pak = _compute_pass_at_k(res_dir)
                if pak:
                    _set(pd, "pass_at_k", pak, changes, f"pass_at_k[{stats_key}]")
    # Aggregate newly_resolved / lost across per-repo summaries into top-level.
    if all_prd:
        nr = sorted({i for prd in all_prd for i in prd.get("summary", {}).get("newly_resolved_by_skillbook", [])})
        lo = sorted({i for prd in all_prd for i in prd.get("summary", {}).get("lost_by_skillbook", [])})
        summary = st.setdefault("summary", {})
        _set(summary, "newly_resolved_by_skillbook", nr, changes, "newly_resolved[top]")
        _set(summary, "lost_by_skillbook", lo, changes, "lost[top]")


def backfill_run(run_dir: Path, apply: bool, verbose: bool) -> list[str]:
    """Backfill one run dir. Returns the list of changed-field labels (empty if none)."""
    stats_path = run_dir / "statistics.json"
    config_path = run_dir / "config.json"
    if not stats_path.exists():
        return []
    with open(stats_path) as f:
        st = json.load(f)
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)

    bench_dir = cr._find_benchmark_dir(run_dir)
    is_split = ("val_skillbook_phase" in st) or ("val_baseline_phase" in st)
    is_iterate = (st.get("mode") == "iterate_repos") or bool(cfg.get("benchmark", {}).get("iterate_repos"))

    changes: list[str] = []

    # --- Top-level: exit-status breakdown (all runs) ---
    es = _norm_es(cr._count_exit_statuses(run_dir))
    if es:
        _set(st, "exit_statuses", es, changes, "exit_statuses[top]")
        _set(st, "exit_status_totals", cr._total_exit_status_counts(es), changes, "exit_totals[top]")

    # --- Top-level: iter_0 resolution (flat runs; split runs use per-phase) ---
    if not is_split:
        ir, it = cr._count_iter0_resolved(run_dir)
        if it > 0:
            _set(st, "iter0", {"resolved": ir, "total": it, "rate": ir / it}, changes, "iter0[top]")

    # --- Per-phase backfill (split single-repo / global) ---
    if is_split and not is_iterate:
        for disk_phase, stats_key in PHASES:
            _backfill_phase(bench_dir, disk_phase, stats_key, st, prefix=None, changes=changes)
        _backfill_new_lost(bench_dir, st, prefix=None, changes=changes)

    # --- iterate_repos: top-level aggregates + per-repo files ---
    if is_iterate:
        _backfill_iterate(run_dir, st, cfg, bench_dir, changes)

    if not changes:
        return []

    st["_backfill"] = {"script": SCRIPT_NAME, "derived": sorted(set(changes))}

    if apply:
        with open(stats_path, "w") as f:
            json.dump(st, f, indent=2, default=str)  # match save_statistics format

    if verbose:
        detail = ", ".join(changes)
        tag = "WROTE " if apply else "would   "
        print(f"  {tag} {run_dir.name}: {detail}")
    return changes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="Run dirs (default: all data/run_* with statistics.json)")
    ap.add_argument("--apply", action="store_true", help="Actually write changes (default: dry-run).")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print per-run detail.")
    args = ap.parse_args()

    if args.runs:
        run_dirs = [Path(r).resolve() for r in args.runs]
    else:
        run_dirs = sorted(p.parent for p in DATA.glob("*/statistics.json"))

    if not run_dirs:
        print("No runs found.", file=sys.stderr)
        sys.exit(1)

    mode = "APPLY" if args.apply else "DRY-RUN (no files written; pass --apply to write)"
    print(f"[{mode}] scanning {len(run_dirs)} run(s)…\n")

    n_changed = n_unchanged = 0
    field_tally: dict[str, int] = {}
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"  skip (missing): {run_dir}", file=sys.stderr)
            continue
        changes = backfill_run(run_dir, apply=args.apply, verbose=args.verbose)
        if changes:
            n_changed += 1
        else:
            n_unchanged += 1
        # Tally by field group (strip [phase] suffix).
        for c in changes:
            key = c.split("[", 1)[0]
            field_tally[key] = field_tally.get(key, 0) + 1

    print(f"\nDone. changed={n_changed}  unchanged={n_unchanged}  (mode={mode})")
    if field_tally:
        print("Field write tally:")
        for k in sorted(field_tally):
            print(f"  {field_tally[k]:>4}  {k}")
    if not args.apply and n_changed:
        print("\nThis was a dry-run. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
