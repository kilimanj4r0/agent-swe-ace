#!/usr/bin/env python3
"""
Live CLI dashboard for monitoring agent-swe-ace experiments.

Usage:
    python scripts/watch_experiments.py          # auto-refresh every 10s
    python scripts/watch_experiments.py -n        # one-shot, no refresh
    python scripts/watch_experiments.py -i 5      # refresh every 5s
    python scripts/watch_experiments.py --all     # show completed runs too
    python scripts/watch_experiments.py --running  # show only active runs
    python scripts/watch_experiments.py --tests   # show only e2e smoke test runs
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIRS = [DATA_DIR, Path(__file__).resolve().parent.parent / "_data"]

# ── helpers ──────────────────────────────────────────────────────────────

def _all_run_dirs(reverse: bool = False):
    """Yield (dir_path, dir_name, test_slug) from all data directories, sorted.

    For data/: flat layout — run_* dirs directly inside.
    For _data/: two-level layout — <test_slug>/run_* dirs nested one level deeper.
    """
    entries = []  # list of (path, name, test_slug)
    for ddir in DATA_DIRS:
        if not ddir.exists():
            continue
        is_test_root = ddir.name == "_data"
        for d in ddir.iterdir():
            if not d.is_dir():
                continue
            if is_test_root and not d.name.startswith("run_"):
                # Two-level: _data/<test_slug>/run_*
                for sub in d.iterdir():
                    if sub.is_dir():
                        entries.append((sub, sub.name, d.name))
            else:
                entries.append((d, d.name, None))
    entries.sort(key=lambda e: e[1], reverse=reverse)
    return entries


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def short_model(model: str) -> str:
    if not model:
        return "?"
    name = model.split("/")[-1]
    name = name.replace("-Instruct", "").replace("-Chat", "")
    return name


def fmt_model_display(agent_model: str, ace_model: str) -> str:
    a = short_model(agent_model)
    b = short_model(ace_model)
    if a == b or b == "?" or not b:
        return a
    return f"{a}/{b}"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m}m"
    if m > 0:
        return f"{m}m{s}s"
    return f"{s}s"


def fmt_time(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        return datetime.fromisoformat(iso).strftime("%m/%d %H:%M")
    except Exception:
        return iso[:16]


def short_dir(dir_name: str) -> str:
    """Abbreviate run folder: 'run_20260426_211426' -> '0426_211426'."""
    if (dir_name.startswith("run_") and len(dir_name) >= 19
            and dir_name[4:8].isdigit() and dir_name[8:12].isdigit()
            and dir_name[12] == '_'):
        return dir_name[8:19]  # "MMDD_HHMMSS"
    if len(dir_name) > 16:
        return dir_name[:14] + ".."
    return dir_name


def bar(resolved: int, failed: int, total: int, width: int = 20) -> str:
    """█ resolved, ░ tried-but-failed, spaces remaining."""
    if total == 0:
        return "[" + " " * width + "]"
    r = round(resolved / total * width)
    u = round(failed / total * width)
    r = min(r, width)
    u = min(u, width - r)
    return "[" + "█" * r + "░" * u + " " * (width - r - u) + "]"


def pct_str(done: int, total: int) -> str:
    return f"{done / total * 100:5.1f}%" if total else "  -  "


def _shorten_url(url: str) -> str:
    if not url:
        return "?"
    url = url.rstrip("/")
    if url.startswith(("http://localhost:", "http://127.0.0.1:")):
        return "lcl:" + url.split(":")[-1].split("/")[0]
    if url.startswith(("http://10.", "http://192.168.")):
        return url.split("://", 1)[1].split("/")[0]
    if "z.ai" in url:
        return "z.ai"
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        return f"{p.hostname}:{p.port}" if p.port else (p.hostname or url[:20])
    except Exception:
        return url[:20]


# ── endpoint health ─────────────────────────────────────────────────────

_HEALTH_CACHE: dict[tuple[str, str], tuple[str, float]] = {}
_HEALTH_TTL = 60  # seconds between health re-checks


def _check_vllm_health(api_base: str) -> str:
    try:
        url = api_base.rstrip("/") + "/models"
        req = urllib.request.Request(url, method="GET")
        resp = urllib.request.urlopen(req, timeout=3)
        return "UP" if resp.status == 200 else "ERR"
    except urllib.error.HTTPError as e:
        return "UP" if e.code == 401 else "ERR"
    except (urllib.error.URLError, OSError, TimeoutError):
        return "DOWN"
    except Exception:
        return "ERR"


def _is_local_url(url: str) -> bool:
    if not url:
        return False
    return any(url.startswith(p) for p in (
        "http://localhost:", "http://127.0.0.1:",
        "http://10.", "http://192.168.", "http://172.",
    ))


def get_endpoint_health(provider: str, api_base: str) -> str:
    # Check any local endpoint regardless of provider label
    if provider != "hosted_vllm" and not _is_local_url(api_base):
        return "cloud"
    key = (provider, api_base)
    now = time.monotonic()
    cached = _HEALTH_CACHE.get(key)
    if cached and now - cached[1] < _HEALTH_TTL:
        return cached[0]
    status = _check_vllm_health(api_base)
    _HEALTH_CACHE[key] = (status, now)
    return status


def _parse_log_start_time(log_path: Path) -> str | None:
    """Parse start time from the first line of a log file."""
    try:
        first_line = log_path.read_text().split("\n")[0]
        ts = first_line.split(" | ")[0].strip()
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return dt.isoformat()
    except Exception:
        return None


def get_start_time(run_dir: Path, stat: dict) -> str | None:
    """Get start time from statistics.json, then earliest experiment.log (incl. rotated)."""
    t = stat.get("start_time") or stat.get("timestamp")
    if t:
        return t
    # Check all experiment*.log files and use the earliest timestamp
    candidates = []
    for log in run_dir.glob("experiment*.log"):
        ts = _parse_log_start_time(log)
        if ts:
            candidates.append(ts)
    if candidates:
        return min(candidates)
    return None


# ── process detection ────────────────────────────────────────────────────

def get_active_run_dirs() -> set[str]:
    """Find run dirs with experiment.log open in a live src.cli.commands process."""
    active = set()
    try:
        result = subprocess.run(
            ["pgrep", "-af", "src.cli.commands"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [l.split()[0] for l in result.stdout.strip().split("\n") if l.strip()]
    except Exception:
        return active

    for pid in pids:
        try:
            for fd in Path(f"/proc/{pid}/fd").iterdir():
                try:
                    target = os.readlink(fd)
                    if "experiment.log" in target and ("/data/run_" in target or "/_data/" in target):
                        for p in target.split("/"):
                            if p.startswith("run_"):
                                active.add(p)
                                break
                except OSError:
                    continue
        except OSError:
            continue
    return active


# ── progress from file system (for runs without statistics.json) ─────────

_DATASET_SIZE_CACHE: dict[str, int] = {}
_DATASET_IDS_CACHE: dict[str, set[str]] = {}
_DATASET_REPO_COUNTS: dict[str, dict[str, int]] = {}  # dataset -> {repo: count}


def _load_dataset_size(dataset: str) -> int | None:
    """Load dataset size from HuggingFace (cached after first call)."""
    if dataset in _DATASET_SIZE_CACHE:
        return _DATASET_SIZE_CACHE[dataset]
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset, split="test")
        size = len(ds)
        _DATASET_SIZE_CACHE[dataset] = size
        _DATASET_IDS_CACHE[dataset] = set(ds['instance_id'])
        # Also cache per-repo counts
        from collections import Counter
        _DATASET_REPO_COUNTS[dataset] = dict(Counter(i['repo'] for i in ds))
        return size
    except Exception:
        return None


def compute_eff_total(dataset: str, exclude_instances: list | None = None) -> int | None:
    """Compute effective total: dataset size minus excluded instances that exist in it."""
    _load_dataset_size(dataset)
    ids = _DATASET_IDS_CACHE.get(dataset)
    if ids is None:
        return None
    effective = ids
    if exclude_instances:
        effective = effective - set(exclude_instances)
    return len(effective)


def find_dataset_total(dataset: str, filter_repos: list | None = None) -> int | None:
    """Find the effective total instances for a dataset from any run.

    Checks statistics.json from any run (not just completed) that used the
    same dataset without max_instances. When filter_repos is given, looks
    for runs with the same filter to find the total for that repo subset.
    """
    if not dataset:
        return None
    for d, _name, _slug in _all_run_dirs(reverse=True):
        cfg = load_json(d / "config.json")
        if cfg.get("benchmark", {}).get("dataset") != dataset:
            continue
        if cfg.get("benchmark", {}).get("max_instances") is not None:
            continue
        cfg_filter = cfg.get("benchmark", {}).get("filter_repos")
        if filter_repos is not None:
            if cfg_filter != filter_repos:
                continue
        else:
            if cfg_filter is not None:
                continue
        stat = load_json(d / "statistics.json")
        t = stat.get("total_instances", 0)
        if t > 0:
            return t
    # No run with statistics — load the dataset itself
    if filter_repos:
        return None  # can't know repo subset size without loading + filtering
    return _load_dataset_size(dataset)


def find_repo_phase_counts(dataset: str, filter_repos: list) -> dict | None:
    """Find train/val instance counts from a sibling run with the same filter.

    Scans for any run (active or completed) with the same filter_repos that
    has val_baseline dirs, so we can estimate the full workload even when
    the current run hasn't reached the val phase yet.
    """
    for d, _name, _slug in _all_run_dirs(reverse=True):
        cfg = load_json(d / "config.json")
        if cfg.get("benchmark", {}).get("dataset") != dataset:
            continue
        if cfg.get("benchmark", {}).get("filter_repos") != filter_repos:
            continue
        results_dir = _results_dir_for(d, dataset)
        if not results_dir.exists():
            continue
        vb_dir = results_dir / "val_baseline"
        val_dir = results_dir / "val"
        train_dir = results_dir / "train"
        if not train_dir.is_dir():
            continue
        train_count = sum(1 for p in train_dir.iterdir() if p.is_dir())
        vb_count = sum(1 for p in vb_dir.iterdir() if p.is_dir()) if vb_dir.is_dir() else 0
        val_count = sum(1 for p in val_dir.iterdir() if p.is_dir()) if val_dir.is_dir() else 0
        val_estimate = max(vb_count, val_count)
        if train_count > 0 and val_estimate > 0:
            return {"train": train_count, "val": val_estimate}
    return None


def _collect_instance_dirs(results_dir: Path):
    """Yield (instance_dir, phase) from both flat and two-phase layouts."""
    phase_names = {"train", "val", "val_baseline"}
    has_phases = any((results_dir / p).is_dir() for p in phase_names)

    if has_phases:
        for phase in ("train", "val_baseline", "val"):
            phase_dir = results_dir / phase
            if not phase_dir.is_dir():
                continue
            for inst_dir in phase_dir.iterdir():
                if inst_dir.is_dir():
                    yield inst_dir, phase
    else:
        for inst_dir in results_dir.iterdir():
            if inst_dir.is_dir():
                yield inst_dir, None


def _results_dir_for(run_dir: Path, dataset: str | None = None) -> Path:
    """Find the results directory from the run's config.json."""
    if dataset is None:
        cfg = load_json(run_dir / "config.json")
        dataset = cfg.get("benchmark", {}).get("dataset")
    if dataset:
        bench_dir = dataset.replace("/", "__")
        candidate = run_dir / bench_dir / "results"
        if candidate.exists():
            return candidate
    # Flat layout
    return run_dir / "results"


def scan_progress(run_dir: Path, max_attempts: int | None = None,
                   total_instances: int | None = None,
                   dataset: str | None = None,
                   filter_repos: list | None = None) -> dict:
    """Count completed instances from results dir when statistics.json is absent.

    Handles both flat and two-phase (train/val) directory layouts.
    An instance is 'processed' only when it has exhausted all attempts
    or resolved early.
    """
    results_dir = _results_dir_for(run_dir, dataset)
    if not results_dir.exists():
        return {"processed": 0, "resolved": 0, "total": 0, "phases": {}}

    phases: dict[str, list[int]] = {}  # phase -> [resolved, processed, total_dirs]

    for inst_dir, phase in _collect_instance_dirs(results_dir):
        iters = sorted(inst_dir.glob("iter_*.json"))
        if not iters:
            if phase:
                phases.setdefault(phase, [0, 0, 0])
                phases[phase][2] += 1
            continue
        last_data = load_json(iters[-1])
        is_done = False
        if last_data.get("resolved"):
            is_done = True
        elif max_attempts is not None and len(iters) >= max_attempts:
            is_done = True
        if phase:
            phases.setdefault(phase, [0, 0, 0])
            phases[phase][2] += 1
            if is_done:
                phases[phase][1] += 1
            if last_data.get("resolved"):
                phases[phase][0] += 1

    # Compute top-level counts, deduplicating val across vb/val phases
    if phases:
        # Train counts always included
        tr = phases.get("train", [0, 0, 0])
        resolved = tr[0]
        processed = tr[1]
        dir_count = tr[2]
        # Val: prefer val over vb (val supersedes vb)
        val_p = phases.get("val", [0, 0, 0])
        vb_p = phases.get("val_baseline", [0, 0, 0])
        if val_p[2] > 0:
            resolved += val_p[0]
            processed += val_p[1]
            dir_count += val_p[2]
        elif vb_p[2] > 0:
            resolved += vb_p[0]
            processed += vb_p[1]
            dir_count += vb_p[2]
    else:
        # Flat layout — count directly
        processed = 0
        resolved = 0
        dir_count = 0
        for inst_dir, _ in _collect_instance_dirs(results_dir):
            dir_count += 1
            iters = sorted(inst_dir.glob("iter_*.json"))
            if not iters:
                continue
            last_data = load_json(iters[-1])
            if last_data.get("resolved"):
                resolved += 1
                processed += 1
            elif max_attempts is not None and len(iters) >= max_attempts:
                processed += 1

    if total_instances is not None and total_instances > 0:
        total = max(total_instances, dir_count)
    elif phases:
        # Two-phase: total is the unique instance count (train + val)
        train_total = phases.get("train", [0, 0, 0])[2]
        vb_total = phases.get("val_baseline", [0, 0, 0])[2]
        val_total = phases.get("val", [0, 0, 0])[2]
        val_estimate = max(vb_total, val_total)
        if val_estimate > 0:
            total = train_total + val_estimate
        else:
            # Val phase hasn't started — try sibling runs for phase counts
            phase_counts = find_repo_phase_counts(dataset, filter_repos) if filter_repos and dataset else None
            if phase_counts:
                total = phase_counts["train"] + phase_counts["val"]
            else:
                # No sibling data — use dataset total or dir_count
                ds_total = find_dataset_total(dataset, filter_repos) if dataset else None
                total = ds_total if ds_total else dir_count
    elif filter_repos:
        total = dir_count
    else:
        dataset_total = find_dataset_total(dataset) if dataset else None
        total = max(dataset_total, dir_count) if dataset_total else dir_count

    return {"processed": processed, "resolved": resolved, "total": total, "phases": phases}


# ── iterate_repos support ─────────────────────────────────────────────────

def _repo_from_instance(instance_id: str) -> str | None:
    """Extract repo name from instance_id like 'django__django-10914' -> 'django/django'."""
    parts = instance_id.split("__", 1)
    if len(parts) != 2:
        return None
    owner, rest = parts
    # rest is like 'django-10914' — project name is everything up to last dash-number
    segments = rest.rsplit("-", 1)
    if len(segments) != 2 or not segments[1].isdigit():
        return None
    return f"{owner}/{segments[0]}"


def _get_repo_expected_sizes(dataset: str, repos: list[str], val_ratio: float | None,
                             exclude_instances: list | None = None
                             ) -> dict[str, dict] | None:
    """Get expected per-repo sizes from the dataset.

    Returns {repo: {"train": N, "val": N, "total": N}} or None.
    Subtracts excluded instances (that actually exist in the dataset) per repo
    before computing train/val split.
    """
    _load_dataset_size(dataset)
    counts = _DATASET_REPO_COUNTS.get(dataset)
    ds_ids = _DATASET_IDS_CACHE.get(dataset)
    if not counts:
        return None
    # Build per-repo exclude counts (only for instances that exist in the dataset)
    repo_excludes: dict[str, int] = {}
    if exclude_instances and ds_ids:
        for iid in exclude_instances:
            if iid not in ds_ids:
                continue
            repo = _repo_from_instance(iid)
            if repo:
                repo_excludes[repo] = repo_excludes.get(repo, 0) + 1
    result = {}
    for repo in repos:
        total = counts.get(repo, 0) - repo_excludes.get(repo, 0)
        if total <= 0:
            continue
        if val_ratio:
            val = round(total * val_ratio)
            train = total - val
        else:
            train = total
            val = 0
        result[repo] = {"train": train, "val": val, "total": total}
    return result


def _detect_repo_phase(phases_detail: dict, max_attempts: int | None) -> str:
    """Determine the current active phase for a repo.

    Returns: "train", "vb", "val", "done", or "pending".
    A phase is "active" if it has dirs without completed iter files.
    """
    has_any = False
    for phase in ("train", "vb", "val"):
        pd = phases_detail.get(phase)
        if not pd or pd["total"] == 0:
            continue
        has_any = True
        # Phase is incomplete if some dirs lack terminal iter files
        if pd["processed"] < pd["total"]:
            return phase

    if not has_any:
        return "pending"
    return "done"


def collect_iterate_repos_progress(
    run_dir: Path, cfg: dict, stat: dict, max_attempts: int | None
) -> dict | None:
    """Collect per-repo progress for iterate_repos experiments.

    Returns dict with:
      - repo_progress: list of per-repo dicts with name, status, phases, totals
      - total/resolved/processed/unresolved: correct aggregates across all repos
      - repo_total: number of repos expected
      - repo_completed: number of repos with completed statistics
    """
    iterate_repos_list = cfg.get("benchmark", {}).get("iterate_repos")
    if not iterate_repos_list:
        return None

    dataset = cfg.get("benchmark", {}).get("dataset")
    val_ratio = cfg.get("experiment", {}).get("split", {}).get("val_ratio")
    results_dir = _results_dir_for(run_dir, dataset)

    # Get expected per-repo sizes from dataset (cached, accounts for excluded instances)
    exclude_instances = cfg.get("benchmark", {}).get("exclude_instances")
    expected = _get_repo_expected_sizes(dataset, iterate_repos_list, val_ratio, exclude_instances)

    # Read per-repo completed stats
    per_repo_dir = run_dir / "statistics_per_repo"
    completed_repo_stats: dict[str, dict] = {}
    if per_repo_dir.is_dir():
        for f in per_repo_dir.glob("*.json"):
            repo_name = f.stem.replace("__", "/")
            completed_repo_stats[repo_name] = load_json(f)

    # Scan filesystem to find per-repo instance progress
    repo_fs: dict[str, dict[str, list[str]]] = {}
    repo_fs_resolved: dict[str, dict[str, int]] = {}
    repo_fs_processed: dict[str, dict[str, int]] = {}

    if results_dir.exists():
        for inst_dir, phase in _collect_instance_dirs(results_dir):
            inst_id = inst_dir.name
            repo = _repo_from_instance(inst_id)
            if not repo:
                continue
            phase_key = phase or "main"
            repo_fs.setdefault(repo, {}).setdefault(phase_key, []).append(inst_id)

            iters = sorted(inst_dir.glob("iter_*.json"))
            is_done = False
            is_resolved = False
            if iters:
                last_data = load_json(iters[-1])
                if last_data.get("resolved"):
                    is_resolved = True
                    is_done = True
                elif max_attempts is not None and len(iters) >= max_attempts:
                    is_done = True
            if is_done:
                repo_fs_processed.setdefault(repo, {}).setdefault(phase_key, 0)
                repo_fs_processed[repo][phase_key] += 1
            if is_resolved:
                repo_fs_resolved.setdefault(repo, {}).setdefault(phase_key, 0)
                repo_fs_resolved[repo][phase_key] += 1

    # Build per-repo progress list
    repo_progress = []
    total_resolved = 0
    total_processed = 0
    total_instances = 0
    total_errors = 0

    for repo in iterate_repos_list:
        repo_stat = completed_repo_stats.get(repo)
        fs_data = repo_fs.get(repo, {})
        exp = expected.get(repo) if expected else None

        if repo_stat and repo_stat.get("status") == "completed":
            tp = repo_stat.get("train_phase", {})
            vbp = repo_stat.get("val_baseline_phase", {})
            vsp = repo_stat.get("val_skillbook_phase", {})
            r_resolved = repo_stat.get("resolved_count", 0)
            r_processed = repo_stat.get("processed_instances", 0)
            r_total = repo_stat.get("total_instances", 0)

            phases_detail = {}
            if tp.get("total_instances"):
                phases_detail["train"] = {
                    "resolved": tp.get("resolved_count", 0),
                    "processed": tp.get("total_instances", 0),
                    "total": tp.get("total_instances", 0),
                }
            if vbp.get("total_instances"):
                phases_detail["vb"] = {
                    "resolved": vbp.get("resolved_count", 0),
                    "processed": vbp.get("total_instances", 0),
                    "total": vbp.get("total_instances", 0),
                }
            if vsp.get("total_instances"):
                phases_detail["val"] = {
                    "resolved": vsp.get("resolved_count", 0),
                    "processed": vsp.get("total_instances", 0),
                    "total": vsp.get("total_instances", 0),
                }

            repo_progress.append({
                "name": repo,
                "status": "done",
                "current_phase": "done",
                "resolved": r_resolved,
                "processed": r_processed,
                "total": r_total,
                "errors": 0,
                "phases": phases_detail,
                "expected": exp,
            })
            total_resolved += r_resolved
            total_processed += r_processed
            total_instances += r_total
        else:
            # In-progress or not started — use filesystem scan
            train_dirs = fs_data.get("train", [])
            vb_dirs = fs_data.get("val_baseline", [])
            val_dirs = fs_data.get("val", [])

            train_resolved = repo_fs_resolved.get(repo, {}).get("train", 0)
            train_processed = repo_fs_processed.get(repo, {}).get("train", 0)
            vb_resolved = repo_fs_resolved.get(repo, {}).get("val_baseline", 0)
            vb_processed = repo_fs_processed.get(repo, {}).get("val_baseline", 0)
            val_resolved = repo_fs_resolved.get(repo, {}).get("val", 0)
            val_processed = repo_fs_processed.get(repo, {}).get("val", 0)

            # Use expected sizes for totals; fall back to filesystem counts
            if exp:
                train_total = exp["train"]
                val_total = exp["val"]
            else:
                train_total = len(train_dirs)
                val_total = max(len(val_dirs), len(vb_dirs))

            phases_detail = {}
            if train_dirs or (exp and exp["train"] > 0):
                phases_detail["train"] = {
                    "resolved": train_resolved,
                    "processed": train_processed,
                    "total": train_total,
                }
            if vb_dirs:
                phases_detail["vb"] = {
                    "resolved": vb_resolved,
                    "processed": vb_processed,
                    "total": val_total,
                }
            if val_dirs:
                phases_detail["val"] = {
                    "resolved": val_resolved,
                    "processed": val_processed,
                    "total": val_total,
                }

            r_total = train_total + val_total
            r_resolved = train_resolved + max(val_resolved, vb_resolved)
            r_processed = train_processed + max(val_processed, vb_processed)

            current_phase = _detect_repo_phase(phases_detail, max_attempts)
            r_status = "active" if current_phase not in ("pending",) else "pending"

            repo_progress.append({
                "name": repo,
                "status": r_status,
                "current_phase": current_phase,
                "resolved": r_resolved,
                "processed": r_processed,
                "total": r_total,
                "errors": 0,
                "phases": phases_detail,
                "expected": exp,
            })
            total_resolved += r_resolved
            total_processed += r_processed
            total_instances += r_total

    return {
        "repo_progress": repo_progress,
        "total": total_instances,
        "resolved": total_resolved,
        "processed": total_processed,
        "unresolved": total_processed - total_resolved,
        "errors": total_errors,
        "repo_total": len(iterate_repos_list),
        "repo_completed": len(completed_repo_stats),
        "rate": total_resolved / total_processed if total_processed else 0.0,
    }


# ── collect ──────────────────────────────────────────────────────────────

def collect_runs(show_all: bool, only_running: bool, only_tests: bool = False):
    if not any(d.exists() for d in DATA_DIRS):
        return []

    active_dirs = get_active_run_dirs()
    entries = []

    for d, _name, test_slug in _all_run_dirs():
        if not d.is_dir():
            continue

        has_config = (d / "config.json").exists()
        has_stats = (d / "statistics.json").exists()
        if not has_config and not has_stats:
            continue

        # Filter by test/non-test
        is_test = test_slug is not None
        if only_tests and not is_test:
            continue
        if not only_tests and is_test and not show_all and not only_running:
            # Hide test runs by default unless --all or --running
            is_active_flag = d.name in active_dirs
            if not is_active_flag:
                continue

        cfg = load_json(d / "config.json") if has_config else {}
        stat = load_json(d / "statistics.json") if has_stats else {}
        is_active = d.name in active_dirs
        is_completed = stat.get("status") == "completed" or stat.get("end_time") is not None

        if only_running and not is_active:
            continue
        if not show_all and not is_active and not is_completed:
            continue
        if not show_all and is_completed and not only_running:
            try:
                end = stat.get("end_time") or stat.get("timestamp", "")
                dt = datetime.fromisoformat(end)
                if (datetime.now() - dt).total_seconds() > 14 * 86400:
                    continue
            except Exception:
                continue

        exp = cfg.get("experiment", {})
        llm = cfg.get("llm", {})
        agent_llm = llm.get("agent", {})
        ace_llm = llm.get("ace", {})
        sb_cfg = exp.get("skillbook", {})
        attempts = exp.get("max_attempts", "?")

        total = stat.get("total_instances", 0)
        processed = stat.get("processed_instances", 0)
        resolved = stat.get("resolved_count", 0)
        unresolved = stat.get("unresolved_count", 0)
        errors = max(processed - resolved - unresolved, 0)
        rate = stat.get("resolution_rate", 0.0)
        sb = stat.get("skillbook_assisted", {})
        sb_count = sb.get("count", 0) if isinstance(sb, dict) else 0
        iterate_repos_progress = None

        # Check for iterate_repos mode
        is_iterate_repos = bool(cfg.get("benchmark", {}).get("iterate_repos"))

        if is_iterate_repos:
            iterate_repos_progress = collect_iterate_repos_progress(
                d, cfg, stat, attempts if isinstance(attempts, int) else None
            )
            if iterate_repos_progress:
                total = iterate_repos_progress["total"]
                processed = iterate_repos_progress["processed"]
                resolved = iterate_repos_progress["resolved"]
                unresolved = iterate_repos_progress["unresolved"]
                errors = iterate_repos_progress["errors"]
                rate = iterate_repos_progress["rate"]
            phases = {}
        elif not has_stats or (total == 0 and is_active):
            # If no statistics.json yet, scan filesystem
            dataset = cfg.get("benchmark", {}).get("dataset")
            exclude = cfg.get("benchmark", {}).get("exclude_instances") or []
            # Compute effective total from config
            eff_total = cfg.get("benchmark", {}).get("max_instances")
            if eff_total is None:
                # Try completed run's statistics first, then compute from dataset
                eff_total = find_dataset_total(dataset, cfg.get("benchmark", {}).get("filter_repos"))
                if eff_total is None:
                    eff_total = compute_eff_total(dataset, exclude)
                elif exclude:
                    # find_dataset_total may not account for excludes; compute properly
                    eff_total = compute_eff_total(dataset, exclude) or eff_total
            elif exclude:
                # max_instances set — subtract only instances actually in dataset
                computed = compute_eff_total(dataset, exclude)
                if computed is not None:
                    eff_total = min(eff_total, computed)
            prog = scan_progress(
                d,
                max_attempts=attempts if isinstance(attempts, int) else None,
                total_instances=eff_total,
                dataset=dataset,
                filter_repos=cfg.get("benchmark", {}).get("filter_repos"),
            )
            total = prog["total"]
            processed = prog["processed"]
            resolved = prog["resolved"]
            unresolved = processed - resolved
            rate = resolved / processed if processed else 0.0
            phases = prog.get("phases", {})
        else:
            phases = {}

        start_time = get_start_time(d, stat)
        # For iterate_repos, statistics.json start_time is per-repo; prefer log file
        if is_iterate_repos:
            log_start = get_start_time(d, {})
            if log_start:
                start_time = log_start
        elapsed = stat.get("experiment_time_seconds")
        if is_active and start_time:
            try:
                elapsed = (datetime.now() - datetime.fromisoformat(start_time)).total_seconds()
            except Exception:
                pass

        # Avg time per instance (wall clock, accounts for concurrency)
        avg_inst_time = None
        if elapsed and processed:
            avg_inst_time = elapsed / processed

        status_str = "RUNNING" if is_active else (stat.get("status") or "done")

        entries.append({
            "dir": d.name,
            "test_slug": test_slug or "",
            "name": stat.get("run_name") or exp.get("name") or d.name,
            "status": status_str,
            "model_display": fmt_model_display(
                agent_llm.get("model", ""), ace_llm.get("model", "")
            ),
            "attempts": attempts,
            "concurrency": exp.get("concurrency", 1),
            "swe_learn": sb_cfg.get("custom_swe_learn", False),
            "total": total,
            "processed": processed,
            "resolved": resolved,
            "unresolved": unresolved,
            "errors": errors,
            "rate": rate,
            "phases": phases,
            "elapsed": elapsed,
            "avg_inst_time": avg_inst_time,
            "sb_assisted": sb_count,
            "start": start_time,
            "llm_config": llm,
            "iterate_repos_progress": iterate_repos_progress,
        })

    entries.sort(key=lambda e: (0 if e["status"] == "RUNNING" else 1, e["start"] or ""))
    return entries


# ── endpoint collection ──────────────────────────────────────────────────

PROVIDER_DEFAULTS = {
    "hosted_vllm": "http://localhost:8000/v1",
    "zai": "https://api.z.ai/api/paas/v4",
}


def collect_endpoints(entries: list[dict]) -> list[dict]:
    """Collect unique LLM endpoints from RUNNING experiments with health status."""
    # Dedup by (provider, api_base), collect roles and models
    ep_map: dict[tuple[str, str], dict] = {}

    for e in entries:
        if e["status"] != "RUNNING":
            continue
        llm = e.get("llm_config", {})
        for role in ("agent", "ace"):
            rcfg = llm.get(role, {})
            if not rcfg:
                continue
            provider = rcfg.get("provider", "?")
            api_base = rcfg.get("api_base") or PROVIDER_DEFAULTS.get(provider, "?")
            model = rcfg.get("model", "?")
            key = (provider, api_base)

            if key not in ep_map:
                ep_map[key] = {
                    "provider": provider,
                    "api_base": api_base,
                    "models": set(),
                    "roles": {},  # role -> count
                }
            ep = ep_map[key]
            ep["models"].add(model)
            ep["roles"][role] = ep["roles"].get(role, 0) + 1

    # Build result with health status
    result = []
    for key, ep in ep_map.items():
        status = get_endpoint_health(ep["provider"], ep["api_base"])
        models = ", ".join(sorted(ep["models"]))
        role_parts = [f"{r}({c})" for r, c in sorted(ep["roles"].items())]
        result.append({
            "status": status,
            "short_url": _shorten_url(ep["api_base"]),
            "api_base": ep["api_base"],
            "provider": ep["provider"],
            "models": models,
            "roles": " ".join(role_parts),
        })

    # Sort: hosted_vllm first, then zai; DOWN first within each group
    result.sort(key=lambda e: (0 if e["provider"] == "hosted_vllm" else 1,
                               0 if e["status"] == "DOWN" else 1))
    return result


# ── render ───────────────────────────────────────────────────────────────

C_RUNNING  = "\033[32m"
C_DONE     = "\033[90m"
C_WARN     = "\033[33m"
C_ERR      = "\033[31m"
RESET      = "\033[0m"
BOLD       = "\033[1m"
DIM        = "\033[2m"

STATUS_COLORS = {
    "RUNNING": C_RUNNING,
    "completed": C_DONE,
    "interrupted": C_WARN,
}


def render(entries, term_width: int):
    print(f"\033[H\033[2J", end="")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{BOLD}  agent-swe-ace experiment dashboard{RESET}  {now}")
    print()

    if not entries:
        print("  No experiments found.")
        return

    col_name = min(max((len(e["name"]) for e in entries), default=20), 42)
    col_model = min(max((len(e["model_display"]) for e in entries), default=15), 30)
    bar_w = min(15, max(8, (term_width - 160) // 2))

    # Legend
    print(f"  {DIM}Legend: █ resolved  ░ failed  · remaining | "
          f"SWE=custom-swe-learn | "
          f"Phases: train  vb=val_baseline  val | "
          f"TEST: e2e smoke test run | "
          f"Endpoints: UP/DOWN/cloud{RESET}")
    print()

    hdr = (
        f"  {'Status':<10} "
        f"{'Folder':<11} "
        f"{'Name':<{col_name}} "
        f"{'Model':<{col_model}} "
        f"{'Att':>3} "
        f"{'Con':>3} "
        f"{'Res/Proc/Total':>16} "
        f"{'Rate':>6} "
        f"{'Elapsed':>9} "
        f"{'Avg/inst':>9} "
        f"{'ETA':>9} "
        f"{'Started':<12}"
    )
    print(f"{BOLD}{hdr}{RESET}")
    print(f"  {'─' * (len(hdr) - 2)}")

    for e in entries:
        sc = STATUS_COLORS.get(e["status"], C_ERR)
        is_test = bool(e.get("test_slug"))
        name = e["name"]
        if len(name) > col_name:
            name = name[: col_name - 2] + ".."

        # Folder display: test slug for test runs, abbreviated run dir for production
        if is_test:
            folder = e["test_slug"][:11] if len(e["test_slug"]) > 11 else e["test_slug"]
        else:
            folder = short_dir(e["dir"])

        b = bar(e["resolved"], e["unresolved"], e["total"], bar_w)
        p = pct_str(e["processed"], e["total"])
        avg = fmt_duration(e["avg_inst_time"]) if e["avg_inst_time"] else "-"

        # ETA: extrapolate remaining instances at current throughput
        eta_str = "-"
        if e["elapsed"] and e["processed"] and e["total"] > e["processed"]:
            remaining = e["total"] - e["processed"]
            secs_per = e["elapsed"] / e["processed"]
            eta_secs = remaining * secs_per
            eta_str = fmt_duration(eta_secs)

        # TEST badge in status column for test runs
        status_display = f"TEST:{e['status']}" if is_test else e["status"]

        line = (
            f"  {sc}{status_display:<10}{RESET} "
            f"{DIM if is_test else ''}{folder:<11}{RESET if is_test else ''} "
            f"{name:<{col_name}} "
            f"{e['model_display']:<{col_model}} "
            f"{e['attempts']:>3} "
            f"{e['concurrency']:>3} "
            f"{sc}{e['resolved']:>3}/{e['processed']:>3}/{e['total']:<3}{RESET} "
            f"{e['rate']:>5.1%} "
            f"{fmt_duration(e['elapsed']):>9} "
            f"{avg:>7} "
            f"{eta_str:>9} "
            f"{fmt_time(e['start']):<12}"
        )
        print(line)
        detail = f"  {DIM}{'':>22} {b} {p}"
        parts = []
        if e["swe_learn"]:
            parts.append("swe-learn")
        if e["errors"] > 0:
            parts.append(f"{e['errors']} err")
        # Phase breakdown for two-phase runs
        if e["phases"]:
            phase_parts = []
            for pname in ("train", "val_baseline", "val"):
                pstat = e["phases"].get(pname)
                if pstat is None:
                    continue
                r, p, t = pstat
                label = "train" if pname == "train" else "vb" if pname == "val_baseline" else "val"
                if t > 0:
                    phase_parts.append(f"{label}:{r}/{t}")
                else:
                    phase_parts.append(f"{label}:-")
            if phase_parts:
                parts.append(" ".join(phase_parts))
        if parts:
            detail += f"  {', '.join(parts)}"
        detail += RESET
        print(detail)

        # Per-repo breakdown for iterate_repos
        irp = e.get("iterate_repos_progress")
        if irp:
            repo_total = irp["repo_total"]
            repo_done = irp["repo_completed"]
            # Count how many repos are in each phase
            phase_counts = {"train": 0, "vb": 0, "val": 0, "done": 0, "pending": 0}
            for rp in irp["repo_progress"]:
                cp = rp.get("current_phase", "pending")
                phase_counts[cp] = phase_counts.get(cp, 0) + 1
            parts = []
            for p, c in phase_counts.items():
                if c > 0:
                    parts.append(f"{c} {p}")
            print(f"  {DIM}{'':>22} repos: {', '.join(parts)}{RESET}")
            for rp in irp["repo_progress"]:
                short_repo = rp["name"].split("/")[-1]
                current_phase = rp.get("current_phase", "pending")
                if current_phase == "done":
                    r_color = C_DONE
                    phase_tag = "done "
                elif current_phase == "pending":
                    r_color = DIM
                    phase_tag = "wait "
                else:
                    r_color = C_RUNNING
                    phase_tag = f"{current_phase:>4}"

                # Build pipeline: train→vb→val with progress
                pipeline = []
                for plabel in ("train", "vb", "val"):
                    pd = rp["phases"].get(plabel)
                    if not pd or pd["total"] == 0:
                        continue
                    is_current = (plabel == current_phase)
                    if pd["processed"] >= pd["total"]:
                        # Phase complete
                        pipeline.append(f"{plabel}:{pd['resolved']}/{pd['total']}✓")
                    elif is_current:
                        # Active phase — show progress bar feel
                        pipeline.append(f"{plabel}:{pd['resolved']}/{pd['processed']}/{pd['total']}")
                    else:
                        pipeline.append(f"{plabel}:{pd['resolved']}/{pd['processed']}/{pd['total']}")
                pipe_str = " → ".join(pipeline) if pipeline else "-"

                r_resolved = rp["resolved"]
                r_total = rp["total"]
                r_rate = f"{r_resolved}/{r_total}" if r_total else "-"

                print(f"  {DIM}{'':>22}{r_color}{phase_tag}{RESET} {DIM}{short_repo:<16} "
                      f"{r_rate:>8}  {pipe_str}{RESET}")

    # Summary
    print()
    sections = [
        ("Active", [e for e in entries if e["status"] == "RUNNING"]),
        ("Completed", [e for e in entries if e["status"] == "completed"]),
    ]
    for label, runs in sections:
        if not runs:
            continue
        t_res = sum(e["resolved"] for e in runs)
        t_proc = sum(e["processed"] for e in runs)
        t_total = sum(e["total"] for e in runs)
        avg = t_res / t_proc if t_proc else 0
        avg_t = fmt_duration(
            sum(e["elapsed"] for e in runs if e["elapsed"]) / len(runs)
            if any(e["elapsed"] for e in runs) else None
        )
        print(f"  {label}: {len(runs)} runs | "
              f"{t_proc}/{t_total} processed | "
              f"{t_res} resolved | "
              f"avg {avg:.1%} | "
              f"~{avg_t}/run")


def render_endpoints(endpoints: list[dict], term_width: int):
    """Render the endpoint health section."""
    if not endpoints:
        return
    HEALTH_COLORS = {"UP": C_RUNNING, "DOWN": C_ERR, "cloud": C_DONE, "ERR": C_WARN}
    print()
    print(f"  {BOLD}Endpoints:{RESET}")
    for ep in endpoints:
        sc = HEALTH_COLORS.get(ep["status"], DIM)
        print(f"  {sc}{ep['status']:<6}{RESET} "
              f"{ep['short_url']:<16} "
              f"{ep['provider']:<14} "
              f"{ep['models']:<30} "
              f"{ep['roles']}")


# ── main loop ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Watch agent-swe-ace experiments")
    parser.add_argument("-n", "--no-refresh", action="store_true", help="One-shot")
    parser.add_argument("-i", "--interval", type=int, default=10, help="Refresh seconds (default: 10)")
    parser.add_argument("--all", action="store_true", help="Show all runs incl. old completed")
    parser.add_argument("--running", action="store_true", help="Show only active runs")
    parser.add_argument("--tests", action="store_true", help="Show only e2e smoke test runs")
    args = parser.parse_args()

    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 180

    if args.no_refresh:
        runs = collect_runs(args.all, args.running, only_tests=args.tests)
        render(runs, term_width)
        render_endpoints(collect_endpoints(runs), term_width)
        return

    alive = True

    def _stop(sig, frame):
        nonlocal alive
        alive = False

    signal.signal(signal.SIGINT, _stop)

    while alive:
        runs = collect_runs(args.all, args.running, only_tests=args.tests)
        render(runs, term_width)
        render_endpoints(collect_endpoints(runs), term_width)
        print(f"\n  {DIM}Refresh {args.interval}s | Ctrl+C to quit{RESET}", flush=True)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            break

    print("\033[H\033[2J", end="")
    print("Dashboard stopped.")


if __name__ == "__main__":
    main()
