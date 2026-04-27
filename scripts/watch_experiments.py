#!/usr/bin/env python3
"""
Live CLI dashboard for monitoring agent-swe-ace experiments.

Usage:
    python scripts/watch_experiments.py          # auto-refresh every 10s
    python scripts/watch_experiments.py -n        # one-shot, no refresh
    python scripts/watch_experiments.py -i 5      # refresh every 5s
    python scripts/watch_experiments.py --all     # show completed runs too
    python scripts/watch_experiments.py --running  # show only active runs
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── helpers ──────────────────────────────────────────────────────────────

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


def get_start_time(run_dir: Path, stat: dict) -> str | None:
    """Get start time from statistics.json, then experiment.log first line."""
    t = stat.get("start_time") or stat.get("timestamp")
    if t:
        return t
    log = run_dir / "experiment.log"
    if log.exists():
        try:
            first_line = log.read_text().split("\n")[0]
            # Format: "2026-04-14 01:51:44 | INFO ..."
            ts = first_line.split(" | ")[0].strip()
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            return dt.isoformat()
        except Exception:
            pass
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
                    if "experiment.log" in target and "data/run_" in target:
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

def scan_progress(run_dir: Path) -> dict:
    """Count processed/resolved from results dir when statistics.json is absent."""
    results_dir = run_dir / "princeton-nlp__SWE-bench_Lite" / "results"
    if not results_dir.exists():
        results_dir = run_dir / "results"
    if not results_dir.exists():
        return {"processed": 0, "resolved": 0, "total": 0}

    processed = 0
    resolved = 0
    for inst_dir in results_dir.iterdir():
        if not inst_dir.is_dir():
            continue
        processed += 1
        iters = sorted(inst_dir.glob("iter_*.json"))
        if iters:
            data = load_json(iters[-1])
            if data.get("resolved"):
                resolved += 1

    return {"processed": processed, "resolved": resolved, "total": 292}


# ── collect ──────────────────────────────────────────────────────────────

def collect_runs(show_all: bool, only_running: bool):
    if not DATA_DIR.exists():
        return []

    active_dirs = get_active_run_dirs()
    entries = []

    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue

        has_config = (d / "config.json").exists()
        has_stats = (d / "statistics.json").exists()
        if not has_config and not has_stats:
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

        # If no statistics.json yet, scan filesystem
        if not has_stats or (total == 0 and is_active):
            prog = scan_progress(d)
            total = prog["total"]
            processed = prog["processed"]
            resolved = prog["resolved"]
            unresolved = processed - resolved
            rate = resolved / processed if processed else 0.0

        start_time = get_start_time(d, stat)
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
            "elapsed": elapsed,
            "avg_inst_time": avg_inst_time,
            "sb_assisted": sb_count,
            "start": start_time,
        })

    entries.sort(key=lambda e: (0 if e["status"] == "RUNNING" else 1, e["start"] or ""))
    return entries


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
    bar_w = min(15, max(8, (term_width - 155) // 2))

    # Legend
    print(f"  {DIM}Legend: █ resolved  ░ failed  · remaining | "
          f"Learn: SWE=custom-swe-learn  SB-assist=skillbook helped solve{RESET}")
    print()

    hdr = (
        f"  {'Status':<10} "
        f"{'Name':<{col_name}} "
        f"{'Model':<{col_model}} "
        f"{'Att':>3} "
        f"{'Con':>3} "
        f"{'Res/Proc/Total':>16} "
        f"{'Rate':>6} "
        f"{'SB':>3} "
        f"{'Elapsed':>9} "
        f"{'Avg/inst':>9} "
        f"{'Started':<12}"
    )
    print(f"{BOLD}{hdr}{RESET}")
    print(f"  {'─' * (len(hdr) - 2)}")

    for e in entries:
        sc = STATUS_COLORS.get(e["status"], C_ERR)
        name = e["name"]
        if len(name) > col_name:
            name = name[: col_name - 2] + ".."

        b = bar(e["resolved"], e["unresolved"], e["total"], bar_w)
        p = pct_str(e["processed"], e["total"])
        avg = fmt_duration(e["avg_inst_time"]) if e["avg_inst_time"] else "-"

        learn_flag = "swe" if e["swe_learn"] else "-"
        line = (
            f"  {sc}{e['status']:<10}{RESET} "
            f"{name:<{col_name}} "
            f"{e['model_display']:<{col_model}} "
            f"{e['attempts']:>3} "
            f"{e['concurrency']:>3} "
            f"{sc}{e['resolved']:>3}/{e['processed']:>3}/{e['total']:<3}{RESET} "
            f"{e['rate']:>5.1%} "
            f"{e['sb_assisted']:>3} "
            f"{fmt_duration(e['elapsed']):>9} "
            f"{avg:>7} "
            f"{fmt_time(e['start']):<12}"
        )
        print(line)
        detail = f"  {DIM}{'':>10} {b} {p}"
        parts = []
        if e["swe_learn"]:
            parts.append("swe-learn")
        if e["errors"] > 0:
            parts.append(f"{e['errors']} err")
        if parts:
            detail += f"  {', '.join(parts)}"
        detail += RESET
        print(detail)

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


# ── main loop ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Watch agent-swe-ace experiments")
    parser.add_argument("-n", "--no-refresh", action="store_true", help="One-shot")
    parser.add_argument("-i", "--interval", type=int, default=10, help="Refresh seconds (default: 10)")
    parser.add_argument("--all", action="store_true", help="Show all runs incl. old completed")
    parser.add_argument("--running", action="store_true", help="Show only active runs")
    args = parser.parse_args()

    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 180

    if args.no_refresh:
        render(collect_runs(args.all, args.running), term_width)
        return

    alive = True

    def _stop(sig, frame):
        nonlocal alive
        alive = False

    signal.signal(signal.SIGINT, _stop)

    while alive:
        render(collect_runs(args.all, args.running), term_width)
        print(f"\n  {DIM}Refresh {args.interval}s | Ctrl+C to quit{RESET}", flush=True)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            break

    print("\033[H\033[2J", end="")
    print("Dashboard stopped.")


if __name__ == "__main__":
    main()
