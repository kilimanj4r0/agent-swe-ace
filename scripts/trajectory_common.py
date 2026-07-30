"""Shared loader + master per-attempt table for trajectory analysis.

Walks the configured Lite (single-phase) and split025 (two-phase) runs, joins each
trajectory with its matching result file, computes length/token metrics and a
rule-based error category, and emits one row per attempt to
``trajectories_analysis_results/trajectories_attempts.csv``.

Both ``analyze_trajectory_length.py`` and ``analyze_trajectory_errors.py`` consume
this CSV read-only.

Run:  uv run python scripts/trajectory_common.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "trajectories_analysis_results"
CSV_PATH = OUT / "trajectories_attempts.csv"

PHASES = ("train", "val_baseline", "val")

# (panel, run_dir, backbone, learn, phase, has_skillbook, sb_mode)
# Lite runs are single-phase -> phase=None. split025 runs are two-phase; we take
# val_baseline (empty skillbook) + val (skillbook).
EXTRACTIONS: list[tuple] = [
    # ---- Panel A: Lite (3-way) ----
    ("Lite", "run_20260414_015225_completed_glm_4a_default", "GLM", "default", None, True, "per_instance"),
    ("Lite", "run_20260414_015144_completed_glm_4a_swe", "GLM", "swe", None, True, "per_instance"),
    ("Lite", "run_20260415_020217_completed_qwen3next_4a_default", "QNext", "default", None, True, "per_instance"),
    ("Lite", "run_20260415_020540_completed_qwen3next_4a_swe", "QNext", "swe", None, True, "per_instance"),
    ("Lite", "run_20260426_211500_completed_qwen3_4a_default", "Q30", "default", None, True, "per_instance"),
    ("Lite", "run_20260426_211426_completed_qwen3_4a_swe", "Q30", "swe", None, True, "per_instance"),
    ("Lite", "run_20260526_133345_completed_qwen3_4a_no_skillbook", "Q30", "no-sb", None, False, "none"),
    # ---- Panel B: split025 Verified (Q30 vs QNext) ----
    # val_baseline = empty skillbook (pure backbone); val = skillbook
    ("split025", "run_20260605_111733_completed_qwen3_global_split025_default_verified_vpk5", "Q30", "default", "val_baseline", False, "global"),
    ("split025", "run_20260605_111733_completed_qwen3_global_split025_default_verified_vpk5", "Q30", "default", "val", True, "global"),
    ("split025", "run_20260605_111708_completed_qwen3_repos_split025_default_verified_vpk5", "Q30", "default", "val_baseline", False, "per_repo"),
    ("split025", "run_20260605_111708_completed_qwen3_repos_split025_default_verified_vpk5", "Q30", "default", "val", True, "per_repo"),
    ("split025", "run_20260615_125626_completed_qwen3next_global_split025_default_verified_vpk5", "QNext", "default", "val_baseline", False, "global"),
    ("split025", "run_20260615_125626_completed_qwen3next_global_split025_default_verified_vpk5", "QNext", "default", "val", True, "global"),
    ("split025", "run_20260622_105529_completed_qwen3next_repos_split025_default_verified_vpk5", "QNext", "default", "val_baseline", False, "per_repo"),
    ("split025", "run_20260622_105529_completed_qwen3next_repos_split025_default_verified_vpk5", "QNext", "default", "val", True, "per_repo"),
]

FIELDS = [
    "panel", "backbone", "run", "learn", "phase", "has_skillbook", "sb_mode",
    "instance_id", "iter", "exit_status", "resolved", "error_category",
    "steps", "prompt_tokens", "completion_tokens", "total_tokens",
    "reasoning_tokens", "api_calls",
]


def find_benchmark_dir(run_dir: Path) -> Path | None:
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and "__" in child.name:
            return child
    return None


def _to_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def load_trajectory(path: Path) -> dict:
    j = json.loads(path.read_text())
    info = j.get("info") or {}
    messages = j.get("messages") or []

    exit_status = info.get("exit_status")
    submission = info.get("submission") or ""

    # steps = assistant turns
    steps = info.get("assistant_message_count")
    if steps is None:
        steps = sum(1 for m in messages if m.get("role") == "assistant")
    steps = _to_int(steps)

    # tokens: sum per-call usage over assistant messages
    prompt = completion = total = reasoning = 0
    api_calls = 0
    model = info.get("model")
    for m in messages:
        if m.get("role") != "assistant":
            continue
        extra = m.get("extra") or {}
        resp = extra.get("response") or {}
        if not model:
            model = resp.get("model")
        usage = resp.get("usage") or {}
        if usage:
            api_calls += 1
            prompt += _to_int(usage.get("prompt_tokens"))
            completion += _to_int(usage.get("completion_tokens"))
            total += _to_int(usage.get("total_tokens"))
            cd = usage.get("completion_tokens_details") or {}
            reasoning += _to_int(cd.get("reasoning_tokens"))

    return {
        "instance_id": info.get("instance_id") or path.parent.name,
        "iter": _to_int(info.get("iteration", path.stem.split("_")[-1])),
        "exit_status": exit_status,
        "steps": steps,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": reasoning,
        "api_calls": api_calls,
        "submission": submission,
        "model": model,
        "message_count": _to_int(info.get("message_count", len(messages))),
    }


def load_result(result_path: Path) -> dict | None:
    if not result_path.exists():
        return None
    j = json.loads(result_path.read_text())
    metrics = j.get("metrics") or {}
    return {
        "resolved": j.get("resolved"),
        "patch_empty": metrics.get("patch_empty"),
        "patch_invalid_format": metrics.get("patch_invalid_format"),
        "feedback": j.get("feedback") or "",
    }


def classify_error(exit_status, resolved, patch_empty, patch_invalid, has_submission) -> str:
    """Rule-based category with precedence (see design spec)."""
    if exit_status == "error":
        return "runtime_error"
    if exit_status == "TimeoutExpired":
        return "timeout"
    if exit_status == "ContextWindowExceeded":
        return "context_window_exceeded"
    if exit_status == "LimitsExceeded":
        return "limit_exceeded"
    # Submitted (or other) -> look at the patch
    if patch_invalid:
        return "invalid_format"
    empty = bool(patch_empty) if patch_empty is not None else (not has_submission)
    if empty:
        return "no_patch"
    if resolved is True:
        return "resolved"
    return "submitted_tests_failed"


def extract() -> list[dict]:
    rows: list[dict] = []
    for panel, run_name, backbone, learn, phase, has_sb, sb_mode in EXTRACTIONS:
        run_dir = DATA / run_name
        if not run_dir.exists():
            print(f"  [skip] missing run dir: {run_name}", file=sys.stderr)
            continue
        bench = find_benchmark_dir(run_dir)
        if bench is None:
            print(f"  [skip] no benchmark dir in {run_name}", file=sys.stderr)
            continue
        if phase is None:
            traj_glob = bench / "trajectories"
        else:
            traj_glob = bench / "trajectories" / phase
        if not traj_glob.exists():
            print(f"  [skip] no trajectories/{phase} in {run_name}", file=sys.stderr)
            continue

        for traj in sorted(traj_glob.glob("*/iter_*.json")):
            instance_dir = traj.parent
            t = load_trajectory(traj)
            result_path = bench / "results" / (phase or "") / instance_dir.name / traj.name
            # In single-phase runs results live at results/<inst>/iter_N.json
            if phase is None:
                result_path = bench / "results" / instance_dir.name / traj.name
            r = load_result(result_path)
            resolved = r["resolved"] if r else None
            cat = classify_error(
                t["exit_status"], resolved,
                r["patch_empty"] if r else None,
                r["patch_invalid_format"] if r else None,
                bool(t["submission"]),
            )
            rows.append({
                "panel": panel, "backbone": backbone, "run": run_name, "learn": learn,
                "phase": phase or "", "has_skillbook": has_sb, "sb_mode": sb_mode,
                "instance_id": t["instance_id"], "iter": t["iter"],
                "exit_status": t["exit_status"] or "", "resolved": "" if resolved is None else resolved,
                "error_category": cat, "steps": t["steps"],
                "prompt_tokens": t["prompt_tokens"], "completion_tokens": t["completion_tokens"],
                "total_tokens": t["total_tokens"], "reasoning_tokens": t["reasoning_tokens"],
                "api_calls": t["api_calls"],
            })
    return rows


def summarize(rows: list[dict]) -> None:
    print(f"\nTotal attempts: {len(rows)}")
    # per panel/backbone/phase
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["panel"], r["backbone"], r["phase"]), []).append(r)
    print(f"\n{'panel/backbone/phase':45s} {'n':>5s} {'res%':>6s}  exit_status(top) / error_category(top)")
    for key in sorted(groups):
        rs = groups[key]
        n = len(rs)
        res = sum(1 for r in rs if r["resolved"] is True) / n * 100 if n else 0
        exits = Counter(r["exit_status"] for r in rs)
        cats = Counter(r["error_category"] for r in rs)
        label = f"{key[0]}/{key[1]}/{key[2] or '-'}"
        print(f"{label:45s} {n:5d} {res:5.1f}%  exit={dict(exits.most_common(3))}")
        print(f"{'':45s} {'':>5s} {'':>6s}  cats={dict(cats.most_common(5))}")
    # length sanity
    print("\nSteps / total_tokens / completion_tokens (median):")
    for key in sorted(groups):
        rs = groups[key]
        sm = median([r["steps"] for r in rs])
        tm = median([r["total_tokens"] for r in rs])
        cm = median([r["completion_tokens"] for r in rs])
        label = f"{key[0]}/{key[1]}/{key[2] or '-'}"
        print(f"  {label:45s} steps={sm:6.0f}  total_tok={tm:8.0f}  compl_tok={cm:7.0f}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    rows = extract()
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {CSV_PATH.relative_to(ROOT)}")
    summarize(rows)


if __name__ == "__main__":
    main()
