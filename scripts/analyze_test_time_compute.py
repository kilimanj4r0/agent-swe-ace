#!/usr/bin/env python3
"""Per-attempt test-time compute analysis for the aggregated val-baseline folder.

For every attempt (instance x run x iter) it extracts, from the stored trajectory
and result:
  - exit_status, resolved (bool), #assistant steps
  - token usage (input / output / total), summed over the agent's LLM calls
  - wall-clock seconds (last message ts - first message ts)

Writes analysis/per_attempt.csv and analysis/compute_summary.json, and prints
summary stats overall + by exit_status and by resolved.

Usage:
    uv run python scripts/analyze_test_time_compute.py
    uv run python scripts/analyze_test_time_compute.py \
        --data-dir data/val_baseline_aggregated_split025_vpk5
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

AGG = Path("data/val_baseline_aggregated_split025_vpk5")


def attempt_compute(traj: dict) -> dict:
    msgs = traj.get("messages", [])
    info = traj.get("info", {})
    in_t = out_t = tot = 0
    n_asst = 0
    t0 = t1 = None
    for m in msgs:
        ts = m.get("timestamp")
        if isinstance(ts, (int, float)):
            t0 = ts if t0 is None else min(t0, ts)
            t1 = ts if t1 is None else max(t1, ts)
        if m.get("role") == "assistant":
            n_asst += 1
            u = (m.get("extra", {}) or {}).get("response", {}).get("usage") or {}
            in_t += u.get("prompt_tokens", 0) or 0
            out_t += u.get("completion_tokens", 0) or 0
            tot += u.get("total_tokens", 0) or 0
    wall = (t1 - t0) if (t0 is not None and t1 is not None and t1 > t0) else None
    return {
        "exit_status": info.get("exit_status"),
        "n_steps": info.get("assistant_message_count", n_asst) or n_asst,
        "in_tokens": in_t,
        "out_tokens": out_t,
        "total_tokens": tot,
        "wall_seconds": wall,
    }


def describe(vals: list, name: str) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    qs = statistics.quantiles(vals, n=100) if len(vals) >= 100 else sorted(vals)
    p10 = qs[9] if len(vals) >= 100 else vals[len(vals) // 10]
    p90 = qs[89] if len(vals) >= 100 else vals[len(vals) * 9 // 10]
    out = {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "p10": p10,
        "p90": p90,
        "min": min(vals),
        "max": max(vals),
        "sum": sum(vals),
    }
    print(f"  {name:<26} mean={out['mean']:>14,.0f}  median={out['median']:>12,.0f}  "
          f"p90={out['p90']:>14,.0f}  max={out['max']:>14,.0f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=AGG)
    args = ap.parse_args()
    D = args.data_dir
    traj_root = D / "trajectories"
    res_root = D / "results"
    out_dir = D / "analysis"
    out_dir.mkdir(exist_ok=True)

    rows = []
    files = sorted(traj_root.glob("*/*.json"))
    for i, tf in enumerate(files):
        instance = tf.parent.name
        run, iter_ = tf.stem.split("_iter")  # rNN, N
        rf = res_root / instance / f"{run}_iter{iter_}.json"
        resolved = None
        if rf.exists():
            resolved = bool(json.loads(rf.read_text()).get("resolved"))
        traj = json.loads(tf.read_text())
        c = attempt_compute(traj)
        rows.append({
            "instance": instance,
            "run": run, "iter": int(iter_),
            "resolved": resolved,
            **c,
        })
        if (i + 1) % 1000 == 0:
            print(f"  ...{i+1}/{len(files)}", flush=True)

    # write CSV
    with open(out_dir / "per_attempt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_dir/'per_attempt.csv'} ({len(rows)} attempts)")

    # summary
    def vals(key, subset=rows):
        return [r[key] for r in subset if r.get(key) is not None]

    summary = {"n_attempts": len(rows)}
    print("\n=== OVERALL per-attempt compute ===")
    summary["total_tokens"] = describe(vals("total_tokens"), "total tokens")
    summary["in_tokens"] = describe(vals("in_tokens"), "input tokens")
    summary["out_tokens"] = describe(vals("out_tokens"), "output tokens")
    summary["wall_seconds"] = describe(vals("wall_seconds"), "wall seconds")
    summary["n_steps"] = describe(vals("n_steps"), "agent steps")

    print("\n=== by exit_status ===")
    by_es = defaultdict(list)
    for r in rows:
        by_es[r["exit_status"]].append(r)
    summary["by_exit_status"] = {}
    for es, sub in sorted(by_es.items()):
        print(f"  [{es}]  n={len(sub)}  resolved_any={sum(1 for r in sub if r['resolved'])}")
        summary["by_exit_status"][es] = {
            "n": len(sub),
            "total_tokens": describe([r["total_tokens"] for r in sub], f"  total tokens"),
            "wall_seconds": describe([r["wall_seconds"] for r in sub], f"  wall seconds"),
        }

    print("\n=== by resolved ===")
    summary["by_resolved"] = {}
    for flag, label in [(True, "RESOLVED"), (False, "UNRESOLVED")]:
        sub = [r for r in rows if r["resolved"] == flag]
        print(f"  [{label}]  n={len(sub)}")
        summary["by_resolved"][label] = {
            "n": len(sub),
            "total_tokens": describe([r["total_tokens"] for r in sub], f"  total tokens"),
            "wall_seconds": describe([r["wall_seconds"] for r in sub], f"  wall seconds"),
            "n_steps": describe([r["n_steps"] for r in sub], f"  agent steps"),
        }

    (out_dir / "compute_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir/'compute_summary.json'}")


if __name__ == "__main__":
    main()
