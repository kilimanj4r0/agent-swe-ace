#!/usr/bin/env python3
"""Build per-run val tables comparing each run's val_skillbook (valSB, 5 attempts)
against the aggregated val_baseline (valBL, 60 attempts shared across all runs).

Output (CSV): one CSV per run (named <run>.csv), rows = real instance ids, cells =
float resolution ratio (`resolved_attempts / total_attempts`) for the valBL and
valSB columns.

Usage:
    uv run python scripts/make_val_per_run_tables.py \
        --aggregated data/val_baseline_aggregated_split025_vpk5_qwen3 \
        --runs data/*completed*vpk5* \
        --output-dir data/val_per_run_tables_split025_vpk5
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys


def ratio(resolved: int, total: int, ndigits: int = 4) -> float:
    """Resolution ratio as a rounded float (0.0 if no attempts)."""
    return round(resolved / total, ndigits) if total else 0.0


def load_valbl(aggregated_dir: str) -> dict[str, tuple[int, int]]:
    """Return {instance_id: (successes, n_attempts)} from the aggregated baseline."""
    path = os.path.join(aggregated_dir, "stats", "per_instance.json")
    with open(path) as f:
        data = json.load(f)
    out = {}
    for inst, rec in data.items():
        out[inst] = (int(rec.get("successes", 0)), int(rec.get("n_attempts", 0)))
    return out


def load_valsb(run_dir: str, benchmark: str) -> dict[str, tuple[int, int]]:
    """Return {instance_id: (resolved_attempts, total_attempts)} for a run's val set."""
    val_results = os.path.join(run_dir, benchmark, "results", "val")
    out = {}
    if not os.path.isdir(val_results):
        return out
    for inst in sorted(os.listdir(val_results)):
        idir = os.path.join(val_results, inst)
        if not os.path.isdir(idir):
            continue
        files = sorted(glob.glob(os.path.join(idir, "iter_*.json")))
        resolved = 0
        total = 0
        for fp in files:
            try:
                with open(fp) as f:
                    rec = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            total += 1
            if rec.get("resolved"):
                resolved += 1
        if total:
            out[inst] = (resolved, total)
    return out


def detect_benchmark(run_dir: str) -> str:
    """Return the benchmark-scoped subdir (e.g. princeton-nlp__SWE-bench_Verified)."""
    for name in os.listdir(run_dir):
        full = os.path.join(run_dir, name)
        if os.path.isdir(full) and os.path.isdir(os.path.join(full, "results")):
            return name
    raise RuntimeError(f"no benchmark subdir with results/ under {run_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aggregated", required=True,
                    help="aggregated val_baseline dir (has stats/per_instance.json)")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="completed run dirs (each with <benchmark>/results/val)")
    ap.add_argument("--benchmark", default=None,
                    help="benchmark subdir name; auto-detected if omitted")
    ap.add_argument("--output-dir", required=True, help="output dir (one CSV per run)")
    args = ap.parse_args()

    valbl = load_valbl(args.aggregated)

    runs = []
    for pattern in args.runs:
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[" ) else [pattern]
        runs.extend(matches)
    # dedupe preserving order
    seen = set()
    runs = [r for r in runs if not (r in seen or seen.add(r))]

    # union of instances, sorted; valBL drives the row set, valSB may add extras
    instance_set = set(valbl)
    per_run_sb = {}
    for run in runs:
        bench = args.benchmark or detect_benchmark(run)
        sb = load_valsb(run, bench)
        per_run_sb[run] = sb
        instance_set |= set(sb)

    instances = sorted(instance_set)
    os.makedirs(args.output_dir, exist_ok=True)

    for run in runs:
        sb = per_run_sb[run]
        run_name = os.path.basename(run.rstrip("/"))
        out_path = os.path.join(args.output_dir, f"{run_name}.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["instance_id", "valBL", "valSB"])
            for inst in instances:
                bl = valbl.get(inst)
                bl_cell = ratio(*bl) if bl else ""
                s = sb.get(inst)
                sb_cell = ratio(*s) if s else ""
                w.writerow([inst, bl_cell, sb_cell])

    print(f"wrote {len(runs)} CSVs ({len(instances)} instances) -> {args.output_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
