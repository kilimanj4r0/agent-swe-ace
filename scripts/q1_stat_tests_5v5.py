#!/usr/bin/env python3
"""Q1 5-vs-5 stat tests: for each qwen3 vpk5 run, pair that run's OWN 5-attempt
val_baseline (empty skillbook, no learning) against its 5-attempt val_skillbook,
over the shared val instances.

This is a **k-symmetric (5 vs 5)** design -- fairer than pairing against the
aggregated 60-attempt baseline, whose higher k structurally favored the baseline's
`resolved_any` (more attempts ⇒ more chances to solve at least once). Here both
conditions draw exactly 5 attempts, so McNemar's BL_only/SB_only are directly
comparable and both conditions share identical coarseness (multiples of 1/5).

Runs four paired tests per run:
  1. Paired Wilcoxon signed-rank
  2. Paired t-test (primary)
  3. McNemar (now meaningful: resolved_any under 5 vs 5)
  4. Bootstrap 95% CI on the mean diff
Benjamini-Hochberg FDR over the run family. Reuses test helpers from q1_stat_tests.py.

Usage:
    uv run python scripts/q1_stat_tests_5v5.py \
        --runs 'data/*completed_qwen3_*split025*vpk5' \
        --original-runs-index data/val_baseline_aggregated_split025_vpk5_qwen3/runs_index.json \
        --output data/val_per_run_tables_split025_vpk5/Q1_5v5_stat_tests_report.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

from q1_stat_tests import (
    bh_fdr,
    boot_ci,
    fmt_p,
    mcnemar_test,
    short_label,
    star,
    ttest_paired,
    wilcoxon,
)


def _resolved(fp: str) -> bool:
    try:
        with open(fp) as f:
            return bool(json.load(f).get("resolved"))
    except (OSError, json.JSONDecodeError):
        return False


def load_phase_rates(run_dir: str, benchmark: str, phase: str) -> dict[str, float]:
    """{instance_id: resolved/total rate} for a run's phase (val_baseline or val)."""
    base = os.path.join(run_dir, benchmark, "results", phase)
    out: dict[str, float] = {}
    if not os.path.isdir(base):
        return out
    for inst in sorted(os.listdir(base)):
        idir = os.path.join(base, inst)
        if not os.path.isdir(idir):
            continue
        files = sorted(glob.glob(os.path.join(idir, "iter_*.json")))
        if not files:
            continue
        res = sum(1 for fp in files if _resolved(fp))
        out[inst] = round(res / len(files), 4)
    return out


def detect_benchmark(run_dir: str) -> str:
    for name in os.listdir(run_dir):
        full = os.path.join(run_dir, name)
        if os.path.isdir(full) and os.path.isdir(os.path.join(full, "results")):
            return name
    raise RuntimeError(f"no benchmark subdir with results/ under {run_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, help="glob of completed run dirs")
    ap.add_argument("--output", required=True, help="output markdown path")
    ap.add_argument("--original-runs-index", default=None,
                    help="aggregated baseline runs_index.json; flags those runs as 'orig'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot-iters", type=int, default=10000)
    args = ap.parse_args()

    runs = sorted(glob.glob(args.runs))
    runs = [r for r in runs if os.path.isdir(r)
            and os.path.exists(os.path.join(r, "statistics.json"))]
    if not runs:
        print(f"no runs matched: {args.runs}", file=sys.stderr)
        return 1

    original: set[str] = set()
    if args.original_runs_index:
        idx = json.load(open(args.original_runs_index))
        original = {os.path.basename(v["run_dir"].rstrip("/")) for v in idx.values()}

    rows = []
    for i, run in enumerate(runs):
        bench = detect_benchmark(run)
        bl = load_phase_rates(run, bench, "val_baseline")
        sb = load_phase_rates(run, bench, "val")
        insts = sorted(set(bl) & set(sb))
        blv = np.array([bl[j] for j in insts], float)
        sbv = np.array([sb[j] for j in insts], float)
        w = wilcoxon(sbv, blv)
        t = ttest_paired(sbv, blv)
        m = mcnemar_test(blv, sbv)
        b = boot_ci(sbv, blv, args.seed + i, args.boot_iters)
        base = os.path.basename(run.rstrip("/"))
        rows.append({"label": short_label(run), "run": base,
                     "orig": (base in original) if original else None,
                     "n_inst": len(insts), "wil": w, "ttest": t, "mc": m, "boot": b})

    for key in ("wil", "ttest", "mc"):
        ps = [r[key]["p"] for r in rows]
        corr = bh_fdr(ps)
        for r, c in zip(rows, corr):
            r[key]["p_fdr"] = c

    write_report(rows, args)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def disp(r):
    tag = " [orig]" if r.get("orig") else ""
    return f"{r['label']}{tag}"


def write_report(rows, args):
    L = []
    n_runs = len(rows)
    n_inst = rows[0]["n_inst"] if rows else 0
    has_orig = bool(rows) and rows[0]["orig"] is not None

    L.append("# Q1 (5-vs-5) — Does the skillbook beat the run's OWN empty baseline?")
    L.append("")
    L.append(f"For each of the {n_runs} qwen3 vpk5 runs, the run's OWN 5-attempt "
             f"val_baseline (empty skillbook, no learning) is paired against its 5-attempt "
             f"val_skillbook over the shared {n_inst} val instances. **k-symmetric (5 vs 5)** — "
             "fairer than the aggregated 60-attempt baseline, whose higher k structurally "
             "favored `resolved_any`. diff = valSB5 − valBL5 (positive ⇒ skillbook helps). "
             f"Two-sided; Benjamini–Hochberg FDR over the {n_runs}-run family. "
             f"Bootstrap: {args.boot_iters} resamples, instance-level, seed={args.seed}. α = 0.05.")
    L.append("")
    if has_orig:
        n_orig = sum(1 for r in rows if r["orig"])
        L.append(f"_{n_orig} of the {n_runs} runs are in the original aggregated 60-attempt "
                 "baseline (marked `[orig]`); the rest are newer retrieval-method runs added "
                 "since. The 5-vs-5 design is independent of that aggregation._")
        L.append("")

    # --- Wilcoxon ---
    L.append("## 1. Paired Wilcoxon signed-rank (non-parametric)")
    L.append("Tests whether per-instance rate differences are systematically off zero. "
             "Zero-difference instances dropped (n_pairs reported). "
             "r = matched-pairs rank-biserial effect size in [−1, +1].")
    L.append("")
    L.append("| run | n_pairs | W | p_raw | p_FDR | r |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        w = r["wil"]
        L.append(f"| {disp(r)} | {w['n_pairs']} | {w['W']:.1f} | "
                 f"{fmt_p(w['p'])} | {fmt_p(w['p_fdr'])} | {w['r']:+.3f} |")
    L.append("")

    # --- t-test ---
    L.append("## 2. Paired t-test (parametric)")
    L.append("Tests the mean per-instance rate difference; reports 95% CI and "
             "Cohen's dz (signed, paired = mean_diff / sd_diff).")
    L.append("")
    L.append("| run | mean_diff | 95% CI | t | p_raw | p_FDR | Cohen's dz |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        t = r["ttest"]
        L.append(f"| {disp(r)} | {t['mean']:+.4f} | "
                 f"[{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}] | {t['t']:+.2f} | "
                 f"{fmt_p(t['p'])} | {fmt_p(t['p_fdr'])} | {t['d']:+.3f} |")
    L.append("")

    # --- McNemar ---
    L.append("## 3. McNemar (resolved-at-least-once: rate > 0) — now fair (5 vs 5)")
    L.append("Pairs each instance as solved-yes/no under each condition. With both sides "
             "drawing exactly 5 attempts, `BL_only` and `SB_only` are now directly comparable "
             "(unlike the aggregated design, where BL's 60 attempts inflated `BL_only`). "
             "Exact two-sided sign test on discordants.")
    L.append("")
    L.append("| run | both | BL_only | SB_only | neither | p_raw | p_FDR |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        m = r["mc"]
        L.append(f"| {disp(r)} | {m['both']} | {m['bl_only']} | {m['sb_only']} | "
                 f"{m['neither']} | {fmt_p(m['p'])} | {fmt_p(m['p_fdr'])} |")
    L.append("")

    # --- Bootstrap ---
    L.append("## 4. Bootstrap 95% CI on the mean difference")
    L.append(f"Resamples the {n_inst} instances; CI excluding 0 ≈ significant at 0.05.")
    L.append("")
    L.append("| run | mean_diff | 95% CI | excludes 0? |")
    L.append("|---|---:|---:|:--:|")
    for r in rows:
        b = r["boot"]
        excl = "yes" if (b["ci_lo"] > 0 or b["ci_hi"] < 0) else "no"
        L.append(f"| {disp(r)} | {b['mean']:+.4f} | "
                 f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {excl} |")
    L.append("")

    # --- at a glance ---
    L.append("## At a glance")
    L.append("FDR-corrected significance (`*` < 0.05). Wilcoxon and t-test are the "
             "trustworthy rate-difference tests; McNemar is now also fair (5 vs 5).")
    L.append("")
    L.append("| run | mean_diff | Wilcoxon | t-test | McNemar | boot CI excl 0 |")
    L.append("|---|---:|:--:|:--:|:--:|:--:|")
    for r in rows:
        b = r["boot"]
        excl = "yes" if (b["ci_lo"] > 0 or b["ci_hi"] < 0) else "no"
        L.append(f"| {disp(r)} | {r['ttest']['mean']:+.4f} | "
                 f"{star(r['wil']['p_fdr'])} | {star(r['ttest']['p_fdr'])} | "
                 f"{star(r['mc']['p_fdr'])} | {excl} |")
    L.append("")

    # --- caveats ---
    L.append("## Caveats")
    L.append("- **valBL is each run's OWN 5-attempt empty-skillbook baseline** (not the "
             "aggregated 60). The comparison is therefore **k-symmetric (5 vs 5)**: McNemar's "
             "`BL_only`/`SB_only` are directly comparable, and both conditions share the same "
             "coarseness (multiples of 1/5). This is the cleaner test of "
             "'skillbook vs matched no-skillbook control under identical sampling'.")
    L.append("- **Trade-off vs the aggregated design**: the baseline is noisier here (5 vs 60 "
             "attempts ⇒ wider CIs), but the pairing is fair. The aggregated design had a "
             "tighter baseline estimate at the cost of k-asymmetry.")
    L.append("- **valBL now varies per run** (each run's own 5 attempts) — it is no longer a "
             "shared control across rows. This is by design (within-run matched control).")
    L.append("- **Direction matters**: a *negative* mean_diff / negative effect means the "
             "skillbook **hurt** that run.")
    L.append(f"- FDR is corrected *within* each test family ({n_runs} runs); cross-family "
             "comparisons are not additionally corrected.")
    L.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
