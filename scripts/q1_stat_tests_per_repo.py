#!/usr/bin/env python3
"""Q1 stat tests PER REPO: does val_skillbook (valSB) beat the aggregated val_baseline
(valBL), per repo, within each repos-split025 run.

Reads the per-run CSVs from make_val_per_run_tables.py (columns: instance_id, valBL,
valSB), joins each instance to its repo via the aggregated baseline's
stats/per_instance.json, then runs paired tests per (run x repo) cell:
  1. Paired t-test (primary)   -- mean_diff, 95% CI, t, p, Cohen's dz
  2. Paired Wilcoxon           -- rank-biserial r
  3. Bootstrap 95% CI on mean_diff
Benjamini-Hochberg FDR is applied over the full (run x repo) family, per test.

Reuses the test helpers from q1_stat_tests.py (so numbers are consistent with the
per-run Q1 report).

Usage:
    uv run python scripts/q1_stat_tests_per_repo.py \
        --csv-dir data/val_per_run_tables_split025_vpk5 \
        --repo-json data/val_baseline_aggregated_split025_vpk5_qwen3/stats/per_instance.json \
        --filter repos \
        --output data/val_per_run_tables_split025_vpk5/Q1_per_repo_stat_tests_report.md \
        --valbl-attempts 60 --sb-attempts 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from q1_stat_tests import (
    bh_fdr,
    boot_ci,
    fmt_p,
    short_label,
    ttest_paired,
    wilcoxon,
)


def load_repo_map(path: str) -> dict[str, str]:
    """{instance_id: repo} from the aggregated baseline's per_instance.json."""
    with open(path) as f:
        data = json.load(f)
    return {inst: rec.get("repo") for inst, rec in data.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", required=True, help="dir of <run>.csv per-run tables")
    ap.add_argument("--repo-json", required=True,
                    help="aggregated baseline stats/per_instance.json (for repo mapping)")
    ap.add_argument("--filter", default="repos",
                    help="substring to select CSVs (default 'repos' = per-repo runs)")
    ap.add_argument("--output", required=True, help="output markdown path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot-iters", type=int, default=10000)
    ap.add_argument("--valbl-attempts", type=int, default=60, help="prose")
    ap.add_argument("--sb-attempts", type=int, default=5, help="prose")
    args = ap.parse_args()

    repo_map = load_repo_map(args.repo_json)

    csvs = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if args.filter:
        csvs = [c for c in csvs if args.filter in os.path.basename(c)]
    if not csvs:
        print(f"no CSVs matching filter '{args.filter}' in {args.csv_dir}", file=sys.stderr)
        return 1

    rows = []
    pooled = {}          # run -> pooled-over-repos mean_diff (for ordering)
    repo_n = {}          # repo -> n instances (for ordering; same set across runs)
    seed_i = 0
    for fp in csvs:
        df = pd.read_csv(fp).dropna(subset=["valBL", "valSB"])
        df["repo"] = df["instance_id"].map(repo_map)
        label = short_label(fp)
        pbl = df["valBL"].to_numpy(float)
        psb = df["valSB"].to_numpy(float)
        pooled[label] = float(np.mean(psb - pbl))
        for repo, g in df.groupby("repo"):
            g = g.sort_values("instance_id")
            bl = g["valBL"].to_numpy(float)
            sb = g["valSB"].to_numpy(float)
            n = len(g)
            repo_n[repo] = max(repo_n.get(repo, 0), n)
            t = ttest_paired(sb, bl)
            w = wilcoxon(sb, bl)
            b = boot_ci(sb, bl, args.seed + seed_i, args.boot_iters)
            seed_i += 1
            rows.append({"label": label, "repo": repo, "n": n,
                         "ttest": t, "wil": w, "boot": b})

    # FDR over the full (run x repo) family, per test
    for key in ("ttest", "wil"):
        ps = [r[key]["p"] for r in rows]
        corr = bh_fdr(ps)
        for r, c in zip(rows, corr):
            r[key]["p_fdr"] = c

    run_order = [l for l, _ in sorted(pooled.items(), key=lambda x: -x[1])]
    repo_order = [r for r, _ in sorted(repo_n.items(), key=lambda x: -x[1])]
    write_report(rows, run_order, repo_order, pooled, args)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def write_report(rows, run_order, repo_order, pooled, args):
    L = []
    n_runs = len(run_order)
    n_repos = len(repo_order)
    n_cells = len(rows)

    L.append("# Q1 (per-repo) — Does the skillbook beat the empty baseline, per repo?")
    L.append("")
    L.append(f"Paired over each repo's val instances (n shown), within each of the {n_runs} "
             f"qwen3 repos-split025 runs (vpk5). diff = valSB − valBL (positive ⇒ skillbook "
             f"helps). valBL = shared {args.valbl_attempts}-attempt aggregated baseline; "
             f"valSB = that run's {args.sb_attempts}-attempt val skillbook. Two-sided tests; "
             f"Benjamini–Hochberg FDR over the {n_cells}-cell (run × repo) family, per test. "
             f"α = 0.05.")
    L.append("")
    L.append("## Column definitions")
    L.append("- **run** — skillbook config: `swe`/`default` (Reflector/SkillManager variant) × "
             "`retk5`/`retk20`/none (retrieval top-k).")
    L.append("- **repo** — SWE-bench repo.")
    L.append("- **n** — # val instances in that repo (drives power; same set across runs).")
    L.append("- **mean_diff** — mean per-instance (valSB − valBL) resolution-rate difference; "
             "+ ⇒ skillbook helps.")
    L.append("- **95% CI** — paired CI on mean_diff; **excludes 0 ⇒ significant at 0.05**.")
    L.append("- **t, p_raw** — paired t-test statistic and raw two-sided p-value.")
    L.append("- **p_FDR** — Benjamini–Hochberg-adjusted p over all run×repo cells (`*` < 0.05).")
    L.append("- **Cohen's dz** — paired effect size = mean_diff / sd_diff "
             "(~0.2 small, 0.5 medium, 0.8 large).")
    L.append("- **Wil p_FDR** — paired Wilcoxon signed-rank, BH-adjusted (non-parametric cross-check).")
    L.append("- **boot0** — does the bootstrap 95% CI on mean_diff exclude 0?")
    L.append("")

    def is_sig(p):
        return p is not None and not (isinstance(p, float) and np.isnan(p)) and p < 0.05

    for run in run_order:
        rrows = [r for r in rows if r["label"] == run]
        rrows.sort(key=lambda r: repo_order.index(r["repo"]))
        L.append(f"### {run}  (pooled mean_diff = {pooled[run]:+.4f})")
        L.append("")
        L.append("| repo | n | mean_diff | 95% CI | t | p_raw | p_FDR | dz | Wil p_FDR | boot0 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
        for r in rrows:
            t, w, b = r["ttest"], r["wil"], r["boot"]
            ci_s = f"[{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}]"
            excl = "yes" if (b["ci_lo"] > 0 or b["ci_hi"] < 0) else "no"
            star = "*" if is_sig(t["p_fdr"]) else ""
            L.append(f"| {r['repo']} | {r['n']} | {t['mean']:+.4f} | {ci_s} | "
                     f"{t['t']:+.2f} | {fmt_p(t['p'])} | {fmt_p(t['p_fdr'])}{star} | "
                     f"{t['d']:+.3f} | {fmt_p(w['p_fdr'])} | {excl} |")
        L.append("")

    # --- at a glance: FDR-significant cells ---
    L.append("## At a glance — significant cells (t-test p_FDR < 0.05)")
    sig = [r for r in rows if is_sig(r["ttest"]["p_fdr"])]
    if sig:
        L.append("| run | repo | n | mean_diff | 95% CI | t p_FDR | dz | Wil p_FDR |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for r in sig:
            t = r["ttest"]
            ci_s = f"[{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}]"
            L.append(f"| {r['label']} | {r['repo']} | {r['n']} | {t['mean']:+.4f} | "
                     f"{ci_s} | {fmt_p(t['p_fdr'])} | {t['d']:+.3f} | {fmt_p(r['wil']['p_fdr'])} |")
    else:
        L.append("_(no run×repo cell reaches t-test p_FDR < 0.05)_")
        near = [r for r in rows if is_sig(r["ttest"]["p"])]
        if near:
            L.append("")
            L.append("Near-misses — cells with **raw** p < 0.05 (do not survive the 48-cell FDR):")
            L.append("| run | repo | n | mean_diff | t | p_raw | p_FDR | dz |")
            L.append("|---|---|---:|---:|---:|---:|---:|---:|")
            for r in sorted(near, key=lambda r: r["ttest"]["p"]):
                t = r["ttest"]
                L.append(f"| {r['label']} | {r['repo']} | {r['n']} | {t['mean']:+.4f} | "
                         f"{t['t']:+.2f} | {fmt_p(t['p'])} | {fmt_p(t['p_fdr'])} | {t['d']:+.3f} |")
    L.append("")

    # --- caveats ---
    L.append("## Caveats")
    L.append(f"- **{n_cells} tests** ({n_runs} runs × {n_repos} repos): "
             f"~{n_cells * 0.05:.1f} false positives expected at α=0.05 before FDR — "
             "BH over all cells is the honest correction.")
    L.append("- **Small repos have ~no power**: astropy/pytest (n=4), xarray (5), "
             "sklearn (6), matplotlib (8) can swing hard on a single instance — treat their "
             "p-values as exploratory. Only django (57) and sympy (18) carry real power.")
    L.append(f"- **valBL is the shared {args.valbl_attempts}-attempt control** (identical in "
             "every run); only valSB varies, so significance reflects the skillbook, not "
             "baseline noise.")
    L.append("- **Direction matters**: a negative mean_diff / CI below 0 means the skillbook "
             "**hurt** that repo.")
    L.append(f"- valSB uses only {args.sb_attempts} attempts → coarse per-instance rates "
             f"(multiples of 1/{args.sb_attempts}).")
    L.append("")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
