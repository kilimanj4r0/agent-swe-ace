#!/usr/bin/env python3
"""Q1 stat tests: does val_skillbook (valSB) beat the aggregated val_baseline (valBL),
per run, paired across the 113 shared instances.

Runs four paired tests per run on per-instance resolution rates read from the CSVs
produced by make_val_per_run_tables.py:
  1. Paired Wilcoxon signed-rank          (non-parametric, on rate diffs)
  2. Paired t-test                         (parametric, on rate diffs)
  3. McNemar                               (dichotomized: resolved_any = rate>0)
  4. Bootstrap 95% CI on the mean diff     (instance-level resampling)

Two-sided throughout. Benjamini-Hochberg FDR applied per test family (12 runs),
via statsmodels. Wilcoxon/t-test use pingouin (built-in effect sizes + CI),
McNemar via statsmodels, bootstrap via scipy.stats.bootstrap.

Usage:
    uv run python scripts/q1_stat_tests.py \
        --csv-dir data/val_per_run_tables_split025_vpk5 \
        --output data/val_per_run_tables_split025_vpk5/Q1_stat_tests_report.md
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


# ---------- helpers ----------

def short_label(run_dir: str) -> str:
    """run_<ts>_completed_qwen3_<CONFIG>_verified_vpk5 -> <CONFIG>."""
    name = os.path.basename(run_dir.rstrip("/")).replace(".csv", "")
    pre, suf = "_completed_qwen3_", "_verified_vpk5"
    i, j = name.find(pre), name.rfind(suf)
    return name[i + len(pre):j] if i != -1 and j != -1 else name


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values via statsmodels; NaNs passed through."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    mask = ~np.isnan(p)
    if mask.any():
        out[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return out


def star(p, alpha=0.05):
    return "*" if (p is not None and not np.isnan(p) and p < alpha) else ""


# ---------- the four tests ----------

def wilcoxon(x, y):
    """Paired Wilcoxon via pingouin; RBC is the rank-biserial effect size [-1, 1]."""
    d = np.asarray(x) - np.asarray(y)
    n = int(np.sum(d != 0))
    if n < 1:
        return {"n_pairs": 0, "W": np.nan, "p": np.nan, "r": np.nan}
    try:
        res = pg.wilcoxon(np.asarray(x), np.asarray(y), alternative="two-sided")
    except (ValueError, AssertionError):
        return {"n_pairs": n, "W": np.nan, "p": np.nan, "r": np.nan}
    return {"n_pairs": n, "W": float(res["W_val"].iloc[0]),
            "p": float(res["p_val"].iloc[0]), "r": float(res["RBC"].iloc[0])}


def ttest_paired(x, y):
    """Paired t-test: pingouin for T/p; CI and Cohen's dz computed directly because
    pingouin rounds CI95 to 2dp and reports cohen_d as a sign-stripped magnitude."""
    d = np.asarray(x) - np.asarray(y)
    n = len(d)
    if n < 2:
        return {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "t": np.nan, "p": np.nan, "d": np.nan, "n": n}
    res = pg.ttest(np.asarray(x), np.asarray(y), paired=True, alternative="two-sided")
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    tcrit = float(stats.t.ppf(0.975, n - 1))
    se = sd / np.sqrt(n)
    return {"mean": mean, "ci_lo": mean - tcrit * se, "ci_hi": mean + tcrit * se,
            "t": float(res["T"].iloc[0]), "p": float(res["p_val"].iloc[0]),
            "d": (mean / sd) if sd > 0 else np.nan, "n": n}


def mcnemar_test(x, y):
    """resolved_any = rate>0 under each condition; exact discordant-pairs test."""
    bl = np.asarray(x) > 0
    sb = np.asarray(y) > 0
    both = int(np.sum(bl & sb))
    bl_only = int(np.sum(bl & ~sb))      # baseline solved, skillbook didn't
    sb_only = int(np.sum(~bl & sb))      # skillbook solved, baseline didn't
    neither = int(np.sum(~bl & ~sb))
    disc = bl_only + sb_only
    if disc == 0:
        return {"both": both, "bl_only": bl_only, "sb_only": sb_only,
                "neither": neither, "p": np.nan, "disc": 0}
    table = np.array([[both, bl_only], [sb_only, neither]])
    p = float(mcnemar(table, exact=True).pvalue)
    return {"both": both, "bl_only": bl_only, "sb_only": sb_only,
            "neither": neither, "p": p, "disc": disc}


def boot_ci(x, y, seed, B=10000):
    """95% bootstrap CI on the mean difference via scipy.stats.bootstrap."""
    d = np.asarray(x) - np.asarray(y)
    rng = np.random.default_rng(seed)
    res = stats.bootstrap((d,), np.mean, n_resamples=B, confidence_level=0.95,
                          method="percentile", random_state=rng)
    return {"mean": float(d.mean()),
            "ci_lo": float(res.confidence_interval.low),
            "ci_hi": float(res.confidence_interval.high)}


# ---------- driver ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", required=True, help="dir of <run>.csv tables")
    ap.add_argument("--output", required=True, help="output markdown path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot-iters", type=int, default=10000)
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if not csvs:
        print(f"no CSVs in {args.csv_dir}", file=sys.stderr)
        return 1

    rows = []
    for i, fp in enumerate(csvs):
        df = pd.read_csv(fp)
        df = df.dropna(subset=["valBL", "valSB"]).sort_values("instance_id")
        bl = df["valBL"].to_numpy(float)
        sb = df["valSB"].to_numpy(float)
        w = wilcoxon(sb, bl)
        t = ttest_paired(sb, bl)
        m = mcnemar_test(bl, sb)
        b = boot_ci(sb, bl, args.seed + i, args.boot_iters)
        rows.append({"label": short_label(fp), "csv": os.path.basename(fp),
                     "n_inst": len(df), "wil": w, "ttest": t, "mc": m, "boot": b})

    # FDR per family
    for key in ("wil", "ttest", "mc"):
        ps = [r[key]["p"] for r in rows]
        corr = bh_fdr(ps)
        for r, c in zip(rows, corr):
            r[key]["p_fdr"] = c

    write_report(rows, args)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def write_report(rows, args):
    L = []
    L.append("# Q1 — Does the skillbook beat the empty baseline? (per run)")
    L.append("")
    L.append(f"Paired over the shared 113 val instances. "
             f"diff = valSB − valBL (positive ⇒ skillbook helps). "
             f"Two-sided tests; Benjamini–Hochberg FDR applied per test family "
             f"(12 runs). Bootstrap: {args.boot_iters} resamples, instance-level, "
             f"seed={args.seed}. α = 0.05.")
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
        L.append(f"| {r['label']} | {w['n_pairs']} | {w['W']:.1f} | "
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
        L.append(f"| {r['label']} | {t['mean']:+.4f} | "
                 f"[{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}] | {t['t']:+.2f} | "
                 f"{fmt_p(t['p'])} | {fmt_p(t['p_fdr'])} | {t['d']:+.3f} |")
    L.append("")

    # --- McNemar ---
    L.append("## 3. McNemar (resolved-at-least-once: rate > 0)")
    L.append("Pairs each instance as solved-yes/no under each condition. "
             "`BL_only` = baseline solved it but skillbook didn't; "
             "`SB_only` = the reverse. Exact two-sided sign test on discordants.")
    L.append("")
    L.append("| run | both | BL_only | SB_only | neither | p_raw | p_FDR |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        m = r["mc"]
        L.append(f"| {r['label']} | {m['both']} | {m['bl_only']} | {m['sb_only']} | "
                 f"{m['neither']} | {fmt_p(m['p'])} | {fmt_p(m['p_fdr'])} |")
    L.append("")

    # --- Bootstrap ---
    L.append("## 4. Bootstrap 95% CI on the mean difference")
    L.append("Resamples the 113 instances; CI excluding 0 ≈ significant at 0.05.")
    L.append("")
    L.append("| run | mean_diff | 95% CI | excludes 0? |")
    L.append("|---|---:|---:|:--:|")
    for r in rows:
        b = r["boot"]
        excl = "yes" if (b["ci_lo"] > 0 or b["ci_hi"] < 0) else "no"
        L.append(f"| {r['label']} | {b['mean']:+.4f} | "
                 f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {excl} |")
    L.append("")

    # --- at a glance ---
    L.append("## At a glance")
    L.append("FDR-corrected significance (`*` < 0.05). "
             "Wilcoxon and t-test are the trustworthy ones here.")
    L.append("")
    L.append("| run | mean_diff | Wilcoxon | t-test | McNemar | boot CI excl 0 |")
    L.append("|---|---:|:--:|:--:|:--:|:--:|")
    for r in rows:
        b = r["boot"]
        excl = "yes" if (b["ci_lo"] > 0 or b["ci_hi"] < 0) else "no"
        L.append(f"| {r['label']} | {r['ttest']['mean']:+.4f} | "
                 f"{star(r['wil']['p_fdr'])} | {star(r['ttest']['p_fdr'])} | "
                 f"{star(r['mc']['p_fdr'])} | {excl} |")
    L.append("")

    L.append("## Caveats")
    L.append("- **valBL is the same 60-attempt baseline in every run** — it's the "
             "shared control, so each run gets an identical reference.")
    L.append("- **valSB has only 5 attempts** → per-instance rates are coarse "
             "(multiples of 0.2), producing many ties/zeros in the Wilcoxon (handled, "
             "n_pairs reported) and a coarse signal for the t-test.")
    L.append("- **McNemar is k-asymmetric**: baseline's `resolved_any` draws on 60 "
             "attempts vs the skillbook's 5, so `BL_only` is structurally inflated. "
             "Read it as 'which instances flip solvability', not as a fair "
             "skill-vs-baseline verdict — prefer Wilcoxon/t-test/bootstrap for that.")
    L.append("- Direction matters: a *negative* mean_diff / negative effect means the "
             "skillbook **hurt** that run.")
    L.append("- FDR is corrected *within* each test family (12 runs); cross-family "
             "comparisons are not additionally corrected.")
    L.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
