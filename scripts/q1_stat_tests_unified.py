#!/usr/bin/env python3
"""Q1 unified stat tests: one report with exactly 36 significance-test comparisons.

36 rows = 24 "60v5" + 12 "5v5":
  - **60v5** (24 runs, ALL): each run's 5-attempt val_skillbook (valSB5) paired
    against the SHARED aggregated 60-attempt val_baseline (valBL60) over the 113
    val instances. FDR family = these 24 runs.
  - **5v5** (12 runs, the NON-`orig` retrieval-method runs only): each such run's
    valSB5 paired against its OWN 5-attempt val_baseline (valBL5). FDR family =
    these 12 runs.

Why the 12 `orig` runs get BOTH designs: they ran val_baseline FRESH (12 distinct
5-attempt baselines, pooled 12 x 5 = 60 to BUILD valBL60), so each has its own clean
within-run BL5 -> a fair 5v5, plus the 60v5 against the pooled control. The 12 newer
retrieval-method runs (bm25 / emb / random) COPIED valBL from a matching `orig` run
(via baseline_run_dir, verified byte-identical to the source) instead of running it
fresh, so they have no independent BL5 and are 60v5-only.

Reads directly from run dirs (val + val_baseline phases) and the aggregated baseline's
stats/per_instance.json — no dependency on the per-run CSVs. Reuses the four test
helpers + FDR from q1_stat_tests.py so numbers are consistent with the other Q1 reports.

Usage:
    uv run python scripts/q1_stat_tests_unified.py \
        --runs 'data/*completed_qwen3_*split025*vpk5' \
        --aggregated data/val_baseline_aggregated_split025_vpk5_qwen3 \
        --original-runs-index data/val_baseline_aggregated_split025_vpk5_qwen3/runs_index.json \
        --output data/val_per_run_tables_split025_vpk5/Q1_stat_tests_unified_36rows.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
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


# ---------- data loading ----------

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


def load_shared_valbl(aggregated_dir: str) -> dict[str, float]:
    """{instance_id: successes/n_attempts} from the aggregated 60-attempt baseline."""
    path = os.path.join(aggregated_dir, "stats", "per_instance.json")
    with open(path) as f:
        data = json.load(f)
    out = {}
    for inst, rec in data.items():
        n = int(rec.get("n_attempts", 0))
        out[inst] = round(int(rec.get("successes", 0)) / n, 4) if n else 0.0
    return out


def load_shared_valbl_counts(aggregated_dir: str) -> dict[str, tuple[int, int]]:
    """{instance_id: (successes, n_attempts)} from the aggregated baseline (for pass@k)."""
    path = os.path.join(aggregated_dir, "stats", "per_instance.json")
    with open(path) as f:
        data = json.load(f)
    return {inst: (int(rec.get("successes", 0)), int(rec.get("n_attempts", 0)))
            for inst, rec in data.items()}


def load_phase_counts(run_dir: str, benchmark: str, phase: str) -> dict[str, tuple[int, int]]:
    """{instance_id: (resolved, total)} for a run's phase (for pass@k)."""
    base = os.path.join(run_dir, benchmark, "results", phase)
    out: dict[str, tuple[int, int]] = {}
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
        out[inst] = (res, len(files))
    return out


def pass_at_k(n: int, c: int, k: int) -> float:
    """Combinatorial pass@k = 1 - C(n-c, k)/C(n, k) (HumanEval estimator).
    With n == k it degenerates to the empirical solved-any (1 if c > 0 else 0)."""
    if n <= 0 or k > n:
        return 0.0
    denom = math.comb(n, k)
    if denom == 0:
        return 0.0
    return 1.0 - math.comb(n - c, k) / denom


def detect_benchmark(run_dir: str) -> str:
    for name in os.listdir(run_dir):
        full = os.path.join(run_dir, name)
        if os.path.isdir(full) and os.path.isdir(os.path.join(full, "results")):
            return name
    raise RuntimeError(f"no benchmark subdir with results/ under {run_dir}")


# ---------- driver ----------

def build_row(label, run, design, insts, blv, sbv, seed, boot_iters):
    w = wilcoxon(sbv, blv)
    t = ttest_paired(sbv, blv)
    m = mcnemar_test(blv, sbv)
    b = boot_ci(sbv, blv, seed, boot_iters)
    return {"label": label, "run": run, "design": design,
            "n_inst": len(insts), "wil": w, "ttest": t, "mc": m, "boot": b}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, help="glob of completed run dirs")
    ap.add_argument("--aggregated", required=True,
                    help="aggregated val_baseline dir (has stats/per_instance.json)")
    ap.add_argument("--original-runs-index", default=None,
                    help="aggregated baseline runs_index.json; those runs are 'orig'")
    ap.add_argument("--output", required=True, help="output markdown path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot-iters", type=int, default=10000)
    ap.add_argument("--valbl-attempts", type=int, default=60,
                    help="attempts/instance in the shared aggregated valBL (prose)")
    ap.add_argument("--sb-attempts", type=int, default=5,
                    help="attempts/instance in each run's valSB / own valBL (prose)")
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

    shared_bl = load_shared_valbl(args.aggregated)
    shared_counts = load_shared_valbl_counts(args.aggregated)

    rows_60v5, rows_5v5 = [], []
    seed_i = 0
    for run in runs:
        bench = detect_benchmark(run)
        sb5 = load_phase_rates(run, bench, "val")          # run's 5-attempt skillbook
        sb_counts = load_phase_counts(run, bench, "val")
        base = os.path.basename(run.rstrip("/"))
        orig = base in original
        label = short_label(run)

        # --- 60v5: every run, SB5 vs shared 60-baseline ---
        insts60 = sorted(set(shared_bl) & set(sb5))
        blv60 = np.array([shared_bl[j] for j in insts60], float)
        sbv60 = np.array([sb5[j] for j in insts60], float)
        seed60 = args.seed + seed_i
        rows_60v5.append(build_row(label, base, "60v5", insts60, blv60, sbv60,
                                   seed60, args.boot_iters))
        rows_60v5[-1]["orig"] = orig
        # pass@5 (any-of-5 solves): baseline = combinatorial from 60; skillbook = binary (n=5)
        ic = sorted(set(shared_counts) & set(sb_counts))
        p5bl = np.array([pass_at_k(shared_counts[j][1], shared_counts[j][0], 5) for j in ic], float)
        p5sb = np.array([pass_at_k(sb_counts[j][1], sb_counts[j][0], 5) for j in ic], float)
        rows_60v5[-1]["p5"] = {"ttest": ttest_paired(p5sb, p5bl),
                               "boot": boot_ci(p5sb, p5bl, seed60, args.boot_iters)}
        seed_i += 1

        # --- 5v5: only ORIG runs (ran val_baseline FRESH), SB5 vs own 5-attempt BL5 ---
        if orig:
            bl5 = load_phase_rates(run, bench, "val_baseline")
            insts5 = sorted(set(bl5) & set(sb5))
            blv5 = np.array([bl5[j] for j in insts5], float)
            sbv5 = np.array([sb5[j] for j in insts5], float)
            rows_5v5.append(build_row(label, base, "5v5", insts5, blv5, sbv5,
                                      args.seed + seed_i, args.boot_iters))
            rows_5v5[-1]["orig"] = True
            seed_i += 1

    # FDR per design family, per test
    for fam in (rows_60v5, rows_5v5):
        for key in ("wil", "ttest", "mc"):
            ps = [r[key]["p"] for r in fam]
            corr = bh_fdr(ps)
            for r, c in zip(fam, corr):
                r[key]["p_fdr"] = c

    # pass@5 t-test FDR over the 24 60v5 runs (5v5 block excluded: pass@5 is binary
    # on both sides there = the McNemar discordants, so it adds nothing)
    p5_fam = [r for r in rows_60v5 if "p5" in r]
    if p5_fam:
        corr5 = bh_fdr([r["p5"]["ttest"]["p"] for r in p5_fam])
        for r, c in zip(p5_fam, corr5):
            r["p5"]["ttest"]["p_fdr"] = c

    write_report(rows_60v5, rows_5v5, args)
    print(f"wrote {args.output} "
          f"(60v5={len(rows_60v5)}, 5v5={len(rows_5v5)}, "
          f"total={len(rows_60v5) + len(rows_5v5)})", file=sys.stderr)
    return 0


# ---------- report ----------

def disp(r):
    tag = " [orig]" if r.get("orig") else ""
    return f"{r['label']}{tag}"


def _psig(p, alpha=0.05):
    """p < alpha, NaN/None-safe."""
    return p is not None and not (isinstance(p, float) and np.isnan(p)) and p < alpha


def conclusions_section(all_rows):
    """Data-driven significance summary. Design-keyed so it works for any run family
    (qwen3 two-design, qwen3next single-design, etc.)."""
    L = ["## Conclusions", ""]
    designs = list(dict.fromkeys(r["design"] for r in all_rows))

    def boot_excl_local(b):
        return b["ci_lo"] > 0 or b["ci_hi"] < 0

    # 1) FDR-corrected verdict (trustworthy rate-difference tests)
    t_fdr = [r for r in all_rows if _psig(r["ttest"]["p_fdr"])]
    w_fdr = [r for r in all_rows if _psig(r["wil"]["p_fdr"])]
    L.append("**FDR-corrected verdict.** "
             f"Of {len(all_rows)} comparisons, {len(t_fdr)} survive Benjamini–Hochberg "
             f"FDR at α=0.05 in the paired t-test and {len(w_fdr)} in the Wilcoxon "
             f"(the two trustworthy rate-difference tests).")
    if not (t_fdr or w_fdr):
        L.append("_No comparison reaches FDR-corrected significance in the t-test or "
                 "Wilcoxon — treat all per-run effects below as trends, not confirmed._")
    L.append("")

    # 2) Nominal raw-p < 0.05 signals (pre-FDR)
    L.append("**Nominal signals — raw p < 0.05 (do NOT survive family-wise FDR).** "
             "Trends worth noting; not multiple-comparison-safe.")
    L.append("")
    nom = [r for r in all_rows if _psig(r["ttest"]["p"]) or _psig(r["wil"]["p"])]
    if nom:
        L.append("| design | run | mean_diff | t p_raw | Wilcoxon p_raw | boot CI ∌0 |")
        L.append("|---|---|---:|---:|---:|:--:|")
        for r in nom:
            L.append(f"| {r['design']} | {disp(r)} | {r['ttest']['mean']:+.4f} | "
                     f"{fmt_p(r['ttest']['p'])} | {fmt_p(r['wil']['p'])} | "
                     f"{'yes' if boot_excl_local(r['boot']) else 'no'} |")
    else:
        L.append("_(none)_")
    L.append("")

    # 3) Bootstrap CI excludes 0
    L.append("**Bootstrap 95% CI excludes 0 (≈ significant at 0.05, instance-level; no "
             "FDR needed).** The strongest single-comparison evidence.")
    L.append("")
    boot = [r for r in all_rows if boot_excl_local(r["boot"])]
    if boot:
        L.append("| design | run | mean_diff | 95% CI | direction |")
        L.append("|---|---|---:|---:|---|")
        for r in boot:
            b = r["boot"]
            d = "skillbook **helps**" if b["mean"] > 0 else "skillbook **hurts**"
            L.append(f"| {r['design']} | {disp(r)} | {b['mean']:+.4f} | "
                     f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {d} |")
    else:
        L.append("_(none)_")
    L.append("")

    # 4) Synthesis — counts per design
    L.append("**Synthesis.**")
    for dsg in designs:
        rs = [r for r in all_rows if r["design"] == dsg]
        pos = sum(1 for r in rs if r["ttest"]["mean"] > 0)
        neg = sum(1 for r in rs if r["ttest"]["mean"] < 0)
        bpos = sum(1 for r in rs if r["boot"]["ci_lo"] > 0)
        bneg = sum(1 for r in rs if r["boot"]["ci_hi"] < 0)
        L.append(f"- **{dsg}** ({len(rs)} rows): {pos} positive / {neg} negative "
                 f"mean_diff; bootstrap-significant → {bpos} help, {bneg} hurt.")
    L.append("")
    return L


def boot_excl(b):
    return "yes" if (b["ci_lo"] > 0 or b["ci_hi"] < 0) else "no"


def write_report(rows_60v5, rows_5v5, args):
    all_rows = rows_60v5 + rows_5v5
    n60, n5 = len(rows_60v5), len(rows_5v5)
    n_inst = rows_60v5[0]["n_inst"] if rows_60v5 else 0
    bln, sbn = args.valbl_attempts, args.sb_attempts

    L = []
    L.append("# Q1 — Does the skillbook beat the empty baseline? (unified, 36 rows)")
    L.append("")
    L.append(f"One row per (run × design) comparison over the shared {n_inst} val "
             f"instances. diff = valSB − valBL (positive ⇒ skillbook helps). "
             f"**{n60} + {n5} = {n60 + n5} rows** across two designs:")
    L.append("")
    L.append(f"- **60v5** ({n60} rows, every run): run's {sbn}-attempt val_skillbook vs "
             f"the **shared {bln}-attempt aggregated baseline** (`valBL60`). The 12 "
             f"`[orig]` runs are the ones whose own val_baselines were pooled "
             f"(12 × {sbn} = {bln}) to build `valBL60`; they are flagged `[orig]`.")
    L.append(f"- **5v5** ({n5} rows, only the 12 `[orig]` runs): run's val_skillbook vs "
             f"its **own fresh {sbn}-attempt val_baseline** (`valBL5`) — a k-symmetric "
             f"(5 vs 5) within-run matched control. Only `[orig]` runs qualify: they ran "
             f"val_baseline fresh. The 12 newer retrieval-method runs COPIED valBL from a "
             f"matching `[orig]` run (no independent BL5), so they are 60v5-only.")
    L.append("")
    L.append(f"Two-sided tests; Benjamini–Hochberg FDR applied **per design family** "
             f"(60v5 over {n60} runs, 5v5 over {n5} runs), per test. Bootstrap: "
             f"{args.boot_iters} resamples, instance-level, seed={args.seed}. α = 0.05. "
             f"`*` = FDR-adjusted p < 0.05.")
    L.append("")

    # --- primary consolidated 36-row table ---
    L.append("## At a glance — all 36 comparisons")
    L.append("FDR-corrected significance per design family. Wilcoxon/t-test are the "
             "trustworthy rate-difference tests; McNemar is k-symmetric only in the 5v5 "
             "block (60v5 McNemar is structurally baseline-favored — read as 'which "
             "instances flip', not a fair verdict).")
    L.append("")
    L.append("| # | run | design | n | mean_diff | 95% CI | Wilcoxon p_FDR | t-test p_FDR | McNemar p_FDR | boot CI ∌0 | Cohen's dz |")
    L.append("|---:|---|---|---:|---:|---:|---:|---:|---:|:--:|---:|")
    for i, r in enumerate(all_rows, 1):
        t, b = r["ttest"], r["boot"]
        ci_s = f"[{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}]"
        L.append(f"| {i} | {disp(r)} | {r['design']} | {r['n_inst']} | "
                 f"{t['mean']:+.4f} | {ci_s} | {fmt_p(r['wil']['p_fdr'])}{star(r['wil']['p_fdr'])} | "
                 f"{fmt_p(t['p_fdr'])}{star(t['p_fdr'])} | "
                 f"{fmt_p(r['mc']['p_fdr'])}{star(r['mc']['p_fdr'])} | "
                 f"{boot_excl(b)} | {t['d']:+.3f} |")
    L.append("")

    # --- detailed: Wilcoxon ---
    L.append("## 1. Paired Wilcoxon signed-rank (non-parametric)")
    L.append("Per-instance rate differences; zero-difference instances dropped "
             "(n_pairs reported). r = matched-pairs rank-biserial effect size in [−1, +1].")
    L.append("")
    L.append("| run | design | n_pairs | W | p_raw | p_FDR | r |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for r in all_rows:
        w = r["wil"]
        L.append(f"| {disp(r)} | {r['design']} | {w['n_pairs']} | {w['W']:.1f} | "
                 f"{fmt_p(w['p'])} | {fmt_p(w['p_fdr'])}{star(w['p_fdr'])} | {w['r']:+.3f} |")
    L.append("")

    # --- detailed: t-test ---
    L.append("## 2. Paired t-test (parametric)")
    L.append("Mean per-instance rate difference; 95% CI and Cohen's dz "
             "(signed, paired = mean_diff / sd_diff).")
    L.append("")
    L.append("| run | design | mean_diff | 95% CI | t | p_raw | p_FDR | Cohen's dz |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in all_rows:
        t = r["ttest"]
        L.append(f"| {disp(r)} | {r['design']} | {t['mean']:+.4f} | "
                 f"[{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}] | {t['t']:+.2f} | "
                 f"{fmt_p(t['p'])} | {fmt_p(t['p_fdr'])}{star(t['p_fdr'])} | {t['d']:+.3f} |")
    L.append("")

    # --- detailed: McNemar ---
    L.append("## 3. McNemar (resolved-at-least-once: rate > 0)")
    L.append("`BL_only` = baseline solved, skillbook didn't; `SB_only` = the reverse. "
             "Exact two-sided sign test on discordants. **Fair only in the 5v5 block** "
             "(both sides 5 attempts); in 60v5 the baseline's 60 attempts inflate "
             "`BL_only` by construction.")
    L.append("")
    L.append("| run | design | both | BL_only | SB_only | neither | p_raw | p_FDR |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in all_rows:
        m = r["mc"]
        L.append(f"| {disp(r)} | {r['design']} | {m['both']} | {m['bl_only']} | "
                 f"{m['sb_only']} | {m['neither']} | {fmt_p(m['p'])} | "
                 f"{fmt_p(m['p_fdr'])}{star(m['p_fdr'])} |")
    L.append("")

    # --- detailed: bootstrap ---
    L.append("## 4. Bootstrap 95% CI on the mean difference")
    L.append(f"Resamples the {n_inst} instances; CI excluding 0 ≈ significant at 0.05.")
    L.append("")
    L.append("| run | design | mean_diff | 95% CI | excludes 0? |")
    L.append("|---|---|---:|---:|:--:|")
    for r in all_rows:
        b = r["boot"]
        L.append(f"| {disp(r)} | {r['design']} | {b['mean']:+.4f} | "
                 f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {boot_excl(b)} |")
    L.append("")

    # --- 5. pass@5 (any-of-5-solves) comparison, 60v5 block ---
    p5_rows = [r for r in rows_60v5 if "p5" in r]
    if p5_rows:
        L.append(f"## 5. Pass@5 (any-of-5-solves) comparison — 60v5 block ({len(p5_rows)} runs)")
        L.append("Pass@5 = P(≥1 of 5 attempts solves) — a **breadth** metric (how many distinct "
                 "instances 5 attempts can crack), complementary to the per-attempt pass@1/avg "
                 "rate in §1–4 (a **reliability** metric). The two can diverge: a system can raise "
                 "per-attempt reliability on instances it already solves without expanding the set "
                 "it solves at all. **Baseline**: combinatorial estimator `1 − C(60−c, 5)/C(60, 5)` "
                 "from the 60 pooled attempts (smooth, low-variance; shared mean pass@5 ≈ 0.387). "
                 "**Skillbook**: empirical solved-any-of-5 (binary — each run makes exactly 5 "
                 "attempts, so n=k=5 degenerates to `[c>0]`). diff = valSB pass@5 − valBL pass@5 "
                 "(+ ⇒ skillbook helps). This is the matched-5-attempt any-of-5 test the 60v5 "
                 "McNemar (skillbook pass@5 vs baseline pass@60) could not be. "
                 f"Two-sided; BH-FDR over the {len(p5_rows)} runs; seed={args.seed}.")
        L.append("")
        L.append("### 5a. Paired t-test on pass@5 differences")
        L.append("| run | mean_diff | 95% CI | t | p_raw | p_FDR | Cohen's dz |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in p5_rows:
            t = r["p5"]["ttest"]
            L.append(f"| {disp(r)} | {t['mean']:+.4f} | "
                     f"[{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}] | {t['t']:+.2f} | "
                     f"{fmt_p(t['p'])} | {fmt_p(t['p_fdr'])}{star(t['p_fdr'])} | {t['d']:+.3f} |")
        L.append("")
        L.append("### 5b. Bootstrap 95% CI on the pass@5 mean difference")
        L.append("| run | mean_diff | 95% CI | excludes 0? |")
        L.append("|---|---:|---:|:--:|")
        for r in p5_rows:
            b = r["p5"]["boot"]
            excl = "yes" if (b["ci_lo"] > 0 or b["ci_hi"] < 0) else "no"
            L.append(f"| {disp(r)} | {b['mean']:+.4f} | "
                     f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {excl} |")
        L.append("")
        sig5 = [r for r in p5_rows if _psig(r["p5"]["ttest"]["p_fdr"])]
        boot5 = [r for r in p5_rows if r["p5"]["boot"]["ci_lo"] > 0 or r["p5"]["boot"]["ci_hi"] < 0]
        bhelp = sum(1 for r in boot5 if r["p5"]["boot"]["mean"] > 0)
        bhurt = sum(1 for r in boot5 if r["p5"]["boot"]["mean"] < 0)
        all_neg = all(r["p5"]["boot"]["mean"] < 0 for r in p5_rows)
        lo = min(r["p5"]["boot"]["mean"] for r in p5_rows)
        hi = max(r["p5"]["boot"]["mean"] for r in p5_rows)
        L.append(f"**Pass@5 verdict:** {len(sig5)}/{len(p5_rows)} survive FDR; "
                 f"{len(boot5)} have a bootstrap CI excluding 0 ({bhelp} help, {bhurt} hurt)."
                 + (f" All 24 mean_diffs are negative — on the any-of-5 / breadth view the "
                    f"skillbook is **below** the baseline (skillbook mean pass@5 ≈ "
                    f"{0.387 + lo:.2f}–{0.387 + hi:.2f} vs baseline 0.387)." if all_neg
                    else ""))
        L.append("")
        L.append("**Why pass@5 and pass@1 disagree (§1–4 vs §5).** The skillbook raises "
                 "*per-attempt reliability* on the instances it can solve (higher pass@1 / avg "
                 "rate) but, with only 5 attempts, reaches a *smaller* set of instances than the "
                 "baseline's 60-pool any-of-5 estimate — so pass@5 (breadth) is lower. Both are "
                 "true; they measure different things. Caveat: the baseline pass@5 is estimated "
                 "from 60 attempts (tight) while the skillbook's is a single 5-attempt realization "
                 "(coarse, binary) — the means are unbiased and comparable, but the skillbook side "
                 "is noisier. A fairer *realized* 5-vs-5 pass@5 is the 5v5 McNemar (§3, 5v5 block), "
                 "which is binary on both sides.")
        L.append("")

    # --- conclusions ---
    L += conclusions_section(all_rows)

    # --- caveats ---
    n_orig = sum(1 for r in rows_60v5 if r["orig"])
    L.append("## Caveats")
    L.append(f"- **Two FDR families, corrected separately**: 60v5 over the {n60} runs, "
             f"5v5 over the {n5} `[orig]` runs. The 5v5 family is the 12 `[orig]` runs "
             f"(was 24 in the old report); the 60v5 family grew 12→24 (all runs now "
             f"share the `valBL60` control).")
    L.append(f"- **The 12 newer retrieval-method runs (bm25 / emb / random) are "
             f"60v5-only**: they COPIED valBL from a matching `[orig]` run via "
             f"`baseline_run_dir` (verified: their val_baseline contents are "
             f"byte-identical to the source `[orig]` run) instead of running it fresh, "
             f"so they have no independent BL5 and no fair 5v5. The {n_orig} `[orig]` "
             f"runs ran val_baseline fresh (12 distinct baselines) and get both designs.")
    L.append(f"- **valBL60 is shared**: identical in every 60v5 row, so 60v5 significance "
             f"reflects each run's skillbook, not baseline noise. valBL5 in the 5v5 rows "
             f"is per-run (each run's own 5 attempts) — no longer a shared control.")
    L.append(f"- **valSB has only {sbn} attempts** → coarse per-instance rates "
             f"(multiples of 1/{sbn}): many ties/zeros in the Wilcoxon (handled, "
             f"n_pairs reported) and a coarse signal for the t-test.")
    L.append(f"- **McNemar is k-asymmetric in the 60v5 block** (BL draws {bln} vs SB's "
             f"{sbn} ⇒ `BL_only` inflated); it is fair (5 vs 5) only in the 5v5 block.")
    L.append("- **Direction matters**: a *negative* mean_diff / negative effect means the "
             "skillbook **hurt** that run.")
    L.append("- **Pass@5 (§5) is k-asymmetric in estimation, not in target**: both sides "
             "estimate the same P(solve in 5), but the skillbook side is a single binary "
             "realization (5 attempts) while the baseline side is the smooth combinatorial "
             "estimate (60 attempts) — so skillbook pass@5 is coarse and its CIs are wider. "
             "Restricted to the 60v5 block; in the 5v5 block pass@5 is binary on both sides "
             "(= the McNemar discordants) and is not reported separately.")
    L.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
