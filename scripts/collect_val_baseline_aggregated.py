#!/usr/bin/env python3
"""Aggregate VAL BASELINE results + trajectories across split025 vpk5 runs.

The split025 val_pass_k=5 experiment family consists of several completed runs
that all share the *identical* val-instance split. Each run makes 5 fresh attempts
per val instance under a uniform val-baseline condition (empty skillbook, no
learning), so aggregating across runs yields (n_runs x 5) independent attempts
per instance.

This script:
  1. Collects every val_baseline result + trajectory from the source runs into one
     self-contained directory, with files namespaced by run index (r00_iter0.json,
     ..., r{NN}_iter4.json) so the attempts per instance never collide.
  2. Computes the per-attempt average resolution rate (avg) and the pass@1..N curve
     using the standard combinatorial estimator (HumanEval/CodeX), macro-averaged
     over instances, plus per-repo breakdowns, a per-instance success histogram,
     and instance-level std / 95% CI for avg and pass@k (plus a pooled Wilson CI).

Usage:
    uv run python scripts/collect_val_baseline_aggregated.py
    uv run python scripts/collect_val_baseline_aggregated.py --no-trajectories
    uv run python scripts/collect_val_baseline_aggregated.py --runs 'data/*completed*split025*vpk5' \
        --out data/val_baseline_aggregated_split025_vpk5 --phase val_baseline --force
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from math import lgamma
from pathlib import Path

LOG_C_CACHE: dict[tuple[int, int], float] = {}


def log_choose(n: int, k: int) -> float:
    """Natural log of C(n, k). Memoized."""
    if k < 0 or k > n:
        # C(n, k) = 0 -> log = -inf
        return float("-inf")
    key = (n, k)
    cached = LOG_C_CACHE.get(key)
    if cached is not None:
        return cached
    val = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    LOG_C_CACHE[key] = val
    return val


def pass_at_k(n: int, c: int, k: int) -> float:
    """Combinatorial pass@k = 1 - C(n-c, k) / C(n, k), clipped to [0, 1].

    n = total attempts, c = successful attempts. Order-independent (treats the n
    attempts as exchangeable draws). For k >= n-c the ratio is 0 -> pass@k = 1.
    Computed via a running product of per-draw factors for numerical stability.
    """
    if n <= 0:
        return 0.0
    k = min(k, n)
    if c >= n or k <= 0:
        return 1.0 if c > 0 else 0.0 if k == 0 else (1.0 if c >= n else 0.0)
    if c == 0:
        return 0.0
    fails = n - c
    if k > fails:
        return 1.0
    # prod_{i=0}^{k-1} (n - c - i) / (n - i)
    prod = 1.0
    for i in range(k):
        prod *= (fails - i) / (n - i)
        if prod == 0.0:
            break
    return 1.0 - prod


# Two-sided 95% normal quantile (P(|Z| < Z_95) = 0.95).
Z_95 = 1.959963984540054


def mean_std_sem_ci95(values: list[float]) -> dict:
    """Instance-level stats: mean, sample std (ddof=1), SEM, and a 95% CI.

    Treats each entry of `values` (e.g. one per-instance resolution rate) as a
    single sample and applies the normal-approximation CI on the mean:
        CI = mean +/- Z_95 * (sample_std / sqrt(n)).
    This is the error bar to report for a macro-averaged benchmark metric: the
    instance is the unit of generalization, and within-instance attempts are
    correlated, so this captures cross-instance spread rather than a pooled
    binomial variance. With n>=~30 instances the mean is well-approximated by a
    normal by the CLT even though per-instance rates are bimodal.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sem": 0.0, "ci95": [0.0, 0.0]}
    m = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": m, "std": 0.0, "sem": 0.0, "ci95": [m, m]}
    var = sum((v - m) ** 2 for v in values) / (n - 1)  # sample variance, ddof=1
    sd = var ** 0.5
    sem = sd / (n ** 0.5)
    return {"n": n, "mean": m, "std": sd, "sem": sem, "ci95": [m - Z_95 * sem, m + Z_95 * sem]}


def wilson_ci95(successes: int, total: int) -> list[float]:
    """Wilson score 95% CI for a binomial proportion (pooled per-attempt rate).

    Treats each attempt as an independent Bernoulli trial. This is a complementary
    interval to the instance-level (macro) CI: it reflects sampling noise on the
    pooled success fraction but UNDERSTATES true uncertainty whenever attempts
    within an instance are positively correlated (they are -- same instance
    difficulty), so prefer the macro CI for headline error bars. Clipped to [0, 1].
    """
    if total <= 0:
        return [0.0, 0.0]
    phat = successes / total
    denom = 1.0 + Z_95 * Z_95 / total
    center = (phat + Z_95 * Z_95 / (2 * total)) / denom
    half = (Z_95 * (phat * (1 - phat) / total + Z_95 * Z_95 / (4 * total * total)) ** 0.5) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def find_benchmark_dir(run_dir: Path) -> Path | None:
    for sub in run_dir.iterdir():
        if sub.is_dir() and "__" in sub.name:
            return sub
    return None


def repo_of(instance_id: str) -> str:
    # "astropy__astropy-13236" -> "astropy__astropy"
    return instance_id.rsplit("-", 1)[0]


def retrieval_of(run_dir_name: str) -> str:
    """Infer the skillbook retrieval method from a run dir name.

    Aggregated val_skillbook runs span retrieval variants, e.g.
    "...retk5bm25..." -> "bm25", "...retk5emb..." -> "embedding",
    "...retk5random..." -> "random", and a bare "...retk5_default..." -> "llm".
    """
    n = run_dir_name.lower()
    if "bm25" in n:
        return "bm25"
    if "random" in n:
        return "random"
    if "emb" in n:
        return "embedding"
    if "llm" in n:
        return "llm"
    if "retk" in n:
        return "llm"  # retrieval enabled (top-k) but no explicit method -> default LLM ranker
    return "none"


def collect(
    run_dirs: list[Path],
    out_dir: Path,
    phase: str,
    copy_trajectories: bool,
    force: bool,
) -> dict:
    """Copy results (+trajectories) into out_dir; return provenance + raw counts."""
    if out_dir.exists():
        if not force:
            raise SystemExit(
                f"Output dir exists: {out_dir}. Use --force to remove and recreate."
            )
        shutil.rmtree(out_dir)

    results_root = out_dir / "results"
    results_root.mkdir(parents=True)
    traj_root: Path | None = None
    if copy_trajectories:
        traj_root = out_dir / "trajectories"
        traj_root.mkdir(parents=True)

    run_index: dict[str, dict] = {}
    # per-instance -> list of (run_idx, iter, resolved_bool); order appended
    per_instance: dict[str, list[tuple[int, int, bool]]] = defaultdict(list)

    n_result_copied = 0
    n_traj_copied = 0
    n_traj_missing = 0

    for run_idx, run_dir in enumerate(run_dirs):
        bench = find_benchmark_dir(run_dir)
        if bench is None:
            print(f"  ! no benchmark dir in {run_dir.name}, skipping", flush=True)
            continue
        res_phase = bench / "results" / phase
        traj_phase = bench / "trajectories" / phase
        tag = f"r{run_idx:02d}"
        run_index[tag] = {
            "run_dir": run_dir.name,
            "retrieval": retrieval_of(run_dir.name),
        }

        instances = sorted(d.name for d in res_phase.iterdir() if d.is_dir())
        for inst in instances:
            inst_res = res_phase / inst
            inst_traj = traj_phase / inst
            out_inst_res = results_root / inst
            out_inst_res.mkdir(parents=True, exist_ok=True)
            if traj_root is not None:
                (traj_root / inst).mkdir(parents=True, exist_ok=True)

            for iter_file in sorted(inst_res.glob("iter_*.json")):
                iter_num = int(iter_file.name.removeprefix("iter_").removesuffix(".json"))
                with open(iter_file) as f:
                    data = json.load(f)
                resolved = bool(data.get("resolved"))
                dst = out_inst_res / f"{tag}_iter{iter_num}.json"
                shutil.copy2(iter_file, dst)
                n_result_copied += 1
                per_instance[inst].append((run_idx, iter_num, resolved))

                if traj_root is not None:
                    src_traj = inst_traj / f"iter_{iter_num}.json"
                    if src_traj.exists():
                        shutil.copy2(
                            src_traj, traj_root / inst / f"{tag}_iter{iter_num}.json"
                        )
                        n_traj_copied += 1
                    else:
                        n_traj_missing += 1

        print(
            f"  [{tag}] {run_dir.name}: {len(instances)} instances "
            f"({n_result_copied} results, {n_traj_copied} trajectories so far)",
            flush=True,
        )

    return {
        "run_index": run_index,
        "per_instance_raw": per_instance,
        "n_result_copied": n_result_copied,
        "n_traj_copied": n_traj_copied,
        "n_traj_missing": n_traj_missing,
    }


def compute_stats(
    per_instance_raw: dict[str, list[tuple[int, int, bool]]],
    max_k: int,
    run_index: dict[str, dict] | None = None,
) -> dict:
    """Compute overall + per-repo + per-instance pass@k and histogram.

    When ``run_index`` is provided, also emits a ``per_run`` aggregate in
    ``overall`` (one row per source run, tagged rNN with its retrieval method) so
    heterogeneous val_skillbook pools — e.g. one run per retrieval variant — can be
    read off side by side instead of only as a pooled whole.
    """
    per_instance: dict[str, dict] = {}
    repo_instances: dict[str, list[str]] = defaultdict(list)

    total_attempts = 0
    total_resolved = 0
    success_histogram: dict[int, int] = defaultdict(int)

    for inst, attempts in per_instance_raw.items():
        # Sort attempts canonically: by (run_idx, iter) so per_run ordering is stable.
        attempts_sorted = sorted(attempts, key=lambda t: (t[0], t[1]))
        n = len(attempts_sorted)
        c = sum(1 for _, _, r in attempts_sorted if r)
        total_attempts += n
        total_resolved += c
        success_histogram[c] += 1

        per_run: dict[int, list[bool]] = defaultdict(list)
        for run_idx, _, r in attempts_sorted:
            per_run[run_idx].append(r)
        per_instance[inst] = {
            "repo": repo_of(inst),
            "n_attempts": n,
            "successes": c,
            "failed": n - c,
            "resolved_any": c > 0,
            "rate": c / n if n else 0.0,  # per-instance resolution rate (== pass@1)
            "ci95": wilson_ci95(c, n),  # binomial 95% CI on this instance's rate
            "per_run": {f"r{ri:02d}": runs for ri, runs in sorted(per_run.items())},
        }
        repo_instances[repo_of(inst)].append(inst)

    # Per-run aggregate (e.g. per-retrieval-method when the pool mixes variants).
    # per_instance's per_run already groups by run index; reuse it to avoid a
    # second pass over per_instance_raw.
    per_run_agg: dict[str, dict] = {}
    for inst, d in per_instance.items():
        for tag, flags in d["per_run"].items():
            agg = per_run_agg.setdefault(
                tag,
                {
                    "retrieval": (run_index or {}).get(tag, {}).get("retrieval"),
                    "run_dir": (run_index or {}).get(tag, {}).get("run_dir"),
                    "n_instances": 0,
                    "n_attempts": 0,
                    "resolved_attempts": 0,
                    "resolved_any": 0,
                },
            )
            agg["n_instances"] += 1
            agg["n_attempts"] += len(flags)
            agg["resolved_attempts"] += sum(1 for f in flags if f)
            if any(flags):
                agg["resolved_any"] += 1
    for agg in per_run_agg.values():
        agg["avg"] = (
            agg["resolved_attempts"] / agg["n_attempts"] if agg["n_attempts"] else 0.0
        )
        agg["resolved_any_rate"] = (
            agg["resolved_any"] / agg["n_instances"] if agg["n_instances"] else 0.0
        )

    n_instances = len(per_instance)

    def macro_stats(values: list[float]) -> dict:
        """mean/std/sem/ci95 for a list of per-instance values (no 'n' key)."""
        s = mean_std_sem_ci95(values)
        return {"mean": s["mean"], "std": s["std"], "sem": s["sem"], "ci95": s["ci95"]}

    def pass_at_k_macro(insts: list[str]) -> dict[str, dict[str, float]]:
        """Macro-averaged pass@k over `insts` with instance-level std/SEM/95% CI.

        Returns {"mean": {k:..}, "std": {k:..}, "sem": {k:..}, "ci95": {k:[lo,hi]}}.
        """
        out: dict[str, dict[str, float]] = {"mean": {}, "std": {}, "sem": {}, "ci95": {}}
        if not insts:
            return out
        for k in range(1, max_k + 1):
            vals = [
                pass_at_k(per_instance[i]["n_attempts"], per_instance[i]["successes"], k)
                for i in insts
            ]
            s = mean_std_sem_ci95(vals)
            out["mean"][str(k)] = s["mean"]
            out["std"][str(k)] = s["std"]
            out["sem"][str(k)] = s["sem"]
            out["ci95"][str(k)] = s["ci95"]
        return out

    overall_pass = pass_at_k_macro(list(per_instance.keys()))
    avg = total_resolved / total_attempts if total_attempts else 0.0
    # Instance-level (macro) stats on the per-attempt resolution rate. Equals the
    # pooled `avg` when attempts/instance is uniform, but reported separately so
    # the error bars are anchored to the macro mean regardless.
    overall_macro = macro_stats(
        [d["rate"] for d in per_instance.values() if d["n_attempts"]]
    )

    per_repo: dict[str, dict] = {}
    for repo, insts in sorted(repo_instances.items()):
        attempts_r = sum(per_instance[i]["n_attempts"] for i in insts)
        resolved_r = sum(per_instance[i]["successes"] for i in insts)
        repo_pass = pass_at_k_macro(insts)
        repo_macro = macro_stats(
            [per_instance[i]["rate"] for i in insts if per_instance[i]["n_attempts"]]
        )
        per_repo[repo] = {
            "n_instances": len(insts),
            "n_attempts": attempts_r,
            "resolved_attempts": resolved_r,
            "avg": resolved_r / attempts_r if attempts_r else 0.0,
            "avg_macro": repo_macro["mean"],
            "avg_std": repo_macro["std"],
            "avg_sem": repo_macro["sem"],
            "avg_ci95": repo_macro["ci95"],  # instance-level 95% CI on macro mean
            "avg_binomial_ci95": wilson_ci95(resolved_r, attempts_r),  # pooled Wilson
            "resolved_any": sum(1 for i in insts if per_instance[i]["resolved_any"]),
            "resolved_any_rate": (
                sum(1 for i in insts if per_instance[i]["resolved_any"]) / len(insts)
            ),
            "pass_at_k": repo_pass["mean"],
            "pass_at_k_std": repo_pass["std"],
            "pass_at_k_sem": repo_pass["sem"],
            "pass_at_k_ci95": repo_pass["ci95"],
        }

    overall = {
        "n_instances": n_instances,
        "n_runs": len({t[0] for atts in per_instance_raw.values() for t in atts}),
        "n_attempts_per_instance": sorted(
            {len(a) for a in per_instance_raw.values()}
        ),
        "total_attempts": total_attempts,
        "total_resolved_attempts": total_resolved,
        "avg": avg,  # pooled per-attempt resolution rate (== pass@1 mean when uniform)
        "avg_macro": overall_macro["mean"],  # macro mean of per-instance rates
        "avg_std": overall_macro["std"],  # instance-level sample std (ddof=1)
        "avg_sem": overall_macro["sem"],
        "avg_ci95": overall_macro["ci95"],  # instance-level 95% CI on macro mean
        "avg_binomial_ci95": wilson_ci95(total_resolved, total_attempts),  # pooled Wilson
        "resolved_any": sum(1 for d in per_instance.values() if d["resolved_any"]),
        "resolved_any_rate": (
            sum(1 for d in per_instance.values() if d["resolved_any"]) / n_instances
            if n_instances
            else 0.0
        ),  # == pass@max
        "pass_at_k": overall_pass["mean"],
        "pass_at_k_std": overall_pass["std"],
        "pass_at_k_sem": overall_pass["sem"],
        "pass_at_k_ci95": overall_pass["ci95"],
        "per_run": per_run_agg,  # one row per source run (rNN + retrieval method)
        "success_histogram": {str(k): v for k, v in sorted(success_histogram.items())},
    }
    return {"overall": overall, "per_repo": per_repo, "per_instance": per_instance}


def write_table_line(k: int, v: float) -> str:
    return f"  pass@{k:<2} = {v * 100:6.2f}%"


def print_report(stats: dict, max_k: int, label: str = "Val-Baseline") -> None:
    ov = stats["overall"]
    apiv = ov["n_attempts_per_instance"]
    print("\n" + "=" * 72)
    print(f"{label.upper()} aggregated (split025, val_pass_k=5, {ov['n_runs']} runs)")
    print("=" * 72)
    print(f"instances            : {ov['n_instances']}")
    print(f"runs aggregated      : {ov['n_runs']}")
    print(f"attempts/instance    : {apiv}")
    print(f"total attempts       : {ov['total_attempts']}")
    print(f"resolved attempts    : {ov['total_resolved_attempts']}")
    print(f"avg (per-attempt)    : {ov['avg'] * 100:.2f}%   (== pass@1 mean)")
    lo, hi = ov["avg_ci95"]
    print(
        f"  macro 95% CI       : [{lo * 100:.2f}%, {hi * 100:.2f}%]   "
        f"(instance-level, n={ov['n_instances']}, "
        f"std={ov['avg_std'] * 100:.2f}pp, sem={ov['avg_sem'] * 100:.2f}pp)"
    )
    blo, bhi = ov["avg_binomial_ci95"]
    print(
        f"  pooled Wilson 95% CI: [{blo * 100:.2f}%, {bhi * 100:.2f}%]   "
        f"(binomial on {ov['total_attempts']} attempts; understates spread)"
    )
    print(
        f"resolved_any (>=1/{max_k}): {ov['resolved_any']}/{ov['n_instances']} "
        f"= {ov['resolved_any_rate'] * 100:.2f}%   (== pass@{max_k})"
    )

    print("\npass@k (combinatorial, macro-averaged over instances):")
    pak = ov["pass_at_k"]
    pak_ci = ov["pass_at_k_ci95"]
    # Print in compact columns of 4
    keys = list(range(1, max_k + 1))
    for i in range(0, len(keys), 4):
        cells = []
        for k in keys[i : i + 4]:
            cells.append(f"p@{k:<2}={pak[str(k)] * 100:6.2f}%")
        print("  " + "   ".join(cells))
    print("  macro 95% CI (selected k):")
    seen = set()
    for k in (1, 5, 10, 20, max_k):
        if k in seen:
            continue
        seen.add(k)
        if str(k) in pak_ci:
            clo, chi = pak_ci[str(k)]
            print(f"    p@{k:<2} = [{clo * 100:.2f}%, {chi * 100:.2f}%]")

    print(f"\nsuccess-count histogram (instances with exactly c/{max_k} solved):")
    hist = ov["success_histogram"]
    for c in range(0, max_k + 1):
        cnt = hist.get(str(c), 0)
        if cnt:
            bar = "#" * cnt
            print(f"  c={c:>2}: {cnt:>3}  {bar}")

    print("\nper-repo:")
    # Columns adapt to max_k: 1/5/10/20 where available, always include max_k.
    pak_cols = [k for k in (1, 5, 10, 20, 40) if k <= max_k]
    if max_k not in pak_cols:
        pak_cols.append(max_k)
    col_hdr = " ".join(f"p@{k}".rjust(7) for k in pak_cols)
    print(f"  {'repo':<28} {'inst':>4} {'avg':>8} {'CI95':>9} {col_hdr} {'any':>8}")
    for repo, d in stats["per_repo"].items():
        pk = d["pass_at_k"]
        rlo, rhi = d["avg_ci95"]
        cells = " ".join(f"{pk[str(k)] * 100:.1f}%".rjust(7) for k in pak_cols)
        ci_cell = f"±{(rhi - rlo) / 2 * 100:.1f}"
        print(
            f"  {repo:<28} {d['n_instances']:>4} {d['avg'] * 100:>7.1f}% {ci_cell:>9} "
            f"{cells} {d['resolved_any_rate'] * 100:>7.1f}%"
        )

    # Per-source-run breakdown (one row per retrieval method when the pool mixes variants).
    per_run = ov.get("per_run", {})
    if per_run:
        print("\nper-run (source run; retrieval method where applicable):")
        print(
            f"  {'run':<5} {'retrieval':<11} {'inst':>4} {'att':>5} "
            f"{'avg':>8} {'pass@max':>9}"
        )
        for tag, d in sorted(per_run.items()):
            print(
                f"  {tag:<5} {str(d.get('retrieval')):<11} {d['n_instances']:>4} "
                f"{d['n_attempts']:>5} {d['avg'] * 100:>7.1f}% "
                f"{d['resolved_any_rate'] * 100:>8.1f}%"
            )


def write_readme(out_dir: Path, run_dirs: list[Path], stats: dict, args) -> None:
    ov = stats["overall"]
    n_runs = ov["n_runs"]
    apiv = ov["n_attempts_per_instance"]
    apiv_val = apiv[-1] if apiv else 0
    vpk = apiv_val // n_runs if n_runs else 0
    rhi = f"r{n_runs - 1:02d}" if n_runs else "r??"
    label = getattr(args, "label", "Val-Baseline")
    condition = getattr(
        args,
        "condition",
        "under a uniform val-baseline condition (empty skillbook, no learning)",
    )
    # Distinct retrieval methods across the pooled runs, if any (val_skillbook pools).
    retrievals = sorted(
        {d.get("retrieval") for d in ov.get("per_run", {}).values() if d.get("retrieval")}
    )
    lines = [
        f"# {label} Aggregated (split025, val_pass_k=5)",
        "",
        f"Collected from **{n_runs}** runs sharing the same {ov['n_instances']} val",
        f"instances, each with **{apiv}** fresh attempts ({n_runs} runs x {vpk}),",
        condition + ".",
    ]
    if len(retrievals) > 1:
        lines += [
            "",
            f"> **Retrieval heterogeneity:** the {n_runs} pooled runs span retrieval "
            f"methods {', '.join(retrievals)} (see `runs_index.json`). The pooled "
            "pass@k treats their attempts as exchangeable draws; per-run (per-method) "
            "rates are in `stats/overall.json` under `per_run`.",
        ]
    lines += [
        "",
        "## Layout",
        "",
        "```",
        "results/<instance>/r{NN}_iter{I}.json          # per-attempt eval results",
        "trajectories/<instance>/r{NN}_iter{I}.json      # per-attempt agent trajectories",
        "runs_index.json                                  # rNN -> source run_dir",
        "stats/overall.json, stats/per_repo.json, stats/per_instance.json",
        "```",
        "",
        f"`rNN` is the run index (r00..{rhi}, see runs_index.json); `iter{{I}}` is the attempt",
        f"within that run (0..{vpk - 1}). Each instance therefore has {apiv_val} files.",
        "",
        "## Methodology",
        "",
        "- **avg** = resolved attempts / total attempts (per-attempt resolution rate).",
        "  Equals pass@1 (pooled == macro mean here since attempts/instance is uniform).",
        "- **pass@k** = 1 - C(n-c, k)/C(n, k) (standard combinatorial / HumanEval estimator),",
        f"  with n = {apiv_val} attempts, c = successes, macro-averaged over",
        f"  the {ov['n_instances']} instances. Order-independent ({apiv_val} attempts treated as exchangeable draws).",
        f"- **pass@{apiv_val}** = fraction of instances solved at least once",
        f"  across all {apiv_val} attempts.",
        "",
        "## Uncertainty (std + 95% CI)",
        "",
        "Two complementary intervals are reported (in overall.json and per_repo.json):",
        "",
        "- **Instance-level (macro) 95% CI** — `avg_std`, `avg_sem`, `avg_ci95`,",
        "  `pass_at_k_std`, `pass_at_k_sem`, `pass_at_k_ci95`. Treats each instance's",
        "  resolution rate as one sample and gives mean +/- Z_95 * sample_std / sqrt(n)",
        "  (sample std, ddof=1; Z_95 = 1.95996). **This is the headline error bar**: the",
        "  instance is the unit of generalization, and within-instance attempts are",
        "  correlated, so this captures cross-instance spread.",
        "- **Pooled binomial 95% CI** — `avg_binomial_ci95` (Wilson score on the pooled",
        "  per-attempt success fraction). Reflects sampling noise on the pooled rate but",
        "  UNDERSTATES true uncertainty (attempts within an instance are positively",
        "  correlated); reported for completeness only.",
        "- **Per-instance** — each entry in per_instance.json carries its own Wilson 95% CI",
        "  (`ci95`) on that instance's c/n rate.",
        "",
        "## Source runs",
        "",
    ]
    for r in run_dirs:
        lines.append(f"- {r.name}")
    lines += [
        "",
        f"Generated by `scripts/{Path(__file__).name}` "
        f"(phase={args.phase}, copy_trajectories={args.copy_trajectories}).",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="data/*completed*split025*vpk5", help="Glob of source run dirs")
    ap.add_argument("--out", default="data/val_baseline_aggregated_split025_vpk5", help="Output dir")
    ap.add_argument("--phase", default="val_baseline", help="Phase subdir to collect (val_baseline or val=val_skillbook)")
    ap.add_argument(
        "--label",
        default=None,
        help="Human label for README/report (default: Val-Baseline for val_baseline, Val-Skillbook for val)",
    )
    ap.add_argument(
        "--condition",
        default=None,
        help="Condition sentence for README (default derived from --phase)",
    )
    ap.add_argument("--no-trajectories", dest="copy_trajectories", action="store_false", help="Skip copying trajectories (results only)")
    ap.add_argument("--force", action="store_true", help="Remove existing output dir first")
    ap.set_defaults(copy_trajectories=True)
    args = ap.parse_args()

    if args.label is None:
        args.label = {"val_baseline": "Val-Baseline", "val": "Val-Skillbook"}.get(
            args.phase, args.phase.replace("_", " ").title()
        )
    if args.condition is None:
        args.condition = {
            "val_baseline": "under a uniform val-baseline condition (empty skillbook, no learning)",
            "val": "under the val-skillbook condition (learned skillbook + retrieval, no learning)",
        }.get(args.phase, f"phase={args.phase}")

    run_dirs = sorted(Path(".").glob(args.runs))
    run_dirs = [r for r in run_dirs if (r / "statistics.json").exists()]
    if not run_dirs:
        raise SystemExit(f"No runs matched: {args.runs}")
    print(f"Collecting {len(run_dirs)} runs -> {args.out} (phase={args.phase})")

    out_dir = Path(args.out)
    collected = collect(
        run_dirs,
        out_dir,
        phase=args.phase,
        copy_trajectories=args.copy_trajectories,
        force=args.force,
    )

    # Sanity: attempts per instance should be uniform.
    attempt_counts = {len(v) for v in collected["per_instance_raw"].values()}
    max_k = max(attempt_counts) if attempt_counts else 0
    print(
        f"\nCollected {collected['n_result_copied']} results, "
        f"{collected['n_traj_copied']} trajectories "
        f"({collected['n_traj_missing']} missing). "
        f"Attempts/instance = {sorted(attempt_counts)} (max_k={max_k})"
    )

    stats = compute_stats(collected["per_instance_raw"], max_k, collected["run_index"])

    stats_dir = out_dir / "stats"
    stats_dir.mkdir(exist_ok=True)
    (stats_dir / "overall.json").write_text(json.dumps(stats["overall"], indent=2))
    (stats_dir / "per_repo.json").write_text(json.dumps(stats["per_repo"], indent=2))
    (stats_dir / "per_instance.json").write_text(json.dumps(stats["per_instance"], indent=2))

    (out_dir / "runs_index.json").write_text(json.dumps(collected["run_index"], indent=2))
    write_readme(out_dir, run_dirs, stats, args)

    print_report(stats, max_k, label=args.label)
    print(f"\nSaved: {out_dir}/stats/{{overall,per_repo,per_instance}}.json + README.md")


if __name__ == "__main__":
    main()
