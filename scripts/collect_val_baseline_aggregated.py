#!/usr/bin/env python3
"""Aggregate VAL BASELINE results + trajectories across split025 vpk5 runs.

The split025 val_pass_k=5 experiment family consists of 8 completed runs that all
share the *identical* 113 val instances. Each run makes 5 fresh attempts per val
instance under a uniform val-baseline condition (empty skillbook, no learning),
so aggregating across runs yields 8 x 5 = 40 independent attempts per instance.

This script:
  1. Collects every val_baseline result + trajectory from the source runs into one
     self-contained directory, with files namespaced by run index (r00_iter0.json,
     ..., r07_iter4.json) so the 40 attempts per instance never collide.
  2. Computes the per-attempt average resolution rate (avg) and the pass@1..N curve
     using the standard combinatorial estimator (HumanEval/CodeX), macro-averaged
     over instances, plus per-repo breakdowns and a per-instance success histogram.

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


def find_benchmark_dir(run_dir: Path) -> Path | None:
    for sub in run_dir.iterdir():
        if sub.is_dir() and "__" in sub.name:
            return sub
    return None


def repo_of(instance_id: str) -> str:
    # "astropy__astropy-13236" -> "astropy__astropy"
    return instance_id.rsplit("-", 1)[0]


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
        run_index[tag] = {"run_dir": run_dir.name}

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


def compute_stats(per_instance_raw: dict[str, list[tuple[int, int, bool]]], max_k: int) -> dict:
    """Compute overall + per-repo + per-instance pass@k and histogram."""
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
            "per_run": {f"r{ri:02d}": runs for ri, runs in sorted(per_run.items())},
        }
        repo_instances[repo_of(inst)].append(inst)

    n_instances = len(per_instance)

    def pass_at_k_macro(insts: list[str]) -> dict[str, float]:
        if not insts:
            return {}
        out: dict[str, float] = {}
        for k in range(1, max_k + 1):
            vals = []
            for inst in insts:
                d = per_instance[inst]
                vals.append(pass_at_k(d["n_attempts"], d["successes"], k))
            out[str(k)] = sum(vals) / len(vals)
        return out

    overall_pass = pass_at_k_macro(list(per_instance.keys()))
    avg = total_resolved / total_attempts if total_attempts else 0.0

    per_repo: dict[str, dict] = {}
    for repo, insts in sorted(repo_instances.items()):
        attempts_r = sum(per_instance[i]["n_attempts"] for i in insts)
        resolved_r = sum(per_instance[i]["successes"] for i in insts)
        per_repo[repo] = {
            "n_instances": len(insts),
            "n_attempts": attempts_r,
            "resolved_attempts": resolved_r,
            "avg": resolved_r / attempts_r if attempts_r else 0.0,
            "resolved_any": sum(1 for i in insts if per_instance[i]["resolved_any"]),
            "resolved_any_rate": (
                sum(1 for i in insts if per_instance[i]["resolved_any"]) / len(insts)
            ),
            "pass_at_k": pass_at_k_macro(insts),
        }

    overall = {
        "n_instances": n_instances,
        "n_runs": len({t[0] for atts in per_instance_raw.values() for t in atts}),
        "n_attempts_per_instance": sorted(
            {len(a) for a in per_instance_raw.values()}
        ),
        "total_attempts": total_attempts,
        "total_resolved_attempts": total_resolved,
        "avg": avg,  # == pass@1 (per-attempt resolution rate)
        "resolved_any": sum(1 for d in per_instance.values() if d["resolved_any"]),
        "resolved_any_rate": (
            sum(1 for d in per_instance.values() if d["resolved_any"]) / n_instances
            if n_instances
            else 0.0
        ),  # == pass@max
        "pass_at_k": overall_pass,
        "success_histogram": {str(k): v for k, v in sorted(success_histogram.items())},
    }
    return {"overall": overall, "per_repo": per_repo, "per_instance": per_instance}


def write_table_line(k: int, v: float) -> str:
    return f"  pass@{k:<2} = {v * 100:6.2f}%"


def print_report(stats: dict, max_k: int) -> None:
    ov = stats["overall"]
    print("\n" + "=" * 72)
    print("VAL BASELINE aggregated (split025, val_pass_k=5, 8 runs)")
    print("=" * 72)
    print(f"instances            : {ov['n_instances']}")
    print(f"runs aggregated      : {ov['n_runs']}")
    print(f"attempts/instance    : {ov['n_attempts_per_instance']}")
    print(f"total attempts       : {ov['total_attempts']}")
    print(f"resolved attempts    : {ov['total_resolved_attempts']}")
    print(f"avg (per-attempt)    : {ov['avg'] * 100:.2f}%   (== pass@1)")
    print(
        f"resolved_any (>=1/40): {ov['resolved_any']}/{ov['n_instances']} "
        f"= {ov['resolved_any_rate'] * 100:.2f}%   (== pass@{max_k})"
    )

    print("\npass@k (combinatorial, macro-averaged over instances):")
    pak = ov["pass_at_k"]
    # Print in compact columns of 4
    keys = list(range(1, max_k + 1))
    for i in range(0, len(keys), 4):
        cells = []
        for k in keys[i : i + 4]:
            cells.append(f"p@{k:<2}={pak[str(k)] * 100:6.2f}%")
        print("  " + "   ".join(cells))

    print("\nsuccess-count histogram (instances with exactly c/40 solved):")
    hist = ov["success_histogram"]
    for c in range(0, max_k + 1):
        cnt = hist.get(str(c), 0)
        if cnt:
            bar = "#" * cnt
            print(f"  c={c:>2}: {cnt:>3}  {bar}")

    print("\nper-repo:")
    print(f"  {'repo':<28} {'inst':>4} {'avg':>8} {'p@1':>8} {'p@5':>8} {'p@10':>8} {'p@20':>8} {'p@40':>8} {'any':>8}")
    for repo, d in stats["per_repo"].items():
        pk = d["pass_at_k"]

        def g(k):
            return f"{pk[str(k)] * 100:.1f}%"

        print(
            f"  {repo:<28} {d['n_instances']:>4} {d['avg'] * 100:>7.1f}% "
            f"{g(1):>8} {g(5):>8} {g(10):>8} {g(20):>8} {g(40):>8} "
            f"{d['resolved_any_rate'] * 100:>7.1f}%"
        )


def write_readme(out_dir: Path, run_dirs: list[Path], stats: dict, args) -> None:
    ov = stats["overall"]
    lines = [
        "# Val-Baseline Aggregated (split025, val_pass_k=5)",
        "",
        f"Collected from **{len(run_dirs)}** completed runs sharing the same 113 val",
        f"instances, each with **{ov['n_attempts_per_instance']}** fresh attempts (8 runs x 5),",
        "under a uniform val-baseline condition (empty skillbook, no learning).",
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
        "`rNN` is the run index (00..07, see runs_index.json); `iter{I}` is the attempt",
        f"within that run (0..4). Each instance therefore has {ov['n_attempts_per_instance']} files.",
        "",
        "## Methodology",
        "",
        "- **avg** = resolved attempts / total attempts (per-attempt resolution rate).",
        "  Equals pass@1.",
        "- **pass@k** = 1 - C(n-c, k)/C(n, k) (standard combinatorial / HumanEval estimator),",
        f"  with n = {ov['n_attempts_per_instance']} attempts, c = successes, macro-averaged over",
        "  the 113 instances. Order-independent (40 attempts treated as exchangeable draws).",
        f"- **pass@{ov['n_attempts_per_instance']}** = fraction of instances solved at least once",
        "  across all 40 attempts.",
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
    ap.add_argument("--phase", default="val_baseline", help="Phase subdir to collect")
    ap.add_argument("--no-trajectories", dest="copy_trajectories", action="store_false", help="Skip copying trajectories (results only)")
    ap.add_argument("--force", action="store_true", help="Remove existing output dir first")
    ap.set_defaults(copy_trajectories=True)
    args = ap.parse_args()

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

    stats = compute_stats(collected["per_instance_raw"], max_k)

    stats_dir = out_dir / "stats"
    stats_dir.mkdir(exist_ok=True)
    (stats_dir / "overall.json").write_text(json.dumps(stats["overall"], indent=2))
    (stats_dir / "per_repo.json").write_text(json.dumps(stats["per_repo"], indent=2))
    (stats_dir / "per_instance.json").write_text(json.dumps(stats["per_instance"], indent=2))

    (out_dir / "runs_index.json").write_text(json.dumps(collected["run_index"], indent=2))
    write_readme(out_dir, run_dirs, stats, args)

    print_report(stats, max_k)
    print(f"\nSaved: {out_dir}/stats/{{overall,per_repo,per_instance}}.json + README.md")


if __name__ == "__main__":
    main()
