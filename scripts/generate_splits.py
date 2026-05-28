#!/usr/bin/env python3
"""Generate fixed train/val split manifests for SWE-bench benchmarks.

Each repo is split independently using a deterministic RNG seeded with
(global_seed, repo_name). The global split is the union of all per-repo splits.

Usage:
    uv run python scripts/generate_splits.py \
        --benchmark princeton-nlp__SWE-bench_Lite \
        --val-ratio 0.25 --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def generate_split(benchmark: str, val_ratio: float, seed: int) -> dict:
    """Generate a per-repo train/val split manifest."""
    dataset = load_dataset(benchmark, split="test")
    instances = list(dataset)

    # Group by repo
    repo_groups: dict[str, list[str]] = defaultdict(list)
    for inst in instances:
        repo_groups[inst["repo"]].append(inst["instance_id"])

    # Sort repos for deterministic ordering
    sorted_repos = sorted(repo_groups.keys())

    per_repo: dict[str, dict] = {}
    all_train: list[str] = []
    all_val: list[str] = []

    for repo in sorted_repos:
        ids = repo_groups[repo]
        # Per-repo deterministic shuffle
        rng = random.Random(f"{seed}:{repo}")
        shuffled = list(ids)
        rng.shuffle(shuffled)

        val_count = max(1, int(len(shuffled) * val_ratio))
        val_ids = shuffled[:val_count]
        train_ids = shuffled[val_count:]

        per_repo[repo] = {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
        }
        all_train.extend(train_ids)
        all_val.extend(val_ids)

    # Sort global lists for readability
    all_train.sort()
    all_val.sort()

    manifest = {
        "benchmark": benchmark,
        "val_ratio": val_ratio,
        "seed": seed,
        "total_instances": len(instances),
        "train_instances": all_train,
        "val_instances": all_val,
        "per_repo": per_repo,
    }

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate train/val split manifest")
    parser.add_argument(
        "--benchmark", required=True,
        help="HuggingFace dataset name (e.g. princeton-nlp__SWE-bench_Lite)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.25,
        help="Fraction of instances per repo for validation",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic splits",
    )
    args = parser.parse_args()

    manifest = generate_split(args.benchmark, args.val_ratio, args.seed)

    # Save to configs/splits/<benchmark>/
    benchmark_dir = args.benchmark.replace("/", "__")
    out_dir = Path("configs/splits") / benchmark_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ratio_str = str(args.val_ratio).replace(".", "_")
    out_path = out_dir / f"val_ratio_{ratio_str}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    n_repos = len(manifest["per_repo"])
    print(f"Benchmark:  {args.benchmark}")
    print(f"Val ratio:  {args.val_ratio}")
    print(f"Seed:       {args.seed}")
    print(f"Repos:      {n_repos}")
    print(f"Train:      {len(manifest['train_instances'])}")
    print(f"Val:        {len(manifest['val_instances'])}")
    print(f"Saved to:   {out_path}")

    # Per-repo breakdown
    print(f"\n{'Repo':<45} {'Train':>6} {'Val':>6} {'Total':>6}")
    print("-" * 65)
    for repo in sorted(manifest["per_repo"]):
        pr = manifest["per_repo"][repo]
        t, v = len(pr["train"]), len(pr["val"])
        print(f"{repo:<45} {t:>6} {v:>6} {t+v:>6}")


if __name__ == "__main__":
    main()
