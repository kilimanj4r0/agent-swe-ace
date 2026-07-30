#!/usr/bin/env python3
"""Prepare Docker images for SWE-bench evaluation.

Wraps swebench.harness.prepare_images with project config.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Prepare Docker images for SWE-bench evaluation")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--instances", nargs="+", help="Specific instance IDs to prepare")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Force rebuild images")
    parser.add_argument("--dataset", default=None, help="Dataset name (overrides config)")
    parser.add_argument("--env-image-tag", default="latest", help="Environment image tag (default: latest)")
    parser.add_argument("--tag", default="latest", help="Instance image tag (default: latest)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    namespace = config.get("environment", {}).get("namespace")
    dataset = args.dataset or config.get("benchmark", {}).get("dataset", "princeton-nlp/SWE-bench_Lite")
    split = config.get("benchmark", {}).get("split", "test")

    # Build swebench prepare_images command
    cmd = [
        sys.executable, "-m", "swebench.harness.prepare_images",
        "--dataset_name", dataset,
        "--split", split,
        "--max_workers", str(args.workers),
        "--env_image_tag", args.env_image_tag,
        "--tag", args.tag,
    ]

    if namespace:
        # Strip trailing slash to avoid double slashes
        cmd.extend(["--namespace", namespace.rstrip("/")])

    if args.instances:
        cmd.extend(["--instance_ids", *args.instances])

    if args.force:
        cmd.extend(["--force_rebuild", "true"])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
