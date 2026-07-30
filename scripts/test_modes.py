#!/usr/bin/env python3
"""
End-to-end smoke test runner for all experiment modes.

Runs configs/test/01-10, verifies outputs after each.
Configs 07-10 depend on output from earlier configs (runner handles path injection).

Usage:
    uv run python scripts/test_modes.py                  # run all 10
    uv run python scripts/test_modes.py --only 01 04      # run specific tests
    uv run python scripts/test_modes.py --keep --verbose  # keep output, show logs
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.loader import deep_merge

CONFIGS_DIR = PROJECT_ROOT / "configs" / "test"
DATA_DIR = PROJECT_ROOT / "_data"
TEACHER_TRAJS_DIR = PROJECT_ROOT / "data" / "teacher_trajs" / "v1.16_opus45_verified"

# Test instances (from analysis of 20 completed runs)
INSTANCE_RESOLVED = "django__django-16527"    # resolved 95% of the time
INSTANCE_UNRESOLVED = "django__django-15695"  # never resolved in 20 runs


# ── Colors ──────────────────────────────────────────────────────────────

def _green(s): return f"\033[92m{s}\033[0m"
def _red(s): return f"\033[91m{s}\033[0m"
def _yellow(s): return f"\033[93m{s}\033[0m"
def _bold(s): return f"\033[1m{s}\033[0m"
def _dim(s): return f"\033[2m{s}\033[0m"
def _cyan(s): return f"\033[96m{s}\033[0m"


# ── Helpers ─────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def find_latest_run(output_dir: Path) -> Path | None:
    """Find the most recent run_* directory in output_dir."""
    if not output_dir.exists():
        return None
    runs = sorted(output_dir.glob("run_*"), key=lambda p: p.name, reverse=True)
    return runs[0] if runs else None


def find_benchmark_dir(run_dir: Path) -> Path | None:
    """Find the princeton-nlp__SWE-bench_* subdirectory inside a run dir."""
    if not run_dir.exists():
        return None
    for d in run_dir.iterdir():
        if d.is_dir() and "__" in d.name and "SWE-bench" in d.name:
            return d
    # Old layout: no benchmark subdir, files directly in run_dir
    if (run_dir / "statistics.json").exists():
        return run_dir
    return None


def run_config(config_path: Path, extra_args: list[str] = None, verbose: bool = False) -> tuple[int, str]:
    """Run a config and return (returncode, output)."""
    cmd = ["uv", "run", "python", "-m", "src.cli.commands", "--config", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)

    if verbose:
        print(_dim(f"  $ {' '.join(cmd)}"))
        # Stream output in real-time
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT,
            text=True, timeout=1800,
        )
        return result.returncode, ""

    # Capture output for parsing
    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=1800,
    )
    output = result.stdout if result.stdout else ""
    return result.returncode, output


def inject_overrides(config_path: Path, overrides: dict) -> Path:
    """Create a temp config with overrides deep-merged on top."""
    base = load_yaml(config_path)
    merged = deep_merge(base, overrides)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=PROJECT_ROOT / "configs" / "test")
    import yaml
    yaml.dump(merged, tmp, default_flow_style=False)
    tmp.close()
    return Path(tmp.name)


# ── Verification ────────────────────────────────────────────────────────

def verify(run_dir: Path, checks: dict) -> tuple[bool, list[str]]:
    """Verify run output against checks. Returns (passed, details)."""
    details = []
    passed = True

    bm_dir = find_benchmark_dir(run_dir)
    if bm_dir is None:
        return False, ["benchmark subdir not found in run dir"]

    check_dir = bm_dir if bm_dir != run_dir else run_dir

    # files_exist
    for f in checks.get("files_exist", []):
        p = run_dir / f
        if not p.exists():
            passed = False
            details.append(f"MISSING file: {f}")

    # dirs_exist
    for d in checks.get("dirs_exist", []):
        p = check_dir / d
        if not p.exists():
            passed = False
            details.append(f"MISSING dir: {d}")

    # dirs_absent
    for d in checks.get("dirs_absent", []):
        p = check_dir / d
        if p.exists():
            passed = False
            details.append(f"UNEXPECTED dir present: {d}")

    # files_absent
    for f in checks.get("files_absent", []):
        p = check_dir / f
        if p.exists():
            passed = False
            details.append(f"UNEXPECTED file present: {f}")

    # Load statistics.json for key/value checks
    stats_path = run_dir / "statistics.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())

        for key in checks.get("stats_keys", []):
            if key not in stats:
                passed = False
                details.append(f"MISSING stats key: {key}")

        for key, expected in checks.get("stats_values", {}).items():
            actual = stats.get(key)
            if actual is None:
                passed = False
                details.append(f"MISSING stats value: {key} (expected {expected})")
            elif isinstance(expected, int) and actual < expected:
                passed = False
                details.append(f"stats[{key}] = {actual}, expected >= {expected}")
            elif not isinstance(expected, int) and actual != expected:
                passed = False
                details.append(f"stats[{key}] = {actual}, expected {expected}")

        # stats_gte: value must be >= expected (for counts that vary)
        for key, expected in checks.get("stats_gte", {}).items():
            actual = stats.get(key)
            if actual is None:
                passed = False
                details.append(f"MISSING stats value: {key} (expected >= {expected})")
            elif actual < expected:
                passed = False
                details.append(f"stats[{key}] = {actual}, expected >= {expected}")

        # Nested key checks (e.g., train_phase.teacher_trajs_found)
        for dotkey, expected in checks.get("stats_nested", {}).items():
            parts = dotkey.split(".")
            val = stats
            for part in parts:
                val = val.get(part) if isinstance(val, dict) else None
                if val is None:
                    break
            if val is None:
                passed = False
                details.append(f"MISSING nested stats: {dotkey} (expected {expected})")
            elif isinstance(expected, int) and val < expected:
                passed = False
                details.append(f"stats[{dotkey}] = {val}, expected >= {expected}")

        # stats_sum_gte: sum of several nested numeric keys must be >= threshold.
        # The check key is a comma-separated list of dot-paths, e.g.
        #   {"retrieval.instances_retrieved,retrieval.instances_no_change": 1}
        # Used to prove a retriever's retrieve() was actually invoked (either it
        # filtered some skills or selected all unchanged) rather than silently skipped.
        for dotkeys, threshold in checks.get("stats_sum_gte", {}).items():
            total = 0
            missing = False
            for dk in dotkeys.split(","):
                val = stats
                for part in dk.strip().split("."):
                    val = val.get(part) if isinstance(val, dict) else None
                    if val is None:
                        break
                if val is None:
                    missing = True
                    passed = False
                    details.append(f"MISSING nested stats for sum: {dk.strip()}")
                    break
                total += val
            if not missing and total < threshold:
                passed = False
                details.append(f"stats sum[{dotkeys}] = {total}, expected >= {threshold}")

    elif checks.get("stats_keys") or checks.get("stats_values"):
        passed = False
        details.append("MISSING statistics.json")

    # run_dir_files: check files relative to run_dir (not benchmark subdir)
    for f in checks.get("run_dir_files", []):
        p = run_dir / f
        if not p.exists():
            passed = False
            details.append(f"MISSING run_dir file: {f}")

    return passed, details


# ── Test Definitions ────────────────────────────────────────────────────

class SmokeTest:
    def __init__(self, verbose=False, keep=False, total=10):
        self.verbose = verbose
        self.keep = keep
        self.total = total
        self.results = {}   # test_num -> bool
        self.run_dirs = {}  # test_num -> run_dir
        self._completed = 0

    def _run(self, num: int, config_name: str, extra_args: list[str] = None,
             overrides: dict = None, checks: dict = None) -> tuple[bool, Path | None]:
        """Run a single test config, verify, return (passed, run_dir)."""
        config_path = CONFIGS_DIR / config_name
        if not config_path.exists():
            print(_red(f"  [{num:02d}/{self.total}] CONFIG NOT FOUND: {config_path}"))
            return False, None

        # Each test gets its own output subdir to avoid parallel collisions
        test_slug = config_name.replace(".yaml", "")
        test_output_dir = DATA_DIR / test_slug
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # Merge output dir override with any other overrides
        all_overrides = {"output": {"dir": str(test_output_dir)}}
        if overrides:
            all_overrides = deep_merge(all_overrides, overrides)

        # Apply overrides
        tmp_config = inject_overrides(config_path, all_overrides)
        config_path = tmp_config

        label = config_name.replace(".yaml", "").replace("_", " ")
        print(f"  [{num:02d}/{self.total}] {label}...", flush=True)

        t0 = time.time()
        rc, output = run_config(config_path, extra_args, verbose=self.verbose)
        elapsed = time.time() - t0

        # Clean up temp config
        try:
            tmp_config.unlink()
        except OSError:
            pass

        if rc != 0:
            print(_red(f"  [{num:02d}/{self.total}] FAIL  {label}  ({elapsed:.0f}s, exit {rc})"))
            if output and not self.verbose:
                for line in output.strip().split("\n")[-3:]:
                    print(_dim(f"         {line}"))
            return False, None

        # Find output dir: parse "Output:" line from subprocess output
        run_dir = None
        for line in output.split("\n"):
            if "Output:" in line and "run_" in line:
                parts = line.split("Output:")
                if len(parts) >= 2:
                    raw = re.sub(r'\x1b\[[0-9;]*m', '', parts[-1].strip())
                    run_dir = Path(raw)
                    break

        if run_dir is None or not run_dir.exists():
            run_dir = find_latest_run(test_output_dir)

        if run_dir is None:
            print(_red(f"  [{num:02d}/{self.total}] FAIL  {label}  (no output dir)"))
            return False, None

        # Verify
        if checks:
            vpassed, vdetails = verify(run_dir, checks)
            self._completed += 1
            if vpassed:
                print(_green(f"  [{num:02d}/{self.total}] PASS  {label}  ({elapsed:.0f}s)  {run_dir.name}"))
            else:
                print(_red(f"  [{num:02d}/{self.total}] FAIL  {label}  ({elapsed:.0f}s)  {run_dir.name}"))
                for d in vdetails:
                    print(_red(f"         {d}"))
            return vpassed, run_dir
        else:
            self._completed += 1
            print(_green(f"  [{num:02d}/{self.total}] PASS  {label}  ({elapsed:.0f}s)  {run_dir.name}"))
            return True, run_dir

    # ── Standalone tests ──

    def test_01_basic(self) -> bool:
        passed, rd = self._run(
            1, "01_basic.yaml",
            extra_args=["--instance", INSTANCE_UNRESOLVED],
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["total_instances", "resolved_count", "unresolved_count"],
                "stats_values": {"total_instances": 1},
            },
        )
        self.run_dirs[1] = rd
        return passed

    def test_02_skip_learn(self) -> bool:
        passed, rd = self._run(
            2, "02_skip_learn.yaml",
            extra_args=["--instance", INSTANCE_UNRESOLVED],
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_values": {"total_instances": 1},
            },
        )
        self.run_dirs[2] = rd
        return passed

    def test_03_concurrent(self) -> bool:
        passed, rd = self._run(
            3, "03_concurrent.yaml",
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_gte": {"total_instances": 2},
            },
        )
        self.run_dirs[3] = rd
        return passed

    def test_04_two_phase(self) -> bool:
        passed, rd = self._run(
            4, "04_two_phase.yaml",
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["train_phase", "val_baseline_phase", "val_skillbook_phase", "summary"],
            },
        )
        self.run_dirs[4] = rd
        return passed

    def test_05_two_phase_global(self) -> bool:
        passed, rd = self._run(
            5, "05_two_phase_global.yaml",
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["train_phase", "val_baseline_phase", "val_skillbook_phase"],
            },
        )
        self.run_dirs[5] = rd
        return passed

    def test_06_iterate_repos(self) -> bool:
        passed, rd = self._run(
            6, "06_iterate_repos.yaml",
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["mode", "repos", "train_phase", "val_baseline_phase", "val_skillbook_phase"],
                "run_dir_files": [
                    "statistics_per_repo/django__django.json",
                    "statistics_per_repo/astropy__astropy.json",
                ],
            },
        )
        self.run_dirs[6] = rd
        return passed

    # ── Data-dependent tests ──

    def test_07_resume(self) -> bool:
        rd_01 = self.run_dirs.get(1)
        if not rd_01:
            print(_yellow(f"  [07/{self.total}] SKIP  resume  (depends on 01)"))
            return True

        passed, rd = self._run(
            7, "07_resume.yaml",
            extra_args=["--instance", INSTANCE_UNRESOLVED, "--resume-dir", str(rd_01)],
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["resumed_complete_count"],
            },
        )
        self.run_dirs[7] = rd
        return passed

    def test_08_baseline_reuse(self) -> bool:
        rd_04 = self.run_dirs.get(4)
        if not rd_04:
            print(_yellow(f"  [08/{self.total}] SKIP  baseline reuse  (depends on 04)"))
            return True

        passed, rd = self._run(
            8, "08_baseline_reuse.yaml",
            overrides={"experiment": {"baseline_run_dir": str(rd_04)}},
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["train_phase", "val_baseline_phase", "val_skillbook_phase"],
            },
        )
        self.run_dirs[8] = rd
        return passed

    def test_09_distillation(self) -> bool:
        if not TEACHER_TRAJS_DIR.exists():
            print(_yellow(f"  [09/{self.total}] SKIP  distillation  (no teacher trajs)"))
            return True

        passed, rd = self._run(
            9, "09_distillation.yaml",
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["train_phase", "val_baseline_phase", "val_skillbook_phase"],
                "stats_nested": {"train_phase.teacher_trajs_found": 1},
            },
        )
        self.run_dirs[9] = rd
        return passed

    def test_10_validation_only(self) -> bool:
        rd_06 = self.run_dirs.get(6)
        if not rd_06:
            print(_yellow(f"  [10/{self.total}] SKIP  validation only  (depends on 06)"))
            return True

        passed, rd = self._run(
            10, "10_validation_only.yaml",
            overrides={"experiment": {"skillbook_source_dir": str(rd_06)}},
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["mode", "repos", "val_baseline_phase", "val_skillbook_phase"],
            },
        )
        self.run_dirs[10] = rd
        return passed

    # ── Standalone tests (continued) ──

    # Retrieval types exercised end-to-end: (config, expected retriever class name).
    # "SkillRetriever" is the LLM two-stage filter+rank retriever.
    RETRIEVAL_TYPES = [
        ("11_retrieval_llm.yaml", "SkillRetriever"),
        ("11_retrieval_bm25.yaml", "BM25Retriever"),
        ("11_retrieval_embedding.yaml", "EmbeddingRetriever"),
        ("11_retrieval_random.yaml", "RandomRetriever"),
    ]

    def test_11_retrieval(self) -> bool:
        """Run each retrieval type end-to-end and verify retrieve() actually ran.

        Per type we assert:
          - statistics.json has a ``retrieval`` block with the expected ``type``
          - ``instances_retrieved + instances_no_change >= 1``: the retriever object
            was built AND its retrieve() ran against a real skillbook (filtered or
            unchanged), not silently skipped — the bug the old single-config test missed.
        Types run sequentially to avoid GPU/LLM contention (embedding model load vs.
        concurrent vLLM calls). Each config targets the never-resolved instance so
        force_learn adds skills on iter_0 and retrieval fires on iter_1.
        """
        all_passed = True
        rd = None
        for config_name, expected_type in self.RETRIEVAL_TYPES:
            passed, rd = self._run(
                11, config_name,
                extra_args=["--instance", INSTANCE_UNRESOLVED],
                checks={
                    "files_exist": ["statistics.json", "config.json"],
                    "stats_keys": ["total_instances", "retrieval"],
                    "stats_nested": {
                        "retrieval.enabled": True,
                        "retrieval.type": expected_type,
                        "retrieval.top_k": 1,
                    },
                    "stats_sum_gte": {
                        "retrieval.instances_retrieved,retrieval.instances_no_change": 1,
                    },
                },
            )
            all_passed = all_passed and passed
        self.run_dirs[11] = rd
        return all_passed


# ── Main ────────────────────────────────────────────────────────────────

ALL_TESTS = {
    1: "test_01_basic",
    2: "test_02_skip_learn",
    3: "test_03_concurrent",
    4: "test_04_two_phase",
    5: "test_05_two_phase_global",
    6: "test_06_iterate_repos",
    7: "test_07_resume",
    8: "test_08_baseline_reuse",
    9: "test_09_distillation",
    10: "test_10_validation_only",
    11: "test_11_retrieval",
}

STANDALONE = {1, 2, 3, 4, 5, 6, 9, 11}
DEPENDENT = {7, 8, 10}


def main():
    parser = argparse.ArgumentParser(description="End-to-end smoke tests for all experiment modes")
    parser.add_argument("--only", nargs="+", type=int, help="Run specific tests by number (e.g., --only 01 04)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Stream subprocess output in real-time")
    parser.add_argument("--keep", action="store_true", help="Keep _data/ output (default: clean up)")
    args = parser.parse_args()

    # Determine which tests to run
    if args.only:
        test_nums = args.only
    else:
        test_nums = sorted(ALL_TESTS.keys())

    total = len(test_nums)
    print(_bold(f"ACE-SWE Smoke Tests: {total} tests"))
    print(_dim(f"Output dir: {DATA_DIR}"))
    print(_dim(f"Config dir: {CONFIGS_DIR}"))

    if not CONFIGS_DIR.exists():
        print(_red(f"Config dir not found: {CONFIGS_DIR}"))
        sys.exit(1)

    test = SmokeTest(verbose=args.verbose, keep=args.keep, total=total)
    t_start = time.time()

    standalone_nums = [n for n in test_nums if n in STANDALONE]
    dependent_nums = [n for n in test_nums if n in DEPENDENT]

    # Phase 1: Run standalone tests in parallel
    if standalone_nums:
        print(_bold(f"\nPhase 1: {len(standalone_nums)} standalone tests (parallel)"))
        with ThreadPoolExecutor(max_workers=len(standalone_nums)) as executor:
            futures = {}
            for num in standalone_nums:
                method = getattr(test, ALL_TESTS[num])
                futures[executor.submit(method)] = num
            done = 0
            for future in as_completed(futures):
                num = futures[future]
                done += 1
                try:
                    test.results[num] = future.result()
                except Exception as e:
                    print(_red(f"  [{num:02d}/{total}] EXCEPTION: {e}"))
                    test.results[num] = False
                if done < len(standalone_nums):
                    remaining = len(standalone_nums) - done
                    print(_dim(f"         Phase 1: {done}/{len(standalone_nums)} done, {remaining} running..."))

    # Phase 2: Run dependent tests sequentially
    if dependent_nums:
        print(_bold(f"\nPhase 2: {len(dependent_nums)} dependent tests (sequential)"))
        for num in dependent_nums:
            method = getattr(test, ALL_TESTS[num])
            test.results[num] = method()

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print(_bold("Summary"))
    print(f"{'='*60}")

    passed_count = 0
    failed_count = 0
    skipped_count = 0
    for num in test_nums:
        if num not in test.results:
            continue
        result = test.results[num]
        label = ALL_TESTS[num].replace("test_", "").replace("_", " ")
        if result is True:
            print(_green(f"  {num:02d} {label}: PASS"))
            passed_count += 1
        elif result is False:
            print(_red(f"  {num:02d} {label}: FAIL"))
            failed_count += 1
        else:
            print(_yellow(f"  {num:02d} {label}: SKIP"))
            skipped_count += 1

    print(f"\n{passed_count} passed, {failed_count} failed, {skipped_count} skipped ({elapsed:.0f}s)")

    # Cleanup
    if not args.keep and DATA_DIR.exists():
        print(_dim(f"Cleaning up {DATA_DIR}..."))
        shutil.rmtree(DATA_DIR, ignore_errors=True)

    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
