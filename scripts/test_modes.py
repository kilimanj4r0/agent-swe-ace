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
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.loader import deep_merge  # noqa: E402

CONFIGS_DIR = PROJECT_ROOT / "configs" / "test"
DATA_DIR = PROJECT_ROOT / "_data"
TEACHER_TRAJS_DIR = PROJECT_ROOT / "data" / "teacher_trajs" / "v1.16_opus45_verified"

# Test instances (from analysis of 20 completed runs)
INSTANCE_RESOLVED = "django__django-16527"    # resolved 95% of the time
INSTANCE_UNRESOLVED = "django__django-15695"  # never resolved in 20 runs


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    BLOCKED = "BLOCKED"


@dataclass
class VerificationResult:
    verdict: Verdict
    details: list[str]


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


def find_fresh_run(
    output_dir: Path,
    existing: set[Path],
    started_at: float,
) -> Path | None:
    """Return the newest run created by the current command only."""
    if not output_dir.exists():
        return None
    candidates = [
        run
        for run in output_dir.glob("run_*")
        if run not in existing and run.stat().st_mtime >= started_at
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


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
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "src.cli.commands",
        "--config",
        str(config_path),
        "--strict",
    ]
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


def resolve_instance_ids(
    config: dict,
    extra_args: list[str] | None = None,
    dataset_loader=None,
) -> list[str]:
    """Resolve the exact instances a config will launch."""
    extra_args = extra_args or []
    for index, arg in enumerate(extra_args):
        if arg == "--instance" and index + 1 < len(extra_args):
            return [extra_args[index + 1]]
        if arg.startswith("--instance="):
            return [arg.split("=", 1)[1]]

    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    benchmark = config.get("benchmark", {})
    instances = list(
        dataset_loader(
            benchmark["dataset"],
            split=benchmark.get("split", "test"),
        )
    )
    max_instances = benchmark.get("max_instances")
    if max_instances:
        instances = instances[:max_instances]

    excluded = set(benchmark.get("exclude_instances", []))
    if excluded:
        instances = [
            instance
            for instance in instances
            if instance["instance_id"] not in excluded
        ]

    repos = benchmark.get("filter_repos") or benchmark.get("iterate_repos")
    if repos:
        repo_set = set(repos)
        instances = [
            instance
            for instance in instances
            if instance.get("repo") in repo_set
        ]
    return [instance["instance_id"] for instance in instances]


def check_docker_images(
    instance_ids: list[str],
    namespace: str | None,
    run_command=subprocess.run,
) -> VerificationResult:
    """Check the exact local image name for every selected instance."""
    prefix = f"{namespace.rstrip('/')}/" if namespace else ""
    missing = []
    for instance_id in instance_ids:
        image = f"{prefix}sweb.eval.x86_64.{instance_id}:latest"
        try:
            completed = run_command(
                ["docker", "image", "inspect", image],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return VerificationResult(
                Verdict.BLOCKED,
                [f"Docker image preflight unavailable: {exc}"],
            )
        if completed.returncode != 0:
            missing.append(image)

    if missing:
        return VerificationResult(
            Verdict.BLOCKED,
            [f"Missing Docker image: {image}" for image in missing],
        )
    return VerificationResult(Verdict.PASS, [])


def combine_verdicts(verdicts: list[Verdict]) -> Verdict:
    """Combine sub-checks using deterministic severity ordering."""
    for verdict in (Verdict.FAIL, Verdict.BLOCKED, Verdict.SKIP):
        if verdict in verdicts:
            return verdict
    return Verdict.PASS


# ── Verification ────────────────────────────────────────────────────────

def verify(run_dir: Path, checks: dict) -> VerificationResult:
    """Verify a fresh run using exact, typed output contracts."""
    failures = []
    blockers = []
    missing = object()

    def fail(message):
        failures.append(message)

    def block(message):
        blockers.append(message)

    def nested_value(mapping, dotkey):
        value = mapping
        for part in dotkey.split("."):
            if not isinstance(value, dict) or part not in value:
                return missing
            value = value[part]
        return value

    def exact(label, actual, expected):
        if type(actual) is not type(expected) or actual != expected:
            fail(f"{label} = {actual!r}, expected exact {expected!r}")

    def gte(label, actual, expected):
        numeric = (int, float)
        if (
            isinstance(actual, bool)
            or isinstance(expected, bool)
            or not isinstance(actual, numeric)
            or not isinstance(expected, numeric)
        ):
            fail(f"{label} = {actual!r}, expected numeric >= {expected!r}")
        elif actual < expected:
            fail(f"{label} = {actual}, expected >= {expected}")

    bm_dir = find_benchmark_dir(run_dir)
    check_dir = bm_dir or run_dir

    for filename in checks.get("files_exist", []):
        if not (run_dir / filename).exists():
            fail(f"MISSING file: {filename}")
    for dirname in checks.get("dirs_exist", []):
        if not (check_dir / dirname).exists():
            fail(f"MISSING dir: {dirname}")
    for dirname in checks.get("dirs_absent", []):
        if (check_dir / dirname).exists():
            fail(f"UNEXPECTED dir present: {dirname}")
    for filename in checks.get("files_absent", []):
        if (check_dir / filename).exists():
            fail(f"UNEXPECTED file present: {filename}")
    for filename in checks.get("run_dir_files", []):
        if not (run_dir / filename).exists():
            fail(f"MISSING run_dir file: {filename}")

    stats_checks = (
        "stats_keys",
        "stats_values",
        "stats_gte",
        "stats_nested",
        "stats_nested_gte",
        "stats_sum_gte",
    )
    stats_path = run_dir / "statistics.json"
    stats = None
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"INVALID statistics.json: {exc}")
    elif any(checks.get(key) for key in stats_checks):
        fail("MISSING statistics.json")

    if stats is not None:
        for key in checks.get("stats_keys", []):
            if key not in stats:
                fail(f"MISSING stats key: {key}")
        for key, expected in checks.get("stats_values", {}).items():
            if key not in stats:
                fail(f"MISSING stats value: {key} (expected {expected!r})")
            else:
                exact(f"stats[{key}]", stats[key], expected)
        for key, expected in checks.get("stats_gte", {}).items():
            if key not in stats:
                fail(f"MISSING stats value: {key} (expected >= {expected})")
            else:
                gte(f"stats[{key}]", stats[key], expected)
        for dotkey, expected in checks.get("stats_nested", {}).items():
            actual = nested_value(stats, dotkey)
            if actual is missing:
                fail(f"MISSING nested stats: {dotkey} (expected {expected!r})")
            else:
                exact(f"stats[{dotkey}]", actual, expected)
        for dotkey, expected in checks.get("stats_nested_gte", {}).items():
            actual = nested_value(stats, dotkey)
            if actual is missing:
                fail(f"MISSING nested stats: {dotkey} (expected >= {expected})")
            else:
                gte(f"stats[{dotkey}]", actual, expected)
        for dotkeys, threshold in checks.get("stats_sum_gte", {}).items():
            values = []
            for dotkey in (item.strip() for item in dotkeys.split(",")):
                actual = nested_value(stats, dotkey)
                if actual is missing:
                    fail(f"MISSING nested stats for sum: {dotkey}")
                    break
                if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                    fail(f"stats[{dotkey}] = {actual!r}, expected numeric")
                    break
                values.append(actual)
            else:
                gte(f"stats sum[{dotkeys}]", sum(values), threshold)

        if (
            stats.get("status") == "degraded"
            and stats.get("infrastructure_error_count", 0) > 0
        ):
            block(
                "Run degraded by infrastructure errors: "
                f"{stats.get('infrastructure_error_ids', [])}"
            )
        elif stats.get("status") == "interrupted":
            fail(f"Run interrupted: {stats.get('error', 'unknown error')}")

    trajectory_paths = sorted(run_dir.glob("**/trajectories/**/*.json"))
    if checks.get("require_trajectories") and not trajectory_paths:
        fail("MISSING trajectories")
    for trajectory_path in trajectory_paths:
        try:
            trajectory = json.loads(trajectory_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"INVALID trajectory {trajectory_path}: {exc}")
            continue
        info = trajectory.get("info", {})
        exit_status = str(info.get("exit_status", ""))
        normalized_status = exit_status.lower()
        messages = trajectory.get("messages", [])
        if normalized_status == "error":
            error_text = str(
                info.get("error")
                or info.get("infrastructure_error")
                or "unknown agent error"
            )
            infrastructure_terms = (
                "docker",
                "image",
                "environment",
                "container",
                "daemon",
            )
            if (
                info.get("error_kind") == "infrastructure"
                or any(term in error_text.lower() for term in infrastructure_terms)
            ):
                block(f"{trajectory_path}: {error_text}")
            else:
                fail(f"{trajectory_path}: agent error: {error_text}")
        elif normalized_status not in {
            "submitted",
            "limitsexceeded",
            "contextwindowexceeded",
            "completed",
        }:
            fail(f"{trajectory_path}: unexpected exit_status={exit_status!r}")

        if checks.get("require_nonempty_trajectory") and not messages:
            fail(f"{trajectory_path}: empty trajectory")

    if failures:
        return VerificationResult(Verdict.FAIL, failures + blockers)
    if blockers:
        return VerificationResult(Verdict.BLOCKED, blockers)
    return VerificationResult(Verdict.PASS, [])


# ── Test Definitions ────────────────────────────────────────────────────

class SmokeTest:
    def __init__(self, verbose=False, keep=False, total=10):
        self.verbose = verbose
        self.keep = keep
        self.total = total
        self.results = {}   # test_num -> Verdict
        self.run_dirs = {}  # test_num -> run_dir
        self._completed = 0

    def _run(self, num: int, config_name: str, extra_args: list[str] = None,
             overrides: dict = None, checks: dict = None) -> tuple[Verdict, Path | None]:
        """Run a single test config and return its explicit verdict."""
        config_path = CONFIGS_DIR / config_name
        if not config_path.exists():
            print(_red(f"  [{num:02d}/{self.total}] CONFIG NOT FOUND: {config_path}"))
            return Verdict.FAIL, None

        # Each test gets its own output subdir to avoid parallel collisions
        test_slug = config_name.replace(".yaml", "")
        test_output_dir = DATA_DIR / test_slug
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # Merge output dir override with any other overrides
        all_overrides = {"output": {"dir": str(test_output_dir)}}
        if overrides:
            all_overrides = deep_merge(all_overrides, overrides)

        label = config_name.replace(".yaml", "").replace("_", " ")
        print(f"  [{num:02d}/{self.total}] {label}...", flush=True)
        tmp_config = None
        try:
            tmp_config = inject_overrides(config_path, all_overrides)
            effective_config = deep_merge(
                load_yaml(PROJECT_ROOT / "config.yaml"),
                load_yaml(tmp_config),
            )

            if effective_config.get("environment", {}).get("type") == "docker":
                try:
                    instance_ids = resolve_instance_ids(
                        effective_config, extra_args
                    )
                except Exception as exc:
                    preflight = VerificationResult(
                        Verdict.BLOCKED,
                        [f"Could not resolve Docker instances: {exc}"],
                    )
                else:
                    preflight = check_docker_images(
                        instance_ids,
                        effective_config.get("environment", {}).get("namespace"),
                    )
                if preflight.verdict is not Verdict.PASS:
                    print(
                        _yellow(
                            f"  [{num:02d}/{self.total}] BLOCKED  {label}  "
                            "(Docker image preflight)"
                        )
                    )
                    for detail in preflight.details:
                        print(_yellow(f"         {detail}"))
                    return preflight.verdict, None

            existing_runs = set(test_output_dir.glob("run_*"))
            started_at = time.time()
            t0 = started_at
            try:
                rc, output = run_config(
                    tmp_config, extra_args, verbose=self.verbose
                )
            except subprocess.TimeoutExpired as exc:
                elapsed = time.time() - t0
                print(
                    _red(
                        f"  [{num:02d}/{self.total}] FAIL  {label}  "
                        f"({elapsed:.0f}s, timeout after {exc.timeout}s)"
                    )
                )
                return Verdict.FAIL, None
            elapsed = time.time() - t0

            run_dir = find_fresh_run(
                test_output_dir, existing_runs, started_at
            )
            if run_dir is None:
                print(
                    _red(
                        f"  [{num:02d}/{self.total}] FAIL  {label}  "
                        "(no fresh output dir)"
                    )
                )
                return Verdict.FAIL, None

            verification = verify(run_dir, checks or {})
            if rc != 0 and verification.verdict is Verdict.PASS:
                verification = VerificationResult(
                    Verdict.FAIL,
                    [f"Command exited {rc} without a classified run failure"],
                )

            self._completed += 1
            if verification.verdict is Verdict.PASS:
                print(
                    _green(
                        f"  [{num:02d}/{self.total}] PASS  {label}  "
                        f"({elapsed:.0f}s)  {run_dir.name}"
                    )
                )
            elif verification.verdict is Verdict.BLOCKED:
                print(
                    _yellow(
                        f"  [{num:02d}/{self.total}] BLOCKED  {label}  "
                        f"({elapsed:.0f}s)  {run_dir.name}"
                    )
                )
            else:
                print(
                    _red(
                        f"  [{num:02d}/{self.total}] FAIL  {label}  "
                        f"({elapsed:.0f}s)  {run_dir.name}"
                    )
                )
            for detail in verification.details:
                color = _red if verification.verdict is Verdict.FAIL else _yellow
                print(color(f"         {detail}"))
            if rc != 0 and output and not self.verbose:
                for line in output.strip().split("\n")[-3:]:
                    print(_dim(f"         {line}"))
            return verification.verdict, run_dir
        finally:
            if tmp_config is not None:
                try:
                    tmp_config.unlink()
                except OSError:
                    pass

    # ── Standalone tests ──

    def test_01_basic(self) -> Verdict:
        passed, rd = self._run(
            1, "01_basic.yaml",
            extra_args=["--instance", INSTANCE_UNRESOLVED],
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["total_instances", "resolved_count", "unresolved_count"],
                "stats_values": {"total_instances": 1},
            },
        )
        self.run_dirs[1] = rd if passed is Verdict.PASS else None
        return passed

    def test_02_skip_learn(self) -> Verdict:
        passed, rd = self._run(
            2, "02_skip_learn.yaml",
            extra_args=["--instance", INSTANCE_UNRESOLVED],
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_values": {"total_instances": 1},
            },
        )
        self.run_dirs[2] = rd if passed is Verdict.PASS else None
        return passed

    def test_03_concurrent(self) -> Verdict:
        passed, rd = self._run(
            3, "03_concurrent.yaml",
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_gte": {"total_instances": 2},
            },
        )
        self.run_dirs[3] = rd if passed is Verdict.PASS else None
        return passed

    def test_04_two_phase(self) -> Verdict:
        passed, rd = self._run(
            4, "04_two_phase.yaml",
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["train_phase", "val_baseline_phase", "val_skillbook_phase", "summary"],
            },
        )
        self.run_dirs[4] = rd if passed is Verdict.PASS else None
        return passed

    def test_05_two_phase_global(self) -> Verdict:
        passed, rd = self._run(
            5, "05_two_phase_global.yaml",
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["train_phase", "val_baseline_phase", "val_skillbook_phase"],
            },
        )
        self.run_dirs[5] = rd if passed is Verdict.PASS else None
        return passed

    def test_06_iterate_repos(self) -> Verdict:
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
        self.run_dirs[6] = rd if passed is Verdict.PASS else None
        return passed

    # ── Data-dependent tests ──

    def test_07_resume(self) -> Verdict:
        rd_01 = self.run_dirs.get(1)
        if not rd_01:
            print(_yellow(f"  [07/{self.total}] SKIP  resume  (depends on 01)"))
            return Verdict.SKIP

        passed, rd = self._run(
            7, "07_resume.yaml",
            extra_args=["--instance", INSTANCE_UNRESOLVED, "--resume-dir", str(rd_01)],
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_gte": {"resumed_complete_count": 1},
            },
        )
        self.run_dirs[7] = rd if passed is Verdict.PASS else None
        return passed

    def test_08_baseline_reuse(self) -> Verdict:
        rd_04 = self.run_dirs.get(4)
        if not rd_04:
            print(_yellow(f"  [08/{self.total}] SKIP  baseline reuse  (depends on 04)"))
            return Verdict.SKIP

        passed, rd = self._run(
            8, "08_baseline_reuse.yaml",
            overrides={"experiment": {"baseline_run_dir": str(rd_04)}},
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["train_phase", "val_baseline_phase", "val_skillbook_phase"],
                "stats_nested_gte": {"train_phase.reused_from_baseline": 1},
            },
        )
        self.run_dirs[8] = rd if passed is Verdict.PASS else None
        return passed

    def test_09_distillation(self) -> Verdict:
        if not TEACHER_TRAJS_DIR.exists():
            print(_yellow(f"  [09/{self.total}] BLOCKED  distillation  (no teacher trajs)"))
            return Verdict.BLOCKED

        passed, rd = self._run(
            9, "09_distillation.yaml",
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["train_phase", "val_baseline_phase", "val_skillbook_phase"],
                "stats_nested_gte": {"train_phase.teacher_trajs_found": 1},
            },
        )
        self.run_dirs[9] = rd if passed is Verdict.PASS else None
        return passed

    def test_10_validation_only(self) -> Verdict:
        rd_06 = self.run_dirs.get(6)
        if not rd_06:
            print(_yellow(f"  [10/{self.total}] SKIP  validation only  (depends on 06)"))
            return Verdict.SKIP

        passed, rd = self._run(
            10, "10_validation_only.yaml",
            overrides={"experiment": {"skillbook_source_dir": str(rd_06)}},
            checks={
                "files_exist": ["statistics.json", "config.json"],
                "stats_keys": ["mode", "repos", "val_baseline_phase", "val_skillbook_phase"],
            },
        )
        self.run_dirs[10] = rd if passed is Verdict.PASS else None
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

    def test_11_retrieval(self) -> Verdict:
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
        verdicts = []
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
            verdicts.append(passed)
        combined = combine_verdicts(verdicts)
        self.run_dirs[11] = rd if combined is Verdict.PASS else None
        return combined


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
                    test.results[num] = Verdict.FAIL
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

    counts = {verdict: 0 for verdict in Verdict}
    for num in test_nums:
        if num not in test.results:
            continue
        result = test.results[num]
        label = ALL_TESTS[num].replace("test_", "").replace("_", " ")
        counts[result] += 1
        if result is Verdict.PASS:
            print(_green(f"  {num:02d} {label}: PASS"))
        elif result is Verdict.FAIL:
            print(_red(f"  {num:02d} {label}: FAIL"))
        elif result is Verdict.BLOCKED:
            print(_yellow(f"  {num:02d} {label}: BLOCKED"))
        else:
            print(_yellow(f"  {num:02d} {label}: SKIP"))

    print(
        f"\n{counts[Verdict.PASS]} passed, "
        f"{counts[Verdict.FAIL]} failed, "
        f"{counts[Verdict.SKIP]} skipped, "
        f"{counts[Verdict.BLOCKED]} blocked ({elapsed:.0f}s)"
    )

    # Cleanup
    if not args.keep and DATA_DIR.exists():
        print(_dim(f"Cleaning up {DATA_DIR}..."))
        shutil.rmtree(DATA_DIR, ignore_errors=True)

    requested_non_pass = any(
        test.results.get(num) is not Verdict.PASS for num in test_nums
    )
    sys.exit(1 if requested_non_pass else 0)


if __name__ == "__main__":
    main()
