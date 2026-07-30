# Test Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make deduplication configuration, infrastructure-error reporting, strict CLI behavior, and E2E verdicts match the behavior promised by configuration and test output.

**Architecture:** Preserve normal task outcomes, but carry infrastructure failures as a disjoint result state from `MiniSWEAgent` through prediction, instance aggregation, experiment statistics, and strict CLI exit handling. Keep E2E policy in `scripts/test_modes.py`, represented by a typed verdict enum and tested entirely with synthetic run artifacts before any Docker smoke run.

**Tech Stack:** Python 3.12, dataclasses, argparse, pathlib, pytest 9, unittest.mock, existing ACE and mini-swe-agent integrations.

## Global Constraints

- Default long experiments continue after an individual infrastructure failure.
- Infrastructure failures never invoke Evaluate or Learn.
- `--strict` exits non-zero only after statistics and artifacts have been saved.
- Missing Docker images are BLOCKED, never PASS.
- Deduplication is enabled only by explicit `deduplication.enabled: true`.
- The caller's configuration mapping is never mutated.
- No dependency versions, registry credentials, or external service settings change.
- Network and Docker calls remain excluded from the non-integration test suite.

---

### Task 1: Honor deduplication enablement without mutating configuration

**Files:**
- Modify: `src/phases/learn.py:62-108`
- Modify: `src/tests/test_phases.py`

**Interfaces:**
- Consumes: `LearnPhase(..., dedup_config: Optional[Dict[str, Any]])`.
- Produces: `LearnPhase.dedup_manager`, either `None` or an initialized `DeduplicationManager`.

- [ ] **Step 1: Write failing deduplication configuration tests**

Add tests that mock `DeduplicationConfig`, `DeduplicationManager`, and
`_get_shared_st_model`:

```python
def test_dedup_disabled_does_not_initialize_or_mutate_config(self, tmp_path):
    from phases.learn import LearnPhase

    source = {"enabled": False, "embedding_device": "cuda"}
    with patch("phases.learn.DeduplicationManager") as manager, \
         patch("phases.learn._get_shared_st_model") as load_model:
        phase = LearnPhase(Mock(), Mock(), tmp_path, dedup_config=source)

    assert phase.dedup_manager is None
    manager.assert_not_called()
    load_model.assert_not_called()
    assert source == {"enabled": False, "embedding_device": "cuda"}


def test_dedup_enabled_uses_clean_copy_and_preserves_source(self, tmp_path):
    from phases.learn import LearnPhase

    source = {
        "enabled": True,
        "embedding_device": "cuda",
        "similarity_threshold": 0.9,
    }
    detector = SimpleNamespace(_model_lock=threading.Lock(), _model=None)
    manager = SimpleNamespace(detector=detector)
    ace_config = SimpleNamespace(
        local_model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.9,
    )
    with patch("phases.learn.DeduplicationConfig", return_value=ace_config) as config_cls, \
         patch("phases.learn.DeduplicationManager", return_value=manager), \
         patch("phases.learn._get_shared_st_model", return_value=object()) as load_model:
        phase = LearnPhase(Mock(), Mock(), tmp_path, dedup_config=source)

    assert phase.dedup_manager is manager
    config_cls.assert_called_once_with(similarity_threshold=0.9)
    load_model.assert_called_once_with("all-MiniLM-L6-v2", "cuda")
    assert source["enabled"] is True
    assert source["embedding_device"] == "cuda"
```

- [ ] **Step 2: Run the tests and confirm the current behavior fails**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_phases.py -k dedup -v
```

Expected: disabled initialization and input immutability assertions fail.

- [ ] **Step 3: Implement explicit, non-mutating configuration parsing**

Replace truthiness-based initialization with:

```python
raw_dedup_config = dict(dedup_config or {})
dedup_enabled = bool(raw_dedup_config.pop("enabled", False))
if dedup_enabled:
    embedding_device = raw_dedup_config.pop("embedding_device", "cpu")
    cfg = DeduplicationConfig(**raw_dedup_config)
    ...
else:
    self.dedup_manager = None
```

Do not mutate `dedup_config`, and do not pass `enabled` or `embedding_device` to
`DeduplicationConfig`.

- [ ] **Step 4: Run focused and phase tests**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_phases.py -v
```

Expected: all phase tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/phases/learn.py src/tests/test_phases.py
git commit -m "fix: respect deduplication enabled flag"
```

---

### Task 2: Carry infrastructure failures through agent and instance results

**Files:**
- Modify: `src/agents/miniswe_agent.py:20-27,98-225`
- Modify: `src/phases/predict.py:14-24,108-155`
- Modify: `src/runners/main_loop.py:46-64,370-455,489-552`
- Modify: `src/tests/test_miniswe_agent.py`
- Modify: `src/tests/test_phases.py`
- Modify: `src/tests/test_main_loop.py`

**Interfaces:**
- Produces: `AgentResult.error_kind: Optional[str]`.
- Produces: `PredictResult.error_kind: Optional[str]`, persisted at
  `trajectory["info"]["error_kind"]`.
- Produces: `InstanceResult.status: str` and
  `InstanceResult.infrastructure_error: Optional[str]`.

- [ ] **Step 1: Add failing agent and prediction propagation tests**

Extend the existing exception tests:

```python
assert result.exit_status == "error"
assert result.error_kind == "infrastructure"
```

Add a `PredictPhase` test whose fake agent returns:

```python
AgentResult(
    exit_status="error",
    patch="",
    trajectory=[],
    error="docker create failed",
    error_kind="infrastructure",
)
```

Assert both `PredictResult.error_kind` and the saved trajectory metadata equal
`"infrastructure"`.

- [ ] **Step 2: Add failing sequential and concurrent loop tests**

For the sequential path:

```python
mock_predict.run.return_value = Mock(
    exit_status="error",
    error_kind="infrastructure",
    error="docker exit 125",
    patch="",
    trajectory=[],
)
result = loop.run_instance(instance)
assert result.status == "infrastructure_error"
assert result.infrastructure_error == "docker exit 125"
assert len(result.iterations) == 1
assert result.iterations[0].evaluate_result is None
mock_evaluate.run.assert_not_called()
mock_learn.run.assert_not_called()
```

Add the equivalent assertion through `_run_instance_concurrent_inner`. Also assert prediction
is not retried even when `max_attempts=3`.

- [ ] **Step 3: Run focused tests and confirm they fail**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_miniswe_agent.py \
  src/tests/test_phases.py \
  src/tests/test_main_loop.py \
  -k "infrastructure or exception or error_kind" -v
```

Expected: new fields are absent and Evaluate/Learn are still invoked.

- [ ] **Step 4: Add result fields and persist the classification**

Add optional `error_kind` fields to `AgentResult` and `PredictResult`. Set it to
`"infrastructure"` for mini-swe import failures and caught execution exceptions; leave it `None`
for successful and `ContextWindowExceeded` results.

In `PredictPhase.run`, add:

```python
if result.error_kind:
    info["error_kind"] = result.error_kind
```

and forward it into `PredictResult`.

- [ ] **Step 5: Short-circuit instance execution before Evaluate**

Add a private helper in `ExperimentLoop`:

```python
@staticmethod
def _is_infrastructure_error(predict_result) -> bool:
    return (
        getattr(predict_result, "exit_status", None) == "error"
        or getattr(predict_result, "error_kind", None) == "infrastructure"
    )
```

Immediately after prediction in both sequential and concurrent paths, append an
`IterationResult(..., evaluate_result=None)`, set:

```python
result.status = "infrastructure_error"
result.infrastructure_error = predict_result.error or "agent infrastructure error"
result.total_attempts = iteration + 1
```

and break/return without Evaluate or Learn. Add defaults
`status="completed"` and `infrastructure_error=None` to `InstanceResult`.

- [ ] **Step 6: Run the new result-propagation tests**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_miniswe_agent.py \
  src/tests/test_phases.py \
  src/tests/test_main_loop.py \
  -k "infrastructure or exception or error_kind" -v
```

Expected: all newly added result-propagation tests pass. The pre-existing isolated import-order
failure is fixed and checked separately in Task 5.

- [ ] **Step 7: Commit**

```bash
git add \
  src/agents/miniswe_agent.py \
  src/phases/predict.py \
  src/runners/main_loop.py \
  src/tests/test_miniswe_agent.py \
  src/tests/test_phases.py \
  src/tests/test_main_loop.py
git commit -m "fix: isolate infrastructure failures from learning"
```

---

### Task 3: Add degraded statistics and strict CLI exit behavior

**Files:**
- Modify: `src/runners/main_loop.py:667-977,1443-1668`
- Modify: `src/cli/commands.py:331-458,1299-1457,1480-1697`
- Modify: `src/tests/test_main_loop.py`
- Modify: `src/tests/test_commands.py`

**Interfaces:**
- Produces statistics keys:
  `infrastructure_error_count`, `infrastructure_error_ids`, and
  `status in {"completed", "degraded", "interrupted"}`.
- Produces: `run_full_experiment(config, args) -> dict`.
- Produces: `_run_iterate_repos(config, args, output_dir) -> dict`.
- Consumes: CLI flag `--strict`.

- [ ] **Step 1: Add failing statistics tests**

Create one resolved, one unresolved, and one infrastructure-error `InstanceResult` through mocked
phases. Assert:

```python
assert statistics["resolved_ids"] == ["resolved"]
assert statistics["unresolved_ids"] == ["unresolved"]
assert statistics["infrastructure_error_ids"] == ["infra"]
assert statistics["infrastructure_error_count"] == 1
assert statistics["status"] == "degraded"
assert set(statistics["resolved_ids"]).isdisjoint(statistics["infrastructure_error_ids"])
assert set(statistics["unresolved_ids"]).isdisjoint(statistics["infrastructure_error_ids"])
```

Add the same counters to `_run_val_pass` tests, including a concurrent worker exception.

- [ ] **Step 2: Add failing strict-mode CLI tests**

Extract a pure helper:

```python
def _strict_exit_code(statistics: dict, strict: bool) -> int:
    if not strict:
        return 0
    return 0 if statistics.get("status") == "completed" else 1
```

Test completed/degraded/interrupted statistics with strict on and off. Test that `main()` raises
`SystemExit(1)` only after the mocked `run_full_experiment` returns degraded statistics.

- [ ] **Step 3: Run the new tests and confirm failure**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_main_loop.py src/tests/test_commands.py \
  -k "infrastructure or degraded or strict" -v
```

- [ ] **Step 4: Implement disjoint aggregation**

Maintain `infrastructure_error_ids` beside resolved/unresolved collections. Centralize
classification:

```python
def _append_outcome(result, instance_id, resolved_ids, unresolved_ids, infra_ids):
    if result.status == "infrastructure_error":
        infra_ids.append(instance_id)
    elif result.final_resolved:
        resolved_ids.append(instance_id)
    else:
        unresolved_ids.append(instance_id)
```

Use it in sequential, concurrent, and validation paths. Unexpected worker exceptions create an
`InstanceResult(status="infrastructure_error", infrastructure_error=str(e))`.

Set `status="degraded"` when `error_info` is absent and any infrastructure-error bucket is
non-empty. Include infrastructure IDs returned by validation and train-evaluation phase summaries
in the top-level count, without duplicating IDs across phases. Preserve `status="interrupted"` for
outer exceptions.

- [ ] **Step 5: Return statistics and implement `--strict`**

Add:

```python
parser.add_argument(
    "--strict",
    action="store_true",
    help="Exit non-zero after saving outputs when the run is degraded or interrupted.",
)
```

Return `statistics` from `ExperimentLoop.run`, return `combined` from `_run_iterate_repos`, and
return the selected result from `run_full_experiment`. In `main()`:

```python
statistics = run_full_experiment(config, args)
exit_code = _strict_exit_code(statistics, args.strict)
if exit_code:
    raise SystemExit(exit_code)
```

Do not change predict/evaluate/learn-only command exit behavior.

Update `_aggregate_iterate_stats` so every repo completed means `completed`, any repo
`degraded`/`infrastructure_error` means `degraded`, and an outer exception remains
`interrupted`. The strict helper treats every non-`completed` aggregate as exit 1.

- [ ] **Step 6: Run loop and CLI tests**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_main_loop.py src/tests/test_commands.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/runners/main_loop.py src/cli/commands.py \
  src/tests/test_main_loop.py src/tests/test_commands.py
git commit -m "feat: report degraded experiment runs"
```

---

### Task 4: Replace boolean E2E results with strict verdicts

**Files:**
- Modify: `scripts/test_modes.py`
- Create: `src/tests/test_test_modes.py`

**Interfaces:**
- Produces: `Verdict(str, Enum)` with `PASS`, `FAIL`, `SKIP`, `BLOCKED`.
- Produces: `VerificationResult(verdict: Verdict, details: list[str])`.
- Produces: `verify(run_dir: Path, checks: dict) -> VerificationResult`.
- Produces: `find_fresh_run(output_dir: Path, existing: set[Path], started_at: float)`.

- [ ] **Step 1: Write synthetic verifier tests**

Load `scripts/test_modes.py` via `importlib.util.spec_from_file_location`. Build `tmp_path` run
fixtures containing `statistics.json` and benchmark/trajectory directories. Cover:

```python
assert verify(run_dir, {"stats_values": {"total_instances": 1}}).verdict is Verdict.PASS
assert verify(run_dir, {"stats_values": {"total_instances": 0}}).verdict is Verdict.FAIL
assert verify(run_dir, {"stats_nested": {"retrieval.type": "BM25Retriever"}}).verdict is Verdict.PASS
assert verify(run_dir, {"stats_nested": {"retrieval.type": "RandomRetriever"}}).verdict is Verdict.FAIL
```

Add boolean exactness (`True` must not compare as integer `1`), explicit nested-GTE behavior, and
missing statistics behavior.

- [ ] **Step 2: Write fresh-run and trajectory tests**

Test that a pre-existing `run_old` is rejected when no new directory appears. Test that only a
post-start `run_new` is returned. Create trajectory metadata with:

```json
{"info": {"exit_status": "error"}, "messages": []}
```

and assert it yields BLOCKED when the error text identifies Docker/image/environment failure, else
FAIL. `LimitsExceeded` remains eligible for PASS when the mode does not require a non-empty
trajectory.

- [ ] **Step 3: Run tests and confirm the old harness fails**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_test_modes.py -v
```

- [ ] **Step 4: Implement typed verdicts and strict comparisons**

Add:

```python
class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    BLOCKED = "BLOCKED"


@dataclass
class VerificationResult:
    verdict: Verdict
    details: list[str]
```

Make exact comparisons type-sensitive:

```python
if type(actual) is not type(expected) or actual != expected:
    fail(...)
```

Keep lower bounds only under `stats_gte` and `stats_nested_gte`.

- [ ] **Step 5: Require a fresh run and guaranteed cleanup**

Before launching:

```python
existing_runs = set(test_output_dir.glob("run_*"))
started_at = time.time()
```

After success, call `find_fresh_run`; never fall back to `find_latest_run`. Wrap temporary config
cleanup in `finally`, and convert `subprocess.TimeoutExpired` to FAIL with a diagnostic.

Always add `--strict` to the command assembled by `run_config`.

- [ ] **Step 6: Convert dependent cases and summary**

Return `Verdict.SKIP` or `Verdict.BLOCKED` from missing prerequisites, never `True`. Resume checks
must use:

```python
"stats_gte": {"resumed_complete_count": 1}
```

Baseline reuse must require `train_phase.reused_from_baseline >= 1`. Retrieval types remain exact
string checks. Summary counts all four verdicts and exits non-zero whenever any requested test is
not PASS.

- [ ] **Step 7: Add exact Docker image preflight**

Resolve configured/CLI instance IDs before launch and inspect local images with:

```bash
docker image inspect <namespace>/<instance-image>:latest
```

Expose the check as a small function whose command execution is injected/mocked in unit tests.
Missing images return BLOCKED without launching the subprocess.

- [ ] **Step 8: Run harness unit tests**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_test_modes.py -v
```

- [ ] **Step 9: Commit**

```bash
git add scripts/test_modes.py src/tests/test_test_modes.py
git commit -m "fix: make E2E verdicts strict"
```

---

### Task 5: Remove import-order masking and strengthen live LLM contracts

**Files:**
- Modify: `src/tests/test_miniswe_agent.py`
- Modify: `src/tests/test_llm_config.py`
- Modify: `src/tests/test_llm_health.py`

**Interfaces:**
- Unit suite performs no network calls.
- Integration suite uses one reusable request helper per role and checks model/content/tool/ACE
  contracts explicitly.

- [ ] **Step 1: Reproduce the isolated import-order failure**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_miniswe_agent.py -q
```

Expected before the fix: `1 failed, 8 passed`, with `KeyError: pydantic.root_model`.

- [ ] **Step 2: Stabilize the unit test import boundary**

Import `litellm` at module load before any fake `minisweagent` modules are installed. Reuse its
`ContextWindowExceededError` in the existing test. Add:

```python
def test_litellm_imports_in_cold_process():
    completed = subprocess.run(
        [sys.executable, "-c", "import litellm; print('ok')"],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
```

- [ ] **Step 3: Consolidate duplicated integration reachability assertions**

Retain unit creation/config tests in `test_llm_config.py`. In `test_llm_health.py`, use one
integration request per configured role and assert:

```python
assert info["content"].strip()
assert info["configured_model"] in info["response_model"]
```

when a response model is supplied. Add a required tool-call request that asserts
`finish_reason == "tool_calls"` and the expected function name. Add a minimal ACE structured-output
probe through `create_ace_client`, marked `integration`.

- [ ] **Step 4: Run isolated and non-integration tests**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_miniswe_agent.py -q
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests -m "not integration" -q
```

Expected: isolated module passes; full non-integration suite passes with the expanded item count.

- [ ] **Step 5: Run live integration tests when the endpoint preflight passes**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_llm_config.py src/tests/test_llm_health.py \
  -m integration -v -s
```

If the endpoint is unavailable, record the run as BLOCKED in the handoff; do not weaken assertions.

- [ ] **Step 6: Commit**

```bash
git add src/tests/test_miniswe_agent.py \
  src/tests/test_llm_config.py src/tests/test_llm_health.py
git commit -m "test: enforce deterministic LLM contracts"
```

---

### Task 6: Final regression and smoke validation

**Files:**
- Modify if counts changed: `README.md`
- Verify: all files changed by Tasks 1-5

**Interfaces:**
- Consumes all preceding result and verdict interfaces.
- Produces a clean branch with verified commits and no generated run data.

- [ ] **Step 1: Run formatting and static checks**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m ruff check \
  src/agents/miniswe_agent.py \
  src/phases/learn.py \
  src/phases/predict.py \
  src/runners/main_loop.py \
  src/cli/commands.py \
  scripts/test_modes.py \
  src/tests/test_phases.py \
  src/tests/test_main_loop.py \
  src/tests/test_commands.py \
  src/tests/test_miniswe_agent.py \
  src/tests/test_test_modes.py
git diff --check
```

- [ ] **Step 2: Run the full non-integration suite twice**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests -m "not integration" -q
PYTHONHASHSEED=42 /root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests -m "not integration" -q
```

- [ ] **Step 3: Run every modified test module alone**

Run:

```bash
for module in \
  src/tests/test_phases.py \
  src/tests/test_main_loop.py \
  src/tests/test_commands.py \
  src/tests/test_miniswe_agent.py \
  src/tests/test_test_modes.py
do
  /root/makharev/agent-swe-ace/.venv/bin/python -m pytest "$module" -q
done
```

- [ ] **Step 4: Run synthetic strict harness tests**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_test_modes.py -v
```

Confirm that stale runs, error trajectories, missing images, and skipped prerequisites are never
PASS.

- [ ] **Step 5: Run Docker smoke modes conditionally**

Run image preflight for all selected test instances. If every exact image is available:

```bash
PATH=/root/miniconda3/bin:$PATH \
  /root/makharev/agent-swe-ace/.venv/bin/python scripts/test_modes.py \
  --only 1 2 3 4 5 6 7 8 9 10 11 --keep --verbose
```

If any image is missing, do not pull or build it in this task; report the affected modes as BLOCKED
and preserve the strict unit evidence.

- [ ] **Step 6: Inspect branch scope and commit documentation updates**

Run:

```bash
git status --short
git diff main...HEAD --stat
git log --oneline --decorate main..HEAD
```

If README test counts or strict-mode usage became stale, update only those exact sections, run
`git diff --check`, and commit:

```bash
git add README.md
git commit -m "docs: document strict experiment validation"
```

No commit is needed when README remains accurate.
