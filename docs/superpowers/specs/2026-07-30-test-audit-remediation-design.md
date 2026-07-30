# Test Audit Remediation Design

Date: 2026-07-30

## Goal

Make experiment configuration and run verdicts truthful: disabled deduplication must stay
disabled, infrastructure failures must not be learned from or reported as ordinary unresolved
attempts, and the E2E harness must never turn skipped, stale, or infrastructure-broken runs into
PASS.

## Scope

This change covers three related areas:

1. Deduplication configuration parsing and immutability.
2. Infrastructure-error propagation from the agent through experiment statistics and strict CLI
   exit behavior.
3. E2E verdict integrity plus deterministic regression tests for the import-order failure and the
   new result contracts.

Building missing SWE-bench images, changing registry credentials, upgrading dependencies, and
redesigning retry/backoff policy are out of scope. Missing images are reported as BLOCKED by the
test harness.

## Design Decisions

### Deduplication configuration

`LearnPhase` treats `deduplication.enabled` as the authoritative switch. It copies the supplied
mapping before removing control-only keys, so it never mutates the resolved experiment config.

- Missing config or `enabled: false`: `dedup_manager` remains `None`; no embedding model is loaded.
- `enabled: true`: remove `enabled` and `embedding_device` from the copied mapping, construct the
  ACE `DeduplicationConfig`, and load the shared embedding model on the requested device.
- Older non-empty mappings without `enabled` remain disabled. The repository's documented config
  already carries the explicit switch, and implicit activation is the behavior being removed.

Regression tests prove disabled/enabled behavior, input immutability, and preservation of the
device across repeated initializations.

### Infrastructure-error propagation

An agent exception other than the existing terminal `ContextWindowExceeded` condition is classified
as `error_kind="infrastructure"`. `AgentResult` and `PredictResult` carry this optional field, and
trajectory metadata persists it.

`InstanceResult` gains:

- `status`: `completed` or `infrastructure_error`;
- `infrastructure_error`: optional diagnostic string.

When prediction returns an infrastructure error:

1. Persist the prediction trajectory as today.
2. Record an `IterationResult` whose `evaluate_result` and `learn_result` are `None`.
3. Mark the instance as `infrastructure_error`.
4. Stop further attempts for that instance.
5. Do not invoke Evaluate or Learn.
6. Continue processing other instances.

Normal unresolved outcomes such as `LimitsExceeded`, `ContextWindowExceeded`, and submitted but
failing patches remain task outcomes and retain current evaluation/learning semantics.

Experiment statistics gain:

- `infrastructure_error_count`;
- `infrastructure_error_ids`;
- `status="degraded"` when no outer exception occurred but at least one instance had an
  infrastructure error.

Infrastructure-error instances are not duplicated in `unresolved_ids`; they form a disjoint third
bucket beside resolved and unresolved instances. Two-phase and validation phase summaries propagate
the same counters.

### Strict CLI behavior

The full experiment functions return their statistics instead of discarding them. `--strict`
preserves the default long-run behavior—other instances continue—but exits non-zero after artifacts
and statistics are written when the final status is not `completed`.

The E2E harness always invokes the project CLI with `--strict`. Normal experiment invocations remain
backward compatible unless users opt into strict mode.

### E2E verdict model

`scripts/test_modes.py` replaces boolean test results with a `Verdict` enum:

- `PASS`: the subprocess and all mode contracts passed;
- `FAIL`: code or contract failure;
- `SKIP`: an optional scenario was not requested or intentionally unavailable;
- `BLOCKED`: an external prerequisite such as a Docker image is missing.

Required test cases that end as SKIP or BLOCKED make the harness exit non-zero. A summary never
prints either state as PASS.

Verification changes:

- `stats_values` means exact typed equality.
- `stats_gte` and `stats_nested_gte` express lower bounds explicitly.
- `stats_nested` compares values of every type, including strings and booleans.
- The harness snapshots `run_*` directories before launch and accepts only a fresh directory
  created by that subprocess.
- Trajectory inspection rejects `exit_status=error` and records Docker/image failures as BLOCKED.
- Dependent scenarios return SKIP/BLOCKED rather than `True`.
- Resume requires a positive resumed count and baseline reuse requires a positive reuse count.
- The retriever class name is compared exactly.
- Temporary YAML cleanup runs in `finally`, including timeout paths.

An image preflight checks the exact configured instances before a Docker smoke run. Missing images
produce BLOCKED without pretending the mode ran successfully.

## Test Design

### Unit tests

- `test_phases.py`: dedup disabled, enabled, non-mutating input, repeated device handling.
- `test_main_loop.py`: infrastructure error skips Evaluate and Learn, stops that instance, continues
  other instances, and writes degraded statistics with disjoint outcome buckets.
- `test_commands.py`: `--strict` handling and returned statistics.
- New `test_test_modes.py`: exact/nested comparisons, fresh-run selection, error-trajectory
  classification, dependent SKIP/BLOCKED, and process exit policy.
- `test_miniswe_agent.py`: import LiteLLM before fake module installation and add a cold-process
  import smoke test so the module passes both alone and in the full suite.

### Integration contracts

Duplicate reachability checks are consolidated around reusable helpers. The assertions cover exact
non-empty content and the served model when the response exposes it. A required tool-call probe and
a minimal ACE structured-output probe remain integration-marked so unit runs make no network calls.

### Verification

The implementation is accepted when:

1. The full non-integration suite passes.
2. `test_miniswe_agent.py` passes in an isolated process.
3. Dedup disabled does not instantiate a manager or load a model, and the source mapping is
   unchanged.
4. A synthetic infrastructure failure invokes neither Evaluate nor Learn and produces
   `status=degraded`.
5. Strict CLI exits non-zero only after statistics are saved for a degraded/interrupted run.
6. Synthetic harness fixtures prove PASS, FAIL, SKIP, and BLOCKED are distinct and stale runs are
   rejected.
7. A Docker E2E run is attempted only when image preflight succeeds; otherwise its verdict is
   BLOCKED.

## Compatibility

Existing result JSON remains readable because all new fields are additive. Default CLI behavior
continues processing and returns normally; only `--strict` changes the exit code. Existing configs
with explicit `deduplication.enabled` gain the documented behavior. No dependency versions or
external service settings change.
