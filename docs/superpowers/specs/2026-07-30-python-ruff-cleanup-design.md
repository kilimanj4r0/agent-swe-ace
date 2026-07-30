# Python Ruff Cleanup Design

**Date:** 2026-07-30  
**Branch:** `fix/test-audit-remediation`

## Objective

Make Ruff pass with zero diagnostics across every tracked `*.py` file while preserving runtime
behavior and test semantics. Jupyter notebooks are explicitly out of scope.

## Baseline

The repository contains 92 Python files with 241 Ruff diagnostics under the existing project
configuration:

- 184 safe automatic fixes;
- 24 fixes Ruff marks unsafe;
- 33 diagnostics without an automatic fix.

The enabled rules remain unchanged: `E`, `F`, `I`, and `W`, with `E501` ignored and Python 3.10 as
the target.

## Scope

Included:

- all tracked `*.py` files under `src/` and `scripts/`;
- root-level tracked Python files, if any;
- imports, whitespace, redundant f-strings, unused names, ambiguous local names, annotation-only
  imports, and multi-statement lines reported by the current Ruff configuration.

Excluded:

- all `*.ipynb` files;
- changes to the Ruff rules or ignore list;
- unrelated refactors, API changes, or behavior changes;
- live-service configuration changes.

## Implementation Strategy

### Safe fixes

Apply Ruff's standard safe fixes to the complete Python-file list. Review the diff before
continuing, with special attention to import ordering around intentional path setup and optional
dependency boundaries.

### Unsafe suggestions

Do not run `--unsafe-fixes` wholesale. Inspect and resolve all 24 suggestions manually:

- retain a right-hand-side expression if evaluating it can have a side effect;
- otherwise remove genuinely dead assignments;
- replace boolean equality checks only where values are intended to be booleans;
- replace the assigned lambda with an equivalent named local function;
- remove unused result bindings in tests while preserving the function calls.

### Manual diagnostics

Resolve the remaining 33 diagnostics explicitly:

- use `TYPE_CHECKING` or postponed annotations for `Skillbook` annotations without introducing
  eager optional-dependency imports;
- preserve intentional `sys.path` or environment initialization and use narrowly scoped `noqa`
  only when import order is semantically required;
- rename ambiguous variables without changing data flow;
- split multi-statement lines without altering control flow.

Broad file-level ignores are not allowed.

## Safety and Verification

After each logical batch, run Ruff over all tracked Python files. After Ruff reaches zero:

1. run the entire non-integration test suite;
2. repeat it with `PYTHONHASHSEED=42`;
3. run every modified test module in isolation;
4. run integration tests only if the configured endpoint preflight succeeds;
5. if the endpoint is unavailable, report integration as `BLOCKED` without weakening assertions;
6. run `git diff --check` and confirm a clean worktree after committing.

## Acceptance Criteria

- Ruff reports zero diagnostics for every tracked `*.py` file.
- No notebook is modified.
- The full non-integration suite passes twice.
- Modified test modules pass independently.
- Live integrations either pass or are explicitly reported as externally blocked.
- No Ruff rule is disabled to hide an existing diagnostic.
- The final branch contains no generated data or temporary files.
