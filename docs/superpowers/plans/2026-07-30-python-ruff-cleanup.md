# Python Ruff Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce Ruff diagnostics for every tracked Python source file to zero without changing Ruff's configured rule set or modifying notebooks.

**Architecture:** Apply Ruff's safe fixes first, then review and implement every unsafe/manual fix in small semantic groups. Preserve import-order side effects with narrow `noqa` markers, retain plotting calls whose return values are unused, and use type-checking-only imports for forward references. Validate the resulting source tree with Ruff, focused tests, the full non-integration suite under two hash seeds, and conditional integration tests.

**Tech Stack:** Python 3.10+, Ruff 0.15.8, pytest, Bash/rg/xargs, Git.

---

### Task 1: Apply and review Ruff's safe fixes

**Files:**
- Modify: the Ruff-reported Python files under `scripts/` and `src/`
- Verify untouched: every `*.ipynb`

**Step 1: Record the exact Python-only baseline**

Run:

```bash
rg --files -g '*.py' -0 | sort -z | \
  xargs -0 /root/makharev/agent-swe-ace/.venv/bin/python -m ruff check --statistics
```

Expected: 241 diagnostics: `F541` 80, `I001` 79, `F401` 25, `F841` 21, `E402` 15, `E741` 12, `E702` 3, `F821` 3, `E712` 2, and `E731` 1.

**Step 2: Apply only Ruff's safe fixes**

Run:

```bash
rg --files -g '*.py' -0 | sort -z | \
  xargs -0 /root/makharev/agent-swe-ace/.venv/bin/python -m ruff check --fix
```

Expected: 184 diagnostics fixed; no unsafe fix is applied.

**Step 3: Review the mechanical diff**

Run:

```bash
git diff --check
git diff --name-only | rg '\.ipynb$' && exit 1 || true
git diff --stat
```

Expected: no whitespace errors and no notebook changes. Inspect the source diff to confirm the edits are limited to import sorting/removal and redundant f-string prefix cleanup.

**Step 4: Verify the reduced diagnostic set**

Run:

```bash
rg --files -g '*.py' -0 | sort -z | \
  xargs -0 /root/makharev/agent-swe-ace/.venv/bin/python -m ruff check
```

Expected: exactly 57 diagnostics remain: 24 unsafe suggestions and 33 manual fixes.

**Step 5: Commit**

```bash
git add scripts src
git commit -m "style: apply safe Ruff fixes"
```

### Task 2: Resolve unsafe fixes without changing behavior

**Files:**
- Modify: `scripts/analyze_skillbook_quality.py`
- Modify: `scripts/analyze_trajectory_behaviors_lite.py`
- Modify: `scripts/analyze_trajectory_behaviors_split025.py`
- Modify: `scripts/analyze_trajectory_errors.py`
- Modify: `scripts/analyze_trajectory_length.py`
- Modify: `scripts/compare_runs.py`
- Modify: `scripts/reeval_run.py`
- Modify: `scripts/watch_experiments.py`
- Modify: `src/cli/commands.py`
- Modify: `src/runners/main_loop.py`
- Modify: `src/tests/test_main_loop.py`

**Step 1: Remove computations assigned to unused names**

Delete the unused pure assignments `gen_pct`, `spec_pct`, `max_iter`, `chart_width`, `qnext_sv_ctx_val`, `qnext_valbl_np_n`, `unit`, `phase_label`, `repo_total`, `repo_done`, `cli_resume_dirs`, `before`, `reused`, and `train_fresh`. In the skillbook chart annotation loop, remove `delta_str` while preserving the `annot` output and iteration indexing.

**Step 2: Preserve the plotting side effect**

In `scripts/analyze_trajectory_behaviors_lite.py`, replace:

```python
bars = ax.bar(...)
```

with the same bare `ax.bar(...)` call. Do not remove the call.

**Step 3: Preserve pandas comparison semantics**

In `scripts/analyze_trajectory_behaviors_split025.py`, replace `series == True` and `series == False` filters with `series.eq(True)` and `series.eq(False)`. This avoids Ruff `E712` without using scalar `not` on a pandas Series or changing null handling.

**Step 4: Replace the assigned lambda**

In `scripts/compare_runs.py`, replace the `rate_str = lambda run: ...` assignment with an equivalent nested `def rate_str(run): ...`.

**Step 5: Preserve test and runner calls while dropping unused results**

In `src/runners/main_loop.py` and `src/tests/test_main_loop.py`, remove only the unused left-hand bindings. Keep each function call, its arguments, ordering, and assertions unchanged.

**Step 6: Check the affected files**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m ruff check \
  scripts/analyze_skillbook_quality.py \
  scripts/analyze_trajectory_behaviors_lite.py \
  scripts/analyze_trajectory_behaviors_split025.py \
  scripts/analyze_trajectory_errors.py \
  scripts/analyze_trajectory_length.py \
  scripts/compare_runs.py scripts/reeval_run.py scripts/watch_experiments.py \
  src/cli/commands.py src/runners/main_loop.py src/tests/test_main_loop.py
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_commands.py src/tests/test_main_loop.py -q
```

Expected: no unsafe-fix diagnostics in these files; focused tests pass.

**Step 7: Commit**

```bash
git add scripts src
git commit -m "style: resolve unsafe Ruff findings"
```

### Task 3: Make intentional late imports explicit

**Files:**
- Modify: `scripts/compare_runs.py`
- Modify: `src/cli/commands.py`
- Modify: `src/runners/main_loop.py`

**Step 1: Document the script path mutation boundary**

Add a narrow `# noqa: E402` to the `collect_val_baseline_aggregated` import in `scripts/compare_runs.py`. Keep the preceding `sys.path.insert` because the script must remain directly executable.

**Step 2: Document the CLI path and environment boundaries**

Add narrow `# noqa: E402` markers to the project imports that follow the `_src_dir` `sys.path` setup in `src/cli/commands.py`. Keep `litellm` after `load_dotenv` and mark that import separately because import-time configuration depends on the environment being loaded first.

**Step 3: Document the runner import boundary**

Add narrow `# noqa: E402` markers to the observer/logging imports in `src/runners/main_loop.py`, which intentionally follow `_build_ground_truth`. Do not add a file-level ignore.

**Step 4: Verify**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m ruff check \
  scripts/compare_runs.py src/cli/commands.py src/runners/main_loop.py
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_commands.py src/tests/test_main_loop.py -q
```

Expected: no `E402` diagnostics; focused tests pass.

**Step 5: Commit**

```bash
git add scripts/compare_runs.py src/cli/commands.py src/runners/main_loop.py
git commit -m "style: mark intentional late imports"
```

### Task 4: Resolve type-only forward references

**Files:**
- Modify: `src/data_io/readers.py`
- Modify: `src/data_io/writers.py`
- Test: `src/tests/test_io.py`

**Step 1: Add type-checking-only imports**

Import `TYPE_CHECKING` from `typing` in both modules and add:

```python
if TYPE_CHECKING:
    from ace import Skillbook
```

Keep the existing quoted annotations and function-local runtime imports. This gives Ruff a definition without eagerly importing ACE at module import time.

**Step 2: Verify the module contracts**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m ruff check \
  src/data_io/readers.py src/data_io/writers.py
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest src/tests/test_io.py -q
```

Expected: no `F821` diagnostics and all I/O tests pass.

**Step 3: Commit**

```bash
git add src/data_io/readers.py src/data_io/writers.py
git commit -m "style: define Skillbook type references"
```

### Task 5: Rename ambiguous locals and split compound statements

**Files:**
- Modify: `scripts/analyze_skillbooks.py`
- Modify: `scripts/analyze_trajectory_behaviors_split025.py`
- Modify: `scripts/analyze_trajectory_errors.py`
- Modify: `scripts/compare_trajectories.py`
- Modify: `scripts/q1_stat_tests_per_repo.py`
- Modify: `scripts/watch_experiments.py`
- Modify: `src/prompts/custom_skill_manager.py`
- Modify: `src/tests/test_custom_swe_learn.py`

**Step 1: Rename each ambiguous `l` by role**

Use role-specific names:

- `lost_count` for lost-domain counts in `analyze_skillbooks.py`
- `legend_label` in `analyze_trajectory_errors.py`
- `label` for run labels in `compare_trajectories.py` and `q1_stat_tests_per_repo.py`
- `line` for process-list lines in `watch_experiments.py`
- `learning` for extracted learning objects in `custom_skill_manager.py` and its tests

Update all references within the same expression or loop without changing values or ordering.

**Step 2: Split semicolon-separated statements**

In `scripts/analyze_trajectory_behaviors_split025.py`, put `set_facecolor` and `set_alpha` on separate lines and split each `set_xticks(...); set_xticklabels(...)` pair into two statements.

**Step 3: Verify**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m ruff check \
  scripts/analyze_skillbooks.py \
  scripts/analyze_trajectory_behaviors_split025.py \
  scripts/analyze_trajectory_errors.py scripts/compare_trajectories.py \
  scripts/q1_stat_tests_per_repo.py scripts/watch_experiments.py \
  src/prompts/custom_skill_manager.py src/tests/test_custom_swe_learn.py
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests/test_custom_swe_learn.py -q
```

Expected: no `E741` or `E702` diagnostics; focused tests pass.

**Step 4: Commit**

```bash
git add scripts src
git commit -m "style: clarify local names and statements"
```

### Task 6: Prove the complete Python tree is clean

**Files:**
- Verify: every tracked `*.py`
- Verify untouched: every `*.ipynb`

**Step 1: Run the final Python-only Ruff check**

Run:

```bash
rg --files -g '*.py' -0 | sort -z | \
  xargs -0 /root/makharev/agent-swe-ace/.venv/bin/python -m ruff check
```

Expected: `All checks passed!`

**Step 2: Run all changed test modules in isolation**

Run each changed test module in its own process:

```bash
for test_file in \
  src/tests/test_commands.py \
  src/tests/test_custom_swe_learn.py \
  src/tests/test_io.py \
  src/tests/test_main_loop.py
do
  /root/makharev/agent-swe-ace/.venv/bin/python -m pytest "$test_file" -q
done
```

Expected: every module passes independently.

**Step 3: Run the complete non-integration suite twice**

Run:

```bash
/root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests -m 'not integration' -q
PYTHONHASHSEED=42 /root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
  src/tests -m 'not integration' -q
```

Expected: both runs pass; current baseline is 410 passed and 4 deselected.

**Step 4: Attempt the integration suite when its service is ready**

Run:

```bash
if curl --silent --show-error --fail --max-time 3 \
  http://127.0.0.1:8800/v1/models >/dev/null
then
  /root/makharev/agent-swe-ace/.venv/bin/python -m pytest \
    src/tests -m integration -q
else
  echo 'BLOCKED: integration endpoint http://127.0.0.1:8800/v1/models is unavailable'
fi
```

Expected: integration tests pass when the endpoint is available; otherwise report the suite as `BLOCKED` with the failed preflight evidence.

**Step 5: Verify scope and repository state**

Run:

```bash
git diff --check
git diff main...HEAD --name-only | rg '\.ipynb$' && exit 1 || true
git status --short
```

Expected: no whitespace errors, no notebook changes, and a clean worktree.
