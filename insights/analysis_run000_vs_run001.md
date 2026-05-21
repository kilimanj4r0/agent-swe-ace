# Analysis: Run #000 vs Run #001 (custom_swe_learn vs default)

**Date**: 2026-05-19

## Runs Compared

| ID | Run Directory | Name | Learning |
|----|---------------|------|----------|
| #000 | `run_20260515_141809_completed_qwen3_4a_swe_verified` | agent-qwen3-ace-qwen3-full-verified-4a-12c | `custom_swe_learn: true` |
| #001 | `run_20260517_202038_completed_qwen3_4a_default_verified` | agent-qwen3-ace-qwen3-full-verified-4a-12c-default | `custom_swe_learn: false` |

Run #001 was initialized with `baseline_run_dir` pointing to Run #000, copying its iter_0 trajectories.

## Summary

| Metric | Run #000 (custom_swe) | Run #001 (default) |
|--------|----------------------|-------------------|
| Total resolved | 137 (28.0%) | 160 (32.7%) |
| iter_0 resolved | 92 (18.8%) | 93 (19.0%) |
| iter_1 new resolved | 35 | 50 |
| iter_2 new resolved | 8 | 11 |
| iter_3 new resolved | 2 | 6 |
| Skillbook-assisted total | 45 (+9.2pp) | 67 (+13.7pp) |
| Experiment time | 52.4h | 32.5h |

Only config difference: `experiment.skillbook.custom_swe_learn` — true vs false.

## iter_0 Discrepancy (92 vs 93)

484 of 489 instances had iter_0 trajectories copied directly from Run #000 (no re-run, no re-evaluation). The +1 comes from `pydata__xarray-7233`, which had `exit_status="error"` in Run #000 and was re-run from scratch in Run #001.

The baseline reuse logic (`_try_load_baseline_iter0()` in `main_loop.py`) only copies trajectories with `exit_status` in `("Submitted", "LimitsExceeded")` — error trajectories are skipped and the agent re-runs.

### All 7 Error Instances at iter_0 in Run #000

| Instance | Run#000 iter_0 | Run#001 iter_0 | Resolved in #001? | Notes |
|---|---|---|---|---|
| `astropy__astropy-13033` | error (126 msgs) | Submitted (132 msgs) | No | Agent hit step limit in #000, submitted patch in #001 but unresolved |
| `django__django-16256` | error (126 msgs) | Submitted (74 msgs) | No | Agent hit step limit in #000, submitted patch in #001 but unresolved |
| `matplotlib__matplotlib-13989` | error (0 msgs) | error (0 msgs) | No | Docker launch failure in both runs |
| `matplotlib__matplotlib-14623` | error (0 msgs) | error (0 msgs) | No | Docker launch failure in both runs |
| `pydata__xarray-6599` | error (126 msgs) | Submitted (68 msgs) | No | Agent hit step limit in #000, submitted patch in #001 but unresolved |
| **`pydata__xarray-7233`** | error (126 msgs) | Submitted (64 msgs) | **Yes** | The single +1 at iter_0 |
| `sympy__sympy-23262` | error (154 msgs) | LimitsExceeded (503 msgs) | No | Hit step limit in both runs |

5 of 7 had the agent run to the step limit (126-154 msgs) in Run #000. In Run #001, 4 of those 5 were re-run and submitted a patch; only `pydata__xarray-7233` was resolved. The 2 matplotlib instances had Docker infrastructure failures (0 messages in both runs).

## iter_1+ Difference: Why Default Outperforms Custom SWE

The `custom_swe_learn` flag controls which reflection/skill-management classes are used:

- **custom_swe_learn=true**: `SWEReflector` + `SWESkillManager` — produces prescriptive, action-oriented skills
- **custom_swe_learn=false**: default `Reflector` + `SkillManager` — produces descriptive, analysis-oriented skills

### Skill Quality Comparison (example: `astropy__astropy-13033`, iter_1)

**Custom SWE skills** — prescriptive patterns:
- `VERIFIED: Use /testbed directory for file modifications (writable)`
- `AVOID: Claiming task completion without verifying git diff shows actual changes`
- `CONSIDER: Modify error message format to be more descriptive`
- `AVOID: Creating multiple bash commands in one response`

**Default ACE skills** — analysis-oriented:
- `Identify misleading error messages that confuse users about actual issues`
- `Improve error messages to clearly indicate missing required columns`
- `Avoid using sed and regex replacements for complex Python multi-line string literals`
- `Locate the _check_required_columns method in BaseTimeSeries class within astropy/timeseries/core.py`

All 1342 skillbook files differ between the two runs (0 matching).

### Resolution by Iteration

The default learner resolved **23 more instances** across skillbook-assisted iterations:

| Iteration | Custom SWE | Default | Delta |
|-----------|-----------|---------|-------|
| iter_1 | 35 new | 50 new | +15 (+43%) |
| iter_2 | 8 new | 11 new | +3 |
| iter_3 | 2 new | 6 new | +4 |

## Conclusion

- **iter_0 gap is negligible** (92 vs 93): caused by a single instance being re-run due to error exit status
- **The 23-instance overall gap comes entirely from skillbook quality**: default ACE learning produces more effective skills than the custom SWE variant, especially at iter_1 where it resolved 43% more new instances
- Default skills are more descriptive and analysis-oriented, providing better guidance for subsequent agent attempts
