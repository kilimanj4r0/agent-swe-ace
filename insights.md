# ACE-SWE Experiment Insights

## Overview

This document captures key insights from analyzing SWE-bench experiment runs, focusing on how skillbook learning affects agent behavior and patch generation.

---

## Key Finding: Skillbook Causes Agents to Fail

### The Problem

Agents with skillbooks (learned from failed attempts) consistently produce **0-char patches**, while agents without skillbooks produce working patches.

### Evidence Summary

| Instance | iter_0 (no skillbook) | iter_1+ (with skillbook) |
|----------|----------------------|--------------------------|
| astropy-12907 | 14495 char patch | All 0 char patches |
| astropy-14182 | 622 char patch | All 0 char patches |
| astropy-14365 | 1391 char patch | All 0 char patches |

**Verification:** Running WITHOUT skillbook (`run_20260322_202646`) produced 11768 char patch.

---

## Root Cause Analysis

### 1. Skills Are Too Abstract

Skills describe WHAT to do but not HOW:

```
❌ "When processing nested compound models, recursively apply separability_matrix to their components"
✅ Better: "Edit /testbed/astropy/modeling/separable.py, add recursive call at line 125"
```

### 2. Agent Misinterprets Skills as "Solution Understood"

Reading skills makes the agent believe it "knows" the solution, causing it to:
- **Not explore `/testbed` properly** (0 mentions vs 45 in baseline)
- **Create test scripts** instead of editing source files
- **Make false claims**: "I cannot modify the installed astropy package in this environment"

**Example agent quote from failed run:**
> "However, since I cannot modify the installed astropy package in this environment, I'll document what needs to be fixed"

This is FALSE - `/testbed` contains writable source code.

### 3. Skills from Failed Attempts Contain Misleading Context

The `evidence` field describes what the FAILED agent did:
```
"Evidence: The fix required identifying that nested CompoundModels needed special handling..."
```

This makes subsequent agents think the solution is understood, when actually the previous agent failed.

---

## Behavioral Patterns Observed

### Message Count Decreases with Skillbook

| Iteration | Assistant Messages | Patch Size |
|-----------|-------------------|------------|
| iter_0 (baseline) | 26-66 | 622-14495 chars |
| iter_1 | 21-43 | 0 chars |
| iter_5 | 2-38 | 0 chars |
| iter_9 | 3-10 | 0 chars |

Agents give up faster in later iterations.

### `/testbed` Usage Drops with Skillbook

| Instance | iter_0 | iter_1 | iter_5 |
|----------|--------|--------|--------|
| astropy-12907 | 45 mentions | 9 mentions | 0 mentions |
| astropy-14182 | 0 mentions | 9 mentions | 0 mentions |
| astropy-14365 | 0 mentions | 11 mentions | 0 mentions |

### Format Errors Don't Correlate with Success

iter_0 for astropy-12907 had 43 format errors but still produced 14495 char patch. The issue isn't format compliance - it's that the agent doesn't edit source files.

---

## Technical Issues Fixed This Session

### 1. Submission Command Missing `git add -A`

**Problem:** Default template only had `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`

**Fix:** Load `swebench.yaml` instead of `default.yaml`:
```python
# src/phases/predict.py
config_path = package_dir / "config" / "extra" / "swebench.yaml"
```

Correct command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached`

### 2. Model Counter Accumulation

**Problem:** `n_calls` and `cost` accumulated across iterations, causing immediate `LimitsExceeded`

**Fix:** Reset counters before each run:
```python
# src/agents/miniswe_agent.py
if hasattr(self.llm_model, 'n_calls'):
    self.llm_model.n_calls = 0
if hasattr(self.llm_model, 'cost'):
    self.llm_model.cost = 0.0
```

### 3. Statistics Not Saved on Crash

**Fix:** Added try/finally block in `main_loop.py`:
```python
try:
    # Process instances...
except Exception as e:
    error_info = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
finally:
    # Always save statistics
    save_statistics(statistics=statistics, run_dir=self.output_dir)
```

### 4. Docker Image Cleanup (Improved)

**Problem:** Disk space exhaustion from accumulated SWE-bench images (2-5GB each)

**Old approach:** `cleanup_interval` parameter with periodic cleanup (didn't work - images in use)

**New approach:** Use SWE-bench's built-in `rm_image` parameter:
```yaml
# config.yaml
evaluation:
  rm_image: true  # Remove Docker image after evaluation (saves disk space)
```

The SWE-bench harness always cleans up containers, and with `rm_image=True` it also removes images after each evaluation. This is more reliable than manual cleanup.

### 5. SWE-bench Logs Redirected

**Problem:** SWE-bench created `logs/run_evaluation/` in project root

**Fix:** Monkey-patch `RUN_EVALUATION_LOG_DIR` to redirect logs to run folder:
```python
# src/evaluation/swebench.py
import swebench.harness.constants as constants
constants.RUN_EVALUATION_LOG_DIR = Path(output_dir) / "swebench_logs"
```

Logs now saved to `{run_dir}/swebench_logs/` instead of cluttering project root.

### 6. Custom SWE Learning System

**Problem:** Default ACE Reflector produces generic skills without type information

**Solution:** New `custom_swe_learn` config option that enables:
- **SWEReflector** - Extracts anti-patterns, discoveries, and unvalidated hypotheses
- **SWESkillManager** - Preserves learning type prefixes in skill content

```yaml
# config.yaml
experiment:
  custom_swe_learn: true  # Use SWE-optimized Reflector + SkillManager
```

**Learning type prefixes in skill content:**
- `AVOID:` - Behaviors that led to failure (from anti-patterns)
- `VERIFIED:` - Factual findings (from discoveries)
- `CONSIDER:` - Unvalidated claims needing verification (from hypotheses)

**Example skills with type prefixes:**
```
AVOID: Claiming task completion without verifying git diff shows actual changes
VERIFIED: Use /testbed directory for file modifications (writable in Docker)
CONSIDER: Modifying RST class constructor to accept header_rows parameter
```

**Files created:**
- `prompts/reflector_prompt.py` - Custom reflector prompt
- `prompts/outputs.py` - SWEReflectorOutput with anti_patterns, discoveries, unvalidated_hypotheses
- `prompts/custom_reflector.py` - SWEReflector class
- `prompts/skill_manager_prompt.py` - Custom skill manager prompt with type preservation
- `prompts/custom_skill_manager.py` - SWESkillManager class

---

## Proposed Solutions for Skillbook Issue

### Option 1: Make Skills More Actionable

Include specific file paths and commands:
```
❌ "Recursively process nested compound models"
✅ "Edit /testbed/astropy/modeling/separable.py:125, add recursive call for nested models"
```

### Option 2: Add Guard Clause to Skillbook Section

```
## Learned Strategies (Skillbook)

⚠️ IMPORTANT: These skills describe approaches from previous attempts, but you MUST still
implement actual code changes in /testbed. Do not assume the solution is already understood.
Always verify you can write to /testbed (it IS writable).
```

### Option 3: Filter Skills by Quality

- Only include skills from **successful** attempts
- Require minimum helpfulness score
- Remove skills with "cannot" or "unable" language

### Option 4: Disable Skillbook Initially

- Only inject skillbook after N failed attempts
- Use skillbook as "last resort" rather than primary guidance

### Option 5: Skill Validation

Before injecting, verify:
1. Skill doesn't mention "cannot" or "installed package"
2. Skill references `/testbed` explicitly
3. Skill includes actionable commands

---

## Data Sources

- `data/run_20260322_180346/` - 3 instances, 10 iterations each, all failed
- `data/run_20260322_201644/` - Single instance with skillbook, 0 char patch
- `data/run_20260322_202646/` - Single instance without skillbook, 11768 char patch
- `data/run_baseline_qwen3coder/` - Original baseline runs with working patches

---

## Detailed Analysis: astropy__astropy-14182 (run_20260322_223509)

### Results Summary

| Iteration | Messages | Patch Size | Key Issue |
|-----------|----------|------------|-----------|
| iter_0 (baseline) | 54 | **622 chars** | No skillbook |
| iter_1 | 52 | **3187 chars** | ✅ Improved 5x! |
| iter_2 | 6 | **0 chars** | Format error |
| iter_3 | 50 | **0 chars** | False confidence |
| iter_4 | 6 | **0 chars** | Format error |

### iter_2 Failure Analysis (6 messages, 0 patch)

**Issue:** ALL 50+ commands in ONE message → **FORMAT ERROR**

Agent tried many edit commands:
```bash
sed -i 's/def __init__(self, **kwargs):/def __init__(self, header_rows=None, **kwargs):/' io/ascii/rst.py
cat <<'EOF' > test_rst_issue.py
# ... many more commands
```

But ALL were in message 3 with multiple bash blocks. Framework rejected the response.

**Result:** Agent immediately submitted → 0 char patch

### iter_3 Failure Analysis (50 messages, 0 patch)

**Issue:** **Edits kept failing** + **False confidence**

Agent made 24 assistant messages with many sed/awk attempts:
- `sed -i '15s/def __init__(self):/def __init__(self, header_rows=None):/'`
- `awk 'NR==15 { print "..." }'`
- Various other approaches

All edits either:
1. Had wrong syntax
2. Didn't produce working code
3. Were overwritten by subsequent attempts

**Critical:** Last message claimed success:
> "I have successfully identified and documented the correct solution. The issue was that the RST class constructor didn't accept the `header_rows` parameter..."

But **NO ACTUAL PATCH** was generated!

### iter_4 Failure Analysis (6 messages, 0 patch)

**Issue:** Same as iter_2 - all commands in one response

Agent repeated the pattern:
- All exploration and edit attempts in message 3
- Framework rejected (multiple bash blocks)
- Immediate submission → 0 char patch

### Skillbook Evolution

```
iter_1: 3 skills  → 3187 char patch ✅ (IMPROVED!)
iter_2: 4 skills  → 0 char patch (format error)
iter_3: 7 skills  → 0 char patch (false confidence)
iter_4: 7 skills  → 0 char patch (format error)
```

Skills became increasingly detailed:
```
iter_1: "Modify the RST class constructor to accept header_rows parameter"
iter_3: "Modify the RST class constructor in io/ascii/rst.py to accept
        header_rows=None as a parameter and pass it through to the parent
        Writer class using super().__init__(header_rows=header_rows)"
```

**Hypothesis:** Detailed skills cause agent to:
1. Rush implementation (put everything in one message → format error)
2. Claim success without verifying ("I have successfully implemented...")

### Key Insights

1. **iter_1 IMPROVED** - Skillbook helped in first iteration (622 → 3187 chars)
2. **Later iterations fail differently** - Not the same issue as astropy-12907
3. **Two failure modes identified:**
   - **Format errors:** Too many commands in one response
   - **False confidence:** Claiming success without actual patch
4. **Skills get more detailed over iterations** - This may cause rushing

---

## Recommended Guard Clause Implementation

### Proposed Text for Skillbook Section

```markdown
## Learned Strategies (Skillbook)

These are strategies learned from previous attempts. Use them to guide your approach:

<skills_content>

⚠️ **CRITICAL REMINDERS:**
1. These skills describe approaches, NOT complete solutions. You MUST implement actual code changes.
2. Do NOT put multiple bash commands in one response - use ONE command per response.
3. Before submitting, verify your patch exists with: `git diff --cached`
4. If you think you've "successfully implemented" but git diff is empty, you haven't actually edited any files.
5. The source code in /testbed IS writable - do not claim you "cannot modify installed packages".

**Submission Checklist:**
- [ ] I have edited source files in /testbed (not just created test scripts)
- [ ] `git diff --cached` shows actual code changes
- [ ] I have run my test to verify the fix works
```

### Implementation Location

In `src/phases/predict.py`, modify `build_instance_template()`:

```python
skillbook_section = f"""

## Learned Strategies (Skillbook)

These are strategies learned from previous attempts. Use them to guide your approach:

{skillbook_context}

⚠️ **CRITICAL REMINDERS:**
1. These skills describe approaches, NOT complete solutions. You MUST implement actual code changes.
2. Do NOT put multiple bash commands in one response - use ONE command per response.
3. Before submitting, verify your patch exists with: `git diff --cached`
4. If you think you've "successfully implemented" but git diff is empty, you haven't actually edited any files.
5. The source code in /testbed IS writable - do not claim you "cannot modify installed packages".

When you apply a strategy successfully, reference it with [skill-id] notation in your reasoning."""
```

---

## Next Steps

1. ✅ Implement skillbook guard clause (quickest fix)
2. ✅ Implement custom SWE learning system with type prefixes
3. ✅ Fix Docker image cleanup with `rm_image` parameter
4. ✅ Redirect SWE-bench logs to run folder
5. Test with `custom_swe_learn: true` on multiple instances
6. Monitor skill content for proper type prefixes (AVOID:, VERIFIED:, CONSIDER:)
7. If still failing, implement skill quality filtering
8. Consider alternative skill extraction from successful patches only
