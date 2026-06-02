# Implementation Progress: Plan 0 + Plan 1

Date: 2026-05-29

## High-Level Overview

These changes fix two systemic problems in the skillbook learning pipeline:

**1. LLM calls ignore configured temperature/max_tokens** — The `llm.ace` config section specified `temperature: 0.0` and `max_tokens: 4096`, but these were never forwarded to any ACE component. Reflector and SkillManager ran at whatever the model API defaults to (~0.6-0.95), producing non-deterministic analysis across identical inputs. This is now wired through `create_model_settings()` → `ModelSettings` at all 6 constructor calls (3 custom SWE + 3 default ACE).

**2. Successful trajectories produce anti-pattern skills** — The reflector treated ALL trajectories as failures (hardcoded "TRAJECTORIES THAT FAILED" in prompt), `feedback` and `ground_truth` were never passed from main_loop, and `resolved` was ignored. Successful solutions generated `[ANTI-PATTERN]`/`AVOID:` skills instead of `[DISCOVERY]`/`VERIFIED:` reinforcement. Now `resolved`, `feedback`, and `ground_truth` (FAIL_TO_PASS/PASS_TO_PASS test lists) flow from main_loop through LearnPhase to the reflector, which uses outcome-dependent prompt framing.

**3. SkillManager loses diagnostic fields** — The reflector computed `error_location`, `confidence_in_analysis`, and `skill_tags`, but `SWESkillManager` inherited ACE's serialization that only forwards 6 fields. These are now included in the skill manager prompt via `hasattr()` checks.

**4. Type hierarchy and defaults** — `SWEReflectorOutput` didn't subclass `ReflectorOutput` (isinstance checks fail). `atomicity_score` defaulted to 0.0 ("Rejected"), silently dropping valid learnings when the LLM omitted the score. `CONSIDER:` prefix was rejected as "meta-commentary". All fixed.

**5. Shared model resolution** — Duplicated ~40 lines of model resolution logic extracted to `model_utils.py`.

---

## Pipeline Architecture After Changes

### Component Construction (commands.py — 3 sites)

All 3 entry points (`_run_iterate_repos`, `run_full_experiment`, `run_learn_cmd`) follow the same pattern:

```
config.yaml llm.ace section
        │
        ▼
  LLMConfig.from_dict(config["llm"]["ace"])
        │
        ├──► create_ace_client(ace_config)  → model string ("zai/glm-4.5-flash" or "hosted_vllm/Qwen/...")
        │
        └──► create_model_settings(ace_config)  → {"temperature": 0.0, "max_tokens": 8912}
                    │
                    ▼
            custom_swe_learn flag?
            ┌─────────YES──────────┐────────────NO────────────┐
            │                       │                           │
            ▼                       ▼                           ▼
    SWEReflector            SWESkillManager            DefaultReflector
    (custom_reflector.py)   (custom_skill_manager.py)  (ace.Reflector)
            │                       │                           │
            │  make_pydantic_agent()│  make_pydantic_agent()   │ ModelSettings(ace_settings)
            │  with model_settings  │  with model_settings     │
            ▼                       ▼                           ▼
    PydanticAgent            PydanticAgent              PydanticAgent
    output_type:             output_type:               output_type:
    SWEReflectorOutput       SkillManagerOutput         ReflectorOutput
```

**Key:** `create_model_settings()` extracts `temperature` + `max_tokens` from the same `llm.ace` config that `create_ace_client()` uses for the model string. Both custom and default paths receive the same `ModelSettings`.

### Model Resolution (model_utils.py)

Shared by both SWEReflector and SWESkillManager:

```
make_pydantic_agent(model, output_type, api_base, api_key, model_settings)
        │
        ├──► resolve_ace_model(model, api_base, api_key)
        │        │
        │        ├── api_base set? ──YES──► AsyncOpenAI → LiteLLMProvider → OpenAIChatModel
        │        │                           (strips provider prefix, e.g. "hosted_vllm/" → bare model name)
        │        │
        │        └── no api_base ────────► ACE resolve_model()
        │                                   (standard provider routing)
        │
        └──► ModelSettings(**model_settings) if provided
                │
                ▼
        PydanticAgent(resolved_model, output_type, model_settings=ms)
```

### Learn Pipeline — Data Flow

```
main_loop.py (4 call sites)
        │
        │  ┌─────────────────────────────────────────────┐
        │  │ Inputs available at each call site:          │
        │  │ • evaluate_result.feedback (sites 1,2,3)    │
        │  │ • evaluate_result.resolved (sites 1,2,3,4)  │
        │  │ • instance["FAIL_TO_PASS"]                   │
        │  │ • instance["PASS_TO_PASS"]                   │
        │  └─────────────────────────────────────────────┘
        │
        ▼
  _build_ground_truth(instance, max_chars=4000)
        │  FAIL_TO_PASS + PASS_TO_PASS → truncated to 4K chars
        │  (django-10097 has 140K chars of tests; prevents prompt explosion)
        │
        ▼
  LearnPhase.run(
      skillbook, instance, trajectory, patch,
      feedback=...,                    # from EvaluateResult or result_data
      ground_truth=_build_ground_truth(instance),
      resolved=evaluate_result.resolved,
  )
        │
        │  ┌─────────────────────────────────────────┐
        │  │ learn.py internal flow:                  │
        │  │                                          │
        │  │ 1. Build AgentOutput from trajectory     │
        │  │ 2. Feedback fallback if None:            │
        │  │    • resolved=True → "All tests passed"  │
        │  │    • resolved=False → "Tests failed"     │
        │  │ 3. Call reflector.reflect(               │
        │  │      question, agent_output, skillbook,  │
        │  │      feedback, ground_truth, resolved)   │
        │  │ 4. Build progress string:                │
        │  │    • resolved → "resolved successfully"  │
        │  │    • !resolved → "tests failed"          │
        │  │ 5. Call skill_manager.update_skills(     │
        │  │      reflections=(reflection,),          │
        │  │      skillbook, question_context,        │
        │  │      progress)                           │
        │  │ 6. Apply UpdateBatch to skillbook        │
        │  │ 7. Run dedup consolidation if enabled    │
        │  │ 8. Save skillbook                        │
        │  └─────────────────────────────────────────┘
```

### Custom SWE Reflector Prompt Pipeline

When `custom_swe_learn: true`, `SWEReflector.reflect()` overrides the base ACE reflector:

```
SWEReflector.reflect(question, agent_output, skillbook, feedback, ground_truth, resolved)
        │
        ├──► Build outcome variables:
        │    resolved=True  → outcome = "SUCCESS — all tests passed"
        │                     outcome_instructions = "Focus on what worked,
        │                       reusable patterns, discoveries, tag helpful"
        │    resolved=False → outcome = "FAILURE — tests did not pass"
        │                     outcome_instructions = "Focus on anti-patterns,
        │                       false assumptions, NEVER extract solutions"
        │
        ├──► Build skillbook excerpt:
        │    make_skillbook_excerpt(skillbook, agent_output.skill_ids)
        │    → formatted skill text or "(No strategies cited)"
        │
        ▼
  CUSTOM_REFLECTOR_PROMPT.format(
      question        = instance["problem_statement"],
      reasoning       = agent_output.reasoning,
      prediction      = agent_output.final_answer (patch),
      ground_truth    = FAIL_TO_PASS / PASS_TO_PASS (truncated),
      feedback        = test evaluation feedback,
      skillbook_excerpt = skillbook excerpt,
      outcome         = "SUCCESS ..." or "FAILURE ...",
      outcome_instructions = outcome-specific analysis focus,
  )
        │
        ▼
  PydanticAgent.run_sync(prompt)
        │
        ▼
  SWEReflectorOutput (extends ReflectorOutput)
        ├── reasoning, error_identification, root_cause_analysis
        ├── error_location          # SWE-specific
        ├── confidence_in_analysis  # SWE-specific (0.0-1.0)
        ├── anti_patterns: [AntiPattern(pattern, why_harmful, atomicity_score=0.8, evidence)]
        ├── discoveries: [Discovery(finding, atomicity_score=0.8, evidence)]
        ├── unvalidated_hypotheses: [UnvalidatedHypothesis(hypothesis, why_unvalidated, ...)]
        ├── key_insight
        ├── skill_tags: [SkillTag(id, tag, justification, impact_score)]
        │
        └── @model_validator computes:
            extracted_learnings → [
              ExtractedLearning(learning="[ANTI-PATTERN] ...", ...),
              ExtractedLearning(learning="[DISCOVERY] ...", ...),
              ExtractedLearning(learning="[HYPOTHESIS] ...", ...),
            ]
```

### Custom SWE SkillManager Prompt Pipeline

When `custom_swe_learn: true`, `SWESkillManager.update_skills()` overrides the base ACE skill manager:

```
SWESkillManager.update_skills(reflections, skillbook, question_context, progress)
        │
        ├──► Serialize each reflection with extended fields:
        │    {
        │      "reasoning": ...,
        │      "error_identification": ...,
        │      "root_cause_analysis": ...,
        │      "correct_approach": ...,
        │      "key_insight": ...,
        │      "extracted_learnings": [{learning: "[ANTI-PATTERN] ...", ...}, ...],
        │      ── SWE-specific fields (hasattr checks) ──
        │      "error_location": "Final submission step",
        │      "confidence_in_analysis": 0.95,
        │      "skill_tags": [{"id": "skill_1", "tag": "helpful"}, ...]
        │    }
        │
        ▼
  CUSTOM_SKILL_MANAGER_PROMPT.format(
      progress        = "Iteration 0: resolved the issue successfully.",
      stats           = skillbook.stats(),
      reflections     = JSON-serialized reflections with diagnostics,
      skillbook       = skillbook.as_prompt() or "(empty skillbook)",
      question_context = "Instance: ...\nRepo: ...\nProblem: ...",
  )
        │
        │  Prompt includes sections:
        │  <atomicity> — scoring criteria for skill quality
        │  <learning_types> — [ANTI-PATTERN]→AVOID:, [DISCOVERY]→VERIFIED:, [HYPOTHESIS]→CONSIDER:
        │  <diagnostic_context> — how to use confidence/error_location/skill_tags  ← NEW
        │  <operations> — ADD/UPDATE/TAG/REMOVE decision table
        │  <rejection_criteria> — bare "consider" removed, CONSIDER: prefix valid  ← FIXED
        │  <output_format> — JSON with learning_index for traceability
        │
        ▼
  SkillManagerOutput
        ├── reasoning
        └── operations: [
              {type: "ADD", content: "AVOID: ...", learning_index: 0},
              {type: "ADD", content: "VERIFIED: ...", learning_index: 1},
              {type: "ADD", content: "CONSIDER: ...", learning_index: 2},
              {type: "TAG", skill_id: "...", metadata: {"helpful": 1}},
            ]
```

### Default ACE Pipeline (custom_swe_learn: false)

When using default ACE components, the same `ModelSettings` and parameters flow through, but without the SWE-specific overrides:

```
DefaultReflector.reflect(question, agent_output, skillbook, feedback, ground_truth, resolved)
        │
        └──► Base ACE Reflector.reflect() uses **kwargs:
             • feedback is forwarded (previously always None)
             • ground_truth is forwarded (previously always None)
             • resolved is silently ignored (base reflector doesn't use it)
             • ModelSettings(temperature=0.0, max_tokens=8912) now applied

SkillManager.update_skills(reflections, skillbook, question_context, progress)
        │
        └──► Base ACE SkillManager serialization:
             Only 6 fields per reflection (no error_location, confidence, skill_tags)
             Standard ACE prompt (no diagnostic_context, no learning type prefixes)
```

### Reflector Prompt Structure (CUSTOM_REFLECTOR_PROMPT)

```
# QUICK REFERENCE
Role: ACE Reflector v2.1-swe
Mission: extract concrete learnings from trajectories     ← was "FAILURES"
Key Rule: failures→anti-patterns; successes→strategies    ← was "extract what NOT to do"

# CORE MISSION
Trajectory Outcome: {outcome}                              ← NEW: "SUCCESS" or "FAILURE"
CRITICAL: Review the outcome, feedback, and ground truth.
{outcome_instructions}                                      ← NEW: outcome-specific focus

# INPUT ANALYSIS CONTEXT
Question: {question}
Reasoning: {reasoning}
Prediction: {prediction}
Ground Truth: {ground_truth}                               ← NEW: FAIL_TO_PASS/PASS_TO_PASS
Feedback: {feedback}                                       ← NEW: actual test feedback
Skillbook Context: {skillbook_excerpt}

# MANDATORY DIAGNOSTIC PROTOCOL
Priority 0: FAILED_ATTEMPT_ANALYSIS
  WHEN: outcome indicates failure                          ← was "agent failed"
Priority 1: SUCCESS_CASE_DETECTED
  WHEN: prediction matches ground truth AND feedback positive
  NOW REACHABLE (ground_truth + feedback actually populated)

# SWE-BENCH SPECIFIC ANTI-PATTERNS
  (5 common failure modes documented)

# ATOMICITY SCORING
  Excellent (95-100%) / Good (85-95%) / Fair (70-85%) / Poor (40-70%) / Rejected (<40%)

# TAGGING CRITERIA
  helpful / harmful / neutral with evidence requirements

# OUTPUT FORMAT (JSON)
  anti_patterns, discoveries, unvalidated_hypotheses, skill_tags
```

### Skill Manager Prompt Structure (CUSTOM_SKILL_MANAGER_PROMPT)

```
<atomicity>
  Scoring criteria for skill quality
  Strategy format: IMPERATIVE COMMANDS, not observations

<learning_types>                    ← TYPE PREFIX PRESERVATION
  [ANTI-PATTERN] → AVOID:          ← negative advice
  [DISCOVERY]   → VERIFIED:        ← positive guidance
  [HYPOTHESIS]  → CONSIDER:        ← conditional guidance

<diagnostic_context>                ← NEW
  error_location → targeted skills
  confidence_in_analysis → weight ADD vs TAG decisions
  skill_tags → respect helpful/harmful assessments

<operations>
  ADD / UPDATE / TAG / REMOVE with decision table
  SKIP rules: vague, duplicate, low atomicity

<rejection_criteria>
  Meta-commentary (bare words): "be careful", "think about", "remember"
  NOTE: "consider" removed as bare word; CONSIDER: prefix is valid  ← FIXED

<output_format>
  JSON with type, content (with prefix), learning_index, justification
```

---

## Plan 0: Custom SWE Skills Fixes

### Step 1: Wire temperature/max_tokens from llm.ace to ACE components
- **Status:** DONE
- **Files:** `src/config/llm.py`, `src/prompts/custom_reflector.py`, `src/prompts/custom_skill_manager.py`, `src/cli/commands.py`
- Added `create_model_settings()` helper in `llm.py` that extracts `temperature` and `max_tokens` from config
- Added `model_settings` param to `SWEReflector` and `SWESkillManager` constructors
- Wired `ModelSettings` at all 3 instantiation sites in `commands.py` (iterate_repos, run_full_experiment, run_learn_cmd)
- Both custom SWE path and default ACE path now receive model settings

### Step 2: Forward error_location, confidence_in_analysis, skill_tags to SkillManager
- **Status:** DONE
- **Files:** `src/prompts/custom_skill_manager.py`, `src/prompts/skill_manager_prompt.py`
- Overrode `update_skills()` in `SWESkillManager` with extended serialization that includes SWE-specific fields
- Added `<diagnostic_context>` section to skill_manager_prompt explaining how to use confidence/error_location/skill_tags

### Step 3: Make SWEReflectorOutput extend ReflectorOutput
- **Status:** DONE
- **Files:** `src/prompts/outputs.py`
- `SWEReflectorOutput` now extends `ReflectorOutput` from ACE
- Uses `@model_validator(mode="after")` to compute `extracted_learnings` from typed categories
- `isinstance(output, ReflectorOutput)` now returns True
- Kept `get_all_learnings_as_dicts()` for backward compatibility with existing tests

### Step 4: Fix atomicity_score default
- **Status:** DONE
- **Files:** `src/prompts/outputs.py`
- Changed default from 0.0 (Rejected) to 0.8 (Fair) in AntiPattern, Discovery, UnvalidatedHypothesis

### Step 5: Fix CONSIDER: rejection criteria conflict
- **Status:** DONE
- **Files:** `src/prompts/skill_manager_prompt.py`
- Removed "consider" from bare-word rejection list, added note about `CONSIDER:` prefix being valid

### Step 6: Extract shared model resolution logic
- **Status:** DONE
- **Files:** `src/prompts/model_utils.py` (NEW), `src/prompts/custom_reflector.py`, `src/prompts/custom_skill_manager.py`
- Created `model_utils.py` with `resolve_ace_model()` and `make_pydantic_agent()` helpers
- Both SWEReflector and SWESkillManager now use `make_pydantic_agent()` — ~40 lines of duplication removed

### Step 7-8: Minor fixes and exports
- **Status:** DONE
- Guard against empty model name from trailing slash in `model_utils.py`
- `__init__.py` unchanged (no new public exports needed)

---

## Plan 1: Fix Reflector Handling of Successful Trajectories

### Change 1: Add resolved/ground_truth to LearnPhase.run()
- **Status:** DONE
- **Files:** `src/phases/learn.py`
- Added `resolved: bool = False` and `ground_truth: Optional[str] = None` params
- Feedback fallback: auto-generates "All tests passed" or "Tests failed" when feedback is None
- Forwards both to `self.reflector.reflect()`
- Progress string now says "resolved the issue successfully" vs "tests failed" based on outcome

### Change 2: Override reflect() in SWEReflector
- **Status:** DONE
- **Files:** `src/prompts/custom_reflector.py`
- Overrode `reflect()` to accept `resolved`, build `{outcome}` / `{outcome_instructions}` variables
- Success: "SUCCEEDED. Focus on what worked" + positive learning extraction
- Failure: "FAILED. Focus on anti-patterns" + negative learning extraction
- Formats prompt template directly (bypasses base Reflector.reflect to avoid double-formatting)

### Change 3: Update reflector_prompt.py with outcome variables
- **Status:** DONE
- **Files:** `src/prompts/reflector_prompt.py`
- Header now says "extract concrete learnings from trajectories" (not just FAILURES)
- Key Rule: "From failures extract anti-patterns; from successes extract reusable strategies"
- Replaced hardcoded "TRAJECTORIES THAT FAILED" with `{outcome}` and `{outcome_instructions}`
- Priority 0 trigger: "outcome indicates failure"

### Change 4: Wire feedback/ground_truth/resolved from main_loop
- **Status:** DONE
- **Files:** `src/runners/main_loop.py`
- Added `_build_ground_truth()` helper — extracts FAIL_TO_PASS/PASS_TO_PASS test lists (not gold patch)
  - Truncates to 4000 chars (some instances like django-10097 have 140K chars)
- Updated all 4 learn.run() call sites:
  1. Sequential path (line ~380): passes evaluate_result.feedback, ground_truth, evaluate_result.resolved
  2. Concurrent path (line ~495): same params
  3. Baseline reuse (line ~1296): passes result_data.get("feedback"), ground_truth, resolved
  4. Teacher trajectory (line ~1364): passes ground_truth, resolved (no feedback in teacher trajs)

### Post-plan fix: ground_truth truncation + max_tokens increase
- **Status:** DONE
- `config.yaml` `llm.ace.max_tokens` increased from 4096 to 8912 (richer output from new prompt)
- `_build_ground_truth()` capped at 4000 chars to prevent prompt explosion on instances with massive test lists

---

## Verification

### Unit tests
```
176 passed, 8 deselected in 4.63s
```

### End-to-end smoke tests (final, 0 errors)
```
10 passed, 0 failed, 0 skipped (300s)
0 errors across all 10 run logs
```

---

## Files Modified

| File | Plan | Changes |
|------|------|---------|
| `config.yaml` | Post-fix | `max_tokens` 4096 → 8912 |
| `src/config/llm.py` | P0 Step 1a | +`create_model_settings()` |
| `src/prompts/model_utils.py` | P0 Step 6a | NEW: shared model resolution |
| `src/prompts/outputs.py` | P0 Steps 3,4 | ReflectorOutput subclass, atomicity fix |
| `src/prompts/custom_reflector.py` | P0 Step 1b+6b, P1 Ch 2 | model_settings, reflect() override |
| `src/prompts/custom_skill_manager.py` | P0 Steps 1c+2a+6c | model_settings, update_skills() override |
| `src/prompts/skill_manager_prompt.py` | P0 Steps 2b, 5a | diagnostic_context, CONSIDER fix |
| `src/prompts/reflector_prompt.py` | P1 Ch 3 | outcome variables |
| `src/phases/learn.py` | P1 Ch 1 | resolved/ground_truth params |
| `src/cli/commands.py` | P0 Step 1d | ModelSettings wiring at 3 sites |
| `src/runners/main_loop.py` | P1 Ch 4 | _build_ground_truth(truncated), 4 call sites |
| `src/prompts/__init__.py` | — | Unchanged |
