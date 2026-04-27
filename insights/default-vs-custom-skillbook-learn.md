# Default vs Custom SWE Learn Phase Skillbooks

## Overview

The ACE framework supports two learning modes for the **Learn Phase** (Reflector + SkillManager pipeline):

| | Default (`ace` package) | Custom SWE (`--custom-swe-learn`) |
|---|---|---|
| **Reflector** | `ace.Reflector` with `ReflectorOutput` | `SWEReflector` with `SWEReflectorOutput` |
| **SkillManager** | `ace.SkillManager` with `SKILL_MANAGER_PROMPT` | `SWESkillManager` with `CUSTOM_SKILL_MANAGER_PROMPT` |
| **Skill type** | Generic atomic learnings | Typed learnings with semantic prefixes |
| **Output model** | `ace.core.outputs.ReflectorOutput` | `prompts.outputs.SWEReflectorOutput` |
| **Prompt** | `ace.implementations.prompts.REFLECTOR_PROMPT` | `prompts.reflector_prompt.CUSTOM_REFLECTOR_PROMPT` |

---

## Reflector Comparison

### Default Reflector

The default `ace.Reflector` (`ace/implementations/reflector.py`) uses a generic prompt (`REFLECTOR_PROMPT`) that treats all outcomes uniformly. It extracts a flat list of `ExtractedLearning` objects with no semantic type differentiation.

**Learning extraction is homogeneous** — every extracted learning is just an `ExtractedLearning` with a free-text `learning` field, an `atomicity_score`, `evidence`, and optional `justification`. There is no distinction between "this was a mistake", "this was a verified fact", or "this is an untested guess".

**Default ReflectorOutput schema** (`ace/core/outputs.py`):

```
ReflectorOutput
├── reasoning: str
├── error_identification: str
├── root_cause_analysis: str
├── correct_approach: str
├── key_insight: str
├── extracted_learnings: List[ExtractedLearning]    # <-- FLAT, UNTYPED
│       ├── learning: str
│       ├── atomicity_score: float
│       ├── evidence: str
│       └── justification: str
├── skill_tags: List[SkillTag]
│       ├── id: str
│       └── tag: str
└── raw: dict
```

**Priority protocol (default):**
1. SUCCESS_CASE_DETECTED
2. CALCULATION_ERROR_DETECTED
3. STRATEGY_MISAPPLICATION_DETECTED
4. WRONG_STRATEGY_SELECTED
5. MISSING_STRATEGY_DETECTED

Success and failure are handled by the same priorities — the reflector doesn't have a dedicated failure-first protocol.

---

### Custom SWEReflector

The `SWEReflector` (`prompts/custom_reflector.py`) uses `CUSTOM_REFLECTOR_PROMPT` with a failure-first diagnostic protocol. It splits extracted learnings into **three typed categories** with distinct semantics.

**Custom SWEReflectorOutput schema** (`prompts/outputs.py`):

```
SWEReflectorOutput
├── reasoning: str
├── error_identification: str
├── error_location: str                          # <-- NEW: exact step
├── root_cause_analysis: str
├── correct_approach: str
├── anti_patterns: List[AntiPattern]             # <-- TYPED: what NOT to do
│       ├── pattern: str
│       ├── why_harmful: str
│       ├── atomicity_score: float
│       └── evidence: str
├── discoveries: List[Discovery]                 # <-- TYPED: verified facts
│       ├── finding: str
│       ├── atomicity_score: float
│       └── evidence: str
├── unvalidated_hypotheses: List[UnvalidatedHypothesis]  # <-- TYPED: untested claims
│       ├── hypothesis: str
│       ├── why_unvalidated: str
│       ├── atomicity_score: float
│       └── evidence: str
├── key_insight: str
├── confidence_in_analysis: float
├── skill_tags: List[SkillTag]
│       ├── id: str
│       ├── tag: str
│       ├── justification: str
│       └── impact_score: float
└── raw: dict
```

**Compatibility bridge**: `SWEReflectorOutput` provides an `extracted_learnings` property that converts typed learnings to `ExtractedLearning` objects with type prefixes for the SkillManager:
- `[ANTI-PATTERN]` prefix from `AntiPattern`
- `[DISCOVERY]` prefix from `Discovery`
- `[HYPOTHESIS]` prefix from `UnvalidatedHypothesis`

**Priority protocol (custom SWE):**
0. **FAILED_ATTEMPT_ANALYSIS** (NEW, highest priority) — dedicated failure-first protocol
1. SUCCESS_CASE_DETECTED
2. CALCULATION_ERROR_DETECTED
3. STRATEGY_MISAPPLICATION_DETECTED
4. WRONG_STRATEGY_SELECTED
5. MISSING_STRATEGY_DETECTED

The critical difference is **Priority 0**: a dedicated protocol for analyzing failures that enforces:
- Extraction of **false assumptions** (what the agent believed that was wrong)
- Extraction of **anti-patterns** (behaviors that led to failure)
- Detection of **false confidence markers** (agent claims success without evidence)
- Explicit **forbidden extractions** — never extract "the solution is..." from a failed attempt

---

## SkillManager Comparison

### Default SkillManager

The default `ace.SkillManager` (`ace/implementations/skill_manager.py`) uses `SKILL_MANAGER_PROMPT`. It processes the flat `extracted_learnings` list and produces `ADD`/`UPDATE`/`TAG`/`REMOVE` operations with plain imperative content strings.

**Default skill content**: plain imperative commands.
```
"Verify intermediate multiplication results"
"Use pandas.read_csv() for CSV files"
"Round financial calculations to 4 decimal places"
```

### Custom SWESkillManager

The `SWESkillManager` (`prompts/custom_skill_manager.py`) uses `CUSTOM_SKILL_MANAGER_PROMPT`. It preserves learning type information by converting the `[ANTI-PATTERN]`/`[DISCOVERY]`/`[HYPOTHESIS]` prefixes into semantic content prefixes:

| Reflector Prefix | SkillManager Content Prefix | Semantic |
|---|---|---|
| `[ANTI-PATTERN]` | `AVOID:` | Negative advice — what NOT to do |
| `[DISCOVERY]` | `VERIFIED:` | Positive guidance — confirmed facts |
| `[HYPOTHESIS]` | `CONSIDER:` | Conditional guidance — needs verification |

**Custom skill content** (with semantic prefixes):
```
"AVOID: Claiming task completion without verifying git diff shows actual changes"
"VERIFIED: Use /testbed directory for file modifications (writable in Docker)"
"CONSIDER: Modify RST class constructor (verify first)"
```

---

## Skill Types Comparison (Extracted from Real Data)

### Default Skillbook (`bug_fix` sections, plain content)

Skills from the default learn pipeline are sectioned by general categories (`bug_fix`, `testing`, etc.) with plain descriptive content:

```
# Section: bug_fix
- "Preserve _original_dpi during unpickling by checking if it exists before setting it"
- "Validate kwargs in set_ticks even when labels=None"
- "Raise TypeError with clear message in set_ticks when kwargs provided without labels"
- "Implement __getstate__ and __setstate__ methods for classes containing weakrefs"

# Section: testing
- "Test pickle round-trip for objects containing weakrefs"
```

**Problem**: These read as if the agent *knows* the solution. When the agent actually failed, future iterations trust these as verified fixes — but they were extracted from failed trajectories.

### Custom Skillbook (semantic prefixes, typed content)

Skills from the custom SWE learn pipeline carry semantic prefixes that signal their epistemic status:

```
# Section: verification
- "AVOID: Claiming task completion without verifying git diff shows actual changes"

# Section: environment
- "VERIFIED: Use /testbed directory for file modifications (writable in Docker)"

# Section: approach
- "CONSIDER: Modify RST class constructor (verify first)"
```

**Advantage**: Future agents can immediately distinguish between warnings (`AVOID`), confirmed facts (`VERIFIED`), and speculative guidance (`CONSIDER`) — preventing the common failure mode of treating untested claims as solutions.

---

## Schemas

### Default Learn Phase Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Default ACE Learn Pipeline                      │
│                                                                      │
│  Trajectory ──► Reflector ──────────────► SkillManager ──► Skillbook │
│                 (REFLECTOR_PROMPT)         (SKILL_MANAGER_PROMPT)     │
│                        │                         │                    │
│                        ▼                         ▼                    │
│                 ReflectorOutput           SkillManagerOutput          │
│                 ┌────────────────┐       ┌──────────────────┐        │
│                 │ reasoning      │       │ update: UpdateBatch│      │
│                 │ error_ident.   │       │  └─ reasoning      │      │
│                 │ root_cause     │       │  └─ operations[]   │      │
│                 │ correct_approach│      │      ├─ ADD         │      │
│                 │ key_insight    │       │      ├─ UPDATE      │      │
│                 │ extracted_     │       │      ├─ TAG         │      │
│                 │  learnings[] ──┼──────►│      └─ REMOVE      │      │
│                 │  (flat, untyped)│      │                     │      │
│                 │ skill_tags[]   │       └──────────────────┘        │
│                 └────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Custom SWE Learn Phase Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Custom SWE Learn Pipeline                              │
│                                                                           │
│  Trajectory ──► SWEReflector ──────────────► SWESkillManager ──► Skillbook│
│                 (CUSTOM_REFLECTOR_PROMPT)      (CUSTOM_SKILL_MANAGER_    │
│                        │                       PROMPT)                    │
│                        ▼                              │                   │
│                 SWEReflectorOutput                    ▼                   │
│                 ┌────────────────────────┐    SkillManagerOutput          │
│                 │ reasoning              │    ┌──────────────────┐        │
│                 │ error_identification   │    │ update: UpdateBatch│      │
│                 │ error_location (NEW)   │    │  └─ reasoning      │      │
│                 │ root_cause_analysis    │    │  └─ operations[]   │      │
│                 │ correct_approach       │    │      ├─ ADD         │      │
│                 │                        │    │      │  (content with     │
│                 │ ┌──────────────────┐   │    │      │   AVOID:/          │
│                 │ │ anti_patterns[]  │   │    │      │   VERIFIED:/       │
│                 │ │  ├─ pattern      │   │    │      │   CONSIDER: prefix)│
│                 │ │  ├─ why_harmful  │   │    │      ├─ UPDATE      │      │
│                 │ │  ├─ atomicity    │   │    │      ├─ TAG         │      │
│                 │ │  └─ evidence     │   │    │      └─ REMOVE      │      │
│                 │ ├──────────────────┤   │    │                     │      │
│                 │ │ discoveries[]    │   │    └──────────────────┘        │
│                 │ │  ├─ finding      │   │                                │
│                 │ │  ├─ atomicity    │   │                                │
│                 │ │  └─ evidence     │   │                                │
│                 │ ├──────────────────┤   │                                │
│                 │ │ unvalidated_     │   │                                │
│                 │ │ hypotheses[]     │   │                                │
│                 │ │  ├─ hypothesis   │   │                                │
│                 │ │  ├─ why_unvalid. │   │                                │
│                 │ │  ├─ atomicity    │   │                                │
│                 │ │  └─ evidence     │   │                                │
│                 │ └──────────────────┘   │                                │
│                 │ key_insight            │                                │
│                 │ confidence             │                                │
│                 │ skill_tags[]           │                                │
│                 │   └─ impact_score (NEW)│                                │
│                 └────────────────────────┘                                │
│                        │                                                  │
│                        ▼  (compatibility bridge)                          │
│                 extracted_learnings property                               │
│                 ┌────────────────────────┐                                │
│                 │ [ANTI-PATTERN] <text>  │                                │
│                 │ [DISCOVERY] <text>     │──► SWESkillManager             │
│                 │ [HYPOTHESIS] <text>    │                                │
│                 └────────────────────────┘                                │
└──────────────────────────────────────────────────────────────────────────┘
```

### Skill Content Schema in Skillbook

```json
// Default skill — plain imperative content
{
  "id": "bug_fix-00001",
  "section": "bug_fix",
  "content": "Preserve _original_dpi during unpickling by checking if it exists",
  "justification": null,
  "evidence": null,
  "helpful": 0,
  "harmful": 0,
  "neutral": 0
}

// Custom SWE skill — content with semantic prefix
{
  "id": "verification-00001",
  "section": "verification",
  "content": "AVOID: Claiming task completion without verifying git diff shows actual changes",
  "justification": "Prevents submission of empty patches",
  "evidence": "Agent said 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' but git diff was empty",
  "helpful": 0,
  "harmful": 0,
  "neutral": 0
}
```

---

## Key Differences Summary

| Aspect | Default | Custom SWE |
|---|---|---|
| **Learning types** | 1 flat type (`ExtractedLearning`) | 3 typed categories (`AntiPattern`, `Discovery`, `UnvalidatedHypothesis`) |
| **Failure handling** | Same priority as success | Dedicated Priority 0 with failure-specific protocol |
| **Content prefixes** | None | `AVOID:` / `VERIFIED:` / `CONSIDER:` |
| **Epistemic status** | Implicit (all treated equally) | Explicit (warning vs fact vs speculation) |
| **SWE-bench anti-patterns** | Not covered | Built-in catalog (false "cannot modify" claims, test-script-as-fix, unverified success claims, following failed skillbook "solutions") |
| **Error location** | Not in output model | `error_location` field (exact step) |
| **Confidence/impact** | Not scored | `confidence_in_analysis` + `impact_score` per skill tag |
| **False confidence detection** | Not covered | Explicit detection of agent claiming knowledge it doesn't have |
| **Skill justification/evidence** | Typically `null` in practice | Populated from structured extraction |

---

## Source Files

| Component | Default (ACE package) | Custom SWE |
|---|---|---|
| Reflector class | `ace/implementations/reflector.py` | `src/prompts/custom_reflector.py` |
| Reflector prompt | `ace/implementations/prompts.py` (`REFLECTOR_PROMPT`) | `src/prompts/reflector_prompt.py` (`CUSTOM_REFLECTOR_PROMPT`) |
| Reflector output | `ace/core/outputs.py` (`ReflectorOutput`) | `src/prompts/outputs.py` (`SWEReflectorOutput`) |
| SkillManager class | `ace/implementations/skill_manager.py` | `src/prompts/custom_skill_manager.py` |
| SkillManager prompt | `ace/implementations/prompts.py` (`SKILL_MANAGER_PROMPT`) | `src/prompts/skill_manager_prompt.py` (`CUSTOM_SKILL_MANAGER_PROMPT`) |
| Skillbook core | `ace/core/skillbook.py` (shared) | `ace/core/skillbook.py` (shared) |
| Learn phase | `src/phases/learn.py` (shared orchestrator) | `src/phases/learn.py` (shared orchestrator) |
