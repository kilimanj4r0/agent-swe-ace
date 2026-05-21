# ACE-SWE: Experiment Pipeline Overview

## Table of Contents

1. [One-Line Summary](#1-one-line-summary)
2. [Core Loop](#2-core-loop)
3. [Execution Modes](#3-execution-modes)
4. [Orchestration Dispatch Tree](#4-orchestration-dispatch-tree)
5. [Agent & Prediction](#5-agent--prediction)
6. [Context Management](#6-context-management)
7. [Evaluation Pipeline](#7-evaluation-pipeline)
8. [Learning Pipeline](#8-learning-pipeline)
9. [Skillbook System](#9-skillbook-system)
10. [Configuration System](#10-configuration-system)
11. [Data Structures](#11-data-structures)
12. [Output Directory Layout](#12-output-directory-layout)
13. [Concurrency Model](#13-concurrency-model)
14. [Resume & Baseline Reuse](#14-resume--baseline-reuse)

---

## 1. One-Line Summary

An LLM agent resolves GitHub issues (SWE-bench), and when it fails, a reflector extracts skills into a skillbook that improves future attempts — with multi-repo train/val experiments measuring the skillbook's impact.

---

## 2. Core Loop

### High-Level

```
Instance → Agent reads issue + skillbook → generates patch → test in Docker
                                                         ↓
                                              Resolved? ── Yes → Done
                                                  │
                                                  No
                                                  ↓
                                        Reflector analyzes failure
                                                  ↓
                                        Skill added to skillbook
                                                  ↓
                                           Retry (up to N)
```

### Detailed

```
┌──────────────────────────────────────────────────────────────────┐
│  Instance Start                                                  │
│  • load instance from HuggingFace dataset                        │
│  • get skillbook (empty | per_repo | global)                     │
│  • determine start iteration (0 or resume point)                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │         PREDICT              │
              │                             │
              │  1. Format skillbook into    │
              │     prompt section           │
              │  2. MiniSWEAgent.run()       │
              │     ┌───────────────────┐    │
              │     │ system prompt     │    │
              │     │ instance + skills │    │
              │     │ agent loop:       │    │
              │     │  query → parse    │    │
              │     │  bash → execute   │    │
              │     │  in Docker /testbed│   │
              │     │  → observe → loop │    │
              │     │  until submit or  │    │
              │     │  limits exceeded  │    │
              │     └───────────────────┘    │
              │  3. Extract patch (git diff) │
              │  4. Save trajectory JSON     │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │         EVALUATE             │
              │                             │
              │  1. Validate patch format    │
              │     (diff --git or unified)  │
              │  2. Acquire eval lock        │
              │  3. Monkey-patch log dir     │
              │  4. Docker: apply patch +    │
              │     run test suite           │
              │     (swebench run_instance)  │
              │  5. Restore log dir          │
              │  6. Release lock             │
              │  7. Save result JSON         │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │   Resolved?     │
                    └──┬──────────┬───┘
                  Yes  │          │ No
                       ▼          │
                  ┌─────────┐     │
                  │  Done   │     │
                  └─────────┘     │
                                  ▼
                    ┌─────────────▼─────────────┐
                    │  Max attempts reached?    │
                    └──┬────────────────────┬───┘
                    Yes │                    │ No
                       ▼                     │
                  ┌──────────┐               │
                  │ Done     │               │
                  │(unsolved)│               │
                  └──────────┘               │
                                             ▼
                    ┌────────────────────────▼────────────────────┐
                    │                  LEARN                       │
                    │                                              │
                    │  1. Extract trajectory → AgentOutput         │
                    │  2. Reflector.analyze(failure trajectory)    │
                    │     ┌─────────────────────────────┐         │
                    │     │ SWE mode:                    │         │
                    │     │  → anti_patterns             │         │
                    │     │  → discoveries               │         │
                    │     │  → unvalidated_hypotheses    │         │
                    │     │                              │         │
                    │     │ Default mode:                │         │
                    │     │  → extracted_learnings       │         │
                    │     └─────────────────────────────┘         │
                    │  3. SkillManager → UpdateBatch(ops)         │
                    │  4. Apply operations to skillbook            │
                    │  5. Deduplicate (cosine sim ≥ 0.85)         │
                    │  6. Save skillbook JSON                      │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                                    iter = iter + 1
                                    ─────────────────
                                    back to PREDICT
```

---

## 3. Execution Modes

### Mode Summary Table

| Mode | Trigger | Train | Val Baseline | Val Skillbook | Skillbook |
|------|---------|-------|-------------|---------------|-----------|
| **Single-phase** | default (no val_ratio) | all instances | — | — | per_instance |
| **Two-phase** | `val_ratio` + single repo | train split | empty SB | learned SB | per_repo or global |
| **Iterate repos** | `iterate_repos` list | per-repo train | per-repo empty | per-repo learned | per_repo |
| **Validation-only** | `skillbook_source_dir` | skipped | empty SB | preloaded SB | loaded from disk |
| **Distillation** | `train_trajs_dir` | learn from teacher | empty SB | distilled SB | per_repo or global |
| **Skip-learn** | `skip_learn: true` | all instances | — | — | none (no skillbooks) |

### Mode 1: Single-Phase (default)

```
Instances → [Predict → Evaluate → (Learn)] × max_attempts → statistics.json
```
- Each instance gets its own skillbook (per_instance) or shares one (per_repo/global).
- Learning happens on failure; skills carry across iterations.

### Mode 2: Two-Phase (single repo)

```
┌─────────── TRAIN ───────────┐     ┌────── VAL BASELINE ──────┐     ┌───── VAL SKILLBOOK ──────┐
│                              │     │                           │     │                           │
│  For each train instance:   │     │  For each val instance:   │     │  For each val instance:   │
│    predict → eval → learn   │     │    predict → eval         │     │    predict → eval         │
│    (1 attempt, force_learn) │     │    (K attempts, no learn) │     │    (K attempts, no learn) │
│    skillbook accumulates    │     │    empty skillbook        │     │    learned skillbook      │
│                              │     │                           │     │                           │
└──────────┬───────────────────┘     └───────────┬───────────────┘     └───────────┬───────────────┘
           │                                     │                                 │
           ▼                                     ▼                                 ▼
    final_skillbook.json             val_baseline stats                 val_skillbook stats
                                     (raw ability)                      (skillbook benefit)
```

### Mode 3: Iterate Repos

```
┌──────────────────────────────────────────────────────────────┐
│                    iterate_repos orchestrator                 │
│                                                              │
│  repos = ["django/django", "flask/flask", "sympy/sympy"]     │
│                                                              │
│  for each repo (sequential or parallel):                     │
│    ┌──────────────────────────────────────────────────┐      │
│    │  _run_single_repo_experiment(repo)                │      │
│    │                                                    │      │
│    │  1. Split repo's instances into train/val          │      │
│    │  2. Run two-phase experiment (train → val)         │      │
│    │  3. Save per_repo/<repo>/final_skillbook.json      │      │
│    │  4. Save statistics_per_repo/<repo>.json            │      │
│    └──────────────────────────────────────────────────┘      │
│                                                              │
│  After all repos: aggregate into combined statistics.json     │
└──────────────────────────────────────────────────────────────┘
```

### Mode 4: Validation-Only

```
skillbook_source_dir → load per-repo skillbooks → skip training entirely
                                                          │
                    ┌─────────────────────────────────────┘
                    │
                    ▼
              Val Baseline (empty SB)  +  Val Skillbook (loaded SB)
```

### Mode 5: Teacher Distillation

```
┌──────────────────────────────────────────────────────────┐
│  TRAIN (distillation)                                     │
│                                                           │
│  For each train instance:                                 │
│    1. Load teacher trajectory from train_trajs_dir        │
│       ({trajs_dir}/{instance_id}/{instance_id}.traj.json) │
│    2. If not found → skip (teacher_trajs_skipped++)       │
│    3. If found → run LEARN only (no predict, no eval)     │
│    4. Skillbook accumulates from teacher insights         │
│                                                           │
│  After training: same val flow as two-phase               │
└──────────────────────────────────────────────────────────┘
```

### Mode 6: Skip-Learn (raw benchmark)

```
Instances → [Predict → Evaluate] × max_attempts → statistics.json
             (no learning, no skillbooks)
```

---

## 4. Orchestration Dispatch Tree

```
main()  [src/cli/commands.py]
  │
  ├── --phase predict  → run_predict_cmd()
  ├── --phase evaluate → run_evaluate_cmd()
  ├── --phase learn    → run_learn_cmd()
  │
  └── --phase all (default) → run_full_experiment()
        │
        ├── dry_run? → _run_dry_run() → exit
        │
        ├── iterate_repos? → _run_iterate_repos()
        │     │
        │     ├── Sequential (concurrency=1 or 1 repo)
        │     │     for repo in repos:
        │     │       _run_single_repo_experiment(repo)
        │     │
        │     └── Parallel (concurrency > 1)
        │           ThreadPoolExecutor(min(concurrency, n_repos))
        │           → _run_single_repo_experiment(repo) per thread
        │
        └── Single experiment (default path)
              │
              split_instances() → train_instances, val_instances
              scan_resume_dirs() → remove completed from train
              │
              ExperimentLoop.run(train, config, val, baseline_run_dir, ...)
                │
                ├── val empty? → SINGLE-PHASE
                │     for instance: _run_instance_inner()
                │       predict → evaluate → learn? → retry?
                │
                └── val present? → TWO-PHASE
                      │
                      ├── TRAIN
                      │   ├── train_trajs_dir? → distillation worker
                      │   ├── baseline_run_dir? → baseline-reuse worker
                      │   └── default → predict+eval+learn worker
                      │   (all: force_learn=True, max_attempts=1)
                      │
                      ├── Post-train dedup sweep
                      ├── Save final_skillbook.json
                      │
                      ├── VAL BASELINE
                      │   └── _run_val_pass(skillbook=empty, frozen=True)
                      │
                      └── VAL SKILLBOOK
                          └── _run_val_pass(skillbook=learned, frozen=True)
```

---

## 5. Agent & Prediction

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    MiniSWEAgent                             │
│                                                            │
│  Config:                                                   │
│    step_limit=100  cost_limit=$5  context_window=65536     │
│                                                            │
│  LLM: LitellmModel (agent LLM via vLLM or Z.AI)           │
│  Env: DockerEnvironment (/testbed in SWE-bench image)      │
│  Templates: system + instance(+skillbook) + observation    │
│                                                            │
│  Tool: bash execution (ONE command per response)           │
│  Submission: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT    │
│              && git add -A && git diff --cached             │
└────────────────────────────────────────────────────────────┘
```

### Agent Loop

```
┌─────────────────────────────────────────────────┐
│  System message (swebench.yaml template)         │
├─────────────────────────────────────────────────┤
│  Instance message (problem + skillbook)          │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──► query(messages) ──► LLM completion         │
│  │         │                                     │
│  │    parse_action()                             │
│  │    extract ```bash ... ``` from response       │
│  │         │                                     │
│  │    execute_action()                           │
│  │    run bash in Docker /testbed                 │
│  │         │                                     │
│  │    has_finished()?                            │
│  │    ├── output starts with FINAL_OUTPUT marker  │
│  │    │   and returncode=0 → raise Submitted     │
│  │    └── no → continue                          │
│  │         │                                     │
│  │    get_observation()                          │
│  │    format output (trunc 10000 chars)           │
│  │         │                                     │
│  └─────────┘                                     │
│                                                  │
│  Exit: Submitted | LimitsExceeded | error        │
└─────────────────────────────────────────────────┘
```

### Skillbook Injection

Skills are injected into the instance template as a `## Learned Strategies (Skillbook)` section:

```
## Learned Strategies (Skillbook)

> CRITICAL REMINDERS:
> 1. Skills describe approaches, NOT complete solutions
> 2. Only ONE command per response
> 3. Verify patch with `git diff --cached`
> 4. Source code in `/testbed` IS writable

### verification-00001

AVOID: Claiming task completion without verifying git diff shows actual changes
**Why this helps:** Prevents empty patch submission

### debugging-00002

AVOID: Using sed command with improper syntax for code modifications
**Why this helps:** Prevents malformed edits that silently fail

<example_response>
... (rest of template)
```

---

## 6. Context Management

### 5-Level Truncation Strategy

When estimated tokens exceed `85% × (context_window - max_tokens - 2000)`:

```
Messages are split into three groups:
┌─────────────────┬──────────────────────────────┬──────────────────┐
│  PROTECTED (0-1) │         MIDDLE               │  RECENT (last 6) │
│  system, task    │  (main truncation target)    │  (kept intact)   │
└─────────────────┴──────────────────────────────┴──────────────────┘

Level 1: Drop format-error messages from middle
         ("Please always provide EXACTLY ONE action")

Level 2: Truncate old observation content (user messages)
         → head/tail 500 chars each: "...[N chars truncated]..."

Level 3: Truncate old assistant messages
         → keep only last ```bash``` block + truncated reasoning

Level 4: Drop oldest middle messages entirely
         → pop from front until budget met

Level 5: Truncate ALL non-protected messages to 500 chars

If still over budget: halve keep_recent (6→3→2) and re-run all levels.
```

---

## 7. Evaluation Pipeline

### Flow

```
patch (string)
    │
    ├── empty/whitespace? → resolved=False, patch_empty=1.0
    ├── invalid format?   → resolved=False, patch_invalid_format=1.0
    │
    ▼ (valid patch)
validate_patch_docker()
    │
    ├── acquire _eval_lock (threading.Lock)
    │     │
    │     ├── monkey-patch swebench RUN_EVALUATION_LOG_DIR → {output_dir}/eval_logs/
    │     ├── build TestSpec from instance + Docker namespace
    │     ├── Docker image pull/create
    │     ├── Start container
    │     ├── Apply patch (git apply)
    │     ├── Run test suite (instance-specific tests)
    │     ├── Capture pass/fail
    │     ├── Optionally remove image (rm_image=True)
    │     └── restore log dir constants (finally)
    │
    └── release lock
    │
    ▼
result: bool (resolved) → EvaluateResult → save_result()
```

### Key Properties

- **Serialized**: global `threading.Lock` — only one Docker eval at a time (swebench mutates global constants)
- **Timeout**: 1800s (30 min) per evaluation
- **Image cleanup**: `rm_image=True` by default to save disk
- **No retry**: if eval fails, returns `resolved=False`; the outer loop may retry from predict

---

## 8. Learning Pipeline

### High-Level Flow

```
Trajectory (messages) → Reflector → SkillManager → Skillbook
```

### Detailed Flow

```
┌────────────────────────────────────────────────────────────────┐
│  LEARN PHASE                                                   │
│                                                                │
│  1. Build AgentOutput from trajectory                          │
│     reasoning = concatenate assistant messages                 │
│     final_answer = patch content                               │
│                                                                │
│  2. REFLECTOR                                                  │
│     input: problem_statement, AgentOutput, Skillbook, feedback │
│     ┌───────────────────────────────────┐                      │
│     │        SWE Reflector               │                      │
│     │  Priority-ordered analysis:        │                      │
│     │   1. FAILED_ATTEMPT_ANALYSIS       │                      │
│     │   2. ERROR_TRACE_DIAGNOSIS         │                      │
│     │   3. ENVIRONMENT_CONSTRAINT        │                      │
│     │   4. STRATEGY_MISMATCH             │                      │
│     │   5. MISSING_STRATEGY_DETECTED     │                      │
│     │                                    │                      │
│     │  Output (3 categories):            │                      │
│     │   • anti_patterns: [AntiPattern]   │                      │
│     │   • discoveries: [Discovery]       │                      │
│     │   • unvalidated_hypotheses: [...]  │                      │
│     └───────────────────────────────────┘                      │
│     ┌───────────────────────────────────┐                      │
│     │      Default ACE Reflector         │                      │
│     │  Generic reflection on failure     │                      │
│     │  Output: extracted_learnings[]     │                      │
│     └───────────────────────────────────┘                      │
│                                                                │
│  3. SKILL MANAGER                                              │
│     input: reflection, skillbook, context                      │
│     ┌───────────────────────────────────┐                      │
│     │      SWE Skill Manager             │                      │
│     │  Prefix mapping:                   │                      │
│     │   [ANTI-PATTERN] → AVOID:          │                      │
│     │   [DISCOVERY]    → VERIFIED:       │                      │
│     │   [HYPOTHESIS]   → CONSIDER:       │                      │
│     │                                    │                      │
│     │  Operations:                       │                      │
│     │   ADD    — genuinely novel skill   │                      │
│     │   UPDATE — refine existing skill   │                      │
│     │   TAG    — mark helpful/harmful     │                      │
│     │   REMOVE — delete obsolete skill   │                      │
│     │                                    │                      │
│     │  Default: UPDATE over ADD           │                      │
│     │  Cap: 50 skills, then prioritize   │                      │
│     └───────────────────────────────────┘                      │
│     ┌───────────────────────────────────┐                      │
│     │    Default ACE Skill Manager       │                      │
│     │  Same operations, no prefixes      │                      │
│     │  Plain prose content               │                      │
│     └───────────────────────────────────┘                      │
│                                                                │
│  4. Apply UpdateBatch → skillbook                              │
│                                                                │
│  5. DEDUPLICATE                                                │
│     if dedup enabled:                                          │
│       embed all skills (all-MiniLM-L6-v2)                      │
│       for each pair with cosine_sim ≥ 0.85:                    │
│         both validated + close scores → KEEP both              │
│         both validated + different scores → MERGE (keep best)  │
│         one unvalidated → DELETE duplicate                     │
│                                                                │
│  6. SAVE skillbook JSON                                        │
└────────────────────────────────────────────────────────────────┘
```

### SWE Learn vs Default Learn

| Aspect | SWE Learn (`custom_swe_learn: true`) | Default Learn |
|--------|--------------------------------------|---------------|
| Reflector | `SWEReflector` — 5-priority diagnostic protocol | `ACE Reflector` — generic PydanticAI |
| Output | 3 categories: anti_patterns, discoveries, hypotheses | 1 list: extracted_learnings |
| Skill content | `AVOID:` / `VERIFIED:` / `CONSIDER:` prefixed | Plain prose |
| Section naming | Domain types (`debugging`, `bug_fixing`, `verification`) | Repo-area based |
| Anti-patterns | First-class, explicit | Implicit in content |
| Prompt | 338-line SWE-optimized prompt | ACE default |
| Skill cap | 50 skills, then prioritize updates | No cap |

---

## 9. Skillbook System

### Skill Data Model

```
Skill
├── id: str                    "{section_prefix}-{5-digit-counter}"
│                              e.g. "verification-00001"
├── section: str               Category for grouping
│                              e.g. "debugging", "django_url_handling"
├── content: str               The actual skill text
│                              "AVOID: Claiming success without..."
├── justification: str | None  Why this skill was created
├── evidence: str | None       Evidence from execution
├── helpful: int               Times tagged helpful
├── harmful: int               Times tagged harmful
├── neutral: int               Times tagged neutral
├── status: "active" | "invalid"  (soft-delete uses "invalid")
├── created_at: str            ISO timestamp
├── updated_at: str            ISO timestamp
├── embedding: float[] | None  Sentence embedding for dedup
└── sources: dict[]            Provenance tracking
```

### Skillbook Modes

```
┌─────────────────┬──────────────────────────┬───────────────────────────┐
│  per_instance    │       per_repo           │         global            │
├─────────────────┼──────────────────────────┼───────────────────────────┤
│ Fresh empty SB   │ One SB per repo          │ One SB for all repos      │
│ per instance     │ Dict[repo, Skillbook]    │ Single Skillbook          │
│                  │                          │                           │
│ Skills never     │ Skills accumulate        │ Skills accumulate         │
│ carry forward    │ within same repo         │ across all repos          │
│                  │                          │                           │
│ Single-phase     │ Two-phase compatible     │ Two-phase compatible      │
│ only             │ iterate_repos uses this  │                           │
│                  │                          │                           │
│ Learning works   │ After train:             │ After train:              │
│ but skills lost  │ per_repo/<repo>/         │ final_skillbook.json      │
│ between insts    │   final_skillbook.json   │                           │
└─────────────────┴──────────────────────────┴───────────────────────────┘
```

### Skillbook Evolution (per_repo, two-phase)

```
Instance A (train)     Instance B (train)     Instance C (val)
────────────────       ────────────────       ───────────────
iter 0: []             iter 0: [s1]           val_baseline: []
  ↓ learn               ↓ learn               (empty SB, no learn)
iter 1: [s1]           iter 1: [s1,s2]        val_skillbook: [s1,s2,s3]
  ↓ done                 ↓ done               (learned SB, no learn)

final_skillbook.json = [s1, s2, s3]
```

### Deduplication

```
Similarity Detection:
  all-MiniLM-L6-v2 embeddings → cosine similarity

  Example:
    Existing: "AVOID: Claiming success without verifying git diff"
    New:      "AVOID: Saying a fix is done without checking diff output"
    sim=0.91  (> 0.85 threshold)

Consolidation:
  ┌──────────────────┬──────────────────────────────────────┐
  │ Both validated,  │ scores close → KEEP both             │
  │ scores similar   │ (record decision, skip in future)    │
  ├──────────────────┼──────────────────────────────────────┤
  │ Both validated,  │ scores differ → MERGE                │
  │ scores differ    │ (keep higher, soft-delete lower)     │
  ├──────────────────┼──────────────────────────────────────┤
  │ One/both         │ DELETE the lower-scoring one         │
  │ unvalidated      │                                      │
  └──────────────────┴──────────────────────────────────────┘
```

---

## 10. Configuration System

### Config Hierarchy

```
config.yaml ──deep-merge──► --config override.yaml ──override──► CLI args
                                                                          │
                                                                          ▼
                                                                    merged config dict
```

Deep-merge: recursive dict merge — override values win; nested dicts recurse.

### Full Config Schema

```yaml
experiment:
  name: str
  description: str
  max_attempts: int          # default 10
  force_learn: bool          # default true
  concurrency: int           # default 1
  random_seed: int           # default 42
  skip_learn: bool           # skip Learn phase entirely
  val_pass_k: int            # attempts per val instance
  baseline_run_dir: str      # reuse baseline results
  skillbook_source_dir: str  # validation-only: load pre-trained skillbooks
  train_trajs_dir: str       # teacher distillation: learn from expert trajs
  train_concurrency: int     # per-repo train concurrency

  skillbook:
    mode: per_instance | per_repo | global
    custom_swe_learn: bool
    deduplication:
      enabled: bool
      similarity_threshold: float    # default 0.85
      embedding_provider: sentence_transformers | litellm
      local_model_name: str          # default all-MiniLM-L6-v2
      embedding_device: cpu | cuda
      within_section_only: bool      # default true

  split:
    val_ratio: float                 # e.g. 0.2 for 80/20 split

  resume_dirs:
    - str                            # paths to previous run dirs

environment:
  type: docker | local
  namespace: str                     # Docker registry namespace

agent:
  step_limit: int                    # default 100
  cost_limit: float                  # default 0.0 (local vLLM)
  context:
    enabled: bool
    context_window: int              # default 65536
    keep_recent_messages: int        # default 6
    truncate_threshold: float        # default 0.85

llm:
  agent:                             # MiniSWEAgent LLM
    provider: zai | hosted_vllm
    model: str
    api_key_env: str
    api_base: str
    temperature: float
    max_tokens: int
  ace:                               # Reflector/SkillManager LLM
    provider: zai | hosted_vllm
    model: str
    api_key_env: str
    api_base: str
    temperature: float
    max_tokens: int

benchmark:
  dataset: str                       # HuggingFace dataset name
  split: str                         # default "test"
  max_instances: int | null
  filter_repos: [str]
  iterate_repos: [str]              # per-repo two-phase experiments
  exclude_instances: [str]

evaluation:
  use_docker: bool                   # default true
  timeout: int                       # default 1800
  rm_image: bool                     # default false

output:
  dir: str                           # default "data"
  save_trajectories: bool
  save_skillbooks: bool
  save_logs: bool
  log_level: str

observability:
  enabled: bool                      # Opik tracing
  project_name: str
```

### LLM Provider Routing

```
┌─────────────────────────────────────────────────────────────┐
│  LLMConfig                                                  │
│                                                             │
│  hosted_vllm:                                               │
│    api_base required (e.g. http://localhost:8000/v1)        │
│    model string: "hosted_vllm/<model_name>"                 │
│    api_key: "not-needed" if unset                           │
│    → LitellmModel via LiteLLM                               │
│                                                             │
│  zai:                                                       │
│    api_key from ZAI_API_KEY env var                         │
│    model string: "zai/<model_name>"                         │
│    → LitellmModel via LiteLLM                               │
│                                                             │
│  agent LLM (llm.agent):                                     │
│    → create_model() → LitellmModel for MiniSWEAgent         │
│                                                             │
│  ace LLM (llm.ace):                                         │
│    → create_ace_client() → model string for PydanticAI      │
│    if api_base set: OPENAI_BASE_URL + "openai:" prefix      │
│    → Used by Reflector + SkillManager                       │
└─────────────────────────────────────────────────────────────┘
```

### Dual LLM Architecture

```
┌──────────────────────────┐    ┌──────────────────────────┐
│  llm.agent               │    │  llm.ace                 │
│  (e.g. Qwen3-30B vLLM)  │    │  (e.g. glm-4.5-flash)   │
│                          │    │                          │
│  Used by:                │    │  Used by:                │
│  MiniSWEAgent            │    │  SWEReflector            │
│  (predict phase)         │    │  SWESkillManager         │
│  bash command loop       │    │  (learn phase)           │
│                          │    │                          │
│  Via: LitellmModel       │    │  Via: PydanticAI         │
└──────────────────────────┘    └──────────────────────────┘
```

---

## 11. Data Structures

### Result Types

```
PredictResult
├── instance_id: str
├── iteration: int
├── exit_status: str           # "Submitted" | "LimitsExceeded" | "error"
├── patch: str                 # git diff output
├── trajectory: dict[]         # full message history
├── error: str | None
└── trajectory_path: Path | None

EvaluateResult
├── instance_id: str
├── iteration: int
├── resolved: bool
├── feedback: str
├── metrics: dict              # {resolved: 0|1, patch_length: int, ...}
└── result_path: Path | None

LearnResult
├── instance_id: str
├── iteration: int
├── skills_added: int
├── skills_updated: int
├── skills_removed: int
└── skillbook_path: Path | None

IterationResult
├── iteration: int
├── predict_result: PredictResult
├── evaluate_result: EvaluateResult
└── learn_result: LearnResult | None

InstanceResult
├── instance_id: str
├── iterations: IterationResult[]
├── final_resolved: bool
└── total_attempts: int

ResumePoint
├── resume_dir: str
├── last_complete_iter: int    # -1 if nothing complete
├── is_fully_complete: bool
├── break_reason: str
└── start_iteration: int       # property: last_complete_iter + 1
```

### SWEReflector Output Model

```
SWEReflectorOutput
├── reasoning: str
├── error_identification: str
├── error_location: str
├── root_cause_analysis: str
├── correct_approach: str
├── anti_patterns: AntiPattern[]
├── discoveries: Discovery[]
├── unvalidated_hypotheses: UnvalidatedHypothesis[]
├── key_insight: str
├── confidence_in_analysis: float
├── skill_tags: SkillTag[]
├── extracted_learnings: ExtractedLearning[]   # compatibility property
└── raw: dict
```

### UpdateOperation

```
UpdateOperation
├── type: "ADD" | "UPDATE" | "TAG" | "REMOVE"
├── section: str
├── content: str | None
├── skill_id: str | None
├── metadata: dict              # e.g. {"helpful": 1} for TAG
├── justification: str | None
├── evidence: str | None
├── insight_source: dict | None
└── learning_index: int | None  # links back to extracted_learning
```

---

## 12. Output Directory Layout

```
data/run_{YYYYMMDD_HHMMSS}/
├── config.json                          # merged config snapshot
├── statistics.json                      # run statistics
├── experiment.log                       # run log
│
├── statistics_per_repo/                 # iterate_repos mode only
│   ├── django__django.json
│   └── sympy__sympy.json
│
└── {benchmark}/                         # e.g. princeton-nlp__SWE-bench_Lite
    │
    ├── trajectories/
    │   ├── {instance_id}/               # single-phase / per_instance
    │   │   ├── iter_0.json
    │   │   └── iter_1.json
    │   ├── train/{instance_id}/         # two-phase
    │   ├── val_baseline/{instance_id}/
    │   └── val/{instance_id}/
    │
    ├── results/
    │   ├── {instance_id}/
    │   │   ├── iter_0.json
    │   │   └── iter_1.json
    │   ├── train/{instance_id}/
    │   ├── val_baseline/{instance_id}/
    │   └── val/{instance_id}/
    │
    ├── eval_logs/                       # Docker eval stdout/stderr
    │
    └── skillbooks/
        ├── {instance_id}/               # per_instance mode
        │   ├── iter_0.json              # (empty, not saved)
        │   ├── iter_1.json
        │   └── iter_2.json
        ├── train/                       # per_repo/global train phase
        │   ├── iter_0.json
        │   ├── iter_1.json
        │   └── ...
        ├── per_repo/                    # iterate_repos mode
        │   ├── django__django/
        │   │   └── final_skillbook.json
        │   └── sympy__sympy/
        │       └── final_skillbook.json
        ├── final_skillbook.json         # global mode
        └── skillbooks_statistics.json   # aggregate skillbook stats
```

### File Formats

**Trajectory JSON** (`iter_N.json`):
```json
{
  "info": {
    "exit_status": "Submitted",
    "submission": "diff --git a/...",
    "iteration": 0,
    "instance_id": "django__django-12184",
    "model": "hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "message_count": 24,
    "assistant_message_count": 12
  },
  "messages": [
    {"role": "system", "content": "...", "timestamp": 1716...},
    {"role": "user", "content": "...", "timestamp": 1716...},
    {"role": "assistant", "content": "...", "timestamp": 1716...}
  ]
}
```

**Result JSON** (`iter_N.json`):
```json
{
  "resolved": false,
  "feedback": "Test failed: ...",
  "metrics": {"resolved": 0.0, "patch_length": 1234},
  "instance_id": "django__django-12184",
  "iteration": 0,
  "timestamp": "2026-05-21T..."
}
```

**Skillbook JSON** (`iter_N.json` or `final_skillbook.json`):
```json
{
  "iteration": 1,
  "timestamp": "2026-05-21T...",
  "instance_id": "django__django-12184",
  "skill_count": 3,
  "skills": {
    "verification-00001": {
      "id": "verification-00001",
      "section": "verification",
      "justification": "Prevents empty patch submission",
      "evidence": "Agent said 'successfully implemented' but git diff was empty",
      "content": "AVOID: Claiming task completion without verifying git diff shows actual changes"
    }
  }
}
```

**Statistics JSON** — key sections:
```json
{
  "status": "completed",
  "total_instances": 300,
  "resolved_count": 45,
  "resolution_rate": 0.15,
  "resolved_ids": ["..."],
  "unresolved_ids": ["..."],

  "train_phase": {
    "total_instances": 240,
    "resolved_count": 40,
    "total_skills_learned": 50,
    "teacher_trajs_found": 200,
    "teacher_trajs_skipped": 5
  },
  "val_baseline_phase": {
    "resolved_count": 5,
    "resolution_rate": 0.083,
    "skillbook_skills": 0,
    "pass_at_k": {"pass@1": {"count": 5, "total": 60, "rate": 0.083}}
  },
  "val_skillbook_phase": {
    "resolved_count": 10,
    "resolution_rate": 0.167,
    "skillbook_skills": 50,
    "pass_at_k": {"pass@1": {"count": 10, "total": 60, "rate": 0.167}}
  },
  "summary": {
    "skillbook_improvement": "+0.083",
    "skillbook_improvement_pct": "+100.0%",
    "newly_resolved_by_skillbook": ["id1", "id2"],
    "lost_by_skillbook": []
  }
}
```

---

## 13. Concurrency Model

```
┌─────────────────────────────────────────────────────────────────┐
│  CONCURRENCY LEVELS                                              │
│                                                                  │
│  Level 1: iterate_repos (ThreadPoolExecutor)                     │
│    └── repos run in parallel (up to concurrency workers)         │
│                                                                  │
│  Level 2: per-repo training (ThreadPoolExecutor)                 │
│    └── train instances run in parallel within a repo              │
│        (requires baseline_run_dir; each worker gets own agent)    │
│                                                                  │
│  Level 3: single-phase instances (ThreadPoolExecutor)            │
│    └── instances predict in parallel; eval serialized             │
│                                                                  │
│  SERIALIZATION POINTS                                            │
│                                                                  │
│  1. _eval_lock (threading.Lock)                                  │
│     → Only one Docker eval at a time (swebench global mutation)  │
│                                                                  │
│  2. _shared_model_lock (for dedup)                               │
│     → SentenceTransformer model shared across threads             │
│                                                                  │
│  3. results_lock (threading.Lock)                                │
│     → Protects all_results, resolved_ids, unresolved_ids          │
│                                                                  │
│  PER-WORKER ISOLATION                                            │
│                                                                  │
│  Each concurrent worker gets:                                    │
│  • Own agent from agent_factory() (separate model + counters)    │
│  • Own PredictPhase                                              │
│  • Shared EvaluatePhase (serialized via _eval_lock)              │
│  • Shared LearnPhase (shared dedup model, own SB per mode)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 14. Resume & Baseline Reuse

### Resume

```
┌─────────────────────────────────────────────────────────────┐
│  --resume-dir data/run_A [data/run_B]                        │
│                                                              │
│  scan_resume_dirs():                                         │
│    For each instance in resume dirs:                         │
│      Walk iter_0, iter_1, ... looking for complete chain:    │
│        ✓ trajectory exists + valid exit_status               │
│        ✓ result exists                                       │
│        ✓ skillbook for next iter exists (unless skip_learn)  │
│      → ResumePoint(last_complete_iter, is_fully_complete)    │
│                                                              │
│    If instance in multiple dirs → pick highest iter          │
│                                                              │
│  Effect:                                                     │
│    • Fully complete instances → removed from train list      │
│    • Partial instances → start from last_complete_iter + 1   │
│    • Artifacts copied from resume dir to new run dir         │
└─────────────────────────────────────────────────────────────┘
```

### Baseline Reuse

```
┌─────────────────────────────────────────────────────────────┐
│  --baseline-run-dir data/run_baseline                        │
│                                                              │
│  Single-phase:                                               │
│    At iter_0: try load baseline trajectory+result            │
│    If found + valid exit_status → skip predict+eval          │
│    Learning and subsequent iterations proceed normally        │
│                                                              │
│  Two-phase train:                                            │
│    Try load baseline artifacts for each train instance       │
│    If baseline skillbook exists + compatible:                 │
│      → merge skills, skip learn                              │
│    If trajectory only:                                       │
│      → copy artifacts, run learn only                        │
│    If not found:                                             │
│      → full predict+eval+learn                               │
│                                                              │
│  Two-phase val baseline:                                     │
│    Load existing val_baseline results from baseline dir      │
│    Skip instances with enough completed iterations            │
│    Re-run only missing instances                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Real Skillbook Examples

### per_instance — `django__django-12184` (SWE learn)

**iter_1.json** — 3 skills after first failure:
```json
{
  "iteration": 1,
  "skill_count": 3,
  "skills": {
    "django_url_handling-00001": {
      "section": "django_url_handling",
      "content": "When optional regex groups in Django URL patterns don't match, do not pass them as positional arguments to view functions to prevent TypeError exceptions"
    },
    "django_development-00003": {
      "section": "django_development",
      "content": "Implement code fixes for framework issues directly in source files rather than just discussing solutions"
    }
  }
}
```

**iter_2.json** — skills refined (updated, not just appended):
```json
{
  "iteration": 2,
  "skill_count": 3,
  "skills": {
    "django_url_handling-00001": {
      "section": "django_url_handling",
      "content": "When optional regex groups in Django URL patterns don't match, do not pass them as **keyword** arguments to view functions to prevent TypeError exceptions"
    }
  }
}
```

### per_repo + SWE learn — `django_split_swe` (263 skills)
```json
{
  "skill_count": 263,
  "skills": {
    "debugging-00001": {
      "section": "debugging",
      "content": "AVOID: Creating test scripts instead of editing source files directly for bug fixes"
    },
    "code_modification-00012": {
      "section": "code_modification",
      "content": "AVOID: Claiming success without verifying git diff shows actual changes"
    }
  }
}
```

### per_repo + Default learn — `django_split_default` (417 skills)
```json
{
  "skill_count": 417,
  "skills": {
    "django_template_system-00001": {
      "section": "django_template_system",
      "content": "Track custom library names from TEMPLATES['OPTIONS']['libraries'] to prevent duplicate detection false positives"
    }
  }
}
```
