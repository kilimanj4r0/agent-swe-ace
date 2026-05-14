# ACE-SWE Experiment Pipeline Overview

## Core Loop (per instance)

```
                         ┌──────────────────────────────────────────┐
                         │           Instance Start                 │
                         │    (load skillbook, iter = 0)            │
                         └──────────────────┬───────────────────────┘
                                            │
                    ┌───────────────────────▼───────────────────────┐
                    │                  PREDICT                       │
                    │                                               │
                    │  MiniSWEAgent reads issue + skillbook          │
                    │  Agent LLM generates patch                    │
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            │
                    ┌───────────────────────▼───────────────────────┐
                    │                 EVALUATE                      │
                    │                                               │
                    │  Apply patch in Docker container              │
                    │  Run SWE-bench test suite                     │
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            │
                                  ┌─────────▼─────────┐
                                  │    Resolved?      │
                                  └────┬─────────┬────┘
                                       │Yes      │No
                                       ▼         │
                               ┌──────────┐      │
                               │   Done   │      │
                               └──────────┘      │
                                                 │
                                       ┌─────────▼─────────┐
                                       │  Max attempts?    │
                                       └────┬─────────┬────┘
                                            │Yes      │No
                                            ▼         │
                                    ┌───────────┐     │
                                    │ Done      │     │
                                    │(unresolved)│    │
                                    └───────────┘     │
                                                      │
                                            ┌─────────▼─────────┐
                                            │       LEARN       │
                                            │                   │
                                            │ Reflector analyzes│
                                            │ failure trajectory│
                                            │ → new skill added │
                                            │   to skillbook    │
                                            └─────────┬─────────┘
                                                      │
                                                      ▼
                                              iter = iter + 1
                                             ───────────────────
                                              back to PREDICT
```

---

## Skillbook Modes

### per_instance (default)

Each instance gets its own skillbook that starts empty and accumulates skills across retry iterations.

```
 Instance A                Instance B                Instance C
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│ iter 0: []   │          │ iter 0: []   │          │ iter 0: []   │
│     ↓ learn  │          │     ↓ learn  │          │              │
│ iter 1: [s1] │          │ iter 1: [s1] │          │   resolved   │
│     ↓ learn  │          │     ↓ learn  │          │              │
│ iter 2:[s1,s2]│         │ iter 2:[s1,s2]│         └──────────────┘
│     ↓ learn  │          │   resolved   │
│ iter 3:[s1,s2,s3]       └──────────────┘
│   resolved   │
└──────────────┘

Skillbooks are independent — no sharing between instances.
```

### per_repo (two-phase experiment)

All instances of the same repo share one skillbook. Uses train/val split.

```
                      REPO: django/django
                    ┌────────────────────────┐
                    │    TRAIN PHASE         │
                    │                        │
                    │  For each train inst:  │
                    │    1 attempt only      │
                    │    learn on success    │
                    │      AND failure       │
                    │    skillbook grows     │
                    │                        │
                    │  inst_A → skill +s1    │
                    │  inst_B → skillbook +s2│
                    │  inst_C → skillbook +s3│
                    │         ...            │
                    └──────────┬─────────────┘
                               │
                        accumulated skillbook
                               │
                    ┌──────────▼─────────────┐
                    │  final_skillbook.json  │
                    │  [s1, s2, s3, ...]     │
                    └──────┬──────────┬──────┘
                           │          │
              ┌────────────▼──┐  ┌────▼──────────────┐
              │  VAL BASELINE │  │  VAL SKILLBOOK    │
              │               │  │                    │
              │  Empty SB     │  │  Learned SB        │
              │  No learning  │  │  No learning       │
              │  1 attempt    │  │  1 attempt         │
              │               │  │                    │
              │  Measures:    │  │  Measures:         │
              │  raw ability  │  │  skillbook benefit │
              └───────────────┘  └────────────────────┘

Compare: val_skillbook resolve rate vs val_baseline resolve rate
       = measured impact of learned skills
```

---

## Learn Modes

### SWE Learn (`custom_swe_learn: true`)

```
  ┌─────────────────────────────────────────────────┐
  │              SWEReflector                        │
  │                                                 │
  │  Analyzes failure trajectory with focus on:     │
  │    • Anti-patterns (what went wrong)             │
  │    • Type-prefixed categories                    │
  │    • SWE-specific failure modes                  │
  │                                                 │
  │  Output: structured failure analysis             │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────┐
  │            SWESkillManager                       │
  │                                                 │
  │  Converts analysis → actionable skill:           │
  │    • Category-prefixed (e.g. "debugging:...")   │
  │    • Includes anti-pattern to avoid              │
  │    • Concrete guidance for future attempts       │
  │                                                 │
  │  Deduplication: cosine similarity >= 0.85        │
  │    merges near-duplicate skills                  │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
                   Updated skillbook
```

### Default Learn (`custom_swe_learn: false`)

```
  ┌─────────────────────────────────────────────────┐
  │            ACE Reflector (PydanticAI)            │
  │                                                 │
  │  Generic reflection on failure trajectory:       │
  │    • What was attempted                          │
  │    • Why it failed                               │
  │    • What to try next                            │
  │                                                 │
  │  Output: general-purpose failure analysis        │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────┐
  │          ACE SkillManager (PydanticAI)           │
  │                                                 │
  │  Converts analysis → generic skill:              │
  │    • Strategy-based format                       │
  │    • No type prefixes or anti-patterns           │
  │    • General problem-solving guidance            │
  │                                                 │
  │  Deduplication: same cosine similarity approach  │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
                   Updated skillbook
```

---

## Full Experiment Matrix

```
                        ┌────────────────────────────────────┐
                        │         Config Layers              │
                        │                                    │
                        │  config.yaml                       │
                        │    ↓ deep-merge                    │
                        │  --config overrides/               │
                        │    ↓ override                      │
                        │  CLI args                          │
                        └──────────────┬─────────────────────┘
                                       │
                        ┌──────────────▼─────────────────────┐
                        │       Experiment Loop              │
                        │                                    │
                        │  For each instance:                │
                        │    Predict → Evaluate → Learn      │
                        │    (repeat until resolved or       │
                        │     max_attempts reached)          │
                        └──────────────┬─────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                         │
    ┌─────────▼──────────┐  ┌─────────▼──────────┐  ┌─────────▼──────────┐
    │   per_instance     │  │     per_repo       │  │      global        │
    │                    │  │                    │  │                    │
    │  Own SB per inst   │  │  Shared SB/repo    │  │  One SB for all    │
    │  N attempts each   │  │  Two-phase train   │  │  Two-phase train   │
    │  Self-learning     │  │  /val split        │  │  /val split        │
    └────────────────────┘  └────────────────────┘  └────────────────────┘
                                       │
                        ┌──────────────┴─────────────────────┐
                        │                                     │
              ┌─────────▼──────────┐           ┌─────────────▼──────────┐
              │   SWE Learn        │           │   Default Learn        │
              │  (custom_swe=true) │           │  (custom_swe=false)    │
              │                    │           │                        │
              │  SWEReflector      │           │  ACE Reflector         │
              │  SWESkillManager   │           │  ACE SkillManager      │
              │  Anti-patterns     │           │  Generic strategies    │
              │  Type prefixes     │           │  PydanticAI-based      │
              └────────────────────┘           └────────────────────────┘
```

---

## Dual LLM Architecture

```
  ┌───────────────────────────────────────────────────┐
  │                    config.yaml                     │
  │                                                    │
  │  llm.agent:          llm.ace:                      │
  │    provider: vllm      provider: zai               │
  │    model: Qwen3-30B   model: glm-4.5-flash        │
  │                                                     │
  │  Used by:            Used by:                      │
  │  MiniSWEAgent        Reflector + SkillManager      │
  │  (predict phase)     (learn phase)                 │
  └────────────────────────────────────────────────────┘
```

Both can be independently swapped via config overrides in `configs/`.

---

## Real Skillbook Examples

### 1. per_instance mode — `django__django-12184` (SWE learn, Qwen3-30B)

Instance starts with empty skillbook. After each failed attempt, SWE learn adds skills with type-prefixed sections.

**iter_0.json** — empty, not saved to disk (no skills yet).

**iter_1.json** — 3 skills after first failure:

```json
{
  "iteration": 1,
  "instance_id": "django__django-12184",
  "skill_count": 3,
  "skills": {
    "django_url_handling-00001": {
      "section": "django_url_handling",
      "justification": "This strategy directly addresses the core issue causing the TypeError in Django 3.0 with optional URL parameters",
      "evidence": "Error message shows 'modules() takes from 1 to 2 positional arguments but 3 were given' when accessing URL without optional parameter",
      "content": "When optional regex groups in Django URL patterns don't match, do not pass them as positional arguments to view functions to prevent TypeError exceptions"
    },
    "django_url_handling-00002": {
      "section": "django_url_handling",
      "justification": "This provides a concrete approach to fixing the underlying mechanism in django/urls/resolvers.py",
      "evidence": "Problem occurs specifically in Django 3.0 with optional URL parameters in URLConf patterns",
      "content": "Modify Django URL resolver's argument construction logic to properly handle optional parameter groups that fail to match regex patterns"
    },
    "django_development-00003": {
      "section": "django_development",
      "justification": "Ensures that theoretical analysis translates into actual working fixes, which was the root cause of the failure",
      "evidence": "Agent analyzed the problem but never executed actual code modification in django/urls/resolvers.py",
      "content": "Implement code fixes for framework issues directly in source files rather than just discussing solutions"
    }
  }
}
```

**iter_2.json** — skills refined (not just appended — existing skills get updated):

```json
{
  "iteration": 2,
  "skill_count": 3,
  "skills": {
    "django_url_handling-00001": {
      "section": "django_url_handling",
      "justification": "Refines existing skill with explicit mention of keyword arguments and TypeError prevention",
      "evidence": "When accessing /module/ with pattern r'^module/(?P<format>(html|json|xml))?/?$', the format group becomes None but should be omitted to allow default parameter usage",
      "content": "When optional regex groups in Django URL patterns don't match, do not pass them as **keyword** arguments to view functions to prevent TypeError exceptions"
    },
    "django_url_handling-00002": {
      "section": "django_url_handling",
      "justification": "Updates skill with specific implementation detail about filtering None values from groupdict before constructing kwargs",
      "evidence": "The fix requires modifying urls/resolvers.py to filter out None values from groupdict before constructing view function kwargs",
      "content": "Modify Django URL resolver's argument construction logic ... by filtering out None values from captured groups"
    }
  }
}
```

> Note: skills are **refined across iterations** — the reflector updates existing skills with better evidence, not just appends new ones.

---

### 2. per_repo + SWE learn — `django_split_swe` (final_skillbook.json, 263 skills)

After training on all django/django train-split instances, the accumulated skillbook has 263 skills with type-prefixed sections:

```json
{
  "skill_count": 263,
  "skills": {
    "debugging-00001": {
      "section": "debugging",
      "content": "AVOID: Creating test scripts instead of editing source files directly for bug fixes (use test scripts for verification but prioritize source code changes)"
    },
    "debugging-00002": {
      "section": "debugging",
      "content": "AVOID: Using sed command with improper syntax for code modifications"
    },
    "bug_fixing-00003": {
      "section": "bug_fixing",
      "content": "VERIFIED: The issue was located in django/core/checks/templates.py in the check_for_template_tags_with_the_same_name function"
    },
    "bug_fixing-00004": {
      "section": "bug_fixing",
      "content": "VERIFIED: The problem stemmed from double-counting when libraries were defined in both custom libraries and get_template_tag_modules()"
    },
    "code_modification-00006": {
      "section": "code_modification",
      "content": "AVOID: Using overly broad sed commands without verifying boundaries"
    },
    "code_modification-00012": {
      "section": "code_modification",
      "justification": "Prevents submission of empty patches by ensuring verification before claiming success",
      "evidence": "Agent said 'successfully implemented' but git diff --cached was empty",
      "content": "AVOID: Claiming success without verifying git diff shows actual changes"
    },
    "verification-00017": {
      "section": "verification",
      "justification": "Critical to ensure patches actually apply and produce expected results",
      "evidence": "Agent created a patch that modified wrong function",
      "content": "AVOID: Submitting patches without verifying the modified file is correct"
    }
  }
}
```

Key patterns in SWE skillbook:
- **`AVOID:`** prefix — anti-patterns extracted from failures
- **`VERIFIED:`** prefix — confirmed correct approaches
- **`CONSIDER:`** prefix — suggested strategies
- **Type-prefixed sections**: `debugging`, `bug_fixing`, `code_modification`, `environment`, `verification`
- **Justification + evidence** fields explain *why* and *where* the skill came from

---

### 3. per_repo + Default learn — `django_split_default` (final_skillbook.json, 417 skills)

Same experiment but with default ACE reflector (no SWE-specific prompting). Notice the different style:

```json
{
  "skill_count": 417,
  "skills": {
    "django_template_system-00001": {
      "section": "django_template_system",
      "content": "Track custom library names from TEMPLATES['OPTIONS']['libraries'] to prevent duplicate detection false positives"
    },
    "django_template_system-00007": {
      "section": "django_template_system",
      "justification": "Adds specific guidance for handling primary keys in Django's deletion logic",
      "evidence": "The fix added 'instance.pk = None' in django/db/models/deletion.py at line 280, right after the fast delete operation.",
      "content": "In Django's deletion logic, when performing a fast delete on instances with no dependencies, the instance's primary key should be explicitly set to None after deletion."
    },
    "django_template_system-00016": {
      "section": "django_template_system",
      "justification": "This directly addresses the specific code change required for the identified bug",
      "evidence": "Issue description explicitly states the method needs to be changed from 'id_%s_%s' % (self.data['name'], self.data['index']) to self.data['attrs']['id']",
      "content": "BoundWidget.id_for_label() method should use self.data['attrs']['id'] instead of constructing ID from name and index"
    }
  }
}
```

Key differences vs SWE learn:

| Aspect | SWE Learn | Default Learn |
|--------|-----------|---------------|
| Skill count (django) | 263 | 417 |
| Section naming | Domain types (`debugging`, `bug_fixing`) | Repo-area based (`django_template_system`) |
| Content prefix | `AVOID:` / `VERIFIED:` / `CONSIDER:` | None — plain prose |
| Anti-patterns | Explicit, first-class | Implicit in content |
| Focus | SWE-specific failure modes | General problem-solving |

---

### Skill Deduplication

Both modes deduplicate skills via cosine similarity (threshold 0.85). When a new skill is too similar to an existing one, it merges them rather than adding a duplicate:

```
  Existing:  "AVOID: Claiming success without verifying git diff shows actual changes"
  New:       "AVOID: Saying a fix is done without checking the diff output"
  Similarity: 0.91  (> 0.85 threshold)
  → Merged into one skill with updated justification/evidence
```
