# src/retrieval/prompts.py
"""Default prompt templates and structured-output schemas for skill retrieval.

Based on scripts/sample_top_skills.py (standalone CLI tool).
The script is not modified — these are independent copies.
"""

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Default prompt templates
# ---------------------------------------------------------------------------

DEFAULT_FILTER_PROMPT = """\
You are filtering skills for a SWE-bench issue resolution agent.

## ISSUE
Repository: {repo}
Title: {title}
Description:
{description}

## CANDIDATE SKILLS  (#num | section [tags] | advice)
{skills_block}

## TASK
Only keep skills plausibly useful for THIS specific issue. Skills from other repos are fine if the technique transfers.

- KEEP if the skill could help resolve this issue
- KEEP if the skill is about the same library/module/API area as the issue
- KEEP if the testing/debugging technique is relevant to the problem type
- KEEP if the AVOID pattern is directly applicable to the issue's domain
- KEEP if the skill is from another repo but describes a technique that applies here

- DISCARD if the skill topic is unrelated to the issue.
- DISCARD if the skill is too vague to be actionable.

Return the indices of all KEEP skills. Keep at most {max_keep} skills. If more than {max_keep} qualify, return only the {max_keep} most relevant."""


DEFAULT_RANK_PROMPT = """\
You are ranking skills for a SWE-bench issue resolution agent.

## ISSUE
Repository: {repo}
Title: {title}
Description:
{description}

## FILTERED SKILLS  (#num | section [tags] | advice)
{skills_block}

## TASK
Pick exactly {k} most useful skills for resolving this issue. Apply these rules:

1. **Quality boost** — prefer skills tagged [evidence] or [justified]. Prefer skills naming specific files, classes, or methods over vague advice.
2. **Diversity** — diversify across topic clusters; avoid selecting many skills from the same cluster.
3. **Type balance** — roughly:
   - 60% domain-specific code-modification skills
   - 20% testing or verification skills
   - 20% cautionary or generic skills (AVOID patterns, environment notes)

Return exactly {k} entries using the # number from the list above as the "idx" field."""


# ---------------------------------------------------------------------------
# Structured-output schemas
# ---------------------------------------------------------------------------


class RelevanceResponse(BaseModel):
    relevant_indices: list[int]


class SkillSelection(BaseModel):
    idx: int
    reason: str


class RankResponse(BaseModel):
    selections: list[SkillSelection]
