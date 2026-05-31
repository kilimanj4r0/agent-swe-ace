# src/retrieval/prompts.py
"""Default prompt templates and structured-output schemas for skill retrieval.

Copied from scripts/sample_top_skills.py (standalone CLI tool).
The script is not modified — these are independent copies.
"""

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Default prompt templates
# ---------------------------------------------------------------------------

DEFAULT_FILTER_PROMPT = r"""\
You are filtering skills for a SWE-bench issue resolution agent.
Be aggressive — only keep skills that are plausibly useful for THIS specific issue.

## ISSUE
Repository: {repo}
Title: {title}
Description:
{description}

## CANDIDATE SKILLS  (#num | section [tags] | advice)
{skills_block}

## TASK
Decide which skills are potentially useful for resolving this issue.

DISCARD a skill if ANY of these apply:
- It is specific to a *different* repository or project (not "{repo}") — e.g. advice about files, modules, or APIs that belong to another project.
- Its topic (the technology, library, module, or concept it discusses) is unrelated to the issue.
- It is too vague or generic to provide actionable guidance (e.g. "write good code", "use version control").

KEEP a skill only if it is:
- About the same library/module/API area as the issue, OR
- A specific testing or debugging technique relevant to the problem type, OR
- An AVOID pattern or cautionary note directly applicable to the issue's domain.

Aim to keep no more than ~50% of the candidate skills.

Return the indices of all KEEP skills."""


DEFAULT_RANK_PROMPT = r"""\
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
