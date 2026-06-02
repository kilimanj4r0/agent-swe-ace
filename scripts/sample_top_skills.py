"""
Retrieve top-k skills from a skillbook for a SWE-bench issue.

Two-stage pipeline:
  1. Filter — remove skills irrelevant to the issue or specific to other repos.
  2. Rank — pick the top-k most useful skills from the filtered set.

Usage:
    # From parquet (preferred):
    python sample_top_skills_v2.py \
        --skillbook final_skillbook_global.json \
        --parquet data.parquet \
        --instance-id astropy__astropy-12907 \
        -k 5

    # Manual mode:
    python sample_top_skills_v2.py \
        --skillbook final_skillbook_global.json \
        --repo django__django \
        --issue-title "Bug in migrations" \
        --issue-body "Description..." \
        -k 5
"""

import argparse
import json
import sys

from openai import OpenAI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Prompts — inlined
# ---------------------------------------------------------------------------

_FILTER_PROMPT = """\
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

_RANK_PROMPT = """\
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


class _RelevanceResponse(BaseModel):
    relevant_indices: list[int]


class _SkillSelection(BaseModel):
    idx: int
    reason: str


class _RankResponse(BaseModel):
    selections: list[_SkillSelection]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_skill(idx: int, s: dict) -> str:
    tags: list[str] = []
    if s.get("evidence"):
        tags.append("evidence")
    if s.get("justification"):
        tags.append("justified")
    tag = f" [{','.join(tags)}]" if tags else ""
    return f"#{idx} | {s['section']}{tag} | {s['content']}"


def _call_structured(
    client: OpenAI,
    model: str,
    prompt: str,
    schema: type[BaseModel],
) -> BaseModel:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
            },
        },
    )
    content = resp.choices[0].message.content or "{}"
    return schema.model_validate_json(content)


# ---------------------------------------------------------------------------
# Two-stage pipeline
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_CHUNK_SIZE = 200
_FILTER_TARGET = 100


def _filter_skills(
    client: OpenAI,
    model: str,
    skill_items: list[tuple[str, dict]],
    issue_title: str,
    issue_body: str,
    repo: str,
    filter_prompt: str,
    filter_target: int,
) -> list[tuple[str, dict]]:
    """Stage 1: return only skills relevant to the issue, processed in chunks."""
    import math

    kept: list[tuple[str, dict]] = []
    total = len(skill_items)

    for start in range(0, total, _CHUNK_SIZE):
        chunk = skill_items[start : start + _CHUNK_SIZE]
        chunk_num = start // _CHUNK_SIZE + 1
        total_chunks = (total + _CHUNK_SIZE - 1) // _CHUNK_SIZE
        max_keep = math.ceil(filter_target * len(chunk) / total)

        skills_block = "\n".join(
            _format_skill(i, s) for i, (_, s) in enumerate(chunk)
        )
        prompt = filter_prompt.format(
            repo=repo,
            title=issue_title,
            description=issue_body,
            skills_block=skills_block,
            max_keep=max_keep,
        )

        print(
            f"Filter chunk {chunk_num}/{total_chunks}: {len(chunk)} candidates...",
            file=sys.stderr,
        )

        valid = set(range(len(chunk)))
        for attempt in range(1, _MAX_RETRIES + 1):
            parsed = _call_structured(client, model, prompt, _RelevanceResponse)
            if not parsed.relevant_indices:
                # Model says nothing in this chunk is relevant — accept
                print(f"  kept 0/{len(chunk)}", file=sys.stderr)
                break
            indices = [i for i in parsed.relevant_indices if i in valid]  # type: ignore[attr-defined]
            if indices:
                kept.extend(chunk[i] for i in indices)
                print(
                    f"  kept {len(indices)}/{len(chunk)}",
                    file=sys.stderr,
                )
                break
            # All returned indices were out of range — retry
            print(
                f"Filter retry {attempt}/{_MAX_RETRIES} (chunk {chunk_num}/{total_chunks}): all indices out of range",
                file=sys.stderr,
            )
        else:
            # Fallback: keep all skills in this chunk
            kept.extend(chunk)
            print(
                f"  kept {len(chunk)}/{len(chunk)} (fallback)",
                file=sys.stderr,
            )

    return kept


def _rank_skills(
    client: OpenAI,
    model: str,
    skill_items: list[tuple[str, dict]],
    issue_title: str,
    issue_body: str,
    repo: str,
    k: int,
    rank_prompt: str,
) -> list[tuple[str, str]]:
    """Stage 2: pick top-k from filtered skills. Returns (skill_id, reason) pairs."""
    idx_to_id = {i: sid for i, (sid, _) in enumerate(skill_items)}
    skills_block = "\n".join(
        _format_skill(i, s) for i, (_, s) in enumerate(skill_items)
    )
    prompt = rank_prompt.format(
        k=k,
        repo=repo,
        title=issue_title,
        description=issue_body,
        skills_block=skills_block,
    )

    results: list[tuple[str, str]] = []
    for attempt in range(1, _MAX_RETRIES + 1):
        parsed = _call_structured(client, model, prompt, _RankResponse)

        seen: set[int] = set()
        results = []
        for entry in parsed.selections:  # type: ignore[attr-defined]
            if entry.idx not in idx_to_id or entry.idx in seen:
                continue
            seen.add(entry.idx)
            results.append((idx_to_id[entry.idx], entry.reason))

        results = results[:k]

        if len(results) == k:
            return results

        print(
            f"Rank retry {attempt}/{_MAX_RETRIES}: got {len(results)}/{k} valid skills",
            file=sys.stderr,
        )

    return results


def retrieve_top_k(
    skills: dict,
    issue_title: str,
    issue_body: str,
    repo: str,
    k: int,
    *,
    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    base_url: str = "http://localhost:8800/v1",
    api_key: str = "EMPTY",
    filter_prompt: str = _FILTER_PROMPT,
    rank_prompt: str = _RANK_PROMPT,
    filter_target: int = _FILTER_TARGET,
) -> list[tuple[str, str]]:
    """Two-stage retrieval: filter irrelevant skills, then rank top-k."""
    client = OpenAI(base_url=base_url, api_key=api_key)
    skill_items = list(skills.items())

    # Stage 1 — filter
    filtered = _filter_skills(
        client, model, skill_items, issue_title, issue_body, repo,
        filter_prompt, filter_target=min(filter_target, len(skill_items)),
    )
    print(
        f"Filter: {len(skill_items)} -> {len(filtered)} skills",
        file=sys.stderr,
    )

    # If already fewer than k, return all with empty reasons
    if len(filtered) <= k:
        return [(sid, "") for sid, _ in filtered]

    # Stage 2 — rank
    return _rank_skills(client, model, filtered, issue_title, issue_body, repo, k, rank_prompt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_from_parquet(parquet_path: str, instance_id: str) -> tuple[str, str, str]:
    """Load issue data from a SWE-bench parquet file. Returns (repo, title, body)."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    row = df[df["instance_id"] == instance_id]
    if row.empty:
        raise ValueError(f"instance_id '{instance_id}' not found in {parquet_path}")
    row = row.iloc[0]
    repo = str(row["repo"]).replace("/", "__")
    ps = str(row["problem_statement"])
    title = ps.split("\n", 1)[0]
    body = ps.split("\n", 1)[1] if "\n" in ps else ""
    return repo, title, body


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skillbook", required=True, help="Path to skillbook JSON")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--parquet", help="Path to SWE-bench parquet (use with --instance-id)")
    src.add_argument("--repo", help="Repo name, e.g. django__django (manual mode)")

    ap.add_argument("--instance-id", help="Row id in parquet (required with --parquet)")
    ap.add_argument("--issue-title", help="Issue title (manual mode)")
    ap.add_argument("--issue-body", help="Issue description (manual mode)")
    ap.add_argument("-k", type=int, default=5, help="Number of skills to retrieve")
    ap.add_argument("--filter-target", type=int, default=_FILTER_TARGET,
                    help=f"Max skills to keep after filtering (default: {_FILTER_TARGET})")
    ap.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8800/v1")
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()

    if args.parquet:
        if not args.instance_id:
            ap.error("--instance-id is required when using --parquet")
        repo, issue_title, issue_body = _load_from_parquet(args.parquet, args.instance_id)
    else:
        if not args.issue_title or not args.issue_body:
            ap.error("--issue-title and --issue-body are required when using --repo")
        repo, issue_title, issue_body = args.repo, args.issue_title, args.issue_body

    with open(args.skillbook) as f:
        book = json.load(f)
    skills = book["skills"]
    print(f"Loaded {len(skills)} skills", file=sys.stderr)

    top_k = retrieve_top_k(
        skills,
        issue_title,
        issue_body,
        repo,
        args.k,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        filter_target=args.filter_target,
    )

    print(f"\n{'=' * 80}")
    print(f"Repo:       {repo}")
    print(f"Title:      {issue_title}")
    print(f"Body:       {issue_body[:500]}{'...' if len(issue_body) > 500 else ''}")
    print(f"{'=' * 80}")

    for i, (skill_id, reason) in enumerate(top_k, 1):
        skill = skills[skill_id]
        print(f"\n{i}. {skill_id}")
        print(f"   Section:  {skill['section']}")
        print(f"   Content:  {skill['content']}")
        if reason:
            print(f"   Reason:   {reason}")


if __name__ == "__main__":
    main()
