# src/retrieval/skill_retriever.py
"""Two-stage LLM-based top-k skill retrieval.

Stage 1 (filter): remove skills irrelevant to the issue or specific to other repos.
Stage 2 (rank): pick the top-k most useful skills from the filtered set.
"""

import sys
from pathlib import Path
from typing import Optional

from ace import Skillbook
from ace.core.skillbook import Skill
from loguru import logger
from openai import OpenAI

from .prompts import (
    DEFAULT_FILTER_PROMPT,
    DEFAULT_RANK_PROMPT,
    RankResponse,
    RelevanceResponse,
)

_MAX_RETRIES = 3


def _load_prompt(path: Optional[str], default: str) -> str:
    """Load prompt template from file, or return default."""
    if path is None:
        return default
    content = Path(path).read_text()
    logger.debug(f"[Retriever] Loaded custom prompt from {path}")
    return content


def _format_skill(idx: int, s: dict) -> str:
    """Format a single skill for the prompt."""
    tags: list[str] = []
    if s.get("evidence"):
        tags.append("evidence")
    if s.get("justification"):
        tags.append("justified")
    tag = f" [{','.join(tags)}]" if tags else ""
    return f"#{idx} | {s['section']}{tag} | {s['content']}"


def _extract_issue_info(instance: dict) -> tuple[str, str, str]:
    """Extract repo, title, body from a SWE-bench instance dict."""
    repo = instance.get("repo", "unknown").replace("/", "__")
    problem = instance.get("problem_statement", "")
    title = problem.split("\n", 1)[0]
    body = problem.split("\n", 1)[1] if "\n" in problem else ""
    return repo, title, body


def _skill_to_dict(skill: Skill) -> dict:
    """Convert a Skill object to a dict for _format_skill."""
    return {
        "section": skill.section,
        "content": skill.content,
        "evidence": getattr(skill, "evidence", None),
        "justification": getattr(skill, "justification", None),
    }


class SkillRetriever:
    """Two-stage LLM-based top-k skill retrieval.

    Args:
        model: LLM model identifier.
        api_base: OpenAI-compatible API base URL.
        api_key: API key string.
        top_k: Number of skills to retrieve.
        skip_threshold: Skip retrieval if skillbook has ≤ this many skills.
        filter_prompt: Path to custom filter prompt file (None = built-in default).
        rank_prompt: Path to custom rank prompt file (None = built-in default).
        temperature: LLM temperature.
        max_tokens: LLM max tokens.
        max_retries: Retries per stage on empty/invalid LLM response.
    """

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        top_k: int = 5,
        skip_threshold: int = 10,
        filter_prompt: Optional[str] = None,
        rank_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.top_k = top_k
        self.skip_threshold = skip_threshold
        self.max_retries = max_retries

        self._client = OpenAI(base_url=api_base, api_key=api_key)
        self._filter_prompt = _load_prompt(filter_prompt, DEFAULT_FILTER_PROMPT)
        self._rank_prompt = _load_prompt(rank_prompt, DEFAULT_RANK_PROMPT)
        self._temperature = temperature
        self._max_tokens = max_tokens

    def retrieve(self, skillbook: Skillbook, instance: dict) -> list[Skill]:
        """Return top-k relevant skills for the given instance.

        Returns all skills if count ≤ skip_threshold (no LLM call).
        Falls back to all skills on retrieval failure.
        """
        skills = skillbook.skills()
        if not skills:
            return []

        if len(skills) <= self.skip_threshold:
            logger.debug(
                f"[Retriever] {len(skills)} skills ≤ skip_threshold={self.skip_threshold}, skipping retrieval"
            )
            return list(skills)

        repo, title, body = _extract_issue_info(instance)
        skill_items = [(s.id, _skill_to_dict(s)) for s in skills]
        id_to_skill = {s.id: s for s in skills}

        # Stage 1 — filter
        filtered = self._filter_skills(skill_items, title, body, repo)
        logger.debug(f"[Retriever] Filter: {len(skill_items)} → {len(filtered)} skills")

        if not filtered:
            logger.warning("[Retriever] Filter returned 0 skills, falling back to all")
            return list(skills)

        # If already fewer than k, return all filtered
        if len(filtered) <= self.top_k:
            return [id_to_skill[sid] for sid, _ in filtered]

        # Stage 2 — rank
        ranked = self._rank_skills(filtered, title, body, repo, self.top_k)
        logger.debug(f"[Retriever] Rank: {len(filtered)} → {len(ranked)} skills")

        if not ranked:
            # Rank failed — return filtered set capped at top_k
            return [id_to_skill[sid] for sid, _ in filtered[: self.top_k]]

        return [id_to_skill[sid] for sid, _ in ranked]

    def _filter_skills(
        self,
        skill_items: list[tuple[str, dict]],
        issue_title: str,
        issue_body: str,
        repo: str,
    ) -> list[tuple[str, dict]]:
        """Stage 1: return only skills relevant to the issue."""
        skills_block = "\n".join(
            _format_skill(i, s) for i, (_, s) in enumerate(skill_items)
        )
        prompt = self._filter_prompt.format(
            repo=repo,
            title=issue_title,
            description=issue_body,
            skills_block=skills_block,
        )

        valid = set(range(len(skill_items)))
        for attempt in range(1, self.max_retries + 1):
            try:
                parsed = self._call_structured(prompt, RelevanceResponse)
                indices = [i for i in parsed.relevant_indices if i in valid]
                if indices:
                    return [skill_items[i] for i in indices]
                logger.debug(
                    f"[Retriever] Filter retry {attempt}/{self.max_retries}: got 0 valid indices"
                )
            except Exception as e:
                logger.debug(f"[Retriever] Filter attempt {attempt} failed: {e}")

        # Fallback: return all
        return list(skill_items)

    def _rank_skills(
        self,
        skill_items: list[tuple[str, dict]],
        issue_title: str,
        issue_body: str,
        repo: str,
        k: int,
    ) -> list[tuple[str, str]]:
        """Stage 2: pick top-k from filtered skills. Returns (skill_id, reason) pairs."""
        idx_to_id = {i: sid for i, (sid, _) in enumerate(skill_items)}
        skills_block = "\n".join(
            _format_skill(i, s) for i, (_, s) in enumerate(skill_items)
        )
        prompt = self._rank_prompt.format(
            k=k,
            repo=repo,
            title=issue_title,
            description=issue_body,
            skills_block=skills_block,
        )

        results: list[tuple[str, str]] = []
        for attempt in range(1, self.max_retries + 1):
            try:
                parsed = self._call_structured(prompt, RankResponse)

                seen: set[int] = set()
                results = []
                for entry in parsed.selections:
                    if entry.idx not in idx_to_id or entry.idx in seen:
                        continue
                    seen.add(entry.idx)
                    results.append((idx_to_id[entry.idx], entry.reason))

                results = results[:k]

                if len(results) == k:
                    return results

                logger.debug(
                    f"[Retriever] Rank retry {attempt}/{self.max_retries}: "
                    f"got {len(results)}/{k} valid skills"
                )
            except Exception as e:
                logger.debug(f"[Retriever] Rank attempt {attempt} failed: {e}")

        return results

    def _call_structured(self, prompt: str, schema: type) -> object:
        """Call LLM with structured output (JSON schema)."""
        from pydantic import BaseModel

        assert issubclass(schema, BaseModel)

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
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
