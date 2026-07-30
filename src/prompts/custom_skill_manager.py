# prompts/custom_skill_manager.py
"""Custom SkillManager that preserves learning type prefixes."""

from __future__ import annotations

import json
from typing import Any, Union

from ace import SkillManager
from ace.core.context import SkillbookView
from ace.core.outputs import ReflectorOutput, SkillManagerOutput
from ace.core.skillbook import Skillbook
from loguru import logger

from .model_utils import make_pydantic_agent
from .skill_manager_prompt import CUSTOM_SKILL_MANAGER_PROMPT


class SWESkillManager(SkillManager):
    """SWE-optimized SkillManager with extended reflection serialization.

    Overrides update_skills() to forward SWE-specific fields
    (error_location, confidence_in_analysis, skill_tags) to the LLM.

    Args:
        model: Model identifier string (e.g. "zai/glm-4.5-flash").
        prompt_template: Custom prompt template (defaults to CUSTOM_SKILL_MANAGER_PROMPT).
        max_retries: Maximum retries for structured output validation.
        api_base: Base URL for the LLM endpoint (required for hosted_vllm).
        api_key: API key for the LLM endpoint.
        model_settings: Optional dict for ModelSettings (temperature, max_tokens).
    """

    def __init__(
        self,
        model: str,
        prompt_template: str = CUSTOM_SKILL_MANAGER_PROMPT,
        *,
        max_retries: int = 3,
        api_base: str | None = None,
        api_key: str | None = None,
        model_settings: dict | None = None,
    ) -> None:
        self._prompt_template = prompt_template
        self._agent = make_pydantic_agent(
            model,
            SkillManagerOutput,
            api_base=api_base,
            api_key=api_key,
            max_retries=max_retries,
            model_settings=model_settings,
        )

    def update_skills(
        self,
        *,
        reflections: tuple[ReflectorOutput, ...],
        skillbook: Union[SkillbookView, Skillbook],
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        """Extended update_skills that forwards SWE-specific reflection fields."""
        reflections_data = []
        for r in reflections:
            entry = {
                "reasoning": r.reasoning,
                "error_identification": r.error_identification,
                "root_cause_analysis": r.root_cause_analysis,
                "correct_approach": r.correct_approach,
                "key_insight": r.key_insight,
                "extracted_learnings": [
                    learning.model_dump() for learning in r.extracted_learnings
                ],
            }
            # SWE-specific fields (only present on SWEReflectorOutput)
            if hasattr(r, "error_location"):
                entry["error_location"] = r.error_location
            if hasattr(r, "confidence_in_analysis"):
                entry["confidence_in_analysis"] = r.confidence_in_analysis
            if hasattr(r, "skill_tags") and r.skill_tags:
                entry["skill_tags"] = [
                    {"id": t.id, "tag": t.tag}
                    for t in r.skill_tags
                ]
            reflections_data.append(entry)

        prompt = self._prompt_template.format(
            progress=progress,
            stats=json.dumps(skillbook.stats()),
            reflections=json.dumps(reflections_data, ensure_ascii=False, indent=2),
            skillbook=skillbook.as_prompt() or "(empty skillbook)",
            question_context=question_context,
        )

        logger.debug(f"[SkillManager] Prompt ({len(prompt)} chars):\n{prompt}")

        result = self._agent.run_sync(prompt)
        output = result.output
        usage = result.usage()
        output.raw = {
            "usage": {
                "prompt_tokens": usage.input_tokens or 0,
                "completion_tokens": usage.output_tokens or 0,
                "total_tokens": usage.total_tokens or 0,
            },
        }
        return output
