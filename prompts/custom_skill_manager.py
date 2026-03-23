# prompts/custom_skill_manager.py
"""Custom SkillManager that preserves learning type prefixes."""

import json
import logging
from typing import Any

from ace_next.core.outputs import SkillManagerOutput
from ace_next.protocols.llm import LLMClientLike

from prompts.skill_manager_prompt import CUSTOM_SKILL_MANAGER_PROMPT

logger = logging.getLogger(__name__)


class SWESkillManager:
    """SWE-optimized SkillManager that preserves learning type prefixes.

    Key differences from default ace_next.SkillManager:
    1. Uses CUSTOM_SKILL_MANAGER_PROMPT with learning type preservation instructions
    2. Instructs LLM to convert [ANTI-PATTERN], [DISCOVERY], [HYPOTHESIS] prefixes
       to AVOID:, VERIFIED:, CONSIDER: prefixes in skill content

    Args:
        llm: An LLM client that satisfies LLMClientLike.
        prompt_template: Custom prompt template (defaults to CUSTOM_SKILL_MANAGER_PROMPT).
    """

    def __init__(
        self,
        llm: LLMClientLike,
        prompt_template: str = CUSTOM_SKILL_MANAGER_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def update_skills(
        self,
        *,
        reflections: tuple,
        skillbook: Any,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        """Generate update operations based on the reflections.

        Args:
            reflections: Tuple of reflection outputs (e.g., SWEReflectorOutput).
            skillbook: Current skillbook to update.
            question_context: Context about the question/task.
            progress: Progress information string.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            SkillManagerOutput containing the update operations.
        """
        # Build reflections data with extracted_learnings
        reflections_data = []
        for r in reflections:
            # Handle both SWEReflectorOutput (with anti_patterns, discoveries, etc.)
            # and standard ReflectorOutput (with extracted_learnings)
            if hasattr(r, 'anti_patterns'):
                # SWEReflectorOutput - use our extracted_learnings property
                learnings = r.extracted_learnings
            else:
                # Standard ReflectorOutput
                learnings = r.extracted_learnings

            reflections_data.append({
                "reasoning": getattr(r, 'reasoning', ''),
                "error_identification": getattr(r, 'error_identification', ''),
                "root_cause_analysis": getattr(r, 'root_cause_analysis', ''),
                "correct_approach": getattr(r, 'correct_approach', ''),
                "key_insight": getattr(r, 'key_insight', ''),
                "extracted_learnings": [l.model_dump() for l in learnings],
            })

        prompt = self.prompt_template.format(
            progress=progress,
            stats=json.dumps(skillbook.stats()),
            reflections=json.dumps(reflections_data, ensure_ascii=False, indent=2),
            skillbook=skillbook.as_prompt() or "(empty skillbook)",
            question_context=question_context,
        )

        return self.llm.complete_structured(
            prompt, SkillManagerOutput, max_retries=self.max_retries
        )
