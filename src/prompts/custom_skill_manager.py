# prompts/custom_skill_manager.py
"""Custom SkillManager that preserves learning type prefixes."""

from __future__ import annotations

from ace import SkillManager
from ace.core.outputs import SkillManagerOutput
from pydantic_ai import Agent as PydanticAgent

from ace.providers.pydantic_ai import resolve_model

from prompts.skill_manager_prompt import CUSTOM_SKILL_MANAGER_PROMPT


class SWESkillManager(SkillManager):
    """SWE-optimized SkillManager that preserves learning type prefixes.

    Key differences from default ace.SkillManager:
    1. Uses CUSTOM_SKILL_MANAGER_PROMPT with learning type preservation instructions
    2. Instructs LLM to convert [ANTI-PATTERN], [DISCOVERY], [HYPOTHESIS] prefixes
       to AVOID:, VERIFIED:, CONSIDER: prefixes in skill content

    Args:
        model: Model identifier string (e.g. "zai/glm-4.5-airx").
        prompt_template: Custom prompt template (defaults to CUSTOM_SKILL_MANAGER_PROMPT).
        max_retries: Maximum retries for structured output validation.
        api_base: Base URL for the LLM endpoint (required for hosted_vllm).
        api_key: API key for the LLM endpoint.

    Example::

        sm = SWESkillManager("zai/glm-4.5-airx")
        output = sm.update_skills(
            reflections=(reflection_output,),
            skillbook=skillbook,
            question_context="Fix bug in Django ORM",
            progress="0/1 resolved",
        )
    """

    def __init__(
        self,
        model: str,
        prompt_template: str = CUSTOM_SKILL_MANAGER_PROMPT,
        *,
        max_retries: int = 3,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        # Bypass SkillManager.__init__ — set up PydanticAgent directly
        # with our custom prompt
        self._prompt_template = prompt_template

        if api_base:
            from openai import AsyncOpenAI
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.litellm import LiteLLMProvider
            openai_client = AsyncOpenAI(
                base_url=api_base,
                api_key=api_key or "not-needed",
                max_retries=int(__import__("os").getenv("ACE_LEARN_MAX_RETRIES", "50")),
            )
            provider = LiteLLMProvider(openai_client=openai_client)
            # Strip LiteLLM provider prefix (e.g. "hosted_vllm/") since
            # OpenAIChatModel sends the model name directly to the endpoint
            model_name = model.split("/", 1)[1] if "/" in model else model
            resolved_model = OpenAIChatModel(model_name=model_name, provider=provider)
        else:
            resolved_model = resolve_model(model)

        self._agent = PydanticAgent(
            resolved_model,
            output_type=SkillManagerOutput,
            retries=max_retries,
            defer_model_check=True,
        )
