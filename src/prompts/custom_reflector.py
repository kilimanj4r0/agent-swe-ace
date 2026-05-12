# prompts/custom_reflector.py
"""Custom Reflector that uses SWE-optimized prompt and output format."""

from __future__ import annotations

from typing import Any, Optional

from ace import Reflector
from ace.core.outputs import AgentOutput
from pydantic_ai import Agent as PydanticAgent

from ace.providers.pydantic_ai import resolve_model

from .reflector_prompt import CUSTOM_REFLECTOR_PROMPT
from .outputs import SWEReflectorOutput


class SWEReflector(Reflector):
    """SWE-optimized Reflector that extracts anti-patterns from failures.

    Subclasses ace.Reflector to use CUSTOM_REFLECTOR_PROMPT and
    SWEReflectorOutput with anti_patterns, discoveries, unvalidated_hypotheses.

    Args:
        model: Model identifier string (e.g. "zai/glm-4.5-flash").
        prompt_template: Custom prompt template (defaults to CUSTOM_REFLECTOR_PROMPT).
        max_retries: Maximum retries for structured output validation.
        api_base: Base URL for the LLM endpoint (required for hosted_vllm).
        api_key: API key for the LLM endpoint.

    Example::

        reflector = SWEReflector("zai/glm-4.5-flash")
        reflection = reflector.reflect(
            question="Fix the bug...",
            agent_output=agent_output,
            skillbook=skillbook,
            feedback="Tests failed: ...",
        )
    """

    def __init__(
        self,
        model: str,
        prompt_template: str = CUSTOM_REFLECTOR_PROMPT,
        *,
        max_retries: int = 3,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        # Bypass Reflector.__init__ — set up PydanticAgent directly
        # with our custom output type and prompt
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
            output_type=SWEReflectorOutput,
            retries=max_retries,
            defer_model_check=True,
        )
