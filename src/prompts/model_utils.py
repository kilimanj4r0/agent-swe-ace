"""Shared model resolution utilities for custom ACE components."""

from __future__ import annotations

import os

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.settings import ModelSettings


def resolve_ace_model(
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
):
    """Resolve model string to a PydanticAI model instance.

    Handles both hosted vLLM endpoints (via AsyncOpenAI) and standard
    provider models (via ACE's resolve_model).

    Returns:
        Resolved PydanticAI model instance.
    """
    if api_base:
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.litellm import LiteLLMProvider

        openai_client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key or "not-needed",
            max_retries=int(os.getenv("ACE_LEARN_MAX_RETRIES", "50")),
        )
        provider = LiteLLMProvider(openai_client=openai_client)
        model_name = model.split("/", 1)[1] if "/" in model else model
        model_name = model_name.strip() or model
        return OpenAIChatModel(model_name=model_name, provider=provider)
    else:
        from ace.providers.pydantic_ai import resolve_model
        return resolve_model(model)


def make_pydantic_agent(
    model: str,
    output_type: type,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    max_retries: int = 3,
    model_settings: dict | None = None,
) -> PydanticAgent:
    """Create a PydanticAgent with shared model resolution logic.

    Args:
        model: Model identifier string.
        output_type: Pydantic model class for structured output.
        api_base: Base URL for hosted vLLM endpoints.
        api_key: API key for the endpoint.
        max_retries: Maximum retries for structured output validation.
        model_settings: Optional dict for ModelSettings (temperature, max_tokens).

    Returns:
        Configured PydanticAgent instance.
    """
    resolved_model = resolve_ace_model(model, api_base, api_key)
    ms = ModelSettings(**model_settings) if model_settings else None

    return PydanticAgent(
        resolved_model,
        output_type=output_type,
        retries=max_retries,
        model_settings=ms,
        defer_model_check=True,
    )
