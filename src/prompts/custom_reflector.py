# prompts/custom_reflector.py
"""Custom Reflector that uses SWE-optimized prompt and output format."""

from __future__ import annotations

from ace import Reflector
from loguru import logger

from .model_utils import make_pydantic_agent
from .reflector_prompt import CUSTOM_REFLECTOR_PROMPT
from .outputs import SWEReflectorOutput


class SWEReflector(Reflector):
    """SWE-optimized Reflector with resolved-dependent prompt framing.

    Overrides reflect() to inject {outcome}/{outcome_instructions} based
    on whether the trajectory succeeded or failed.

    Args:
        model: Model identifier string (e.g. "zai/glm-4.5-flash").
        prompt_template: Custom prompt template (defaults to CUSTOM_REFLECTOR_PROMPT).
        max_retries: Maximum retries for structured output validation.
        api_base: Base URL for the LLM endpoint (required for hosted_vllm).
        api_key: API key for the LLM endpoint.
        model_settings: Optional dict for ModelSettings (temperature, max_tokens).
    """

    def __init__(
        self,
        model: str,
        prompt_template: str = CUSTOM_REFLECTOR_PROMPT,
        *,
        max_retries: int = 3,
        api_base: str | None = None,
        api_key: str | None = None,
        model_settings: dict | None = None,
    ) -> None:
        self._prompt_template = prompt_template
        self._agent = make_pydantic_agent(
            model,
            SWEReflectorOutput,
            api_base=api_base,
            api_key=api_key,
            max_retries=max_retries,
            model_settings=model_settings,
        )

    def reflect(
        self,
        *,
        question,
        agent_output,
        skillbook,
        ground_truth=None,
        feedback=None,
        resolved=False,
        **kwargs,
    ):
        """Reflect with resolved-dependent prompt framing."""
        # Build outcome variables for the prompt
        if resolved:
            outcome = "SUCCESS — all tests passed, patch resolved the issue"
            outcome_instructions = (
                "The agent SUCCEEDED. Focus your analysis on:\n"
                "- What strategies and approaches worked well\n"
                "- Reusable patterns that led to the correct solution\n"
                "- Discoveries about the codebase that were verified\n"
                "- Tag cited skills as 'helpful' where applicable\n"
                "Still extract any anti-patterns you notice, but prioritize positive learnings."
            )
        else:
            outcome = "FAILURE — tests did not pass, patch did not resolve the issue"
            outcome_instructions = (
                "The agent FAILED. Focus your analysis on:\n"
                "- Anti-patterns and behaviors that led to failure\n"
                "- False assumptions and false confidence markers\n"
                "- What the agent did WRONG, not what it claims to know\n"
                "- NEVER extract 'the solution is...' or 'the fix requires...' from failures\n"
                "- Tag cited skills as 'harmful' where they led the agent astray"
            )

        # Format the prompt template
        from ace.implementations.helpers import format_optional, make_skillbook_excerpt
        skillbook_excerpt = make_skillbook_excerpt(skillbook, agent_output.skill_ids)
        if skillbook_excerpt:
            skillbook_context = f"Strategies Applied:\n{skillbook_excerpt}"
        else:
            skillbook_context = "(No strategies cited - outcome-based learning)"

        prompt = self._prompt_template.format(
            question=question,
            reasoning=agent_output.reasoning,
            prediction=agent_output.final_answer,
            ground_truth=format_optional(ground_truth),
            feedback=format_optional(feedback),
            skillbook_excerpt=skillbook_context,
            outcome=outcome,
            outcome_instructions=outcome_instructions,
        )

        logger.debug(f"[Reflector] Prompt ({len(prompt)} chars):\n{prompt}")

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
