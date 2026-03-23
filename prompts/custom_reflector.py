# prompts/custom_reflector.py
"""Custom Reflector that uses SWE-optimized prompt and output format."""

import logging
from typing import Any, Optional

from ace_next.core.outputs import AgentOutput
from ace_next.protocols.llm import LLMClientLike
from ace_next.implementations.helpers import format_optional, make_skillbook_excerpt

from .reflector_prompt import CUSTOM_REFLECTOR_PROMPT
from .outputs import SWEReflectorOutput

logger = logging.getLogger(__name__)


class SWEReflector:
    """SWE-optimized Reflector that extracts anti-patterns from failures.

    Key differences from default ace_next.Reflector:
    1. Uses CUSTOM_REFLECTOR_PROMPT with failure-aware analysis
    2. Returns SWEReflectorOutput with anti_patterns, discoveries, unvalidated_hypotheses
    3. Prioritizes extracting what NOT to do from failed attempts

    Args:
        llm: An LLM client that satisfies LLMClientLike.
        prompt_template: Custom prompt template (defaults to CUSTOM_REFLECTOR_PROMPT).
        output_class: Output class for parsing (defaults to SWEReflectorOutput).

    Example::

        reflector = SWEReflector(llm)
        reflection = reflector.reflect(
            question="Fix the bug...",
            agent_output=agent_output,
            skillbook=skillbook,
            feedback="Tests failed: ...",
        )
        # reflection.anti_patterns contains behaviors to avoid
        # reflection.discoveries contains verified facts
        # reflection.unvalidated_hypotheses contains untested claims
    """

    def __init__(
        self,
        llm: LLMClientLike,
        prompt_template: str = CUSTOM_REFLECTOR_PROMPT,
        output_class: type = SWEReflectorOutput,
        *,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.output_class = output_class
        self.max_retries = max_retries

    def reflect(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Any,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> SWEReflectorOutput:
        """Analyze agent performance and extract learnings.

        This method signature matches ReflectorLike protocol.

        Args:
            question: The original question/problem statement.
            agent_output: The agent's output to analyze.
            skillbook: Current skillbook (duck-typed, needs get_skill).
            ground_truth: Expected correct answer (if available).
            feedback: Environment feedback text (e.g., test results).
            **kwargs: Accepted for protocol compatibility but not forwarded.

        Returns:
            SWEReflectorOutput with anti_patterns, discoveries, unvalidated_hypotheses.
        """
        skillbook_excerpt = make_skillbook_excerpt(skillbook, agent_output.skill_ids)

        if skillbook_excerpt:
            skillbook_context = f"Strategies Applied:\n{skillbook_excerpt}"
        else:
            skillbook_context = "(No strategies cited - outcome-based learning)"

        prompt = self.prompt_template.format(
            question=question,
            reasoning=agent_output.reasoning,
            prediction=agent_output.final_answer,
            ground_truth=format_optional(ground_truth),
            feedback=format_optional(feedback),
            skillbook_excerpt=skillbook_context,
        )

        return self.llm.complete_structured(
            prompt, self.output_class, max_retries=self.max_retries
        )
