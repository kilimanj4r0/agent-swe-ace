# prompts/__init__.py
"""Custom prompts for ACE-SWE skillbook learning."""

from .reflector_prompt import CUSTOM_REFLECTOR_PROMPT
from .skill_manager_prompt import CUSTOM_SKILL_MANAGER_PROMPT
from .outputs import (
    SWEReflectorOutput,
    AntiPattern,
    Discovery,
    UnvalidatedHypothesis,
    SkillTag,
)
from .custom_reflector import SWEReflector
from .custom_skill_manager import SWESkillManager

__all__ = [
    "CUSTOM_REFLECTOR_PROMPT",
    "CUSTOM_SKILL_MANAGER_PROMPT",
    "SWEReflectorOutput",
    "SWEReflector",
    "SWESkillManager",
    "AntiPattern",
    "Discovery",
    "UnvalidatedHypothesis",
    "SkillTag",
]
