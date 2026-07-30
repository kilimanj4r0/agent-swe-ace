# prompts/__init__.py
"""Custom prompts for ACE-SWE skillbook learning."""

from .custom_reflector import SWEReflector
from .custom_skill_manager import SWESkillManager
from .outputs import (
    AntiPattern,
    Discovery,
    SkillTag,
    SWEReflectorOutput,
    UnvalidatedHypothesis,
)
from .reflector_prompt import CUSTOM_REFLECTOR_PROMPT
from .skill_manager_prompt import CUSTOM_SKILL_MANAGER_PROMPT

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
