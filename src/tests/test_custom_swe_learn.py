# src/tests/test_custom_swe_learn.py
"""Tests for custom_swe_learn flag and SWE-optimized learning components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch


class TestSWEReflectorOutput:
    """Tests for SWEReflectorOutput extracted_learnings property."""

    def test_extracted_learnings_includes_anti_pattern_prefix(self):
        """Anti-patterns should have [ANTI-PATTERN] prefix in learning field."""
        from prompts import SWEReflectorOutput, AntiPattern

        output = SWEReflectorOutput(
            reasoning="test",
            error_identification="test error",
            root_cause_analysis="test cause",
            correct_approach="test approach",
            key_insight="test insight",
            anti_patterns=[
                AntiPattern(
                    pattern="Claim success without verification",
                    why_harmful="Wastes iterations",
                    atomicity_score=0.9,
                    evidence="git diff was empty"
                )
            ],
            discoveries=[],
            unvalidated_hypotheses=[],
        )

        learnings = output.extracted_learnings
        assert len(learnings) == 1
        assert "[ANTI-PATTERN]" in learnings[0].learning
        assert "Claim success without verification" in learnings[0].learning

    def test_extracted_learnings_includes_discovery_prefix(self):
        """Discoveries should have [DISCOVERY] prefix in learning field."""
        from prompts import SWEReflectorOutput, Discovery

        output = SWEReflectorOutput(
            reasoning="test",
            error_identification="test error",
            root_cause_analysis="test cause",
            correct_approach="test approach",
            key_insight="test insight",
            anti_patterns=[],
            discoveries=[
                Discovery(
                    finding="/testbed is writable",
                    atomicity_score=1.0,
                    evidence="Docker mounts /testbed as read-write"
                )
            ],
            unvalidated_hypotheses=[],
        )

        learnings = output.extracted_learnings
        assert len(learnings) == 1
        assert "[DISCOVERY]" in learnings[0].learning
        assert "/testbed is writable" in learnings[0].learning

    def test_extracted_learnings_includes_hypothesis_prefix(self):
        """Unvalidated hypotheses should have [HYPOTHESIS] prefix in learning field."""
        from prompts import SWEReflectorOutput, UnvalidatedHypothesis

        output = SWEReflectorOutput(
            reasoning="test",
            error_identification="test error",
            root_cause_analysis="test cause",
            correct_approach="test approach",
            key_insight="test insight",
            anti_patterns=[],
            discoveries=[],
            unvalidated_hypotheses=[
                UnvalidatedHypothesis(
                    hypothesis="Fix requires modifying RST class",
                    why_unvalidated="Never tested this approach",
                    atomicity_score=0.8,
                    evidence="Agent's reasoning"
                )
            ],
        )

        learnings = output.extracted_learnings
        assert len(learnings) == 1
        assert "[HYPOTHESIS]" in learnings[0].learning
        assert "Fix requires modifying RST class" in learnings[0].learning

    def test_extracted_learnings_combines_all_types(self):
        """All learning types should be combined in extracted_learnings."""
        from prompts import SWEReflectorOutput, AntiPattern, Discovery, UnvalidatedHypothesis

        output = SWEReflectorOutput(
            reasoning="test",
            error_identification="test error",
            root_cause_analysis="test cause",
            correct_approach="test approach",
            key_insight="test insight",
            anti_patterns=[
                AntiPattern(pattern="AP1", why_harmful="h1", atomicity_score=0.9, evidence="e1")
            ],
            discoveries=[
                Discovery(finding="D1", atomicity_score=0.9, evidence="e2")
            ],
            unvalidated_hypotheses=[
                UnvalidatedHypothesis(hypothesis="H1", why_unvalidated="u1", atomicity_score=0.8, evidence="e3")
            ],
        )

        learnings = output.extracted_learnings
        assert len(learnings) == 3

        learning_texts = [l.learning for l in learnings]
        assert any("[ANTI-PATTERN]" in t for t in learning_texts)
        assert any("[DISCOVERY]" in t for t in learning_texts)
        assert any("[HYPOTHESIS]" in t for t in learning_texts)

    def test_get_all_learnings_as_dicts(self):
        """get_all_learnings_as_dicts should return flat list with type info."""
        from prompts import SWEReflectorOutput, AntiPattern, Discovery

        output = SWEReflectorOutput(
            reasoning="test",
            error_identification="test error",
            root_cause_analysis="test cause",
            correct_approach="test approach",
            key_insight="test insight",
            anti_patterns=[
                AntiPattern(pattern="AP1", why_harmful="h1", atomicity_score=0.9, evidence="e1")
            ],
            discoveries=[
                Discovery(finding="D1", atomicity_score=0.9, evidence="e2")
            ],
            unvalidated_hypotheses=[],
        )

        learnings = output.get_all_learnings_as_dicts()
        assert len(learnings) == 2

        ap_learning = next(l for l in learnings if l["type"] == "anti_pattern")
        assert ap_learning["learning"] == "AP1"
        assert ap_learning["why_harmful"] == "h1"

        disc_learning = next(l for l in learnings if l["type"] == "discovery")
        assert disc_learning["learning"] == "D1"


class TestSWESkillManager:
    """Tests for SWESkillManager class."""

    def test_skill_manager_subclasses_ace(self):
        """SWESkillManager should subclass ace.SkillManager."""
        from ace import SkillManager
        from prompts import SWESkillManager

        assert issubclass(SWESkillManager, SkillManager)

    def test_skill_manager_uses_custom_prompt(self):
        """SWESkillManager should use CUSTOM_SKILL_MANAGER_PROMPT by default."""
        from prompts import SWESkillManager
        from prompts.skill_manager_prompt import CUSTOM_SKILL_MANAGER_PROMPT

        manager = SWESkillManager("zai/glm-4.5-airx")

        assert manager._prompt_template == CUSTOM_SKILL_MANAGER_PROMPT

    def test_skill_manager_has_update_skills(self):
        """SWESkillManager should inherit update_skills from ace.SkillManager."""
        from prompts import SWESkillManager

        manager = SWESkillManager("zai/glm-4.5-airx")
        assert hasattr(manager, 'update_skills')


class TestCustomSWELearnFlag:
    """Tests for custom_swe_learn configuration flag."""

    def test_custom_swe_learn_default_false(self):
        """custom_swe_learn should default to False."""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["experiment"].get("custom_swe_learn", False) == False

    def test_custom_swe_learn_enables_swe_components(self):
        """When custom_swe_learn is True, SWEReflector and SWESkillManager should be used."""
        # This is tested indirectly through integration tests
        # Here we just verify the imports work
        from prompts import SWEReflector, SWESkillManager

        assert SWEReflector is not None
        assert SWESkillManager is not None


class TestSWEReflector:
    """Tests for SWEReflector class."""

    def test_swe_reflector_subclasses_ace(self):
        """SWEReflector should subclass ace.Reflector."""
        from ace import Reflector
        from prompts import SWEReflector

        assert issubclass(SWEReflector, Reflector)

    def test_swe_reflector_has_custom_prompt(self):
        """SWEReflector should use CUSTOM_REFLECTOR_PROMPT by default."""
        from prompts import SWEReflector
        from prompts.reflector_prompt import CUSTOM_REFLECTOR_PROMPT

        reflector = SWEReflector("zai/glm-4.5-airx")

        assert reflector._prompt_template == CUSTOM_REFLECTOR_PROMPT

    def test_swe_reflector_has_reflect(self):
        """SWEReflector should inherit reflect from ace.Reflector."""
        from prompts import SWEReflector

        reflector = SWEReflector("zai/glm-4.5-airx")
        assert hasattr(reflector, 'reflect')

    def test_swe_reflector_output_type(self):
        """SWEReflector should use SWEReflectorOutput as output type."""
        from prompts import SWEReflector, SWEReflectorOutput

        reflector = SWEReflector("zai/glm-4.5-airx")
        assert reflector._agent._output_type is SWEReflectorOutput


class TestLearningTypePreservation:
    """Tests for learning type preservation through the pipeline."""

    def test_anti_pattern_becomes_avoid_prefix(self):
        """Anti-patterns should result in AVOID: prefix guidance in skills."""
        from prompts.skill_manager_prompt import CUSTOM_SKILL_MANAGER_PROMPT

        # The prompt should instruct LLM to convert [ANTI-PATTERN] to AVOID:
        assert "AVOID:" in CUSTOM_SKILL_MANAGER_PROMPT
        assert "[ANTI-PATTERN]" in CUSTOM_SKILL_MANAGER_PROMPT

    def test_discovery_becomes_verified_prefix(self):
        """Discoveries should result in VERIFIED: prefix guidance in skills."""
        from prompts.skill_manager_prompt import CUSTOM_SKILL_MANAGER_PROMPT

        # The prompt should instruct LLM to convert [DISCOVERY] to VERIFIED:
        assert "VERIFIED:" in CUSTOM_SKILL_MANAGER_PROMPT
        assert "[DISCOVERY]" in CUSTOM_SKILL_MANAGER_PROMPT

    def test_hypothesis_becomes_consider_prefix(self):
        """Hypotheses should result in CONSIDER: prefix guidance in skills."""
        from prompts.skill_manager_prompt import CUSTOM_SKILL_MANAGER_PROMPT

        # The prompt should instruct LLM to convert [HYPOTHESIS] to CONSIDER:
        assert "CONSIDER:" in CUSTOM_SKILL_MANAGER_PROMPT
        assert "[HYPOTHESIS]" in CUSTOM_SKILL_MANAGER_PROMPT

    def test_prompt_has_type_prefix_examples(self):
        """Skill manager prompt should have examples showing type prefixes."""
        from prompts.skill_manager_prompt import CUSTOM_SKILL_MANAGER_PROMPT

        # Should have example JSON output with type prefixes
        assert '"AVOID:' in CUSTOM_SKILL_MANAGER_PROMPT
        assert '"VERIFIED:' in CUSTOM_SKILL_MANAGER_PROMPT
