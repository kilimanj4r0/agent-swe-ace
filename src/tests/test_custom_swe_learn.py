# src/tests/test_custom_swe_learn.py
"""Tests for custom_swe_learn flag and SWE-optimized learning components."""

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

    def test_skill_manager_uses_custom_prompt(self):
        """SWESkillManager should use CUSTOM_SKILL_MANAGER_PROMPT by default."""
        from prompts import SWESkillManager
        from prompts.skill_manager_prompt import CUSTOM_SKILL_MANAGER_PROMPT

        mock_llm = MagicMock()
        manager = SWESkillManager(mock_llm)

        assert manager.prompt_template == CUSTOM_SKILL_MANAGER_PROMPT

    def test_skill_manager_handles_swe_reflector_output(self):
        """SWESkillManager should handle SWEReflectorOutput correctly."""
        from prompts import SWESkillManager, SWEReflectorOutput, AntiPattern

        mock_llm = MagicMock()
        mock_llm.complete_structured.return_value = MagicMock()

        manager = SWESkillManager(mock_llm)

        # Create SWEReflectorOutput
        output = SWEReflectorOutput(
            reasoning="test reasoning",
            error_identification="test error",
            root_cause_analysis="test cause",
            correct_approach="test approach",
            key_insight="test insight",
            anti_patterns=[
                AntiPattern(pattern="AP1", why_harmful="h1", atomicity_score=0.9, evidence="e1")
            ],
            discoveries=[],
            unvalidated_hypotheses=[],
        )

        mock_skillbook = MagicMock()
        mock_skillbook.stats.return_value = {"total": 0}
        mock_skillbook.as_prompt.return_value = "(empty skillbook)"

        manager.update_skills(
            reflections=(output,),
            skillbook=mock_skillbook,
            question_context="test context",
            progress="test progress",
        )

        # Verify complete_structured was called
        assert mock_llm.complete_structured.called

        # Get the prompt that was used
        call_args = mock_llm.complete_structured.call_args
        prompt = call_args[0][0]

        # Verify the prompt contains the extracted learning with prefix
        assert "[ANTI-PATTERN]" in prompt
        assert "AP1" in prompt


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

    def test_swe_reflector_has_custom_prompt(self):
        """SWEReflector should use CUSTOM_REFLECTOR_PROMPT by default."""
        from prompts import SWEReflector
        from prompts.reflector_prompt import CUSTOM_REFLECTOR_PROMPT

        mock_llm = MagicMock()
        reflector = SWEReflector(mock_llm)

        assert reflector.prompt_template == CUSTOM_REFLECTOR_PROMPT

    def test_swe_reflector_uses_structured_output(self):
        """SWEReflector should use complete_structured with SWEReflectorOutput."""
        from prompts import SWEReflector, SWEReflectorOutput

        mock_llm = MagicMock()
        mock_llm.complete_structured.return_value = SWEReflectorOutput(
            reasoning="test",
            error_identification="test",
            root_cause_analysis="test",
            correct_approach="test",
            key_insight="test",
            anti_patterns=[],
            discoveries=[],
            unvalidated_hypotheses=[],
        )

        reflector = SWEReflector(mock_llm)

        from ace_next.core.outputs import AgentOutput
        from ace_next import Skillbook

        result = reflector.reflect(
            question="test question",
            agent_output=AgentOutput(
                reasoning="test reasoning",
                final_answer="test answer",
                skill_ids=[],
                raw={}
            ),
            skillbook=Skillbook(),
        )

        assert isinstance(result, SWEReflectorOutput)
        assert mock_llm.complete_structured.called


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
