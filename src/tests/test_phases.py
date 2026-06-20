# src/tests/test_phases.py
"""Tests for phase scripts."""
import json
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.miniswe_agent import AgentResult


class TestPredictPhase:
    """Test the predict (agent) phase."""

    def test_predict_phase_creates_trajectory(self, tmp_path):
        """Test that predict phase creates a trajectory file."""
        from phases.predict import PredictPhase

        # Mock the agent
        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            exit_status="submitted",
            patch="diff --git a/file.py...",
            trajectory=[{"role": "user", "content": "Fix"}],
            error=None,
        )

        instance = {
            "instance_id": "test__repo-123",
            "repo": "test/repo",
            "problem_statement": "Fix the bug",
        }

        phase = PredictPhase(
            agent=mock_agent,
            output_dir=tmp_path,
        )

        result = phase.run(
            instance=instance,
            skillbook=None,
            iteration=0,
        )

        assert result.exit_status == "submitted"
        assert result.patch  # Check patch is not empty
        assert result.trajectory is not None

    def test_predict_phase_saves_trajectory(self, tmp_path):
        """Test that predict phase saves trajectory to file."""
        from phases.predict import PredictPhase

        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            exit_status="submitted",
            patch="patch content",
            trajectory=[{"role": "user", "content": "Fix"}],
            error=None,
        )

        instance = {"instance_id": "test__repo-123", "problem_statement": "Fix"}

        phase = PredictPhase(agent=mock_agent, output_dir=tmp_path)
        phase.run(instance=instance, skillbook=None, iteration=0)

        # Check trajectory file was created
        traj_file = tmp_path / "swebench-lite" / "trajectories" / "test__repo-123" / "iter_0.json"
        assert traj_file.exists()


class TestSkillbookInjection:
    """Test skillbook injection edge cases."""

    MOCK_MINI_CONFIG = {
        "agent": {
            "system_template": "You are a coding assistant.",
            "instance_template": "Problem:\n{{ problem_statement }}\n\n<example_response>...</example_response>",
            "action_observation_template": "{{ observation }}",
        }
    }

    def test_empty_skillbook_returns_default_template(self):
        """Test that empty skillbook returns unmodified template."""
        from phases.predict import build_instance_template

        with patch("phases.predict._load_mini_swe_config", return_value=self.MOCK_MINI_CONFIG):
            # Empty skillbook
            mock_skillbook = Mock()
            mock_skillbook.skills.return_value = []

            template = build_instance_template(skillbook=mock_skillbook)

            # Should NOT contain skillbook section
            assert "## Learned Strategies" not in template

    def test_none_skillbook_returns_default_template(self):
        """Test that None skillbook returns unmodified template."""
        from phases.predict import build_instance_template

        with patch("phases.predict._load_mini_swe_config", return_value=self.MOCK_MINI_CONFIG):
            template = build_instance_template(skillbook=None)

            assert "## Learned Strategies" not in template

    def test_skillbook_injection_adds_section(self):
        """Test that skillbook with skills injects Learned Strategies section."""
        from phases.predict import build_instance_template

        with patch("phases.predict._load_mini_swe_config", return_value=self.MOCK_MINI_CONFIG):
            # Mock skillbook with one skill
            mock_skill = Mock()
            mock_skill.id = "skill-1"
            mock_skill.content = "Always check imports first."
            mock_skill.justification = "Prevents NameError failures."

            mock_skillbook = Mock()
            mock_skillbook.skills.return_value = [mock_skill]

            template = build_instance_template(skillbook=mock_skillbook)

            # Should contain the skillbook section
            assert "## Learned Strategies" in template
            # Should contain the skill content
            assert "Always check imports first." in template
            # Should appear before <example_response>
            assert template.index("## Learned Strategies") < template.index("<example_response>")


class TestRetrieveSkillsPreservesIds:
    """Retrieval must preserve the original skill IDs in the narrowed skillbook/prompt."""

    MOCK_MINI_CONFIG = {
        "agent": {
            "system_template": "You are a coding assistant.",
            "instance_template": "Problem:\n{{ problem_statement }}\n\n<example_response>...</example_response>",
            "action_observation_template": "{{ observation }}",
        }
    }

    def test_narrowed_skillbook_keeps_original_ids(self):
        """Selected skills keep their real IDs; they must not be renumbered to 00001..k."""
        from phases.predict import PredictPhase, build_instance_template

        # High-numbered IDs that a fresh Skillbook's _generate_id counter (starts at 1)
        # could never produce — so any renumbering is unambiguous.
        full = []
        for sid in [
            "file_modification-00740",
            "file_modification-00247",
            "problem_analysis-00540",
            "environment-00004",
            "verification-00515",
        ]:
            skill = Mock()
            skill.id = sid
            skill.section = sid.rsplit("-", 1)[0]  # "file_modification", etc.
            skill.content = f"AVOID: guidance from {sid}"
            skill.justification = None
            skill.evidence = None
            full.append(skill)

        # Retriever narrows the full pool down to the first 3 skills.
        selected = full[:3]
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = selected

        skillbook = Mock()
        skillbook.skills.return_value = full

        phase = PredictPhase(
            agent=Mock(), output_dir=Path("/tmp"), skill_retriever=mock_retriever
        )

        filtered_sb, stats = phase._retrieve_skills(skillbook, {"instance_id": "x"})

        # retrieval_stats is the source of truth for which skills were selected.
        assert stats["selected_ids"] == [s.id for s in selected]

        with patch("phases.predict._load_mini_swe_config", return_value=self.MOCK_MINI_CONFIG):
            template = build_instance_template(skillbook=filtered_sb)

        # Every real selected ID must appear as a section header in the injected prompt.
        for sid in stats["selected_ids"]:
            assert f"### {sid}" in template, f"{sid} missing from injected template"

        # Fresh-counter IDs (the renumbering bug) must NOT appear.
        assert "file_modification-00001" not in template
        assert "problem_analysis-00003" not in template


class TestIsValidPatch:
    """Test _is_valid_patch from evaluate.py."""

    @pytest.mark.parametrize("patch,expected", [
        ("", False),
        ("   ", False),
        ("\n\t\n", False),
        ("diff --git a/file.py b/file.py\n...", True),
        ("   diff --git a/f b/f\n...", True),  # leading whitespace ok
        ("--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new", True),
        ("just some text", False),
        ("--- a/file.py\n+++ b/file.py\nno hunk marker", False),  # missing @@
        ("+++ b/file.py\n@@ -1 +1 @@\n-old\n+new", False),  # missing ---
        ("--- a/file.py\nmissing +++\n@@ -1 +1 @@\n-old\n+new", False),  # missing +++
    ])
    def test_is_valid_patch(self, patch, expected):
        """Test _is_valid_patch with various inputs."""
        from phases.evaluate import _is_valid_patch

        assert _is_valid_patch(patch) == expected


class TestEvaluatePhase:
    """Test the evaluate phase."""

    def test_evaluate_phase_resolved(self, tmp_path):
        """Test evaluate phase with resolved patch."""
        from phases.evaluate import EvaluatePhase

        # Mock the validator
        with patch("phases.evaluate.validate_patch", return_value=True):
            phase = EvaluatePhase(
                use_docker=True,
                output_dir=tmp_path,
            )

            instance = {"instance_id": "test__repo-123"}
            result = phase.run(
                instance=instance,
                patch="diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
                iteration=0,
            )

            assert result.resolved is True
            assert "resolved" in result.metrics

    def test_evaluate_phase_not_resolved(self, tmp_path):
        """Test evaluate phase with unresolved patch."""
        from phases.evaluate import EvaluatePhase

        with patch("phases.evaluate.validate_patch", return_value=False):
            phase = EvaluatePhase(
                use_docker=True,
                output_dir=tmp_path,
            )

            instance = {"instance_id": "test__repo-123"}
            result = phase.run(
                instance=instance,
                patch="diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
                iteration=0,
            )

            assert result.resolved is False

    def test_evaluate_phase_empty_patch(self, tmp_path):
        """Test evaluate phase with empty patch."""
        from phases.evaluate import EvaluatePhase

        phase = EvaluatePhase(
            use_docker=True,
            output_dir=tmp_path,
        )

        instance = {"instance_id": "test__repo-123"}
        result = phase.run(
            instance=instance,
            patch="",
            iteration=0,
        )

        assert result.resolved is False
        assert "No patch" in result.feedback

    def test_evaluate_phase_invalid_format_patch(self, tmp_path):
        """Test evaluate phase with invalid format patch (no Docker needed)."""
        from phases.evaluate import EvaluatePhase

        phase = EvaluatePhase(
            use_docker=True,
            output_dir=tmp_path,
        )

        instance = {"instance_id": "test__repo-123"}
        result = phase.run(
            instance=instance,
            patch="just some text that is not a diff",
            iteration=0,
        )

        assert result.resolved is False
        assert "not a valid diff format" in result.feedback
        assert result.metrics["patch_invalid_format"] == 1.0


class TestLearnPhase:
    """Test the learn phase."""

    def test_learn_phase_creates_skill(self, tmp_path):
        """Test that learn phase creates a skill from failure."""
        from phases.learn import LearnPhase
        from ace import Skillbook
        from ace.core.outputs import SkillManagerOutput, UpdateBatch
        from ace.core.skillbook import UpdateOperation

        # Mock ACE components
        mock_reflector = Mock()
        mock_reflector.reflect.return_value = Mock(
            error_identification="Wrong approach",
            root_cause_analysis="Misunderstood the issue",
            key_insight="Check imports first",
        )

        # Build update result with 1 ADD and 1 UPDATE operation
        add_op = UpdateOperation(
            type="ADD",
            section="root_cause",
            content="Missing import statement",
            skill_id="skill-1",
        )
        update_op = UpdateOperation(
            type="UPDATE",
            section="approach",
            content="Check imports before fixing logic",
            skill_id="skill-0",
        )
        update_batch = UpdateBatch(
            reasoning="Found missing import and updated approach",
            operations=[add_op, update_op],
        )
        mock_skill_manager = Mock()
        mock_skill_manager.update_skills.return_value = SkillManagerOutput(
            update=update_batch,
            raw={},
        )

        phase = LearnPhase(
            reflector=mock_reflector,
            skill_manager=mock_skill_manager,
            output_dir=tmp_path,
        )

        skillbook = Skillbook()
        instance = {"instance_id": "test__repo-123"}
        trajectory = {"messages": [{"role": "user", "content": "Fix"}], "info": {"exit_status": "submitted"}}
        patch = "bad patch"

        result = phase.run(
            skillbook=skillbook,
            instance=instance,
            trajectory=trajectory,
            patch=patch,
            iteration=0,
        )

        assert result.skills_added == 1
        assert result.skills_updated == 1
        mock_reflector.reflect.assert_called_once()

    def test_learn_phase_handles_reflection_failure(self, tmp_path):
        """Test that learn phase handles reflection failures gracefully."""
        from phases.learn import LearnPhase
        from ace import Skillbook

        mock_reflector = Mock()
        mock_reflector.reflect.side_effect = Exception("Reflection failed")

        mock_skill_manager = Mock()

        phase = LearnPhase(
            reflector=mock_reflector,
            skill_manager=mock_skill_manager,
            output_dir=tmp_path,
        )

        skillbook = Skillbook()
        instance = {"instance_id": "test__repo-123"}
        trajectory = {"messages": [], "info": {"exit_status": "error"}}
        patch = ""

        result = phase.run(
            skillbook=skillbook,
            instance=instance,
            trajectory=trajectory,
            patch=patch,
            iteration=0,
        )

        # Should not crash, returns zero skills added
        assert result.skills_added == 0

    def test_learn_phase_no_update_batch(self, tmp_path):
        """Test that learn phase handles SkillManager output with no .update attribute."""
        from phases.learn import LearnPhase
        from ace import Skillbook

        mock_reflector = Mock()
        mock_reflector.reflect.return_value = Mock(
            error_identification="Wrong approach",
            root_cause_analysis="Misunderstood the issue",
            key_insight="Check imports first",
        )

        # Return a plain dict (no .update attribute) — simulates unexpected output shape
        mock_skill_manager = Mock()
        mock_skill_manager.update_skills.return_value = {"skills_added": ["skill-1"]}

        phase = LearnPhase(
            reflector=mock_reflector,
            skill_manager=mock_skill_manager,
            output_dir=tmp_path,
        )

        skillbook = Skillbook()
        instance = {"instance_id": "test__repo-123"}
        trajectory = {"messages": [{"role": "user", "content": "Fix"}], "info": {"exit_status": "submitted"}}
        patch = "bad patch"

        result = phase.run(
            skillbook=skillbook,
            instance=instance,
            trajectory=trajectory,
            patch=patch,
            iteration=0,
        )

        # When update_result has no .update, skills_added/skills_updated should be 0
        assert result.skills_added == 0
        assert result.skills_updated == 0
