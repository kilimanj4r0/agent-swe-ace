# src/tests/test_phases.py
"""Tests for phase scripts."""
import json
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPredictPhase:
    """Test the predict (agent) phase."""

    def test_predict_phase_creates_trajectory(self, tmp_path):
        """Test that predict phase creates a trajectory file."""
        from phases.predict import PredictPhase

        # Mock the agent
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.exit_status = "submitted"
        mock_result.patch = "diff --git a/file.py..."
        mock_result.trajectory = [{"role": "user", "content": "Fix"}]
        mock_result.error = None
        mock_agent.run.return_value = mock_result

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
        mock_result = Mock()
        mock_result.exit_status = "submitted"
        mock_result.patch = "patch content"
        mock_result.trajectory = [{"role": "user", "content": "Fix"}]
        mock_result.error = None
        mock_agent.run.return_value = mock_result

        instance = {"instance_id": "test__repo-123", "problem_statement": "Fix"}

        phase = PredictPhase(agent=mock_agent, output_dir=tmp_path)
        phase.run(instance=instance, skillbook=None, iteration=0)

        # Check trajectory file was created
        traj_file = tmp_path / "swebench-lite" / "trajectories" / "test__repo-123" / "iter_0.json"
        assert traj_file.exists()


class TestSkillbookInjection:
    """Test skillbook injection edge cases."""

    def test_empty_skillbook_returns_default_template(self):
        """Test that empty skillbook returns unmodified template."""
        from phases.predict import build_instance_template

        # Empty skillbook
        mock_skillbook = Mock()
        mock_skillbook.skills.return_value = []

        template = build_instance_template(skillbook=mock_skillbook)

        # Should NOT contain skillbook section
        assert "## Learned Strategies" not in template

    def test_none_skillbook_returns_default_template(self):
        """Test that None skillbook returns unmodified template."""
        from phases.predict import build_instance_template

        template = build_instance_template(skillbook=None)

        assert "## Learned Strategies" not in template


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
                patch="valid patch",
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
                patch="invalid patch",
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


class TestLearnPhase:
    """Test the learn phase."""

    def test_learn_phase_creates_skill(self, tmp_path):
        """Test that learn phase creates a skill from failure."""
        from phases.learn import LearnPhase
        from ace_next import Skillbook

        # Mock ACE components
        mock_reflector = Mock()
        mock_reflector.reflect.return_value = Mock(
            error_identification="Wrong approach",
            root_cause_analysis="Misunderstood the issue",
            key_insight="Check imports first",
        )

        mock_skill_manager = Mock()
        mock_skill_manager.update_skills.return_value = {
            "skills_added": ["skill-1"],
            "skills_updated": [],
        }

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

        assert result.skills_added >= 0
        mock_reflector.reflect.assert_called_once()

    def test_learn_phase_handles_reflection_failure(self, tmp_path):
        """Test that learn phase handles reflection failures gracefully."""
        from phases.learn import LearnPhase
        from ace_next import Skillbook

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
