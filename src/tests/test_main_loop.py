# src/tests/test_main_loop.py
"""Tests for main loop runner."""
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMainLoop:
    """Test the main experiment loop."""

    def test_main_loop_single_instance_resolved_first_try(self, tmp_path):
        """Test loop exits early when resolved on first try."""
        from runners.main_loop import ExperimentLoop

        # Mock all phases
        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="test__repo-123",
            exit_status="submitted",
            patch="good patch",
            trajectory=[],
        )

        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id="test__repo-123",
            resolved=True,  # Resolved!
            feedback="Great!",
            metrics={"resolved": 1.0},
        )

        mock_learn = Mock()

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="test-run",
            max_attempts=3,
        )

        instance = {"instance_id": "test__repo-123", "problem_statement": "Fix"}
        results = loop.run_instance(instance)

        # Should only run once (resolved first try)
        assert results.final_resolved is True
        assert mock_predict.run.call_count == 1
        mock_learn.run.assert_not_called()  # No learning needed

    def test_main_loop_retries_on_failure(self, tmp_path):
        """Test loop retries when not resolved."""
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="test__repo-123",
            exit_status="submitted",
            patch="patch",
            trajectory=[],
        )

        # First attempt fails, second succeeds
        mock_evaluate = Mock()
        mock_evaluate.run.side_effect = [
            Mock(instance_id="test__repo-123", resolved=False, feedback="Bad"),
            Mock(instance_id="test__repo-123", resolved=True, feedback="Good"),
        ]

        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="test-run",
            max_attempts=3,
        )

        instance = {"instance_id": "test__repo-123", "problem_statement": "Fix"}
        results = loop.run_instance(instance)

        # Should run twice
        assert mock_predict.run.call_count == 2
        assert mock_learn.run.call_count == 1  # Learn after first failure

    def test_main_loop_max_attempts(self, tmp_path):
        """Test loop respects max_attempts."""
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="test__repo-123",
            exit_status="submitted",
            patch="patch",
            trajectory=[],
        )

        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id="test__repo-123",
            resolved=False,  # Always fails
            feedback="Bad",
        )

        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="test-run",
            max_attempts=2,  # Only 2 attempts
        )

        instance = {"instance_id": "test__repo-123", "problem_statement": "Fix"}
        results = loop.run_instance(instance)

        # Should stop at max_attempts
        assert mock_predict.run.call_count == 2
        assert results.final_resolved is False


class TestSkillbookModes:
    """Test skillbook mode handling."""

    def test_per_instance_mode(self, tmp_path):
        """Test per-instance skillbook mode (default)."""
        from runners.main_loop import ExperimentLoop
        from ace import Skillbook

        mock_predict = Mock()
        mock_evaluate = Mock()
        mock_learn = Mock()

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            skillbook_mode="per_instance",
        )

        # Each instance gets fresh skillbook
        skillbook = loop.get_skillbook("repo1")
        assert isinstance(skillbook, Skillbook)
        assert len(skillbook.skills()) == 0

    def test_global_mode(self, tmp_path):
        """Test global skillbook mode."""
        from runners.main_loop import ExperimentLoop
        from ace import Skillbook, Skill

        mock_predict = Mock()
        mock_evaluate = Mock()
        mock_learn = Mock()

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            skillbook_mode="global",
        )

        # Add a skill to global skillbook
        skill = Skill(id="test-skill", section="debugging", content="Test content")
        loop.global_skillbook._skills["test-skill"] = skill

        # All repos get same skillbook
        skillbook1 = loop.get_skillbook("repo1")
        skillbook2 = loop.get_skillbook("repo2")

        assert skillbook1 is skillbook2
        assert len(skillbook1.skills()) == 1
