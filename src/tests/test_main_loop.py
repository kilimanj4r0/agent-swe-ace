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
            max_attempts=1,
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
            max_attempts=2,
        )

        instance = {"instance_id": "test__repo-123", "problem_statement": "Fix"}
        results = loop.run_instance(instance)

        # Should run twice (fail→learn→resolve→break since max_attempts=2)
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


class TestResumeSupport:
    """Test resume state handling in ExperimentLoop."""

    def test_resume_skips_complete_instance(self, tmp_path):
        """Fully complete instances should be skipped (start_iteration=-1)."""
        from runners.main_loop import ExperimentLoop
        from data_io.resume_scanner import ResumePoint

        mock_predict = Mock()
        mock_evaluate = Mock()
        mock_learn = Mock()

        resume_state = {
            "test__repo-123": ResumePoint(
                resume_dir=tmp_path,
                last_complete_iter=0,
                is_fully_complete=True,
            )
        }

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            resume_state=resume_state,
        )

        start = loop._get_resume_start("test__repo-123")
        assert start == -1

    def test_resume_continues_partial_instance(self, tmp_path):
        """Partial instances should continue from last_complete_iter + 1."""
        from runners.main_loop import ExperimentLoop
        from data_io.resume_scanner import ResumePoint

        mock_predict = Mock()
        mock_evaluate = Mock()
        mock_learn = Mock()

        resume_state = {
            "test__repo-123": ResumePoint(
                resume_dir=tmp_path,
                last_complete_iter=1,
                is_fully_complete=False,
            )
        }

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            max_attempts=4,
            resume_state=resume_state,
        )

        start = loop._get_resume_start("test__repo-123")
        assert start == 2  # Continue from iter_2

    def test_resume_copies_artifacts(self, tmp_path):
        """Partial resume should copy artifacts from source directory."""
        from runners.main_loop import ExperimentLoop
        from data_io.resume_scanner import ResumePoint
        import json

        # Create source directory with artifacts
        source_dir = tmp_path / "source"
        benchmark = "princeton-nlp__SWE-bench_Lite"
        instance_id = "django__django-12345"
        for subdir in ("trajectories", "results", "skillbooks"):
            d = source_dir / benchmark / subdir / instance_id
            d.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                (d / f"iter_{i}.json").write_text(json.dumps({"test": i}))

        mock_predict = Mock()
        mock_evaluate = Mock()
        mock_learn = Mock()

        resume_state = {
            instance_id: ResumePoint(
                resume_dir=source_dir,
                last_complete_iter=1,
                is_fully_complete=False,
            )
        }

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=dest_dir,
            benchmark=benchmark,
            resume_state=resume_state,
        )

        loop._copy_resume_artifacts(instance_id)

        # Verify artifacts were copied
        for subdir in ("trajectories", "results"):
            for i in range(2):
                f = dest_dir / benchmark / subdir / instance_id / f"iter_{i}.json"
                assert f.exists(), f"Missing {subdir}/iter_{i}.json"

    def test_no_resume_starts_from_zero(self, tmp_path):
        """Instances not in resume_state should start from iter_0."""
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_evaluate = Mock()
        mock_learn = Mock()

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
        )

        start = loop._get_resume_start("unknown__instance-999")
        assert start == 0


class TestForceLearn:
    """Test force_learn and frozen_skillbook flags."""

    def test_force_learn_runs_learn_on_resolved(self, tmp_path):
        """When force_learn=True, learn runs even if resolved."""
        from runners.main_loop import ExperimentLoop

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
            resolved=True,
            feedback="Great!",
            metrics={"resolved": 1.0},
        )

        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="test-run",
            max_attempts=1,
        )

        instance = {"instance_id": "test__repo-123", "problem_statement": "Fix"}
        result = loop.run_instance(instance, force_learn=True)

        # Should have called learn even though resolved
        assert result.final_resolved is True
        mock_learn.run.assert_called_once()

    def test_frozen_skillbook_skips_learn(self, tmp_path):
        """When frozen_skillbook=True, learn never runs."""
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="test__repo-123",
            exit_status="submitted",
            patch="bad patch",
            trajectory=[],
        )

        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id="test__repo-123",
            resolved=False,
            feedback="Bad",
            metrics={"resolved": 0.0},
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

        # Frozen path now calls predict.prepare_skillbook once per instance (a no-op
        # without a retriever); echo the skillbook so the mocked runs proceed.
        mock_predict.prepare_skillbook.side_effect = lambda instance, skillbook, phase=None: (skillbook, None)

        instance = {"instance_id": "test__repo-123", "problem_statement": "Fix"}
        result = loop.run_instance(instance, frozen_skillbook=True, max_attempts_override=1)

        # Learn should not be called at all
        assert result.final_resolved is False
        mock_learn.run.assert_not_called()

    def test_force_learn_false_skips_learn_on_single_attempt(self, tmp_path):
        """When force_learn=False and max_attempts=1, learn is skipped entirely."""
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="test__repo-123",
            exit_status="submitted",
            patch="bad patch",
            trajectory=[],
        )

        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id="test__repo-123",
            resolved=False,
            feedback="Bad",
            metrics={"resolved": 0.0},
        )

        mock_learn = Mock()

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="test-run",
            max_attempts=1,
            force_learn=False,
        )

        instance = {"instance_id": "test__repo-123", "problem_statement": "Fix"}
        result = loop.run_instance(instance, max_attempts_override=1)

        assert result.final_resolved is False
        mock_learn.run.assert_not_called()

    def test_force_learn_true_runs_learn_on_single_attempt(self, tmp_path):
        """When force_learn=True and max_attempts=1, learn still runs on unresolved."""
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="test__repo-123",
            exit_status="submitted",
            patch="bad patch",
            trajectory=[],
        )

        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id="test__repo-123",
            resolved=False,
            feedback="Bad",
            metrics={"resolved": 0.0},
        )

        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="test-run",
            max_attempts=1,
            force_learn=True,
        )

        instance = {"instance_id": "test__repo-123", "problem_statement": "Fix"}
        result = loop.run_instance(instance, max_attempts_override=1)

        assert result.final_resolved is False
        mock_learn.run.assert_called_once()

    def test_max_attempts_override_limits_iterations(self, tmp_path):
        """max_attempts_override=1 forces single attempt."""
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
            resolved=False,
            feedback="Bad",
            metrics={"resolved": 0.0},
        )

        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="test-run",
            max_attempts=10,  # Would normally try 10 times
        )

        instance = {"instance_id": "test__repo-123", "problem_statement": "Fix"}
        result = loop.run_instance(instance, max_attempts_override=1)

        # Should only run once despite max_attempts=10
        assert mock_predict.run.call_count == 1
        assert result.total_attempts == 1


class TestPerRepoMode:
    """Test per_repo skillbook mode."""

    def test_per_repo_mode(self, tmp_path):
        """Skillbook accumulates across instances from the same repo."""
        from runners.main_loop import ExperimentLoop
        from ace import Skillbook

        loop = ExperimentLoop(
            predict_phase=Mock(),
            evaluate_phase=Mock(),
            learn_phase=Mock(),
            output_dir=tmp_path,
            skillbook_mode="per_repo",
        )

        # Same repo gets same skillbook
        sb1 = loop.get_skillbook("django/django")
        sb2 = loop.get_skillbook("django/django")
        assert sb1 is sb2

        # Different repo gets different skillbook
        sb3 = loop.get_skillbook("flask/flask")
        assert sb1 is not sb3

    def test_two_phase_run(self, tmp_path):
        """Test two-phase run produces correct statistics structure."""
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="test__repo-123",
            exit_status="submitted",
            patch="patch",
            trajectory=[],
        )

        def evaluate_side_effect(*args, **kwargs):
            instance = kwargs.get("instance") or args[0]
            iid = instance.get("instance_id", "unknown")
            if iid == "train-1":
                return Mock(instance_id=iid, resolved=True, feedback="OK")
            elif iid == "train-2":
                return Mock(instance_id=iid, resolved=False, feedback="Bad")
            else:
                phase = kwargs.get("phase", "")
                if "baseline" in phase:
                    return Mock(instance_id=iid, resolved=False, feedback="Bad")
                else:
                    return Mock(instance_id=iid, resolved=True, feedback="OK")

        mock_evaluate = Mock()
        mock_evaluate.run.side_effect = evaluate_side_effect

        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="test-run",
            max_attempts=3,
            skillbook_mode="per_repo",
        )

        train = [
            {"instance_id": "train-1", "repo": "django/django"},
            {"instance_id": "train-2", "repo": "django/django"},
        ]
        val = [
            {"instance_id": "val-1", "repo": "django/django"},
        ]

        # Val pass (frozen) now calls predict.prepare_skillbook once per instance
        # (a no-op without a retriever); echo the skillbook for the mocked runs.
        mock_predict.prepare_skillbook.side_effect = lambda instance, skillbook, phase=None: (skillbook, None)

        stats = loop.run(train, val_instances=val)

        # Check statistics structure
        assert "train_phase" in stats
        assert "val_baseline_phase" in stats
        assert "val_skillbook_phase" in stats
        assert "summary" in stats

        assert stats["train_phase"]["total_instances"] == 2
        assert stats["train_phase"]["resolved_count"] == 1
        assert stats["val_baseline_phase"]["total_instances"] == 1
        assert stats["val_baseline_phase"]["resolved_count"] == 0
        assert stats["val_skillbook_phase"]["total_instances"] == 1
        assert stats["val_skillbook_phase"]["resolved_count"] == 1

        # Skillbook improvement
        assert stats["summary"]["newly_resolved_by_skillbook"] == ["val-1"]
        assert stats["summary"]["lost_by_skillbook"] == []

    def test_backward_compat_no_split(self, tmp_path):
        """Without val_instances, run() works identically to before."""
        from runners.main_loop import ExperimentLoop

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
            resolved=True,
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
            skillbook_mode="per_instance",
        )

        instance = {"instance_id": "test__repo-123", "problem_statement": "Fix"}
        stats = loop.run([instance])

        # Old-style statistics
        assert "train_phase" not in stats
        assert "val_baseline_phase" not in stats
        assert "skillbook_assisted" in stats
        assert stats["resolved_count"] == 1


class TestTrainBaselineReuse:
    """Test reusing existing trajectories from baseline_run_dir in train phase."""

    def _setup_baseline_dir(self, tmp_path, instance_id, exit_status="Submitted",
                            resolved=True, has_traj=True, has_result=True):
        """Helper: create a fake baseline run directory with artifacts."""
        baseline_dir = tmp_path / "baseline"
        benchmark = "princeton-nlp__SWE-bench_Lite"

        if has_traj:
            traj_dir = baseline_dir / benchmark / "trajectories" / instance_id
            traj_dir.mkdir(parents=True, exist_ok=True)
            import json
            (traj_dir / "iter_0.json").write_text(json.dumps({
                "info": {"exit_status": exit_status, "submission": "patch content"},
                "messages": [{"role": "user", "content": "fix it"}],
            }))

        if has_result:
            result_dir = baseline_dir / benchmark / "results" / instance_id
            result_dir.mkdir(parents=True, exist_ok=True)
            import json
            (result_dir / "iter_0.json").write_text(json.dumps({
                "resolved": resolved,
                "feedback": "test feedback",
            }))

        return baseline_dir

    def test_reuse_valid_trajectory_only_learns(self, tmp_path):
        """When baseline has valid trajectory (Submitted), only run learn phase."""
        from runners.main_loop import ExperimentLoop

        instance_id = "django__django-12345"
        baseline_dir = self._setup_baseline_dir(tmp_path, instance_id, "Submitted", resolved=False)

        mock_predict = Mock()
        mock_evaluate = Mock()
        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=2)

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="test",
            skillbook_mode="per_repo",
        )

        instance = {"instance_id": instance_id, "repo": "django/django"}
        result = loop._run_train_instance_reusing_baseline(instance, baseline_dir)

        # Learn was called, predict/evaluate were not
        mock_learn.run.assert_called_once()
        mock_predict.run.assert_not_called()
        mock_evaluate.run.assert_not_called()

        # Result reflects baseline resolved status
        assert result.final_resolved is False
        assert result.total_attempts == 1

        # Artifacts copied to train/ subdirs
        assert (tmp_path / "princeton-nlp__SWE-bench_Lite" / "trajectories" / "train" /
                instance_id / "iter_0.json").exists()
        assert (tmp_path / "princeton-nlp__SWE-bench_Lite" / "results" / "train" /
                instance_id / "iter_0.json").exists()

    def test_reuse_limits_exceeded_status(self, tmp_path):
        """LimitsExceeded is also a valid exit status for reuse."""
        from runners.main_loop import ExperimentLoop

        instance_id = "django__django-12345"
        baseline_dir = self._setup_baseline_dir(tmp_path, instance_id, "LimitsExceeded", resolved=False)

        mock_predict = Mock()
        mock_evaluate = Mock()
        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="test",
            skillbook_mode="per_repo",
        )

        instance = {"instance_id": instance_id, "repo": "django/django"}
        result = loop._run_train_instance_reusing_baseline(instance, baseline_dir)

        mock_learn.run.assert_called_once()
        mock_predict.run.assert_not_called()

    def test_invalid_exit_status_falls_back(self, tmp_path):
        """When exit_status is not Submitted/LimitsExceeded, run full pipeline."""
        from runners.main_loop import ExperimentLoop

        instance_id = "django__django-12345"
        baseline_dir = self._setup_baseline_dir(tmp_path, instance_id, "UnknownError", resolved=False)

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id=instance_id, exit_status="submitted", patch="p", trajectory=[],
        )
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id=instance_id, resolved=False, feedback="Bad",
        )
        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="test",
            skillbook_mode="per_repo",
        )

        instance = {"instance_id": instance_id, "repo": "django/django"}
        result = loop._run_train_instance_reusing_baseline(instance, baseline_dir)

        # Falls back to full predict→eval→learn
        mock_predict.run.assert_called_once()
        mock_evaluate.run.assert_called_once()
        mock_learn.run.assert_called_once()

    def test_missing_artifacts_falls_back(self, tmp_path):
        """When baseline is missing traj or result, run full pipeline."""
        from runners.main_loop import ExperimentLoop

        instance_id = "django__django-12345"
        baseline_dir = self._setup_baseline_dir(
            tmp_path, instance_id, has_traj=False, has_result=True
        )

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id=instance_id, exit_status="submitted", patch="p", trajectory=[],
        )
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id=instance_id, resolved=True, feedback="OK",
        )
        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="test",
            skillbook_mode="per_repo",
        )

        instance = {"instance_id": instance_id, "repo": "django/django"}
        result = loop._run_train_instance_reusing_baseline(instance, baseline_dir)

        mock_predict.run.assert_called_once()

    def test_two_phase_with_baseline_reuse_stats(self, tmp_path):
        """Full two-phase run with baseline reuse produces correct statistics."""
        from runners.main_loop import ExperimentLoop
        import json

        # Setup baseline dir with one train instance that has valid data
        baseline_dir = tmp_path / "baseline"
        benchmark = "princeton-nlp__SWE-bench_Lite"
        train_id = "django__django-11111"
        traj_dir = baseline_dir / benchmark / "trajectories" / train_id
        traj_dir.mkdir(parents=True)
        (traj_dir / "iter_0.json").write_text(json.dumps({
            "info": {"exit_status": "Submitted", "submission": "patch"},
            "messages": [{"role": "user", "content": "fix"}],
        }))
        result_dir = baseline_dir / benchmark / "results" / train_id
        result_dir.mkdir(parents=True)
        (result_dir / "iter_0.json").write_text(json.dumps({
            "resolved": True, "feedback": "OK",
        }))

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="django__django-22222", exit_status="submitted",
            patch="p", trajectory=[],
        )
        mock_evaluate = Mock()
        mock_evaluate.run.side_effect = [
            Mock(instance_id="django__django-22222", resolved=False, feedback="Bad"),
            # val baseline
            Mock(instance_id="django__django-33333", resolved=False, feedback="Bad"),
            # val skillbook
            Mock(instance_id="django__django-33333", resolved=True, feedback="OK"),
        ]
        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="test",
            skillbook_mode="per_repo",
        )

        train = [
            {"instance_id": train_id, "repo": "django/django"},          # reused
            {"instance_id": "django__django-22222", "repo": "django/django"},  # fresh
        ]
        val = [{"instance_id": "django__django-33333", "repo": "django/django"}]

        stats = loop.run(train, val_instances=val, baseline_run_dir=baseline_dir)

        assert stats["train_phase"]["reused_from_baseline"] == 1
        assert stats["train_phase"]["freshly_run"] == 1
        assert stats["train_phase"]["total_instances"] == 2

    def test_skip_learn_no_learning(self, tmp_path):
        """When skip_learn=True, Learn phase never runs even on failure."""
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="test__repo-123", exit_status="submitted",
            patch="patch", trajectory=[],
        )
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id="test__repo-123", resolved=False, feedback="Bad",
        )
        mock_learn = Mock()

        loop = ExperimentLoop(
            predict_phase=mock_predict, evaluate_phase=mock_evaluate,
            learn_phase=mock_learn, output_dir=tmp_path, run_name="test",
            max_attempts=3, skip_learn=True,
        )
        result = loop.run_instance({"instance_id": "test__repo-123", "repo": "test/repo"})

        assert mock_predict.run.call_count == 3  # All 3 attempts
        mock_learn.run.assert_not_called()       # Never learns
        assert result.final_resolved is False

    def test_skip_learn_breaks_on_resolve(self, tmp_path):
        """When skip_learn=True, stops on resolve even with max_attempts > 1."""
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="test__repo-123", exit_status="submitted",
            patch="patch", trajectory=[],
        )
        mock_evaluate = Mock()
        mock_evaluate.run.side_effect = [
            Mock(instance_id="test__repo-123", resolved=False, feedback="Bad"),
            Mock(instance_id="test__repo-123", resolved=True, feedback="Good"),
        ]
        mock_learn = Mock()

        loop = ExperimentLoop(
            predict_phase=mock_predict, evaluate_phase=mock_evaluate,
            learn_phase=mock_learn, output_dir=tmp_path, run_name="test",
            max_attempts=4, skip_learn=True,
        )
        result = loop.run_instance({"instance_id": "test__repo-123", "repo": "test/repo"})

        assert mock_predict.run.call_count == 2  # Stopped after resolve
        assert result.final_resolved is True
        mock_learn.run.assert_not_called()

    def test_skip_learn_default_still_learns(self, tmp_path):
        """When skip_learn=False (default), Learn runs on unresolved with force_learn=True."""
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="test__repo-123", exit_status="submitted",
            patch="patch", trajectory=[],
        )
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id="test__repo-123", resolved=False, feedback="Bad",
        )
        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict, evaluate_phase=mock_evaluate,
            learn_phase=mock_learn, output_dir=tmp_path, run_name="test",
            max_attempts=1, force_learn=True,  # Default force_learn in production
        )
        result = loop.run_instance({"instance_id": "test__repo-123", "repo": "test/repo"})

        assert mock_predict.run.call_count == 1
        mock_learn.run.assert_called_once()  # Learned from failure
        assert result.final_resolved is False


class TestPredictPhaseInjection:
    """run_instance(predict_phase=X) must use X for retrieval + predict, not self.predict."""

    def test_injected_predict_phase_is_used(self, tmp_path):
        from ace import Skillbook
        from runners.main_loop import ExperimentLoop

        injected = Mock()
        injected.prepare_skillbook.return_value = (Skillbook(), None)
        injected.run.return_value = Mock(
            instance_id="repo__i-1", exit_status="submitted",
            patch="p", trajectory=[],
        )

        own_predict = Mock()  # must NOT be called when predict_phase is injected
        evaluate = Mock()
        evaluate.run.return_value = Mock(
            instance_id="repo__i-1", resolved=True, feedback="", metrics={},
        )

        loop = ExperimentLoop(
            predict_phase=own_predict, evaluate_phase=evaluate, learn_phase=Mock(),
            output_dir=tmp_path, run_name="t", max_attempts=1,
        )
        instance = {"instance_id": "repo__i-1", "problem_statement": "Fix"}
        loop.run_instance(
            instance, frozen_skillbook=True, phase="val",
            predict_phase=injected,
        )

        injected.prepare_skillbook.assert_called_once()
        injected.run.assert_called_once()
        own_predict.run.assert_not_called()
        own_predict.prepare_skillbook.assert_not_called()
