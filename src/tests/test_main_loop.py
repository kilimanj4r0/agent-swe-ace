# src/tests/test_main_loop.py
"""Tests for main loop runner."""
import sys
from pathlib import Path
from unittest.mock import Mock

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

    def test_infrastructure_error_skips_evaluate_and_learn(self, tmp_path):
        """A broken runner stops only this instance before evaluation."""
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="test__repo-123",
            exit_status="error",
            error="docker exit 125",
            error_kind="infrastructure",
            patch="",
            trajectory=[],
        )
        mock_evaluate = Mock()
        mock_learn = Mock()
        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            max_attempts=3,
        )

        result = loop.run_instance(
            {"instance_id": "test__repo-123", "problem_statement": "Fix"}
        )

        assert result.status == "infrastructure_error"
        assert result.infrastructure_error == "docker exit 125"
        assert result.total_attempts == 1
        assert len(result.iterations) == 1
        assert result.iterations[0].evaluate_result is None
        mock_predict.run.assert_called_once()
        mock_evaluate.run.assert_not_called()
        mock_learn.run.assert_not_called()

    def test_concurrent_infrastructure_error_skips_evaluate_and_learn(self, tmp_path):
        """Concurrent workers apply the same infrastructure-error boundary."""
        from runners.main_loop import ExperimentLoop

        worker_predict = Mock()
        worker_predict.run.return_value = Mock(
            instance_id="test__repo-123",
            exit_status="error",
            error="docker daemon unavailable",
            error_kind="infrastructure",
            patch="",
            trajectory=[],
        )
        mock_evaluate = Mock()
        mock_learn = Mock()
        loop = ExperimentLoop(
            predict_phase=Mock(),
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            max_attempts=3,
            concurrency=2,
            agent_factory=Mock(),
        )
        loop._make_worker_predict = Mock(return_value=worker_predict)

        result = loop._run_instance_concurrent_inner(
            {"instance_id": "test__repo-123", "repo": "test/repo"}
        )

        assert result.status == "infrastructure_error"
        assert result.infrastructure_error == "docker daemon unavailable"
        assert result.total_attempts == 1
        assert len(result.iterations) == 1
        assert result.iterations[0].evaluate_result is None
        worker_predict.run.assert_called_once()
        mock_evaluate.run.assert_not_called()
        mock_learn.run.assert_not_called()

    def test_concurrent_partial_resume_copies_flat_artifacts(self, tmp_path):
        """Concurrent resume must not reference a nonexistent phase variable."""
        from data_io.resume_scanner import ResumePoint
        from runners.main_loop import ExperimentLoop

        worker_predict = Mock()
        worker_predict.run.return_value = Mock(
            exit_status="submitted",
            error_kind=None,
            patch="patch",
            trajectory=[],
        )
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            resolved=True,
            feedback="ok",
        )
        resume_state = {
            "test__repo-123": ResumePoint(
                resume_dir=tmp_path,
                last_complete_iter=0,
                is_fully_complete=False,
            )
        }
        loop = ExperimentLoop(
            predict_phase=Mock(),
            evaluate_phase=mock_evaluate,
            learn_phase=Mock(),
            output_dir=tmp_path,
            max_attempts=2,
            concurrency=2,
            agent_factory=Mock(),
            resume_state=resume_state,
        )
        loop._make_worker_predict = Mock(return_value=worker_predict)
        loop._copy_resume_artifacts = Mock()

        result = loop._run_instance_concurrent_inner(
            {"instance_id": "test__repo-123", "repo": "test/repo"}
        )

        loop._copy_resume_artifacts.assert_called_once_with("test__repo-123")
        assert result.final_resolved is True

    def test_statistics_keep_infrastructure_errors_disjoint(self, tmp_path):
        """Infrastructure failures are reported separately from task outcomes."""
        from runners.main_loop import ExperimentLoop, InstanceResult

        outcomes = {
            "resolved": InstanceResult(
                instance_id="resolved", final_resolved=True, total_attempts=1
            ),
            "unresolved": InstanceResult(
                instance_id="unresolved", final_resolved=False, total_attempts=1
            ),
            "infra": InstanceResult(
                instance_id="infra",
                status="infrastructure_error",
                infrastructure_error="docker unavailable",
                total_attempts=1,
            ),
        }
        mock_predict = Mock()
        mock_predict.get_retrieval_summary.return_value = None
        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=Mock(),
            learn_phase=Mock(),
            output_dir=tmp_path,
            max_attempts=1,
        )
        loop.run_instance = Mock(
            side_effect=lambda instance, **kwargs: outcomes[instance["instance_id"]]
        )

        statistics = loop.run(
            [{"instance_id": instance_id} for instance_id in outcomes]
        )

        assert statistics["resolved_ids"] == ["resolved"]
        assert statistics["unresolved_ids"] == ["unresolved"]
        assert statistics["infrastructure_error_ids"] == ["infra"]
        assert statistics["infrastructure_error_count"] == 1
        assert statistics["status"] == "degraded"
        assert set(statistics["resolved_ids"]).isdisjoint(
            statistics["infrastructure_error_ids"]
        )
        assert set(statistics["unresolved_ids"]).isdisjoint(
            statistics["infrastructure_error_ids"]
        )


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


class TestTrainSequentialInTwoPhase:
    """Two-phase train must run sequentially even with concurrency > 1; the old
    guard that raised ValueError for two_phase + concurrency>1 + no baseline is gone."""

    def test_no_guard_error_without_baseline(self, tmp_path):
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="x", exit_status="submitted", patch="p", trajectory=[],
        )
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id="x", resolved=True, feedback="", metrics={},
        )
        loop = ExperimentLoop(
            predict_phase=mock_predict, evaluate_phase=mock_evaluate, learn_phase=Mock(),
            output_dir=tmp_path, run_name="t", max_attempts=1,
            skillbook_mode="global", concurrency=4, agent_factory=lambda: Mock(),
        )
        train = [{"instance_id": "r__t-0", "problem_statement": "x", "repo": "r"}]
        val = [{"instance_id": "r__v-0", "problem_statement": "x", "repo": "r"}]

        # Must NOT raise (old behavior: ValueError without baseline_run_dir)
        stats = loop.run(train, val_instances=val)
        assert "train_phase" in stats

    def test_train_runs_one_at_a_time(self, tmp_path):
        import threading
        import time
        from ace import Skillbook
        from runners.main_loop import ExperimentLoop

        active = [0]
        max_active = [0]
        lock = threading.Lock()

        def predict_run(**kw):
            with lock:
                active[0] += 1
                max_active[0] = max(max_active[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return Mock(instance_id="x", exit_status="submitted",
                        patch="p", trajectory=[])

        mock_predict = Mock()
        mock_predict.run.side_effect = predict_run
        mock_predict.prepare_skillbook.return_value = (Skillbook(), None)
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id="x", resolved=False, feedback="", metrics={},
        )
        loop = ExperimentLoop(
            predict_phase=mock_predict, evaluate_phase=mock_evaluate, learn_phase=Mock(),
            output_dir=tmp_path, run_name="t", max_attempts=1,
            skillbook_mode="global", concurrency=4, agent_factory=lambda: Mock(),
        )
        # 4 train instances (would overlap if train were concurrent); 1 val instance
        # (val stays sequential here: concurrent val is added in Task 4, and even then
        # a single val instance runs inline).
        train = [{"instance_id": f"r__t-{i}", "problem_statement": "x", "repo": "r"}
                 for i in range(4)]
        val = [{"instance_id": "r__v-0", "problem_statement": "x", "repo": "r"}]
        loop.run(train, val_instances=val)

        assert max_active[0] == 1, (
            f"two-phase train must be sequential, max in-flight={max_active[0]}"
        )


class TestWorkerPredictFactory:
    """_make_worker_predict builds a fresh PredictPhase per call from agent_factory."""

    def test_distinct_predict_per_call(self, tmp_path):
        from phases.predict import PredictPhase
        from runners.main_loop import ExperimentLoop

        created = []

        def factory():
            a = Mock()
            created.append(a)
            return a

        base = PredictPhase(
            agent=Mock(), output_dir=tmp_path, run_name="t", benchmark="b",
        )
        loop = ExperimentLoop(
            predict_phase=base, evaluate_phase=Mock(), learn_phase=Mock(),
            output_dir=tmp_path, run_name="t", max_attempts=1, agent_factory=factory,
        )
        w1 = loop._make_worker_predict()
        w2 = loop._make_worker_predict()

        assert isinstance(w1, PredictPhase) and isinstance(w2, PredictPhase)
        assert w1 is not w2
        assert w1.agent is created[0]
        assert w2.agent is created[1]
        assert created[0] is not created[1]


class TestConcurrentValPass:
    """_run_val_pass with concurrency > 1 must match sequential results, handle
    worker exceptions, and actually execute instances in parallel."""

    def _make_loop(self, tmp_path):
        from runners.main_loop import ExperimentLoop
        loop = ExperimentLoop(
            predict_phase=Mock(), evaluate_phase=Mock(), learn_phase=Mock(),
            output_dir=tmp_path, run_name="t", max_attempts=1,
            agent_factory=lambda: Mock(),
        )
        # Avoid real PredictPhase construction in the concurrency mechanics tests.
        loop._make_worker_predict = lambda: Mock()
        return loop

    @staticmethod
    def _instances(n):
        return [{"instance_id": f"r__i-{i}", "problem_statement": "x"} for i in range(n)]

    def test_concurrent_matches_sequential(self, tmp_path):
        from ace import Skillbook
        from runners.main_loop import InstanceResult

        def fake_run_instance(inst, **kw):
            iid = inst["instance_id"]
            idx = int(iid.split("-")[-1])
            return InstanceResult(
                instance_id=iid, final_resolved=(idx % 2 == 0), total_attempts=1,
            )

        for conc in (1, 4):
            loop = self._make_loop(tmp_path)
            loop.concurrency = conc
            loop.run_instance = Mock(side_effect=fake_run_instance)
            stats = loop._run_val_pass(
                [dict(i) for i in self._instances(6)],
                Skillbook(), phase="val", max_attempts=1,
            )
            assert stats["resolved_count"] == 3, f"concurrency={conc}"
            assert stats["unresolved_count"] == 3, f"concurrency={conc}"
            assert set(stats["resolved_ids"]) == {f"r__i-{i}" for i in range(6) if i % 2 == 0}
            assert loop.run_instance.call_count == 6

    def test_worker_exception_recorded_as_infrastructure_error(self, tmp_path):
        from ace import Skillbook
        from runners.main_loop import InstanceResult

        def fake_run_instance(inst, **kw):
            if inst["instance_id"] == "r__i-1":
                raise RuntimeError("boom")
            return InstanceResult(
                instance_id=inst["instance_id"], final_resolved=True, total_attempts=1,
            )

        loop = self._make_loop(tmp_path)
        loop.concurrency = 3
        loop.run_instance = Mock(side_effect=fake_run_instance)
        stats = loop._run_val_pass(
            [dict(i) for i in self._instances(4)],
            Skillbook(), phase="val", max_attempts=1,
        )
        assert set(stats["resolved_ids"]) == {"r__i-0", "r__i-2", "r__i-3"}
        assert stats["unresolved_ids"] == []
        assert stats["infrastructure_error_ids"] == ["r__i-1"]
        assert stats["infrastructure_error_count"] == 1
        assert stats["status"] == "degraded"
        assert stats["resolved_count"] == 3

    def test_instances_actually_run_in_parallel(self, tmp_path):
        import threading
        import time
        from ace import Skillbook
        from runners.main_loop import InstanceResult

        active = [0]
        max_active = [0]
        lock = threading.Lock()

        def fake_run_instance(inst, **kw):
            with lock:
                active[0] += 1
                max_active[0] = max(max_active[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return InstanceResult(
                instance_id=inst["instance_id"], final_resolved=True, total_attempts=1,
            )

        loop = self._make_loop(tmp_path)
        loop.concurrency = 4
        loop.run_instance = Mock(side_effect=fake_run_instance)
        loop._run_val_pass(
            [dict(i) for i in self._instances(4)],
            Skillbook(), phase="val", max_attempts=1,
        )
        assert max_active[0] >= 2, (
            f"expected concurrent val execution, max in-flight={max_active[0]}"
        )

    def test_learning_loop_skipped_when_preloaded(self, tmp_path):
        """With preloaded_skillbook set, the train learning loop must NOT run,
        even when train instances are non-empty. Unblocks eval_on_train +
        skillbook_source_dir, and preserves validation-only semantics."""
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="x", exit_status="submitted", patch="p", trajectory=[]
        )
        mock_predict.prepare_skillbook.side_effect = (
            lambda instance, skillbook, phase=None: (skillbook, None)
        )
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(instance_id="x", resolved=True, feedback="ok")
        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        from ace import Skillbook
        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="t",
            skillbook_mode="global",
        )

        train = [{"instance_id": "t1", "repo": "django/django"}]
        val = [{"instance_id": "v1", "repo": "django/django"}]
        loop.run(train, val_instances=val, preloaded_skillbook=Skillbook())

        # The learning loop must be skipped entirely -> learn never runs.
        assert not mock_learn.run.called

    def test_eval_on_train_runs_two_train_passes_and_skips_val(self, tmp_path):
        """eval_on_train=True: _run_val_pass called twice on TRAIN instances
        (train_eval_baseline empty, train_eval learned) and NOT on val."""
        from runners.main_loop import ExperimentLoop
        from ace import Skillbook

        mock_predict = Mock()
        mock_predict.prepare_skillbook.side_effect = (
            lambda instance, skillbook, phase=None: (skillbook, None)
        )
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(instance_id="x", resolved=True, feedback="ok")
        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict,
            evaluate_phase=mock_evaluate,
            learn_phase=mock_learn,
            output_dir=tmp_path,
            run_name="t",
            skillbook_mode="global",
        )

        empty_stats = {
            "total_instances": 1, "resolved_count": 0, "unresolved_count": 1,
            "resolution_rate": 0.0, "resolved_ids": [], "unresolved_ids": [],
            "skillbook_skills": 0,
        }
        loop._run_val_pass = Mock(return_value=dict(empty_stats))

        train = [{"instance_id": "t1", "repo": "django/django"}]
        val = [{"instance_id": "v1", "repo": "django/django"}]
        loop.run(train, val_instances=val, eval_on_train=True, eval_on_train_pass_k=3)

        assert loop._run_val_pass.call_count == 2
        calls = loop._run_val_pass.call_args_list
        phases = [c.kwargs["phase"] for c in calls]
        assert phases == ["train_eval_baseline", "train_eval"]
        # Both passes run on the TRAIN instances, not val.
        assert [i["instance_id"] for i in calls[0].kwargs["val_instances"]] == ["t1"]
        assert [i["instance_id"] for i in calls[1].kwargs["val_instances"]] == ["t1"]
        # TrainBL uses an empty book; TrainSB uses the final learned book. Learning
        # is mocked (it does not mutate global_skillbook), so assert identity with
        # loop.global_skillbook rather than a non-zero skill count.
        assert len(calls[0].kwargs["skillbook"].skills()) == 0
        assert calls[1].kwargs["skillbook"] is loop.global_skillbook
        # TrainBL runs a SINGLE attempt (pass@1 reference); TrainSB runs pass_k.
        assert calls[0].kwargs["max_attempts"] == 1
        assert calls[1].kwargs["max_attempts"] == 3

    def test_eval_on_train_statistics_blocks(self, tmp_path):
        """Returned stats carry train_eval_phase + train_eval_baseline_phase and
        a train summary; val_*_phase absent; top-level mirrors TrainSB."""
        from runners.main_loop import ExperimentLoop
        from ace import Skillbook

        mock_predict = Mock()
        mock_predict.prepare_skillbook.side_effect = (
            lambda instance, skillbook, phase=None: (skillbook, None)
        )
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(instance_id="x", resolved=True, feedback="ok")
        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict, evaluate_phase=mock_evaluate,
            learn_phase=mock_learn, output_dir=tmp_path, run_name="t",
            skillbook_mode="global",
        )

        def fake_pass(val_instances, skillbook, phase, max_attempts=1, **kw):
            is_te = phase == "train_eval"
            total = 1
            resolved = 1 if is_te else 0
            base = {
                "total_instances": total,
                "resolved_count": resolved,
                "unresolved_count": total - resolved,
                "resolution_rate": resolved / total,  # cumulative pass@k
                "resolved_ids": ["t1"] if is_te else [],
                "unresolved_ids": [] if is_te else ["t1"],
                "skillbook_skills": len(skillbook.skills()),
                "max_attempts": max_attempts,
            }
            # TrainSB (max_attempts>1) carries per-attempt + cumulative pass@k so the
            # summary can compute pass@1 / pass@k / avg@k. TrainBL (max_attempts=1) does not.
            if max_attempts > 1:
                per = {}
                for i in range(max_attempts):
                    r = 1.0 if (is_te and i == 0) else 0.0
                    per[f"iter_{i}"] = {"resolved": int(r), "total": total, "rate": r}
                base["per_attempt_rate"] = per
                pak = {}
                for n in range(1, max_attempts + 1):
                    # TE resolves at iter0 -> cumulative pass@n = 1.0 for all n>=1.
                    rate = 1.0 if is_te else 0.0
                    pak[f"pass@{n}"] = {"count": int(rate), "total": total, "rate": rate}
                base["pass_at_k"] = pak
            return base

        loop._run_val_pass = Mock(side_effect=fake_pass)

        train = [{"instance_id": "t1", "repo": "django/django"}]
        val = [{"instance_id": "v1", "repo": "django/django"}]
        stats = loop.run(
            train, val_instances=val, eval_on_train=True, eval_on_train_pass_k=3,
        )

        assert "train_eval_phase" in stats
        assert "train_eval_baseline_phase" in stats
        assert "val_skillbook_phase" not in stats
        assert "val_baseline_phase" not in stats
        # Top-level mirrors TrainSB (pass@k ceiling).
        assert stats["resolved_count"] == 1
        assert stats["total_instances"] == 1
        # Summary reports pass@1 / pass@k / avg@k for TrainSB and pass@1 for TrainBL,
        # with the skillbook effect at pass@1 (clean) and avg@k.
        s = stats["summary"]
        assert s["train_eval_pass1_rate"] == 1.0
        assert s["train_eval_resolution_rate"] == 1.0      # pass@k (cumulative ceiling)
        assert round(s["train_eval_avg_rate"], 3) == 0.333  # mean(iter0=1, iter1=0, iter2=0)
        assert s["train_eval_baseline_resolution_rate"] == 0.0  # TrainBL pass@1
        assert s["train_skillbook_improvement"] == "+1.000"     # Δ@pass@1
        assert s["train_skillbook_improvement_avg"] == "+0.333"  # Δ@avg@k

    def test_train_eval_baseline_reuses_baseline_train_results(self, tmp_path):
        """TrainBL reuses empty-skillbook results from baseline_dir/results/train
        when present, instead of re-executing."""
        from runners.main_loop import ExperimentLoop
        from ace import Skillbook

        bench = "princeton-nlp__SWE-bench_Lite"
        baseline_dir = tmp_path / "baseline"
        # Baseline has an empty-skillbook result for t1 under results/train.
        traj_dir = baseline_dir / bench / "trajectories" / "train" / "t1"
        res_dir = baseline_dir / bench / "results" / "train" / "t1"
        traj_dir.mkdir(parents=True)
        res_dir.mkdir(parents=True)
        (traj_dir / "iter_0.json").write_text(
            '{"info": {"exit_status": "Submitted"}, "messages": []}'
        )
        (res_dir / "iter_0.json").write_text('{"resolved": true}')

        loop = ExperimentLoop(
            predict_phase=Mock(), evaluate_phase=Mock(), learn_phase=Mock(),
            output_dir=tmp_path, run_name="t", benchmark=bench,
            agent_factory=lambda: Mock(),
        )
        loop._make_worker_predict = lambda: Mock()

        instances = [{"instance_id": "t1", "problem_statement": "x"}]
        stats = loop._run_val_pass(
            val_instances=instances, skillbook=Skillbook(),
            phase="train_eval_baseline",
            baseline_run_dir=baseline_dir, max_attempts=1,
        )

        # Reused -> t1 counted as resolved without running the agent.
        assert stats["resolved_count"] == 1
        assert "t1" in stats["resolved_ids"]


class TestTrainEmptySkillbook:
    """Two-phase train must SOLVE with an empty skillbook (distillation) while
    learn still accumulates into the real book. Val phases keep using the real book."""

    def test_train_predict_receives_empty_skillbook(self, tmp_path):
        """run_instance(phase='train') hands predict an empty Skillbook even when
        the accumulated global book is non-empty; learn receives the accumulated one."""
        from ace import Skillbook, Skill
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="django__django-1", exit_status="submitted",
            patch="p", trajectory=[],
        )
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id="django__django-1", resolved=False, feedback="Bad",
        )
        mock_learn = Mock()
        mock_learn.run.return_value = Mock(skills_added=1)

        loop = ExperimentLoop(
            predict_phase=mock_predict, evaluate_phase=mock_evaluate,
            learn_phase=mock_learn, output_dir=tmp_path, run_name="t",
            max_attempts=1, skillbook_mode="global",
        )
        # Seed the accumulated global book so we can prove predict does NOT see it.
        loop.global_skillbook._skills["seed"] = Skill(
            id="seed", section="debugging", content="should not reach predict",
        )

        instance = {"instance_id": "django__django-1", "repo": "django/django"}
        loop.run_instance(instance, force_learn=True, phase="train")

        # Predict got an EMPTY book (0 skills), not the seeded accumulated book.
        pred_sb = mock_predict.run.call_args.kwargs["skillbook"]
        assert isinstance(pred_sb, Skillbook)
        assert len(pred_sb.skills()) == 0

        # Learn got the accumulated book (the seeded skill is there).
        learn_sb = mock_learn.run.call_args.kwargs["skillbook"]
        assert len(learn_sb.skills()) == 1
        assert learn_sb is loop.global_skillbook

    def test_train_accumulates_across_instances_with_empty_predict(self, tmp_path):
        """Two sequential train instances: each predict gets an empty book, but
        learn still accumulates into the real book (instance 2's learn sees
        instance 1's skill)."""
        from ace import Skillbook, Skill
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.run.return_value = Mock(
            instance_id="x", exit_status="submitted", patch="p", trajectory=[],
        )
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id="x", resolved=False, feedback="Bad",
        )

        # learn.run actually adds a skill to the passed (accumulated) book.
        def learn_run(**kw):
            iid = kw["instance"]["instance_id"]
            kw["skillbook"]._skills[f"lesson-{iid}"] = Skill(
                id=f"lesson-{iid}", section="debugging", content="x"
            )
            return Mock(skills_added=1)

        mock_learn = Mock()
        mock_learn.run.side_effect = learn_run

        loop = ExperimentLoop(
            predict_phase=mock_predict, evaluate_phase=mock_evaluate,
            learn_phase=mock_learn, output_dir=tmp_path, run_name="t",
            max_attempts=1, skillbook_mode="global",
        )

        instances = [
            {"instance_id": "django__django-1", "repo": "django/django"},
            {"instance_id": "django__django-2", "repo": "django/django"},
        ]
        for inst in instances:
            loop.run_instance(inst, force_learn=True, phase="train")

        # Accumulation: the global book now holds both lessons.
        assert len(loop.global_skillbook.skills()) == 2

        # But every predict call still received an EMPTY book.
        assert mock_predict.run.call_count == 2
        for call in mock_predict.run.call_args_list:
            assert len(call.kwargs["skillbook"].skills()) == 0

    def test_val_predict_receives_real_skillbook(self, tmp_path):
        """val (skillbook) pass: predict gets the real book, not an empty one."""
        from ace import Skillbook, Skill
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.prepare_skillbook.side_effect = (
            lambda instance, skillbook, phase=None: (skillbook, None)
        )
        mock_predict.run.return_value = Mock(
            instance_id="v-1", exit_status="submitted", patch="p", trajectory=[],
        )
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id="v-1", resolved=False, feedback="Bad",
        )

        loop = ExperimentLoop(
            predict_phase=mock_predict, evaluate_phase=mock_evaluate,
            learn_phase=Mock(), output_dir=tmp_path, run_name="t",
            max_attempts=1,
        )
        real_book = Skillbook()
        real_book._skills["s1"] = Skill(id="s1", section="debugging", content="use me")

        instance = {"instance_id": "v-1", "repo": "django/django"}
        loop.run_instance(
            instance, initial_skillbook=real_book, frozen_skillbook=True,
            phase="val", max_attempts_override=1,
        )

        pred_sb = mock_predict.run.call_args.kwargs["skillbook"]
        assert len(pred_sb.skills()) == 1  # the real book, not empty

    def test_val_baseline_empty_and_global_book_not_mutated(self, tmp_path):
        """val_baseline: predict gets the empty book passed in (not global_skillbook),
        and global_skillbook is not mutated by frozen val passes."""
        from ace import Skillbook, Skill
        from runners.main_loop import ExperimentLoop

        mock_predict = Mock()
        mock_predict.prepare_skillbook.side_effect = (
            lambda instance, skillbook, phase=None: (skillbook, None)
        )
        mock_predict.run.return_value = Mock(
            instance_id="vb-1", exit_status="submitted", patch="p", trajectory=[],
        )
        mock_evaluate = Mock()
        mock_evaluate.run.return_value = Mock(
            instance_id="vb-1", resolved=False, feedback="Bad",
        )
        loop = ExperimentLoop(
            predict_phase=mock_predict, evaluate_phase=mock_evaluate,
            learn_phase=Mock(), output_dir=tmp_path, run_name="t",
            max_attempts=1,
        )
        # Seed global book; val_baseline must NOT see or mutate it.
        loop.global_skillbook._skills["seed"] = Skill(
            id="seed", section="debugging", content="x"
        )
        before = len(loop.global_skillbook.skills())

        instance = {"instance_id": "vb-1", "repo": "django/django"}
        loop.run_instance(
            instance, initial_skillbook=Skillbook(), frozen_skillbook=True,
            phase="val_baseline", max_attempts_override=1,
        )

        pred_sb = mock_predict.run.call_args.kwargs["skillbook"]
        assert len(pred_sb.skills()) == 0                 # empty baseline book
        assert len(loop.global_skillbook.skills()) == before  # global untouched
