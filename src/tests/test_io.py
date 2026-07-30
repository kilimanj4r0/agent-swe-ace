# src/tests/test_io.py
"""Tests for IO module."""
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_io import readers, writers


class TestExtractBenchmarkName:
    """Test benchmark name extraction."""

    def test_extract_from_hf_dataset(self):
        """Test extracting from HuggingFace dataset name."""
        assert readers.extract_benchmark_name("princeton-nlp/SWE-bench_Lite") == "princeton-nlp__SWE-bench_Lite"
        assert readers.extract_benchmark_name("princeton-nlp/SWE-bench") == "princeton-nlp__SWE-bench"

    def test_extract_from_simple_name(self):
        """Test extracting from simple name."""
        assert readers.extract_benchmark_name("SWE-bench_Lite") == "SWE-bench_Lite"

    # --- I9: additional extract_benchmark_name tests ---

    def test_extract_from_writers(self):
        """Test extract_benchmark_name from writers module."""
        assert writers.extract_benchmark_name("princeton-nlp/SWE-bench_Lite") == "princeton-nlp__SWE-bench_Lite"

    def test_both_modules_identical(self):
        """Readers and writers produce identical results for various inputs."""
        test_inputs = [
            "princeton-nlp/SWE-bench_Lite",
            "princeton-nlp/SWE-bench_Verified",
            "SWE-bench_Lite",
            "some-org/some-dataset",
            "no-slashes",
            "a/b/c",
        ]
        for inp in test_inputs:
            assert readers.extract_benchmark_name(inp) == writers.extract_benchmark_name(inp), (
                f"Mismatch for input: {inp}"
            )


class TestReaders:
    """Test data loading functions."""

    def test_load_instance_from_file(self, tmp_path):
        """Test loading a SWE-bench instance from JSON file."""
        instance_data = {
            "instance_id": "test__repo-123",
            "repo": "test/repo",
            "problem_statement": "Fix the bug",
            "base_commit": "abc123",
        }
        instance_file = tmp_path / "test__repo-123.json"
        instance_file.write_text(json.dumps(instance_data))

        instance = readers.load_instance(instance_file)

        assert instance["instance_id"] == "test__repo-123"
        assert instance["repo"] == "test/repo"

    def test_load_skillbook_empty(self, tmp_path):
        """Test loading an empty skillbook."""
        from ace import Skillbook

        skillbook = readers.load_skillbook(None)
        assert isinstance(skillbook, Skillbook)
        assert len(skillbook.skills()) == 0

    def test_load_skillbook_from_file(self, tmp_path):
        """Test loading a skillbook from JSON file."""
        skillbook_data = {
            "skills": {
                "skill-1": {
                    "id": "skill-1",
                    "section": "debugging",
                    "content": "Check imports first",
                }
            }
        }
        skillbook_file = tmp_path / "skillbook.json"
        skillbook_file.write_text(json.dumps(skillbook_data))

        skillbook = readers.load_skillbook(skillbook_file)

        assert len(skillbook.skills()) == 1
        assert skillbook.skills()[0].id == "skill-1"

    def test_load_trajectory(self, tmp_path):
        """Test loading a trajectory from JSON file."""
        traj_data = {
            "info": {"exit_status": "submitted"},
            "messages": [
                {"role": "user", "content": "Fix this"},
                {"role": "assistant", "content": "I'll fix it"},
            ],
        }
        traj_file = tmp_path / "trajectory.json"
        traj_file.write_text(json.dumps(traj_data))

        trajectory = readers.load_trajectory(traj_file)

        assert trajectory["info"]["exit_status"] == "submitted"
        assert len(trajectory["messages"]) == 2


class TestWriters:
    """Test data saving functions."""

    def test_get_run_dir(self, tmp_path):
        """Test run directory creation with timestamp."""
        run_dir = writers.get_run_dir(tmp_path, timestamp="20260319_143052")
        assert run_dir == tmp_path / "run_20260319_143052"

    def test_save_trajectory(self, tmp_path):
        """Test saving a trajectory with new structure."""
        trajectory = {
            "info": {"exit_status": "submitted", "submission": "patch content"},
            "messages": [{"role": "user", "content": "Fix this"}],
        }

        run_dir = tmp_path / "run_20260319_143052"

        output_path = writers.save_trajectory(
            trajectory=trajectory,
            run_dir=run_dir,
            benchmark="swebench-lite",
            instance_id="test__repo-123",
            iteration=0,
        )

        expected_path = run_dir / "swebench-lite" / "trajectories" / "test__repo-123" / "iter_0.json"
        assert output_path == expected_path
        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert loaded["info"]["exit_status"] == "submitted"

    def test_save_skillbook_per_instance(self, tmp_path):
        """Test saving a skillbook in per-instance mode."""
        from ace import Skill, Skillbook

        skillbook = Skillbook()
        skillbook._skills["skill-1"] = Skill(
            id="skill-1",
            section="debugging",
            content="Check imports first",
        )

        run_dir = tmp_path / "run_20260319_143052"

        output_path = writers.save_skillbook(
            skillbook=skillbook,
            run_dir=run_dir,
            benchmark="swebench-lite",
            iteration=1,
            instance_id="test__repo-123",
        )

        expected_path = run_dir / "swebench-lite" / "skillbooks" / "test__repo-123" / "iter_1.json"
        assert output_path == expected_path
        assert output_path.exists()

    def test_save_skillbook_per_run(self, tmp_path):
        """Test saving a skillbook in per-run mode."""
        from ace import Skill, Skillbook

        skillbook = Skillbook()
        skillbook._skills["skill-1"] = Skill(
            id="skill-1",
            section="debugging",
            content="Check imports first",
        )

        run_dir = tmp_path / "run_20260319_143052"

        output_path = writers.save_skillbook(
            skillbook=skillbook,
            run_dir=run_dir,
            benchmark="swebench-lite",
            iteration=1,
            instance_id=None,  # Per-run mode
        )

        expected_path = run_dir / "swebench-lite" / "skillbooks" / "iter_1.json"
        assert output_path == expected_path
        assert output_path.exists()

    def test_skill_to_dict_includes_sources(self):
        """skill_to_dict must serialize provenance `sources`.

        Regression: the per-repo final_skillbook.json previously used an inline
        copy of this serialization that dropped `sources`, so per-repo skills
        lost their instance/repo provenance on disk while global did not.
        skill_to_dict is now the single source shared by all skillbook files.
        """
        from ace import Skill

        skill = Skill(
            id="verification-00001",
            section="verification",
            content="AVOID: claiming success without verifying git diff",
            justification="empty patches submitted as complete",
            evidence="git diff --cached was empty",
            sources=[{"instance_id": "django__django-14376", "repo": "django/django"}],
        )

        d = writers.skill_to_dict(skill)
        assert d["id"] == "verification-00001"
        assert d["section"] == "verification"
        assert d["sources"] == [
            {"instance_id": "django__django-14376", "repo": "django/django"}
        ]
        # Canonical field set — drift guard across global/per_repo/per_instance.
        assert set(d.keys()) == {
            "id",
            "section",
            "content",
            "justification",
            "evidence",
            "sources",
        }

    def test_save_skillbook_writes_sources(self, tmp_path):
        """Saved skillbook JSON must carry `sources` for every skill on disk."""
        from ace import Skill, Skillbook

        skillbook = Skillbook()
        skillbook._skills["skill-1"] = Skill(
            id="skill-1",
            section="debugging",
            content="Check imports first",
            sources=[{"instance_id": "test__repo-123", "repo": "test/repo"}],
        )

        run_dir = tmp_path / "run_20260319_143052"
        output_path = writers.save_skillbook(
            skillbook=skillbook,
            run_dir=run_dir,
            benchmark="swebench-lite",
            iteration=1,
            instance_id=None,
        )

        data = json.loads(output_path.read_text())
        assert data["skills"]["skill-1"]["sources"] == [
            {"instance_id": "test__repo-123", "repo": "test/repo"}
        ]

    def test_save_result(self, tmp_path):
        """Test saving an evaluation result."""
        result = {
            "resolved": False,
            "patch_length": 500,
            "feedback": "Tests failed",
        }

        run_dir = tmp_path / "run_20260319_143052"

        output_path = writers.save_result(
            result=result,
            run_dir=run_dir,
            benchmark="swebench-lite",
            instance_id="test__repo-123",
            iteration=0,
        )

        expected_path = run_dir / "swebench-lite" / "results" / "test__repo-123" / "iter_0.json"
        assert output_path == expected_path
        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert loaded["resolved"] is False
        assert loaded["instance_id"] == "test__repo-123"
        assert loaded["iteration"] == 0

    def test_save_config(self, tmp_path):
        """Test saving run config."""
        config = {"experiment": {"name": "test"}, "benchmark": {"dataset": "swebench-lite"}}

        run_dir = tmp_path / "run_20260319_143052"
        run_dir.mkdir(parents=True, exist_ok=True)

        output_path = writers.save_config(config=config, run_dir=run_dir)

        assert output_path == run_dir / "config.json"
        assert output_path.exists()

    def test_save_statistics(self, tmp_path):
        """Test saving run statistics."""
        statistics = {
            "run_name": "run_20260319_143052",
            "benchmark": "swebench-lite",
            "total_instances": 10,
            "resolved_count": 3,
            "unresolved_count": 7,
            "resolution_rate": 0.3,
            "resolved_ids": ["a", "b", "c"],
            "unresolved_ids": ["d", "e"],
        }

        run_dir = tmp_path / "run_20260319_143052"
        run_dir.mkdir(parents=True, exist_ok=True)

        output_path = writers.save_statistics(statistics=statistics, run_dir=run_dir)

        assert output_path == run_dir / "statistics.json"
        assert output_path.exists()


# --- I4: Additional test classes ---


class TestLoadSkillbookForRepo:
    """Tests for load_skillbook_for_repo."""

    def test_per_repo_found(self, tmp_path):
        """Per-repo skillbook file is loaded when it exists."""

        benchmark = "princeton-nlp__SWE-bench_Verified"
        repo = "django/django"
        repo_name = repo.replace("/", "__")

        skillbook_data = {
            "skills": {
                "skill-1": {
                    "id": "skill-1",
                    "section": "general",
                    "content": "Check migrations",
                }
            }
        }

        per_repo_dir = tmp_path / benchmark / "skillbooks" / "per_repo" / repo_name
        per_repo_dir.mkdir(parents=True)
        (per_repo_dir / "final_skillbook.json").write_text(json.dumps(skillbook_data))

        result = readers.load_skillbook_for_repo(tmp_path, benchmark, repo)

        assert len(result.skills()) == 1
        assert result.skills()[0].id == "skill-1"

    def test_fallback_to_global(self, tmp_path):
        """Falls back to global final_skillbook.json when per-repo not found."""
        benchmark = "princeton-nlp__SWE-bench_Verified"
        repo = "django/django"

        global_data = {
            "skills": {
                "skill-g": {
                    "id": "skill-g",
                    "section": "general",
                    "content": "Global skill",
                }
            }
        }

        skillbooks_dir = tmp_path / benchmark / "skillbooks"
        skillbooks_dir.mkdir(parents=True)
        (skillbooks_dir / "final_skillbook.json").write_text(json.dumps(global_data))

        result = readers.load_skillbook_for_repo(tmp_path, benchmark, repo)

        assert len(result.skills()) == 1
        assert result.skills()[0].id == "skill-g"

    def test_empty_when_nothing_found(self, tmp_path):
        """Returns empty Skillbook when no file exists."""
        from ace import Skillbook

        result = readers.load_skillbook_for_repo(
            tmp_path, "princeton-nlp__SWE-bench_Verified", "django/django"
        )

        assert isinstance(result, Skillbook)
        assert len(result.skills()) == 0


class TestLoadTeacherTrajectory:
    """Tests for load_teacher_trajectory."""

    def test_returns_none_when_not_found(self, tmp_path):
        """Returns None when trajectory file does not exist."""
        result = readers.load_teacher_trajectory(tmp_path, "django__django-12345")
        assert result is None

    def test_returns_dict_when_found(self, tmp_path):
        """Returns dict with info and messages when trajectory exists."""
        instance_id = "django__django-12345"
        traj_data = {
            "info": {"exit_status": "submitted", "model": "opus"},
            "messages": [
                {"role": "user", "content": "Fix the bug"},
                {"role": "assistant", "content": "Here is the fix"},
            ],
        }

        traj_dir = tmp_path / instance_id
        traj_dir.mkdir()
        (traj_dir / f"{instance_id}.traj.json").write_text(json.dumps(traj_data))

        result = readers.load_teacher_trajectory(tmp_path, instance_id)

        assert result is not None
        assert result["info"]["exit_status"] == "submitted"
        assert len(result["messages"]) == 2


class TestLoadResults:
    """Tests for load_results."""

    def test_empty_when_no_dir(self, tmp_path):
        """Returns empty dict when results directory does not exist."""
        result = readers.load_results(tmp_path, "princeton-nlp__SWE-bench_Lite")
        assert result == {}

    def test_loads_latest_iteration(self, tmp_path):
        """Returns the latest iteration file for each instance."""
        benchmark = "princeton-nlp__SWE-bench_Lite"
        instance_id = "django__django-12345"

        results_dir = tmp_path / benchmark / "results" / instance_id
        results_dir.mkdir(parents=True)

        # iter_0: unresolved
        (results_dir / "iter_0.json").write_text(json.dumps({
            "resolved": False,
            "feedback": "Tests failed",
        }))
        # iter_1: resolved
        (results_dir / "iter_1.json").write_text(json.dumps({
            "resolved": True,
            "feedback": "All tests passed",
        }))

        result = readers.load_results(tmp_path, benchmark)

        assert instance_id in result
        assert result[instance_id]["resolved"] is True
        assert result[instance_id]["feedback"] == "All tests passed"

    def test_multiple_instances(self, tmp_path):
        """Loads results for multiple instances."""
        benchmark = "princeton-nlp__SWE-bench_Lite"

        for iid in ["django__django-111", "django__django-222"]:
            results_dir = tmp_path / benchmark / "results" / iid
            results_dir.mkdir(parents=True)
            (results_dir / "iter_0.json").write_text(json.dumps({
                "resolved": False,
                "instance_id": iid,
            }))

        result = readers.load_results(tmp_path, benchmark)

        assert len(result) == 2
        assert "django__django-111" in result
        assert "django__django-222" in result


class TestLoadStatistics:
    """Tests for load_statistics."""

    def test_returns_none_when_missing(self, tmp_path):
        """Returns None when statistics.json does not exist."""
        result = readers.load_statistics(tmp_path)
        assert result is None

    def test_returns_parsed_json(self, tmp_path):
        """Returns parsed JSON dict when statistics.json exists."""
        stats = {
            "run_name": "run_20260319_143052",
            "total_instances": 10,
            "resolved_count": 3,
        }
        (tmp_path / "statistics.json").write_text(json.dumps(stats))

        result = readers.load_statistics(tmp_path)

        assert result is not None
        assert result["run_name"] == "run_20260319_143052"
        assert result["total_instances"] == 10
        assert result["resolved_count"] == 3
