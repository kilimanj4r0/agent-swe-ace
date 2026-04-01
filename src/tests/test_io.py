# src/tests/test_io.py
"""Tests for IO module."""
import json
import sys
import pytest
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
        from ace import Skillbook, Skill

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
        from ace import Skillbook, Skill

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
