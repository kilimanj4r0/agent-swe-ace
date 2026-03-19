# ACE-SWE Project Reorganization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the agent-swe-ace project into a clean three-phase architecture with standalone scripts, unified configuration, and proper data/logging separation.

**Architecture:** Three-phase loop (Predict → Evaluate → Learn) orchestrated by a main runner. Each phase is a standalone script that reads/writes to structured data directories. The skillbook accumulates across iterations for unresolved instances.

**Tech Stack:** Python 3.10+, mini-swe-agent v1, Agentic Context Engine (ACE), swebench, LiteLLM (Z.AI/vLLM), Opik observability, uv for dependencies.

---

## File Structure Overview

### New Files to Create
```
src/
├── phases/
│   ├── __init__.py
│   ├── predict.py          # Phase 1: Run agent with skillbook
│   ├── evaluate.py         # Phase 2: Evaluate patch with swebench
│   └── learn.py            # Phase 3: Update skillbook from failures
├── runners/
│   ├── __init__.py
│   └── main_loop.py        # Main orchestration loop
├── io/
│   ├── __init__.py
│   ├── readers.py          # Load instances, skillbooks, trajectories
│   └── writers.py          # Save results, skillbooks, logs
└── cli/
    ├── __init__.py
    └── commands.py         # CLI entry points for each phase
```

### Output Data Structure

```
data/
└── run_20260319_143052/              # Run name + compact timestamp
    ├── config.json                    # Full config used for this run
    ├── statistics.json                # Counts, resolved/unresolved lists, skills
    ├── experiment.log                 # Main log file
    └── swebench-lite/                 # Benchmark name (extracted from config)
        ├── trajectories/
        │   └── django__django-12345/
        │       ├── iter_0.json
        │       └── iter_1.json
        ├── results/
        │   └── django__django-12345/
        │       ├── iter_0.json
        │       └── iter_1.json
        └── skillbooks/
            # Per-instance mode (default):
            └── django__django-12345/
                ├── iter_0.json        # Empty (initial)
                └── iter_1.json        # After learning from iter_0 failure
            # Per-run mode:
            └── iter_0.json            # Empty
            └── iter_1.json            # Accumulated skills
```

### Benchmark Name Extraction

From config `benchmark.dataset = "princeton-nlp/SWE-bench_Lite"`:
- Extract last part: `SWE-bench_Lite`
- Lowercase and replace `_` with `-`: `swebench-lite`

### Instance Filtering (Per-Run Mode)

When `skillbook_mode = "per_run"` with `--filter-repo "django"`:
- Only instances matching `django` in repo name are processed
- Skillbook learns from all filtered instances
- Filter stored in `config.json`, NOT in directory name

### Files to Modify
```
src/
├── agents/miniswe_agent.py      # Simplify, add skillbook param
├── config/llm.py                # Keep as-is, already good
├── experiments/online_ace_runner.py  # Refactor to use phases
└── scripts/run_experiment.py    # Simplify to use main_loop

config.yaml                      # Already good, minor additions
README.md                        # Rewrite from scratch
```

### Files to Remove/Deprecate
```
src/learning/trajectory.py       # Move to io/readers.py
src/learning/templates.py        # Move to phases/predict.py
src/evaluation/simple.py         # Remove, only use swebench
src/scripts/run_baseline.py      # Fold into predict.py
src/utils/baseline_cache.py      # Remove, use data/ structure
```

### statistics.json Format

```json
{
  "run_name": "run_20260319_143052",
  "benchmark": "swebench-lite",
  "total_instances": 300,
  "resolved_count": 45,
  "unresolved_count": 255,
  "resolution_rate": 0.15,
  "resolved_ids": ["django__django-12345", "astropy__astropy-12907", ...],
  "unresolved_ids": ["django__django-11111", ...],
  "per_iteration": {
    "0": {
      "resolved": 30,
      "avg_trajectory_length": 45.2,
      "skills_count": 0
    },
    "1": {
      "resolved": 15,
      "avg_trajectory_length": 38.7,
      "skills_count": 25
    }
  },
  "total_skills_learned": 25,
  "skill_ids": ["skill-001", "skill-002", ...]
}
```

---

## Chunk 1: Data Layer (IO Module)

### Task 1: Create IO Module Structure

**Files:**
- Create: `src/io/__init__.py`
- Create: `src/io/readers.py`
- Create: `src/io/writers.py`
- Test: `src/tests/test_io.py`

- [ ] **Step 1: Write the failing test for readers**

```python
# src/tests/test_io.py
"""Tests for IO module."""
import json
import pytest
from pathlib import Path
from io import readers, writers


class TestExtractBenchmarkName:
    """Test benchmark name extraction."""

    def test_extract_from_hf_dataset(self):
        """Test extracting from HuggingFace dataset name."""
        assert readers.extract_benchmark_name("princeton-nlp/SWE-bench_Lite") == "swebench-lite"
        assert readers.extract_benchmark_name("princeton-nlp/SWE-bench") == "swebench"

    def test_extract_from_simple_name(self):
        """Test extracting from simple name."""
        assert readers.extract_benchmark_name("SWE-bench_Lite") == "swebench-lite"

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
        from ace_next import Skillbook

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
        from ace_next import Skillbook, Skill

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
        from ace_next import Skillbook, Skill

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/tests/test_io.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'io'" or similar

- [ ] **Step 3: Create io module init**

```python
# src/io/__init__.py
"""IO module for reading and writing experiment data."""

from .readers import load_instance, load_skillbook, load_trajectory
from .writers import save_trajectory, save_skillbook, save_result

__all__ = [
    "load_instance",
    "load_skillbook",
    "load_trajectory",
    "save_trajectory",
    "save_skillbook",
    "save_result",
]
```

- [ ] **Step 4: Write readers module**

```python
# src/io/readers.py
"""Data loading functions."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def extract_benchmark_name(dataset: str) -> str:
    """
    Extract benchmark name from dataset string.

    "princeton-nlp/SWE-bench_Lite" -> "princeton-nlp__SWE-bench_Lite"

    Args:
        dataset: Full dataset name from config

    Returns:
        Normalized benchmark name
    """
    name = name.replace("/", "__")
    return name


def load_instance(source: Union[Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load a SWE-bench instance from file or dict.

    Args:
        source: Path to JSON file or instance dict

    Returns:
        Instance dictionary with instance_id, repo, problem_statement, etc.
    """
    if isinstance(source, dict):
        return source

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Instance file not found: {source}")

    with open(source) as f:
        instance = json.load(f)

    logger.debug(f"Loaded instance: {instance.get('instance_id', 'unknown')}")
    return instance


def load_skillbook(source: Optional[Union[Path, str, Dict]]) -> "Skillbook":
    """
    Load a skillbook from file, dict, or create empty.

    Args:
        source: Path to JSON file, skillbook dict, or None for empty

    Returns:
        Skillbook instance
    """
    from ace_next import Skillbook, Skill

    skillbook = Skillbook()

    if source is None:
        logger.debug("Created empty skillbook")
        return skillbook

    if isinstance(source, Skillbook):
        return source

    # Load from dict or file
    if isinstance(source, dict):
        data = source
    else:
        source = Path(source)
        if not source.exists():
            logger.warning(f"Skillbook file not found: {source}, using empty")
            return skillbook
        with open(source) as f:
            data = json.load(f)

    # Populate skillbook from data
    for skill_id, skill_data in data.get("skills", {}).items():
        skill = Skill(
            id=skill_data["id"],
            section=skill_data.get("section", "general"),
            content=skill_data.get("content", ""),
            title=skill_data.get("title"),
            description=skill_data.get("description"),
        )
        skillbook._skills[skill_id] = skill

    logger.debug(f"Loaded skillbook with {len(skillbook.skills())} skills")
    return skillbook


def load_trajectory(source: Union[Path, Dict]) -> Dict:
    """
    Load an agent trajectory from file or dict.

    Args:
        source: Path to JSON file or trajectory dict

    Returns:
        Trajectory dict with 'info' and 'messages' keys
    """
    if isinstance(source, dict):
        return source

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Trajectory file not found: {source}")

    with open(source) as f:
        trajectory = json.load(f)

    logger.debug(f"Loaded trajectory with {len(trajectory.get('messages', []))} messages")
    return trajectory


def load_results(run_dir: Path, benchmark: str) -> Dict[str, Dict]:
    """
    Load all results for a run.

    Args:
        run_dir: Path to run directory
        benchmark: Benchmark name

    Returns:
        Dict mapping instance_id to result dict (latest iteration)
    """
    results = {}
    results_dir = run_dir / benchmark / "results"

    if not results_dir.exists():
        return results

    for instance_dir in results_dir.iterdir():
        if not instance_dir.is_dir():
            continue
        # Get latest iteration
        iter_files = sorted(instance_dir.glob("iter_*.json"))
        if iter_files:
            with open(iter_files[-1]) as f:
                result = json.load(f)
            instance_id = instance_dir.name
            results[instance_id] = result

    logger.debug(f"Loaded {len(results)} results from {run_dir}")
    return results


def load_statistics(run_dir: Path) -> Optional[Dict]:
    """
    Load statistics for a run.

    Args:
        run_dir: Path to run directory

    Returns:
        Statistics dict or None
    """
    stats_file = run_dir / "statistics.json"
    if not stats_file.exists():
        return None

    with open(stats_file) as f:
        return json.load(f)
```

- [ ] **Step 5: Write writers module**

```python
# src/io/writers.py
"""Data saving functions."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def extract_benchmark_name(dataset: str) -> str:
    """
    Extract benchmark name from dataset string.

    "princeton-nlp/SWE-bench_Lite" -> "princeton-nlp__SWE-bench_Lite"
    """
    name = name.replace("/", "__")
    return name


def get_run_dir(base_dir: Path, timestamp: Optional[str] = None) -> Path:
    """
    Get run directory path with timestamp.

    Args:
        base_dir: Base data directory
        timestamp: Optional timestamp string (default: now)

    Returns:
        Path to run directory
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"run_{timestamp}"


def save_trajectory(
    trajectory: Dict,
    run_dir: Path,
    benchmark: str,
    instance_id: str,
    iteration: int,
) -> Path:
    """
    Save an agent trajectory to JSON file.

    Args:
        trajectory: Trajectory dict with 'info' and 'messages'
        run_dir: Run directory (e.g., data/run_20260319_143052)
        benchmark: Benchmark name (e.g., "swebench-lite")
        instance_id: SWE-bench instance ID
        iteration: Iteration number (0-indexed)

    Returns:
        Path to saved file
    """
    output_dir = run_dir / benchmark / "trajectories" / instance_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"iter_{iteration}.json"

    with open(output_path, "w") as f:
        json.dump(trajectory, f, indent=2, default=str)

    logger.debug(f"Saved trajectory to {output_path}")
    return output_path


def save_skillbook(
    skillbook: "Skillbook",
    run_dir: Path,
    benchmark: str,
    iteration: int,
    instance_id: Optional[str] = None,
) -> Path:
    """
    Save a skillbook to JSON file.

    Args:
        skillbook: Skillbook instance
        run_dir: Run directory
        benchmark: Benchmark name
        iteration: Iteration number (0-indexed)
        instance_id: Optional instance ID for per-instance mode

    Returns:
        Path to saved file
    """
    if instance_id:
        # Per-instance mode
        output_dir = run_dir / benchmark / "skillbooks" / instance_id
    else:
        # Per-run mode
        output_dir = run_dir / benchmark / "skillbooks"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"iter_{iteration}.json"

    # Convert skillbook to dict
    data = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "instance_id": instance_id,
        "skills": {},
    }

    for skill in skillbook.skills():
        data["skills"][skill.id] = {
            "id": skill.id,
            "section": getattr(skill, "section", "general"),
            "title": getattr(skill, "title", None),
            "description": getattr(skill, "description", None),
            "content": getattr(skill, "content", ""),
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.debug(f"Saved skillbook ({len(skillbook.skills())} skills) to {output_path}")
    return output_path


def save_result(
    result: Dict[str, Any],
    run_dir: Path,
    benchmark: str,
    instance_id: str,
    iteration: int,
) -> Path:
    """
    Save an evaluation result to JSON file.

    Args:
        result: Result dict with resolved, feedback, metrics, etc.
        run_dir: Run directory
        benchmark: Benchmark name
        instance_id: SWE-bench instance ID
        iteration: Iteration number (0-indexed)

    Returns:
        Path to saved file
    """
    output_dir = run_dir / benchmark / "results" / instance_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"iter_{iteration}.json"

    # Add metadata
    result["instance_id"] = instance_id
    result["iteration"] = iteration
    result["timestamp"] = datetime.now().isoformat()

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.debug(f"Saved result to {output_path}")
    return output_path


def save_config(config: Dict, run_dir: Path) -> Path:
    """
    Save config for the run.

    Args:
        config: Configuration dict
        run_dir: Run directory

    Returns:
        Path to saved file
    """
    output_path = run_dir / "config.json"
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    logger.debug(f"Saved config to {output_path}")
    return output_path


def save_statistics(
    statistics: Dict,
    run_dir: Path,
) -> Path:
    """
    Save statistics for the run.

    Args:
        statistics: Statistics dict (see format below)
        run_dir: Run directory

    Returns:
        Path to saved file

    Statistics format:
    {
        "run_name": "run_20260319_143052",
        "benchmark": "swebench-lite",
        "total_instances": 300,
        "resolved_count": 45,
        "unresolved_count": 255,
        "resolution_rate": 0.15,
        "resolved_ids": [...],
        "unresolved_ids": [...],
        "per_iteration": {
            "0": {"resolved": 30, "avg_trajectory_length": 45.2, "skills_count": 0},
            "1": {"resolved": 15, "avg_trajectory_length": 38.7, "skills_count": 25}
        },
        "total_skills_learned": 25,
        "skill_ids": [...]
    }
    """
    output_path = run_dir / "statistics.json"
    with open(output_path, "w") as f:
        json.dump(statistics, f, indent=2, default=str)
    logger.info(f"Saved statistics to {output_path}")
    return output_path


def setup_run_logging(run_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging for a run.

    Args:
        run_dir: Run directory
        log_level: Log level string

    Returns:
        Configured logger
    """
    log_file = run_dir / "experiment.log"

    logger = logging.getLogger("experiment")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper()))
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)

    return logger
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest src/tests/test_io.py -v`
Expected: All tests PASS

---

## Chunk 2: Phase Scripts

### Task 2: Create Phase 1 - Predict

**Files:**
- Create: `src/phases/__init__.py`
- Create: `src/phases/predict.py`
- Test: `src/tests/test_phases.py`

- [ ] **Step 1: Write the failing test for predict phase**

```python
# src/tests/test_phases.py (append to file)
"""Tests for phase scripts."""
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestPredictPhase:
    """Test the predict (agent) phase."""

    def test_predict_phase_creates_trajectory(self, tmp_path):
        """Test that predict phase creates a trajectory file."""
        from phases.predict import PredictPhase

        # Mock the agent
        mock_agent = Mock()
        mock_agent.run.return_value = Mock(
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

        assert result["exit_status"] == "submitted"
        assert "patch" in result
        assert "trajectory" in result

    def test_predict_phase_saves_trajectory(self, tmp_path):
        """Test that predict phase saves trajectory to file."""
        from phases.predict import PredictPhase

        mock_agent = Mock()
        mock_agent.run.return_value = Mock(
            exit_status="submitted",
            patch="patch content",
            trajectory=[{"role": "user", "content": "Fix"}],
            error=None,
        )

        instance = {"instance_id": "test__repo-123", "problem_statement": "Fix"}

        phase = PredictPhase(agent=mock_agent, output_dir=tmp_path)
        phase.run(instance=instance, skillbook=None, iteration=0)

        # Check trajectory file was created
        traj_file = tmp_path / "trajectories" / "test__repo-123" / "iter_0.json"
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
        assert "skill-id" not in template

    def test_none_skillbook_returns_default_template(self):
        """Test that None skillbook returns unmodified template."""
        from phases.predict import build_instance_template

        template = build_instance_template(skillbook=None)

        assert "## Learned Strategies" not in template

    def test_skillbook_with_empty_context_returns_default_template(self):
        """Test that skillbook with skills but empty context returns default."""
        from phases.predict import build_instance_template

        # Skillbook with skills but wrap_skillbook_context returns empty
        mock_skillbook = Mock()
        mock_skillbook.skills.return_value = [{"id": "skill-1", "content": "..."}]

        with patch("phases.predict.wrap_skillbook_context", return_value="   "):
            template = build_instance_template(skillbook=mock_skillbook)

        # Should NOT inject empty section
        assert "## Learned Strategies" not in template

    def test_skillbook_injects_before_example_response(self):
        """Test that skillbook is injected before <example_response> tag."""
        from phases.predict import build_instance_template

        mock_skillbook = Mock()
        mock_skillbook.skills.return_value = [{"id": "skill-1", "content": "strategy"}]

        with patch(
            "phases.predict.wrap_skillbook_context",
            return_value="### skill-1\nstrategy content",
        ):
            template = build_instance_template(skillbook=mock_skillbook)

        # Skillbook section should appear BEFORE example_response
        skillbook_pos = template.find("## Learned Strategies")
        example_pos = template.find("<example_response>")
        assert skillbook_pos < example_pos

    def test_skillbook_injection_logged(self):
        """Test that skillbook injection is logged for debugging."""
        from phases.predict import build_instance_template

        mock_skillbook = Mock()
        mock_skillbook.skills.return_value = [{"id": "skill-1", "content": "..."}]

        with patch(
            "phases.predict.wrap_skillbook_context",
            return_value="skill content",
        ):
            with patch("phases.predict.logger") as mock_logger:
                build_instance_template(skillbook=mock_skillbook)

        # Should log injection
        mock_logger.debug.assert_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/tests/test_phases.py::TestPredictPhase -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'phases'"

- [ ] **Step 3: Create phases module init**

```python
# src/phases/__init__.py
"""Phase scripts for the ACE-SWE experiment loop."""

from .predict import (
    PredictPhase,
    run_predict,
    build_instance_template,
    build_system_template,
)
from .evaluate import EvaluatePhase, run_evaluate
from .learn import LearnPhase, run_learn

__all__ = [
    "PredictPhase",
    "run_predict",
    "build_instance_template",
    "build_system_template",
    "EvaluatePhase",
    "run_evaluate",
    "LearnPhase",
    "run_learn",
]
```

- [ ] **Step 4: Write predict phase**

```python
# src/phases/predict.py
"""Phase 1: Run mini-swe-agent with skillbook injection."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ace_next import Skillbook

from io.writers import save_trajectory

logger = logging.getLogger(__name__)


@dataclass
class PredictResult:
    """Result from predict phase."""

    instance_id: str
    iteration: int
    exit_status: str
    patch: str
    trajectory: list
    error: Optional[str] = None
    trajectory_path: Optional[Path] = None


class PredictPhase:
    """
    Phase 1: Run agent to generate patch with skillbook.

    This phase:
    1. Takes a SWE-bench instance and optional skillbook
    2. Runs mini-swe-agent with skillbook injected into prompt
    3. Saves trajectory to data/trajectories/
    4. Returns patch and trajectory for next phases
    """

    def __init__(
        self,
        agent,  # MiniSWEAgent instance
        output_dir: Path,
        run_name: str = "default",
    ):
        """
        Initialize predict phase.

        Args:
            agent: MiniSWEAgent instance
            output_dir: Base directory for outputs
            run_name: Name of the experiment run
        """
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.run_name = run_name

    def run(
        self,
        instance: Dict[str, Any],
        skillbook: Optional[Skillbook],
        iteration: int = 0,
    ) -> PredictResult:
        """
        Run agent on instance with skillbook.

        Args:
            instance: SWE-bench instance dict
            skillbook: Optional skillbook for prompt injection
            iteration: Current iteration number

        Returns:
            PredictResult with patch, trajectory, and metadata
        """
        instance_id = instance.get("instance_id", "unknown")
        logger.info(f"[Predict] Running agent for {instance_id} (iter {iteration})")

        # Run agent
        result = self.agent.run(
            problem=instance.get("problem_statement", ""),
            instance=instance,
            skillbook=skillbook,
        )

        # Build trajectory dict
        trajectory = {
            "info": {
                "exit_status": result.exit_status,
                "submission": result.patch,
                "iteration": iteration,
                "instance_id": instance_id,
            },
            "messages": result.trajectory,
        }

        # Save trajectory
        trajectory_path = save_trajectory(
            trajectory=trajectory,
            run_name=self.run_name,
            instance_id=instance_id,
            iteration=iteration,
            base_dir=self.output_dir,
        )

        logger.info(
            f"[Predict] Agent finished: {result.exit_status}, "
            f"patch={len(result.patch)} chars, "
            f"traj={len(result.trajectory)} messages"
        )

        return PredictResult(
            instance_id=instance_id,
            iteration=iteration,
            exit_status=result.exit_status,
            patch=result.patch,
            trajectory=result.trajectory,
            error=result.error,
            trajectory_path=trajectory_path,
        )


def run_predict(
    instance: Dict[str, Any],
    skillbook: Optional[Skillbook],
    agent,
    output_dir: Path,
    run_name: str,
    iteration: int = 0,
) -> PredictResult:
    """
    Convenience function to run predict phase.

    Args:
        instance: SWE-bench instance dict
        skillbook: Optional skillbook
        agent: MiniSWEAgent instance
        output_dir: Output directory
        run_name: Run name
        iteration: Iteration number

    Returns:
        PredictResult
    """
    phase = PredictPhase(agent=agent, output_dir=output_dir, run_name=run_name)
    return phase.run(instance=instance, skillbook=skillbook, iteration=iteration)


# --- Helper functions for skillbook injection ---

def _load_mini_swe_config() -> dict:
    """Load mini-swe-agent's default config from YAML file (cached)."""
    # Implementation loads from minisweagent package
    pass


def build_system_template() -> str:
    """Get mini-swe-agent's default system template."""
    config = _load_mini_swe_config()
    return config['agent']['system_template']


def build_instance_template(skillbook: Optional[Skillbook] = None) -> str:
    """
    Build instance template with skillbook context injected.

    Loads mini-swe-agent's default instance template and injects
    the skillbook context with defensive checks.

    Args:
        skillbook: Skillbook to inject into template (optional)

    Returns:
        Instance template string with skillbook section if valid skills exist
    """
    config = _load_mini_swe_config()
    default_template = config['agent']['instance_template']

    # Early return for None/empty skillbook
    if not skillbook:
        logger.debug("No skillbook provided - using default template")
        return default_template

    skills = skillbook.skills()
    if not skills:
        logger.debug("Skillbook has no skills - using default template")
        return default_template

    # Get formatted context
    skillbook_context = wrap_skillbook_context(skillbook)

    # Guard against empty context even when skills exist
    # (prevents injecting section with no content)
    if not skillbook_context or not skillbook_context.strip():
        logger.warning(
            f"Skillbook has {len(skills)} skill(s) but context is empty - "
            "using default template"
        )
        return default_template

    skillbook_section = f"""

## Learned Strategies (Skillbook)

These are strategies learned from previous attempts. Use them to guide your approach:

{skillbook_context}

When you apply a strategy successfully, reference it with [skill-id] notation in your reasoning."""

    # Inject before <example_response> if it exists
    if "<example_response>" in default_template:
        parts = default_template.split("<example_response>", 1)
        logger.debug(f"Injected skillbook with {len(skills)} skill(s) before <example_response>")
        return parts[0] + skillbook_section + "\n\n<example_response>" + parts[1]

    # Fallback: append at end (log warning since this may not be ideal placement)
    logger.warning(
        "<example_response> tag not found in template - "
        "appending skillbook at end"
    )
    return default_template + skillbook_section
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest src/tests/test_phases.py::TestPredictPhase -v`
Expected: All tests PASS

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest src/tests/test_phases.py::TestPredictPhase -v`
Expected: All tests PASS

### Task 3: Create Phase 2 - Evaluate

**Files:**
- Modify: `src/tests/test_phases.py`
- Create: `src/phases/evaluate.py`

- [ ] **Step 1: Write the failing test for evaluate phase**

```python
# src/tests/test_phases.py (append)
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
                run_name="test-run",
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
                run_name="test-run",
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
            run_name="test-run",
        )

        instance = {"instance_id": "test__repo-123"}
        result = phase.run(
            instance=instance,
            patch="",
            iteration=0,
        )

        assert result.resolved is False
        assert "No patch" in result.feedback
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/tests/test_phases.py::TestEvaluatePhase -v`
Expected: FAIL with "ImportError" or "ModuleNotFoundError"

- [ ] **Step 3: Write evaluate phase**

```python
# src/phases/evaluate.py
"""Phase 2: Evaluate patch using SWE-bench harness."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from evaluation import validate_patch
from io.writers import save_result

logger = logging.getLogger(__name__)


@dataclass
class EvaluateResult:
    """Result from evaluate phase."""

    instance_id: str
    iteration: int
    resolved: bool
    feedback: str
    metrics: Dict[str, Any]
    result_path: Optional[Path] = None


class EvaluatePhase:
    """
    Phase 2: Evaluate patch using SWE-bench Docker harness.

    This phase:
    1. Takes patch from predict phase
    2. Runs SWE-bench Docker evaluation
    3. Saves result to data/results/
    4. Returns resolved status and feedback
    """

    def __init__(
        self,
        use_docker: bool = True,
        timeout: int = 1800,
        output_dir: Optional[Path] = None,
        run_name: str = "default",
    ):
        """
        Initialize evaluate phase.

        Args:
            use_docker: Use Docker harness (recommended)
            timeout: Evaluation timeout in seconds
            output_dir: Base directory for outputs
            run_name: Name of the experiment run
        """
        self.use_docker = use_docker
        self.timeout = timeout
        self.output_dir = Path(output_dir) if output_dir else Path("data")
        self.run_name = run_name

    def run(
        self,
        instance: Dict[str, Any],
        patch: str,
        iteration: int = 0,
    ) -> EvaluateResult:
        """
        Evaluate patch against SWE-bench test suite.

        Args:
            instance: SWE-bench instance dict
            patch: Generated patch from predict phase
            iteration: Current iteration number

        Returns:
            EvaluateResult with resolved status and feedback
        """
        instance_id = instance.get("instance_id", "unknown")
        logger.info(f"[Evaluate] Evaluating patch for {instance_id} (iter {iteration})")

        # Handle empty patch
        if not patch or not patch.strip():
            logger.warning(f"[Evaluate] Empty patch for {instance_id}")
            result = EvaluateResult(
                instance_id=instance_id,
                iteration=iteration,
                resolved=False,
                feedback="No patch submitted. Agent did not produce a valid patch.",
                metrics={"resolved": 0.0, "patch_empty": 1.0},
            )
            # Save result
            result.result_path = save_result(
                result={
                    "resolved": result.resolved,
                    "feedback": result.feedback,
                    "metrics": result.metrics,
                },
                run_name=self.run_name,
                instance_id=instance_id,
                iteration=iteration,
                base_dir=self.output_dir,
            )
            return result

        # Run evaluation
        try:
            resolved = validate_patch(
                instance=instance,
                patch=patch,
                use_docker=self.use_docker,
                timeout=self.timeout,
            )
        except Exception as e:
            logger.error(f"[Evaluate] Error evaluating {instance_id}: {e}")
            resolved = False

        # Build result
        if resolved:
            feedback = "Patch resolved all tests successfully!"
            logger.info(f"[Evaluate] {instance_id} RESOLVED!")
        else:
            feedback = "Patch did not resolve the issue. Tests failed or patch invalid."
            logger.info(f"[Evaluate] {instance_id} NOT resolved")

        result = EvaluateResult(
            instance_id=instance_id,
            iteration=iteration,
            resolved=resolved,
            feedback=feedback,
            metrics={
                "resolved": 1.0 if resolved else 0.0,
                "patch_length": len(patch),
            },
        )

        # Save result
        result.result_path = save_result(
            result={
                "resolved": result.resolved,
                "feedback": result.feedback,
                "metrics": result.metrics,
                "patch": patch[:1000] + "..." if len(patch) > 1000 else patch,
            },
            run_name=self.run_name,
            instance_id=instance_id,
            iteration=iteration,
            base_dir=self.output_dir,
        )

        return result


def run_evaluate(
    instance: Dict[str, Any],
    patch: str,
    output_dir: Path,
    run_name: str,
    iteration: int = 0,
    use_docker: bool = True,
    timeout: int = 1800,
) -> EvaluateResult:
    """
    Convenience function to run evaluate phase.

    Args:
        instance: SWE-bench instance dict
        patch: Patch to evaluate
        output_dir: Output directory
        run_name: Run name
        iteration: Iteration number
        use_docker: Use Docker evaluation
        timeout: Evaluation timeout

    Returns:
        EvaluateResult
    """
    phase = EvaluatePhase(
        use_docker=use_docker,
        timeout=timeout,
        output_dir=output_dir,
        run_name=run_name,
    )
    return phase.run(instance=instance, patch=patch, iteration=iteration)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/tests/test_phases.py::TestEvaluatePhase -v`
Expected: All tests PASS

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest src/tests/test_phases.py::TestEvaluatePhase -v`
Expected: All tests PASS

### Task 4: Create Phase 3 - Learn

**Files:**
- Modify: `src/tests/test_phases.py`
- Create: `src/phases/learn.py`

- [ ] **Step 1: Write the failing test for learn phase**

```python
# src/tests/test_phases.py (append)
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
        mock_skill_manager.update_skills.return_value = Mock(
            skills_added=["skill-1"],
            skills_updated=[],
        )

        phase = LearnPhase(
            reflector=mock_reflector,
            skill_manager=mock_skill_manager,
            output_dir=tmp_path,
            run_name="test-run",
        )

        skillbook = Skillbook()
        instance = {"instance_id": "test__repo-123"}
        trajectory = {"messages": [{"role": "user", "content": "Fix"}]}
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

    def test_learn_phase_skips_resolved(self, tmp_path):
        """Test that learn phase skips when instance is resolved."""
        from phases.learn import LearnPhase
        from ace_next import Skillbook

        mock_reflector = Mock()
        mock_skill_manager = Mock()

        phase = LearnPhase(
            reflector=mock_reflector,
            skill_manager=mock_skill_manager,
            output_dir=tmp_path,
            run_name="test-run",
        )

        skillbook = Skillbook()

        # Pass resolved=True
        result = phase.run(
            skillbook=skillbook,
            instance={"instance_id": "test__repo-123"},
            trajectory={},
            patch="good patch",
            iteration=0,
            resolved=True,  # Already resolved
        )

        # Should not call reflector
        mock_reflector.reflect.assert_not_called()
        assert result.skills_added == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/tests/test_phases.py::TestLearnPhase -v`
Expected: FAIL with "ImportError" or "ModuleNotFoundError"

- [ ] **Step 3: Write learn phase**

```python
# src/phases/learn.py
"""Phase 3: Learn skillbook from failed attempts."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ace_next import Skillbook, Reflector, SkillManager
from ace.roles import AgentOutput

from io.writers import save_skillbook
from learning.trajectory import extract_agent_output

logger = logging.getLogger(__name__)


@dataclass
class LearnResult:
    """Result from learn phase."""

    instance_id: str
    iteration: int
    skills_added: int
    skills_updated: int
    skillbook_path: Optional[Path] = None
    reflection: Optional[Any] = None


class LearnPhase:
    """
    Phase 3: Update skillbook from failed attempts.

    This phase:
    1. Takes trajectory from failed attempt
    2. Uses ACE Reflector to analyze failure
    3. Uses ACE SkillManager to create/update skills
    4. Saves updated skillbook to data/skillbooks/
    5. Returns updated skillbook for next iteration
    """

    def __init__(
        self,
        reflector: Reflector,
        skill_manager: SkillManager,
        output_dir: Optional[Path] = None,
        run_name: str = "default",
    ):
        """
        Initialize learn phase.

        Args:
            reflector: ACE Reflector for failure analysis
            skill_manager: ACE SkillManager for skill creation
            output_dir: Base directory for outputs
            run_name: Name of the experiment run
        """
        self.reflector = reflector
        self.skill_manager = skill_manager
        self.output_dir = Path(output_dir) if output_dir else Path("data")
        self.run_name = run_name

    def run(
        self,
        skillbook: Skillbook,
        instance: Dict[str, Any],
        trajectory: List[Dict],
        patch: str,
        iteration: int = 0,
        resolved: bool = False,
        feedback: Optional[str] = None,
    ) -> LearnResult:
        """
        Learn from trajectory if not resolved.

        Args:
            skillbook: Current skillbook to update
            instance: SWE-bench instance dict
            trajectory: Agent trajectory from predict phase
            patch: Generated patch
            iteration: Current iteration number
            resolved: Whether the patch resolved the issue
            feedback: Evaluation feedback

        Returns:
            LearnResult with updated skillbook info
        """
        instance_id = instance.get("instance_id", "unknown")

        # Skip if resolved
        if resolved:
            logger.info(f"[Learn] Skipping {instance_id} - already resolved")
            return LearnResult(
                instance_id=instance_id,
                iteration=iteration,
                skills_added=0,
                skills_updated=0,
            )

        logger.info(f"[Learn] Analyzing failure for {instance_id} (iter {iteration})")

        # Extract agent output for ACE
        agent_output = extract_agent_output(trajectory, patch)

        # Get problem statement
        question = instance.get("problem_statement", "")

        # Run reflector
        try:
            reflection = self.reflector.reflect(
                question=question,
                agent_output=agent_output,
                skillbook=skillbook,
                feedback=feedback or "Patch did not resolve the issue",
            )
            logger.info(f"[Learn] Reflection complete for {instance_id}")
        except Exception as e:
            logger.error(f"[Learn] Reflection failed for {instance_id}: {e}")
            return LearnResult(
                instance_id=instance_id,
                iteration=iteration,
                skills_added=0,
                skills_updated=0,
            )

        # Update skills
        try:
            skill_result = self.skill_manager.update_skills(
                reflection=reflection,
                skillbook=skillbook,
                question_context=question,
                progress="failed",
            )
            skills_added = len(getattr(skill_result, "skills_added", []))
            skills_updated = len(getattr(skill_result, "skills_updated", []))
            logger.info(
                f"[Learn] Skills updated for {instance_id}: "
                f"+{skills_added} added, ~{skills_updated} updated"
            )
        except Exception as e:
            logger.error(f"[Learn] Skill update failed for {instance_id}: {e}")
            skills_added = 0
            skills_updated = 0

        # Save skillbook
        skillbook_path = save_skillbook(
            skillbook=skillbook,
            run_name=self.run_name,
            iteration=iteration + 1,  # Next iteration
            base_dir=self.output_dir,
        )

        return LearnResult(
            instance_id=instance_id,
            iteration=iteration,
            skills_added=skills_added,
            skills_updated=skills_updated,
            skillbook_path=skillbook_path,
            reflection=reflection,
        )


def run_learn(
    skillbook: Skillbook,
    instance: Dict[str, Any],
    trajectory: List[Dict],
    patch: str,
    reflector: Reflector,
    skill_manager: SkillManager,
    output_dir: Path,
    run_name: str,
    iteration: int = 0,
    resolved: bool = False,
    feedback: Optional[str] = None,
) -> LearnResult:
    """
    Convenience function to run learn phase.

    Args:
        skillbook: Current skillbook
        instance: SWE-bench instance
        trajectory: Agent trajectory
        patch: Generated patch
        reflector: ACE Reflector
        skill_manager: ACE SkillManager
        output_dir: Output directory
        run_name: Run name
        iteration: Iteration number
        resolved: Whether resolved
        feedback: Evaluation feedback

    Returns:
        LearnResult
    """
    phase = LearnPhase(
        reflector=reflector,
        skill_manager=skill_manager,
        output_dir=output_dir,
        run_name=run_name,
    )
    return phase.run(
        skillbook=skillbook,
        instance=instance,
        trajectory=trajectory,
        patch=patch,
        iteration=iteration,
        resolved=resolved,
        feedback=feedback,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/tests/test_phases.py::TestLearnPhase -v`
Expected: All tests PASS

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest src/tests/test_phases.py::TestLearnPhase -v`
Expected: All tests PASS

---

## Chunk 3: Main Loop Runner

### Task 5: Create Main Loop Runner

**Files:**
- Create: `src/runners/__init__.py`
- Create: `src/runners/main_loop.py`
- Test: `src/tests/test_main_loop.py`

- [ ] **Step 1: Write the failing test**

```python
# src/tests/test_main_loop.py
"""Tests for main loop runner."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


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
        assert results[-1].resolved is True
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
        assert results[-1].resolved is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/tests/test_main_loop.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'runners'"

- [ ] **Step 3: Create runners module**

```python
# src/runners/__init__.py
"""Experiment runners."""

from .main_loop import ExperimentLoop, run_experiment

__all__ = ["ExperimentLoop", "run_experiment"]
```

- [ ] **Step 4: Write main loop runner**

```python
# src/runners/main_loop.py
"""Main experiment loop: Predict → Evaluate → Learn."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ace_next import Skillbook

from io.writers import save_run_summary

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    """Result from a single iteration."""

    iteration: int
    predict_result: Any
    evaluate_result: Any
    learn_result: Optional[Any] = None


@dataclass
class InstanceResult:
    """Result from all iterations for an instance."""

    instance_id: str
    iterations: List[IterationResult] = field(default_factory=list)
    final_resolved: bool = False
    total_attempts: int = 0


class ExperimentLoop:
    """
    Main experiment loop: Predict → Evaluate → Learn.

    For each instance:
    1. Predict: Run agent with current skillbook
    2. Evaluate: Test patch with SWE-bench
    3. Learn (if failed): Update skillbook

    Repeat until resolved or max_attempts reached.
    """

    def __init__(
        self,
        predict_phase,  # PredictPhase instance
        evaluate_phase,  # EvaluatePhase instance
        learn_phase,  # LearnPhase instance
        output_dir: Path,
        run_name: str = "default",
        max_attempts: int = 3,
        skillbook_mode: str = "per_instance",  # per_instance, per_repo, global
    ):
        """
        Initialize experiment loop.

        Args:
            predict_phase: Phase 1 runner
            evaluate_phase: Phase 2 runner
            learn_phase: Phase 3 runner
            output_dir: Output directory
            run_name: Name of this run
            max_attempts: Maximum attempts per instance
            skillbook_mode: How to manage skillbooks
        """
        self.predict = predict_phase
        self.evaluate = evaluate_phase
        self.learn = learn_phase
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.max_attempts = max_attempts
        self.skillbook_mode = skillbook_mode

        # Global skillbook for 'global' mode
        self.global_skillbook = Skillbook()
        # Per-repo skillbooks for 'per_repo' mode
        self.repo_skillbooks: Dict[str, Skillbook] = {}

    def get_skillbook(self, repo: str) -> Skillbook:
        """Get skillbook based on mode."""
        if self.skillbook_mode == "global":
            return self.global_skillbook
        elif self.skillbook_mode == "per_repo":
            if repo not in self.repo_skillbooks:
                self.repo_skillbooks[repo] = Skillbook()
            return self.repo_skillbooks[repo]
        else:  # per_instance
            return Skillbook()

    def update_skillbook(self, repo: str, skillbook: Skillbook):
        """Update skillbook based on mode."""
        if self.skillbook_mode == "global":
            self.global_skillbook = skillbook
        elif self.skillbook_mode == "per_repo":
            self.repo_skillbooks[repo] = skillbook
        # per_instance: skillbook is not persisted

    def run_instance(
        self,
        instance: Dict[str, Any],
        initial_skillbook: Optional[Skillbook] = None,
    ) -> InstanceResult:
        """
        Run experiment loop for a single instance.

        Args:
            instance: SWE-bench instance dict
            initial_skillbook: Optional starting skillbook

        Returns:
            InstanceResult with all iteration results
        """
        instance_id = instance.get("instance_id", "unknown")
        repo = instance.get("repo", "unknown")

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting instance: {instance_id}")
        logger.info(f"Repo: {repo}")
        logger.info(f"{'='*60}")

        # Get skillbook for this instance
        skillbook = initial_skillbook or self.get_skillbook(repo)

        result = InstanceResult(instance_id=instance_id)

        for iteration in range(self.max_attempts):
            logger.info(f"\n--- Iteration {iteration + 1}/{self.max_attempts} ---")

            # Phase 1: Predict
            predict_result = self.predict.run(
                instance=instance,
                skillbook=skillbook,
                iteration=iteration,
            )

            # Phase 2: Evaluate
            evaluate_result = self.evaluate.run(
                instance=instance,
                patch=predict_result.patch,
                iteration=iteration,
            )

            # Record iteration
            iter_result = IterationResult(
                iteration=iteration,
                predict_result=predict_result,
                evaluate_result=evaluate_result,
            )

            # Check if resolved
            if evaluate_result.resolved:
                logger.info(f"Instance {instance_id} RESOLVED at iteration {iteration + 1}")
                result.final_resolved = True
                result.total_attempts = iteration + 1
                result.iterations.append(iter_result)
                break

            # Phase 3: Learn (only if not resolved)
            learn_result = self.learn.run(
                skillbook=skillbook,
                instance=instance,
                trajectory=predict_result.trajectory,
                patch=predict_result.patch,
                iteration=iteration,
                resolved=False,
                feedback=evaluate_result.feedback,
            )
            iter_result.learn_result = learn_result

            # Update skillbook for next iteration
            # (skillbook is modified in-place by learn phase)
            self.update_skillbook(repo, skillbook)

            result.iterations.append(iter_result)
            result.total_attempts = iteration + 1

        if not result.final_resolved:
            logger.info(f"Instance {instance_id} NOT resolved after {self.max_attempts} attempts")

        return result

    def run(
        self,
        instances: List[Dict[str, Any]],
        config: Optional[Dict] = None,
    ) -> Dict[str, InstanceResult]:
        """
        Run experiment loop for multiple instances.

        Args:
            instances: List of SWE-bench instance dicts
            config: Optional config to save with summary

        Returns:
            Dict mapping instance_id to InstanceResult
        """
        logger.info(f"\n{'#'*60}")
        logger.info(f"Starting experiment: {self.run_name}")
        logger.info(f"Instances: {len(instances)}")
        logger.info(f"Max attempts: {self.max_attempts}")
        logger.info(f"Skillbook mode: {self.skillbook_mode}")
        logger.info(f"{'#'*60}\n")

        results = {}
        resolved_count = 0

        for i, instance in enumerate(instances):
            instance_id = instance.get("instance_id", f"unknown-{i}")
            logger.info(f"\n[{i+1}/{len(instances)}] Processing {instance_id}")

            instance_result = self.run_instance(instance)
            results[instance_id] = instance_result

            if instance_result.final_resolved:
                resolved_count += 1

            # Log progress
            rate = resolved_count / (i + 1) * 100
            logger.info(
                f"\nProgress: {i+1}/{len(instances)} | "
                f"Resolved: {resolved_count} ({rate:.1f}%)"
            )

        # Save summary
        statistics = {
            "total_instances": len(instances),
            "resolved": resolved_count,
            "unresolved": len(instances) - resolved_count,
            "resolution_rate": resolved_count / len(instances) if instances else 0,
            "total_skills": len(self.global_skillbook.skills())
            if self.skillbook_mode == "global"
            else sum(len(s.skills()) for s in self.repo_skillbooks.values()),
        }

        save_run_summary(
            run_name=self.run_name,
            config=config or {},
            statistics=statistics,
            base_dir=self.output_dir,
        )

        logger.info(f"\n{'#'*60}")
        logger.info(f"Experiment complete: {self.run_name}")
        logger.info(f"Resolution rate: {statistics['resolution_rate']:.1%}")
        logger.info(f"{'#'*60}\n")

        return results


def run_experiment(
    instances: List[Dict[str, Any]],
    predict_phase,
    evaluate_phase,
    learn_phase,
    output_dir: Path,
    run_name: str,
    max_attempts: int = 3,
    skillbook_mode: str = "per_instance",
    config: Optional[Dict] = None,
) -> Dict[str, InstanceResult]:
    """
    Convenience function to run experiment.

    Args:
        instances: SWE-bench instances
        predict_phase: Predict phase
        evaluate_phase: Evaluate phase
        learn_phase: Learn phase
        output_dir: Output directory
        run_name: Run name
        max_attempts: Max attempts per instance
        skillbook_mode: Skillbook mode
        config: Config dict

    Returns:
        Results dict
    """
    loop = ExperimentLoop(
        predict_phase=predict_phase,
        evaluate_phase=evaluate_phase,
        learn_phase=learn_phase,
        output_dir=output_dir,
        run_name=run_name,
        max_attempts=max_attempts,
        skillbook_mode=skillbook_mode,
    )
    return loop.run(instances, config)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest src/tests/test_main_loop.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest src/tests/test_main_loop.py -v`
Expected: All tests PASS

---

## Chunk 4: CLI Commands and Integration

### Task 6: Create CLI Commands

**Files:**
- Create: `src/cli/__init__.py`
- Create: `src/cli/commands.py`
- Modify: `src/scripts/run_experiment.py`

- [ ] **Step 1: Write CLI commands module**

```python
# src/cli/__init__.py
"""CLI commands for ACE-SWE experiment."""

from .commands import main, run_predict_cmd, run_evaluate_cmd, run_learn_cmd

__all__ = ["main", "run_predict_cmd", "run_evaluate_cmd", "run_learn_cmd"]
```

```python
# src/cli/commands.py
"""CLI entry points for ACE-SWE experiment phases."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from datasets import load_dataset
from dotenv import load_dotenv

# Setup path for imports
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from config.llm import LLMConfig, create_model, create_ace_client
from agents.miniswe_agent import MiniSWEAgent
from phases.predict import PredictPhase, run_predict
from phases.evaluate import EvaluatePhase, run_evaluate
from phases.learn import LearnPhase, run_learn
from runners.main_loop import ExperimentLoop
from io.readers import load_instance, load_skillbook, load_trajectory

load_dotenv()

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Setup logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_instances(config: dict) -> list:
    """Load instances from SWE-bench or file."""
    # Check for cached instances first
    cache_dir = Path(config.get("output", {}).get("dir", "data")) / "instances"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load from dataset
    logger.info(f"Loading dataset: {config['benchmark']['dataset']}")
    dataset = load_dataset(
        config["benchmark"]["dataset"],
        split=config["benchmark"]["split"],
    )
    instances = list(dataset)

    # Limit if specified
    max_instances = config["benchmark"].get("max_instances")
    if max_instances:
        instances = instances[:max_instances]

    return instances


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ACE-SWE Experiment: Skillbook learning with mini-swe-agent"
    )
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--max-instances", "-n", type=int, help="Max instances")
    parser.add_argument("--max-attempts", "-a", type=int, help="Max attempts per instance")
    parser.add_argument(
        "--phase",
        choices=["all", "predict", "evaluate", "learn"],
        default="all",
        help="Run specific phase only",
    )
    parser.add_argument("--instance", help="Run specific instance ID")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number")
    parser.add_argument("--skillbook", help="Path to skillbook JSON")
    parser.add_argument("--trajectory", help="Path to trajectory JSON (for evaluate/learn)")
    parser.add_argument("--patch", help="Patch string (for evaluate)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--observe", action="store_true", help="Enable Opik observability")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.max_instances:
        config.setdefault("benchmark", {})["max_instances"] = args.max_instances
    if args.max_attempts:
        config.setdefault("experiment", {})["max_attempts"] = args.max_attempts
    if args.output:
        config.setdefault("output", {})["dir"] = args.output

    # Run appropriate phase
    if args.phase == "all":
        run_full_experiment(config, args)
    elif args.phase == "predict":
        run_predict_cmd(config, args)
    elif args.phase == "evaluate":
        run_evaluate_cmd(config, args)
    elif args.phase == "learn":
        run_learn_cmd(config, args)


def run_full_experiment(config: dict, args):
    """Run full experiment loop."""
    from utils.llm_observer import enable_observability

    # Setup run name and output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config["experiment"].get("name", "experiment")
    output_dir = Path(config["output"]["dir"]) / f"{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enable observability
    if args.observe or config.get("observability", {}).get("enabled"):
        enable_observability(
            project_name=config.get("observability", {}).get("project_name", run_name)
        )

    # Create LLM configs
    agent_config = LLMConfig.from_dict(config["llm"]["agent"])
    ace_config = LLMConfig.from_dict(config["llm"]["ace"])

    # Create components
    agent_model = create_model(agent_config)
    ace_client = create_ace_client(ace_config.to_dict())

    agent = MiniSWEAgent(
        llm_model=agent_model,
        use_docker=config.get("environment", {}).get("type") == "docker",
        step_limit=config.get("agent", {}).get("step_limit", 100),
        cost_limit=config.get("agent", {}).get("cost_limit", 5.0),
        output_dir=output_dir,
    )

    from ace_next import Reflector, SkillManager

    predict_phase = PredictPhase(agent=agent, output_dir=output_dir, run_name=run_name)
    evaluate_phase = EvaluatePhase(
        use_docker=config.get("evaluation", {}).get("use_docker", True),
        timeout=config.get("evaluation", {}).get("timeout", 1800),
        output_dir=output_dir,
        run_name=run_name,
    )
    learn_phase = LearnPhase(
        reflector=Reflector(ace_client),
        skill_manager=SkillManager(ace_client),
        output_dir=output_dir,
        run_name=run_name,
    )

    # Get instances
    instances = get_instances(config)

    # Filter to specific instance if requested
    if args.instance:
        instances = [i for i in instances if i.get("instance_id") == args.instance]
        if not instances:
            logger.error(f"Instance not found: {args.instance}")
            sys.exit(1)

    # Run experiment
    loop = ExperimentLoop(
        predict_phase=predict_phase,
        evaluate_phase=evaluate_phase,
        learn_phase=learn_phase,
        output_dir=output_dir,
        run_name=run_name,
        max_attempts=config["experiment"].get("max_attempts", 2),
        skillbook_mode=config["experiment"].get("skillbook_mode", "per_instance"),
    )

    loop.run(instances, config)


def run_predict_cmd(config: dict, args):
    """Run predict phase only."""
    if not args.instance:
        logger.error("--instance required for predict phase")
        sys.exit(1)

    # Setup
    output_dir = Path(config["output"]["dir"])
    run_name = config["experiment"].get("name", "experiment")

    # Create agent
    agent_config = LLMConfig.from_dict(config["llm"]["agent"])
    agent_model = create_model(agent_config)
    agent = MiniSWEAgent(
        llm_model=agent_model,
        use_docker=config.get("environment", {}).get("type") == "docker",
        step_limit=config.get("agent", {}).get("step_limit", 100),
        cost_limit=config.get("agent", {}).get("cost_limit", 5.0),
        output_dir=output_dir,
    )

    # Load instance
    instances = get_instances(config)
    instance = next((i for i in instances if i["instance_id"] == args.instance), None)
    if not instance:
        logger.error(f"Instance not found: {args.instance}")
        sys.exit(1)

    # Load skillbook
    skillbook = load_skillbook(args.skillbook)

    # Run predict
    phase = PredictPhase(agent=agent, output_dir=output_dir, run_name=run_name)
    result = phase.run(instance=instance, skillbook=skillbook, iteration=args.iteration)

    print(f"\nPredict result:")
    print(f"  Exit status: {result.exit_status}")
    print(f"  Patch length: {len(result.patch)} chars")
    print(f"  Trajectory: {result.trajectory_path}")


def run_evaluate_cmd(config: dict, args):
    """Run evaluate phase only."""
    if not args.instance:
        logger.error("--instance required for evaluate phase")
        sys.exit(1)

    output_dir = Path(config["output"]["dir"])
    run_name = config["experiment"].get("name", "experiment")

    # Load instance
    instances = get_instances(config)
    instance = next((i for i in instances if i["instance_id"] == args.instance), None)
    if not instance:
        logger.error(f"Instance not found: {args.instance}")
        sys.exit(1)

    # Get patch
    if args.patch:
        patch = args.patch
    elif args.trajectory:
        traj = load_trajectory(args.trajectory)
        patch = traj.get("info", {}).get("submission", "")
    else:
        logger.error("--patch or --trajectory required for evaluate phase")
        sys.exit(1)

    # Run evaluate
    phase = EvaluatePhase(
        use_docker=config.get("evaluation", {}).get("use_docker", True),
        timeout=config.get("evaluation", {}).get("timeout", 1800),
        output_dir=output_dir,
        run_name=run_name,
    )
    result = phase.run(instance=instance, patch=patch, iteration=args.iteration)

    print(f"\nEvaluate result:")
    print(f"  Resolved: {result.resolved}")
    print(f"  Feedback: {result.feedback}")
    print(f"  Result: {result.result_path}")


def run_learn_cmd(config: dict, args):
    """Run learn phase only."""
    if not args.instance or not args.trajectory:
        logger.error("--instance and --trajectory required for learn phase")
        sys.exit(1)

    output_dir = Path(config["output"]["dir"])
    run_name = config["experiment"].get("name", "experiment")

    # Load instance
    instances = get_instances(config)
    instance = next((i for i in instances if i["instance_id"] == args.instance), None)
    if not instance:
        logger.error(f"Instance not found: {args.instance}")
        sys.exit(1)

    # Load trajectory
    traj = load_trajectory(args.trajectory)
    patch = traj.get("info", {}).get("submission", "")
    messages = traj.get("messages", [])

    # Load skillbook
    skillbook = load_skillbook(args.skillbook)

    # Create ACE client
    ace_config = LLMConfig.from_dict(config["llm"]["ace"])
    ace_client = create_ace_client(ace_config.to_dict())

    from ace_next import Reflector, SkillManager

    # Run learn
    phase = LearnPhase(
        reflector=Reflector(ace_client),
        skill_manager=SkillManager(ace_client),
        output_dir=output_dir,
        run_name=run_name,
    )
    result = phase.run(
        skillbook=skillbook,
        instance=instance,
        trajectory=messages,
        patch=patch,
        iteration=args.iteration,
        resolved=False,
    )

    print(f"\nLearn result:")
    print(f"  Skills added: {result.skills_added}")
    print(f"  Skills updated: {result.skills_updated}")
    print(f"  Skillbook: {result.skillbook_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update run_experiment.py to use new structure**

```python
# src/scripts/run_experiment.py
"""
Main experiment entry point.

Usage:
    uv run python -m scripts.run_experiment
    uv run python -m scripts.run_experiment --config config.yaml
    uv run python -m scripts.run_experiment --max-instances 10
    uv run python -m scripts.run_experiment --phase predict --instance django__django-12345
"""

from cli.commands import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Test CLI help**

Run: `uv run python -m scripts.run_experiment --help`
Expected: Shows help with all options

- [ ] **Step 4: Test CLI help**

Run: `uv run python -m scripts.run_experiment --help`
Expected: Shows help with all options

---

## Chunk 5: Documentation and Demo

### Task 7: Write README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write README**

```markdown
# ACE-SWE: Skillbook Learning for SWE-bench

Integrates ACE (Automated Capability Enhancement) skillbook learning with mini-swe-agent to improve SWE-bench Lite issue resolution rates through iterative learning.

## Overview

The system learns from failed attempts by reflecting on trajectories and updating a skillbook of strategies. Each unresolved issue triggers learning that can help future attempts.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Loop                           │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Predict  │───▶│ Evaluate │───▶│  Learn   │──┐           │
│  │ (Agent)  │    │(SWE-bench)│   │  (ACE)   │  │           │
│  └──────────┘    └──────────┘    └──────────┘  │           │
│       ▲              │ Resolved?                │           │
│       │              └──────────────────────────┘           │
│       │                      No                             │
│       └──────────────────────────────────┐                 │
│                                          │                  │
│                          With updated    │                  │
│                          skillbook       │                  │
│                                          ▼                  │
│                                   Max attempts?             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run experiment
uv run python -m scripts.run_experiment --max-instances 10

# Run specific instance
uv run python -m scripts.run_experiment --instance django__django-12345

# Run with observability
uv run python -m scripts.run_experiment --observe
```

## Usage

### Full Experiment

```bash
# Run on all instances with 2 attempts each
uv run python -m scripts.run_experiment

# Limit instances and attempts
uv run python -m scripts.run_experiment --max-instances 50 --max-attempts 3
```

### Individual Phases

```bash
# Phase 1: Predict (run agent)
uv run python -m scripts.run_experiment \
    --phase predict \
    --instance django__django-12345 \
    --skillbook data/skillbooks/run_001/iter_0.json

# Phase 2: Evaluate (test patch)
uv run python -m scripts.run_experiment \
    --phase evaluate \
    --instance django__django-12345 \
    --trajectory data/trajectories/run_001/django__django-12345/iter_0.json

# Phase 3: Learn (update skillbook)
uv run python -m scripts.run_experiment \
    --phase learn \
    --instance django__django-12345 \
    --trajectory data/trajectories/run_001/django__django-12345/iter_0.json
```

## Configuration

See `config.yaml` for all options:

```yaml
experiment:
  name: "mini-swe-skillbook"
  max_attempts: 2
  skillbook_mode: "per_instance"  # per_instance, per_repo, global

llm:
  agent:
    provider: "zai"
    model: "glm-4.7-flashx"
  ace:
    provider: "zai"
    model: "glm-4.7-flashx"

benchmark:
  dataset: "princeton-nlp/SWE-bench_Lite"
  max_instances: null  # all instances
```

## Project Structure

```
src/
├── phases/          # Predict, Evaluate, Learn
├── runners/         # Main experiment loop
├── agents/          # mini-swe-agent wrapper
├── io/              # Data loading/saving
├── config/          # LLM configuration
└── scripts/         # Entry points

data/
├── instances/       # Cached SWE-bench instances
├── trajectories/    # Agent trajectories
├── skillbooks/      # Learned skillbooks
└── results/         # Evaluation results

logs/                # Per-run logs
```

## Dependencies

- **mini-swe-agent** (v1) - SWE resolution agent
- **ace-framework** - Skillbook learning
- **swebench** - Docker harness for evaluation
- **litellm** - Unified LLM API (Z.AI/vLLM)
- **opik** - LLM observability (optional)

## License

MIT
```

- [ ] **Step 2: Review README**

Open `README.md` and verify it documentation is complete and accurate.

### Task 8: Create Demo Notebook

**Files:**
- Create: `notebooks/demo_phases.ipynb`

- [ ] **Step 1: Create demo notebook showing each phase**

Create a simplified notebook that demonstrates each phase on a single instance.

---

## Summary

This plan reorganizes the project into:

1. **Three Phase Scripts** - Predict, Evaluate, Learn - each runnable standalone
2. **Main Loop** - Orchestrates the three phases with retry logic
3. **IO Module** - Clean data loading/saving with new directory structure
4. **CLI Commands** - Easy command-line access to all functionality
5. **Updated Documentation** - README and demo notebook

### Output Directory Structure

```
data/
└── run_20260319_143052/              # run_<compact_timestamp>
    ├── config.json                    # Config used for this run
    ├── statistics.json                # Counts, resolved/unresolved lists, skills
    ├── experiment.log                 # Main log file
    └── princeton-nlp__SWE-bench_Lite/                 # Benchmark from config
        ├── trajectories/
        │   └── django__django-12345/
        │       ├── iter_0.json
        │       └── iter_1.json
        ├── results/
        │   └── django__django-12345/
        │       ├── iter_0.json
        │       └── iter_1.json
        └── skillbooks/
            # Per-instance mode (default):
            └── django__django-12345/
                ├── iter_0.json        # Empty (initial)
                └── iter_1.json        # After learning from iter_0 failure
            # Per-run mode:
            └── iter_0.json            # Empty
            └── iter_1.json            # Accumulated skills
```

### Baseline Data Conversion

A separate script will convert existing baseline trajectories from:
```
data/baseline_trajectories/swebench-lite/qwen3_coder_30ba3b/
    └── django__django-12345/django__django-12345.traj.json
```

To the new format:
```
data/run_baseline_qwen3coder/
    ├── config.json
    ├── statistics.json
    └── swebench-lite/
        └── trajectories/
            └── django__django-12345/
                └── iter_0.json
```
