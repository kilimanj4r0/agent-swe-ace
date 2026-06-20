# src/tests/test_resume_scanner.py
"""Tests for resume_scanner — chain-walking logic for experiment resumption."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_io.resume_scanner import (
    ResumePoint,
    _detect_resume_phase,
    scan_resume_state,
    scan_resume_dirs,
    copy_instance_artifacts,
)

BENCHMARK = "princeton-nlp__SWE-bench_Lite"
INSTANCE = "django__django-12345"


def _write_iter(
    base_dir: Path,
    subdir: str,
    instance_id: str,
    iteration: int,
    data: dict,
):
    """Helper to write an iteration file."""
    d = base_dir / BENCHMARK / subdir / instance_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"iter_{iteration}.json").write_text(json.dumps(data))


def _make_trajectory(exit_status: str, messages=None) -> dict:
    return {"info": {"exit_status": exit_status}, "messages": messages or []}


def _make_result(resolved: bool) -> dict:
    return {"resolved": resolved}


def _make_skillbook(skills=None) -> dict:
    return {"skills": skills or []}


class TestScanResumeState:
    """Tests for scan_resume_state chain-walking logic."""

    def test_instance_not_found_returns_none(self, tmp_path):
        """If instance has no trajectory dir, return None."""
        result = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4)
        assert result is None

    def test_empty_trajectory_dir_returns_none(self, tmp_path):
        """If trajectory dir exists but has no files, return None."""
        (tmp_path / BENCHMARK / "trajectories" / INSTANCE).mkdir(parents=True)
        result = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4)
        assert result is None

    def test_iter0_resolved_is_fully_complete(self, tmp_path):
        """iter_0 resolved → is_fully_complete=True, last_complete_iter=0."""
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results", INSTANCE, 0, _make_result(resolved=True))

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4)
        assert rp is not None
        assert rp.is_fully_complete is True
        assert rp.last_complete_iter == 0
        assert rp.start_iteration == -1

    def test_iter0_not_resolved_no_skillbook_chain_breaks(self, tmp_path):
        """iter_0 not resolved + no skillbook → last_complete_iter=-1."""
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results", INSTANCE, 0, _make_result(resolved=False))
        # No skillbook for iter_1

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4)
        assert rp is not None
        assert rp.last_complete_iter == -1
        assert rp.is_fully_complete is False
        assert rp.start_iteration == 0

    def test_two_iters_second_resolved(self, tmp_path):
        """iter_0 ok + iter_1 resolved → is_fully_complete=True."""
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results", INSTANCE, 0, _make_result(resolved=False))
        _write_iter(tmp_path, "skillbooks", INSTANCE, 1, _make_skillbook())
        _write_iter(tmp_path, "trajectories", INSTANCE, 1, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results", INSTANCE, 1, _make_result(resolved=True))

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4)
        assert rp.is_fully_complete is True
        assert rp.last_complete_iter == 1

    def test_partial_chain_two_ok_then_fail(self, tmp_path):
        """iter_0 ok, iter_1 ok, iter_2 fail → last_complete_iter=1."""
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results", INSTANCE, 0, _make_result(resolved=False))
        _write_iter(tmp_path, "skillbooks", INSTANCE, 1, _make_skillbook())

        _write_iter(tmp_path, "trajectories", INSTANCE, 1, _make_trajectory("LimitsExceeded"))
        _write_iter(tmp_path, "results", INSTANCE, 1, _make_result(resolved=False))
        _write_iter(tmp_path, "skillbooks", INSTANCE, 2, _make_skillbook())

        # iter_2 has bad exit_status → chain breaks
        _write_iter(tmp_path, "trajectories", INSTANCE, 2, _make_trajectory("Error"))
        _write_iter(tmp_path, "results", INSTANCE, 2, _make_result(resolved=False))

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4)
        assert rp.is_fully_complete is False
        assert rp.last_complete_iter == 1
        assert rp.start_iteration == 2

    def test_all_attempts_exhausted_is_complete(self, tmp_path):
        """All max_attempts exhausted without resolving → is_fully_complete=True."""
        max_attempts = 2
        for k in range(max_attempts):
            _write_iter(tmp_path, "trajectories", INSTANCE, k, _make_trajectory("Submitted"))
            _write_iter(tmp_path, "results", INSTANCE, k, _make_result(resolved=False))
            if k < max_attempts - 1:
                _write_iter(tmp_path, "skillbooks", INSTANCE, k + 1, _make_skillbook())

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=max_attempts)
        assert rp.is_fully_complete is True
        assert rp.last_complete_iter == 1

    def test_bad_exit_status_breaks_chain(self, tmp_path):
        """Bad exit_status (e.g. 'Error') breaks the chain."""
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("Error"))
        _write_iter(tmp_path, "results", INSTANCE, 0, _make_result(resolved=False))

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4)
        assert rp.last_complete_iter == -1

    def test_bad_json_breaks_chain(self, tmp_path):
        """Malformed JSON in trajectory file breaks the chain."""
        d = tmp_path / BENCHMARK / "trajectories" / INSTANCE
        d.mkdir(parents=True)
        (d / "iter_0.json").write_text("{bad json")

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4)
        assert rp.last_complete_iter == -1

    def test_missing_result_file_breaks_chain(self, tmp_path):
        """Missing result file breaks the chain even if trajectory is OK."""
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("Submitted"))
        # No result file

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4)
        assert rp.last_complete_iter == -1

    def test_limits_exceeded_is_good_exit(self, tmp_path):
        """LimitsExceeded is a good exit status (not a failure)."""
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("LimitsExceeded"))
        _write_iter(tmp_path, "results", INSTANCE, 0, _make_result(resolved=True))

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4)
        assert rp.is_fully_complete is True
        assert rp.last_complete_iter == 0

    def test_context_window_exceeded_is_good_exit(self, tmp_path):
        """ContextWindowExceeded is a good exit status — chain doesn't break on it."""
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("ContextWindowExceeded"))
        _write_iter(tmp_path, "results", INSTANCE, 0, _make_result(resolved=True))

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4)
        assert rp.is_fully_complete is True
        assert rp.last_complete_iter == 0

    def test_context_window_exceeded_single_attempt_complete(self, tmp_path):
        """ContextWindowExceeded with max_attempts=1 is fully complete."""
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("ContextWindowExceeded"))
        _write_iter(tmp_path, "results", INSTANCE, 0, _make_result(resolved=False))

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=1)
        assert rp.is_fully_complete is True
        assert rp.last_complete_iter == 0


class TestScanResumeDirs:
    """Tests for scan_resume_dirs — multi-directory merging."""

    def test_picks_best_directory_per_instance(self, tmp_path):
        """When instance exists in multiple dirs, highest last_complete_iter wins."""
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"

        # dir1: iter_0 resolved
        _write_iter(dir1, "trajectories", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(dir1, "results", INSTANCE, 0, _make_result(resolved=True))

        # dir2: iter_0 not resolved, iter_1 resolved (longer chain)
        _write_iter(dir2, "trajectories", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(dir2, "results", INSTANCE, 0, _make_result(resolved=False))
        _write_iter(dir2, "skillbooks", INSTANCE, 1, _make_skillbook())
        _write_iter(dir2, "trajectories", INSTANCE, 1, _make_trajectory("Submitted"))
        _write_iter(dir2, "results", INSTANCE, 1, _make_result(resolved=True))

        result = scan_resume_dirs([dir1, dir2], BENCHMARK, [INSTANCE], max_attempts=4)
        assert INSTANCE in result
        assert result[INSTANCE].last_complete_iter == 1
        assert result[INSTANCE].resume_dir == dir2

    def test_missing_directory_skipped(self, tmp_path):
        """Non-existent directories are skipped without error."""
        missing = tmp_path / "nonexistent"
        result = scan_resume_dirs([missing], BENCHMARK, [INSTANCE], max_attempts=4)
        assert INSTANCE not in result


class TestCopyInstanceArtifacts:
    """Tests for copy_instance_artifacts."""

    def test_copies_trajectories_and_results_up_to_iter(self, tmp_path):
        """Copies iter_0..iter_N for trajectories and results."""
        source = tmp_path / "source"
        dest = tmp_path / "dest"

        for subdir in ("trajectories", "results"):
            for i in range(3):
                _write_iter(source, subdir, INSTANCE, i, {"data": i})

        copy_instance_artifacts(source, dest, BENCHMARK, INSTANCE, up_to_iter=1)

        # iter_0 and iter_1 should be copied
        for subdir in ("trajectories", "results"):
            for i in range(2):
                f = dest / BENCHMARK / subdir / INSTANCE / f"iter_{i}.json"
                assert f.exists(), f"Missing {subdir}/iter_{i}.json"
            # iter_2 should NOT be copied
            f = dest / BENCHMARK / subdir / INSTANCE / "iter_2.json"
            assert not f.exists(), f"Unexpected {subdir}/iter_2.json"

    def test_copies_skillbooks_up_to_iter_plus_one(self, tmp_path):
        """Skillbooks are copied for iter_0..iter_{N+1} (learn phase output)."""
        source = tmp_path / "source"
        dest = tmp_path / "dest"

        for i in range(3):
            _write_iter(source, "skillbooks", INSTANCE, i, {"skills": []})

        copy_instance_artifacts(source, dest, BENCHMARK, INSTANCE, up_to_iter=1)

        # iter_0, iter_1, iter_2 (= up_to_iter + 1) should be copied
        for i in range(3):
            f = dest / BENCHMARK / "skillbooks" / INSTANCE / f"iter_{i}.json"
            assert f.exists(), f"Missing skillbooks/iter_{i}.json"

    def test_creates_subdirectories(self, tmp_path):
        """Creates destination subdirectories if they don't exist."""
        source = tmp_path / "source"
        dest = tmp_path / "dest"

        _write_iter(source, "trajectories", INSTANCE, 0, {"data": 0})

        copy_instance_artifacts(source, dest, BENCHMARK, INSTANCE, up_to_iter=0)

        assert (dest / BENCHMARK / "trajectories" / INSTANCE).is_dir()

    def test_skips_non_json_files(self, tmp_path):
        """Non-JSON files in instance directories are ignored."""
        source = tmp_path / "source"
        dest = tmp_path / "dest"

        d = source / BENCHMARK / "trajectories" / INSTANCE
        d.mkdir(parents=True)
        (d / "iter_0.json").write_text("{}")
        (d / "notes.txt").write_text("ignore me")

        copy_instance_artifacts(source, dest, BENCHMARK, INSTANCE, up_to_iter=0)

        assert (dest / BENCHMARK / "trajectories" / INSTANCE / "iter_0.json").exists()
        assert not (dest / BENCHMARK / "trajectories" / INSTANCE / "notes.txt").exists()

    def test_skip_learn_no_skillbook_required(self, tmp_path):
        """With skip_learn=True, unresolved iteration chain continues without skillbook files."""
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results", INSTANCE, 0, _make_result(resolved=False))
        # No skillbook file for iter_1 — normally breaks the chain

        # Without skip_learn: chain breaks at iter_0 (no skillbook)
        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4)
        assert rp is not None
        assert rp.last_complete_iter == -1  # broken chain

        # With skip_learn: chain continues, skillbook not required
        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4, skip_learn=True)
        assert rp is not None
        assert rp.last_complete_iter == 0  # iter_0 is complete
        assert rp.is_fully_complete is False

    def test_skip_learn_resolved_still_detected(self, tmp_path):
        """With skip_learn=True, resolved instances are still fully complete."""
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results", INSTANCE, 0, _make_result(resolved=True))
        # No skillbook file — but resolved, so should be fully complete either way

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4, skip_learn=True)
        assert rp is not None
        assert rp.is_fully_complete is True

    def test_skip_learn_multi_iter_chain(self, tmp_path):
        """With skip_learn=True, multi-iteration chain works without skillbooks."""
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results", INSTANCE, 0, _make_result(resolved=False))
        _write_iter(tmp_path, "trajectories", INSTANCE, 1, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results", INSTANCE, 1, _make_result(resolved=False))
        # No skillbook files at all

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4, skip_learn=True)
        assert rp is not None
        assert rp.last_complete_iter == 1
        assert rp.is_fully_complete is False


class TestDetectResumePhase:
    """Tests for _detect_resume_phase helper."""

    def test_train_phase_detected(self, tmp_path):
        _write_iter(tmp_path, "trajectories/train", INSTANCE, 0, _make_trajectory("Submitted"))
        assert _detect_resume_phase(tmp_path, BENCHMARK, INSTANCE) == "train"

    def test_val_baseline_phase_detected(self, tmp_path):
        _write_iter(tmp_path, "trajectories/val_baseline", INSTANCE, 0, _make_trajectory("Submitted"))
        assert _detect_resume_phase(tmp_path, BENCHMARK, INSTANCE) == "val_baseline"

    def test_val_phase_detected(self, tmp_path):
        _write_iter(tmp_path, "trajectories/val", INSTANCE, 0, _make_trajectory("Submitted"))
        assert _detect_resume_phase(tmp_path, BENCHMARK, INSTANCE) == "val"

    def test_flat_layout_returns_none(self, tmp_path):
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("Submitted"))
        assert _detect_resume_phase(tmp_path, BENCHMARK, INSTANCE) is None

    def test_not_found_returns_none(self, tmp_path):
        assert _detect_resume_phase(tmp_path, BENCHMARK, INSTANCE) is None

    def test_train_priority_over_val(self, tmp_path):
        """train takes priority over val_baseline and val."""
        for phase in ("train", "val_baseline", "val"):
            _write_iter(tmp_path, f"trajectories/{phase}", INSTANCE, 0, _make_trajectory("Submitted"))
        assert _detect_resume_phase(tmp_path, BENCHMARK, INSTANCE) == "train"

    def test_val_baseline_priority_over_val(self, tmp_path):
        """val_baseline takes priority over val."""
        for phase in ("val_baseline", "val"):
            _write_iter(tmp_path, f"trajectories/{phase}", INSTANCE, 0, _make_trajectory("Submitted"))
        assert _detect_resume_phase(tmp_path, BENCHMARK, INSTANCE) == "val_baseline"


class TestPhaseAwareScanResumeState:
    """Tests for scan_resume_state with phase parameter."""

    def test_phase_param_finds_train_subdir(self, tmp_path):
        """With phase='train', looks in trajectories/train/<instance>/."""
        _write_iter(tmp_path, "trajectories/train", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results/train", INSTANCE, 0, _make_result(resolved=True))

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4, phase="train")
        assert rp is not None
        assert rp.is_fully_complete is True
        assert rp.last_complete_iter == 0

    def test_phase_param_finds_val_subdir(self, tmp_path):
        """With phase='val', looks in trajectories/val/<instance>/."""
        _write_iter(tmp_path, "trajectories/val", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results/val", INSTANCE, 0, _make_result(resolved=False))

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4, phase="val")
        assert rp is not None
        assert rp.is_fully_complete is False

    def test_phase_param_not_found_without_matching_data(self, tmp_path):
        """With phase='train' but data only in flat layout → None."""
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results", INSTANCE, 0, _make_result(resolved=True))

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=4, phase="train")
        assert rp is None

    def test_phase_param_val_baseline(self, tmp_path):
        """With phase='val_baseline', finds data in val_baseline subdir."""
        _write_iter(tmp_path, "trajectories/val_baseline", INSTANCE, 0, _make_trajectory("LimitsExceeded"))
        _write_iter(tmp_path, "results/val_baseline", INSTANCE, 0, _make_result(resolved=False))

        rp = scan_resume_state(tmp_path, BENCHMARK, INSTANCE, max_attempts=1, phase="val_baseline")
        assert rp is not None
        assert rp.is_fully_complete is True  # max_attempts=1, exhausted


class TestPhaseAwareScanResumeDirs:
    """Tests for scan_resume_dirs auto-detecting phase from two-phase layout."""

    def test_auto_detects_train_phase(self, tmp_path):
        _write_iter(tmp_path, "trajectories/train", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results/train", INSTANCE, 0, _make_result(resolved=True))

        result = scan_resume_dirs([tmp_path], BENCHMARK, [INSTANCE], max_attempts=4)
        assert INSTANCE in result
        assert result[INSTANCE].phase == "train"
        assert result[INSTANCE].is_fully_complete is True

    def test_auto_detects_val_phase(self, tmp_path):
        _write_iter(tmp_path, "trajectories/val", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results/val", INSTANCE, 0, _make_result(resolved=True))

        result = scan_resume_dirs([tmp_path], BENCHMARK, [INSTANCE], max_attempts=4)
        assert result[INSTANCE].phase == "val"

    def test_flat_layout_phase_is_none(self, tmp_path):
        _write_iter(tmp_path, "trajectories", INSTANCE, 0, _make_trajectory("Submitted"))
        _write_iter(tmp_path, "results", INSTANCE, 0, _make_result(resolved=True))

        result = scan_resume_dirs([tmp_path], BENCHMARK, [INSTANCE], max_attempts=4)
        assert result[INSTANCE].phase is None


class TestPhaseAwareCopyArtifacts:
    """Tests for copy_instance_artifacts with source_phase/dest_phase."""

    def test_copy_from_train_to_train(self, tmp_path):
        source = tmp_path / "source"
        dest = tmp_path / "dest"
        _write_iter(source, "trajectories/train", INSTANCE, 0, {"data": 0})
        _write_iter(source, "results/train", INSTANCE, 0, {"data": 0})

        copy_instance_artifacts(
            source, dest, BENCHMARK, INSTANCE,
            up_to_iter=0, source_phase="train", dest_phase="train",
        )

        assert (dest / BENCHMARK / "trajectories" / "train" / INSTANCE / "iter_0.json").exists()
        assert (dest / BENCHMARK / "results" / "train" / INSTANCE / "iter_0.json").exists()

    def test_copy_from_val_to_val(self, tmp_path):
        source = tmp_path / "source"
        dest = tmp_path / "dest"
        _write_iter(source, "trajectories/val", INSTANCE, 0, {"data": 0})
        _write_iter(source, "results/val", INSTANCE, 0, {"data": 0})

        copy_instance_artifacts(
            source, dest, BENCHMARK, INSTANCE,
            up_to_iter=0, source_phase="val", dest_phase="val",
        )

        assert (dest / BENCHMARK / "trajectories" / "val" / INSTANCE / "iter_0.json").exists()

    def test_no_phase_uses_flat_paths(self, tmp_path):
        source = tmp_path / "source"
        dest = tmp_path / "dest"
        _write_iter(source, "trajectories", INSTANCE, 0, {"data": 0})

        copy_instance_artifacts(source, dest, BENCHMARK, INSTANCE, up_to_iter=0)

        assert (dest / BENCHMARK / "trajectories" / INSTANCE / "iter_0.json").exists()

    def test_source_phase_not_found_skips_gracefully(self, tmp_path):
        source = tmp_path / "source"
        dest = tmp_path / "dest"
        # No data in source
        copy_instance_artifacts(
            source, dest, BENCHMARK, INSTANCE,
            up_to_iter=0, source_phase="train", dest_phase="train",
        )
        # Should not create anything
        assert not (dest / BENCHMARK).exists()
