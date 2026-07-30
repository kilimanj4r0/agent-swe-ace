"""Unit tests for scripts/plot_r1.py pure data functions.

Covers the per-instance resolution / skill-count math that feeds the R1 figures:
pass@k cumulative curves, first-resolved-attempt, within-run lift (Δ), and
skillbook-size extraction. Plotting code itself is not unit-tested (visual).

The script lives in scripts/ (not a package), so load it by path via importlib.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "plot_r1.py"
_spec = importlib.util.spec_from_file_location("plot_r1", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass(frozen=True) can resolve cls.__module__.
sys.modules["plot_r1"] = mod
_spec.loader.exec_module(mod)

resolved_within_k = mod.resolved_within_k
passk_curve = mod.passk_curve
first_resolved_at = mod.first_resolved_at
per_instance_delta = mod.per_instance_delta
final_skill_sizes = mod.final_skill_sizes
deployed_skill_counts = mod.deployed_skill_counts
load_resolved_per_attempt = mod.load_resolved_per_attempt
load_skill_counts = mod.load_skill_counts


# ---------------------------------------------------------------------------
# resolved_within_k / first_resolved_at
# ---------------------------------------------------------------------------

class TestResolvedHelpers:
    def test_resolved_within_k_basic(self):
        attempts = [False, True, False]
        assert resolved_within_k(attempts, 1) is False
        assert resolved_within_k(attempts, 2) is True
        assert resolved_within_k(attempts, 3) is True

    def test_resolved_within_k_first_attempt(self):
        assert resolved_within_k([True, False], 1) is True

    def test_resolved_within_k_never(self):
        assert resolved_within_k([False, False, False], 3) is False

    def test_resolved_within_k_k_beyond_length(self):
        # k longer than the attempt list only sees the attempts that exist
        assert resolved_within_k([False, True], 99) is True
        assert resolved_within_k([False], 99) is False

    def test_first_resolved_returns_one_indexed(self):
        assert first_resolved_at([False, True, False]) == 2
        assert first_resolved_at([True]) == 1
        assert first_resolved_at([False, False, True]) == 3

    def test_first_resolved_none_when_unresolved(self):
        assert first_resolved_at([False, False]) is None
        assert first_resolved_at([]) is None


# ---------------------------------------------------------------------------
# passk_curve
# ---------------------------------------------------------------------------

class TestPasskCurve:
    def test_curve_matches_cumulative_resolution(self):
        run = {
            "a": [False, True],   # first solved at attempt 2
            "b": [True],          # solved at attempt 1
            "c": [False, False],  # never solved
        }
        curve = passk_curve(run, max_k=2)
        # k=1: only b -> 1/3 ; k=2: a and b -> 2/3
        np.testing.assert_allclose(curve, [1 / 3, 2 / 3])

    def test_curve_caps_at_max_k(self):
        run = {"a": [False, True, True]}  # would be solved at attempt 2
        curve = passk_curve(run, max_k=1)
        np.testing.assert_allclose(curve, [0.0])

    def test_curve_all_resolved_first_attempt(self):
        run = {"a": [True], "b": [True, False]}
        np.testing.assert_allclose(passk_curve(run, max_k=2), [1.0, 1.0])

    def test_curve_empty_run(self):
        np.testing.assert_allclose(passk_curve({}, max_k=3), [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# per_instance_delta (within-run lift, for bootstrap / t-test)
# ---------------------------------------------------------------------------

class TestPerInstanceDelta:
    def test_delta_arrays_match_passk_lift(self):
        run = {"a": [False, True], "b": [True], "c": [False, False]}
        any_, i0 = per_instance_delta(run, max_k=2)
        np.testing.assert_array_equal(any_, [1, 1, 0])
        np.testing.assert_array_equal(i0, [0, 1, 0])
        # mean(any_ - i0) == pass@N - pass@1
        lift = passk_curve(run, max_k=2)
        assert any_.mean() - i0.mean() == pytest.approx(lift[-1] - lift[0])

    def test_delta_respects_max_k_window(self):
        # solved at attempt 3 only -> not within max_k=2
        run = {"a": [False, False, True]}
        any_, i0 = per_instance_delta(run, max_k=2)
        assert any_.tolist() == [0]
        assert i0.tolist() == [0]


# ---------------------------------------------------------------------------
# final_skill_sizes
# ---------------------------------------------------------------------------

class TestFinalSkillSizes:
    def test_takes_last_iter_count(self):
        sc = {"a": [5, 6, 7], "b": [3]}
        assert final_skill_sizes(sc) == {"a": 7, "b": 3}

    def test_falls_back_to_skill_list_length(self):
        sc = {"a": [5]}
        assert final_skill_sizes(sc) == {"a": 5}

    def test_empty(self):
        assert final_skill_sizes({}) == {}


# ---------------------------------------------------------------------------
# deployed_skill_counts (drop the unused trailing iteration)
# ---------------------------------------------------------------------------

class TestDeployedSkillCounts:
    def test_drops_trailing_unused_iter(self):
        # 4-attempt run: iter_4 is learned after the 4th (final) attempt and
        # never deployed (no 5th attempt) -> dropped. Shorter instance unchanged.
        sc = {"a": [1, 2, 3, 4], "b": [5, 6]}
        assert deployed_skill_counts(sc, attempts=4) == {"a": [1, 2, 3], "b": [5, 6]}

    def test_no_trailing_iter_unchanged(self):
        assert deployed_skill_counts({"a": [1, 2, 3]}, attempts=4) == {"a": [1, 2, 3]}

    def test_six_attempt_drops_iter6(self):
        assert deployed_skill_counts({"a": [1, 2, 3, 4, 5, 6]}, attempts=6) == \
            {"a": [1, 2, 3, 4, 5]}

    def test_one_attempt_no_deployed(self):
        # 1 attempt: the only book used is the empty initial one; iter_1 trailing.
        assert deployed_skill_counts({"a": [1]}, attempts=1) == {"a": []}

    def test_empty(self):
        assert deployed_skill_counts({}, attempts=4) == {}


# ---------------------------------------------------------------------------
# Filesystem loaders (flat layout; find_benchmark_dir falls back to run_dir)
# ---------------------------------------------------------------------------

def _write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class TestLoaders:
    def test_load_resolved_per_attempt_flat_layout(self, tmp_path):
        _write(tmp_path / "results" / "inst1" / "iter_0.json", {"resolved": False})
        _write(tmp_path / "results" / "inst1" / "iter_1.json", {"resolved": True})
        _write(tmp_path / "results" / "inst2" / "iter_0.json", {"resolved": True})
        out = load_resolved_per_attempt(tmp_path)
        assert out == {"inst1": [False, True], "inst2": [True]}

    def test_load_resolved_handles_missing_field(self, tmp_path):
        _write(tmp_path / "results" / "inst1" / "iter_0.json", {"instance_id": "inst1"})
        out = load_resolved_per_attempt(tmp_path)
        assert out == {"inst1": [False]}

    def test_load_resolved_benchmark_scoped_layout(self, tmp_path):
        bench = tmp_path / "princeton-nlp__SWE-bench_Verified"
        _write(bench / "results" / "inst1" / "iter_0.json", {"resolved": True})
        _write(bench / "results" / "inst1" / "iter_1.json", {"resolved": False})
        out = load_resolved_per_attempt(tmp_path)
        assert out == {"inst1": [True, False]}

    def test_load_skill_counts_flat_layout(self, tmp_path):
        _write(tmp_path / "skillbooks" / "inst1" / "iter_1.json", {"skill_count": 5})
        _write(tmp_path / "skillbooks" / "inst1" / "iter_2.json", {"skill_count": 6})
        _write(tmp_path / "skillbooks" / "inst2" / "iter_1.json", {"skill_count": 2})
        out = load_skill_counts(tmp_path)
        assert out == {"inst1": [5, 6], "inst2": [2]}

    def test_load_skill_counts_uses_skills_len_when_no_count(self, tmp_path):
        _write(tmp_path / "skillbooks" / "inst1" / "iter_1.json",
               {"skills": [{"id": 1}, {"id": 2}, {"id": 3}]})
        out = load_skill_counts(tmp_path)
        assert out == {"inst1": [3]}
