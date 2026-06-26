"""Tests for scripts/analyze_skill_references.py (stdlib-only script)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # project root
SCRIPT = ROOT / "scripts" / "analyze_skill_references.py"


@pytest.fixture(scope="module")
def srf():
    """Load the standalone script module by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("analyze_skill_references", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_skill_references"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_traj(tmp_path: Path, messages: list[dict], info: dict | None = None) -> Path:
    traj = tmp_path / "trajectories" / "val" / "django__django-1" / "iter_1.json"
    traj.parent.mkdir(parents=True)
    traj.write_text(json.dumps({
        "info": info or {"instance_id": "django__django-1", "iteration": 1},
        "messages": messages,
    }))
    return traj


def test_module_loads(srf):
    assert hasattr(srf, "main")
    assert hasattr(srf, "scan_trajectory")
    assert hasattr(srf, "load_skillbook_ids")


def test_bracket_refs_still_detected(srf, tmp_path):
    traj = _write_traj(tmp_path, [
        {"role": "user", "content": "### testing-00001\n\nDo thing."},
        {"role": "assistant", "content": "Using [skill-id: testing-00001] now."},
    ])
    hits, injected, prose, prose_refs = srf.scan_trajectory(traj, "run_x")
    assert len(hits) == 1
    assert hits[0].skill_id == "skill-testing-00001"
    assert {i.skill_id for i in injected} == {"testing-00001"}
    # bracket reference must NOT also be counted as a prose mention or skill-ref
    assert prose == []
    assert prose_refs == []


def test_prose_word_skill_engagement_detected(srf, tmp_path):
    traj = _write_traj(tmp_path, [
        {"role": "user", "content": "### query_handling-00199\n\nResolve."},
        {"role": "assistant", "content": "Let me check the skills list for a pattern."},
    ])
    hits, injected, prose, prose_refs = srf.scan_trajectory(traj, "run_x")
    assert hits == []
    assert len(prose) == 1
    # "skills list" mentions no specific skill ID nearby
    assert prose_refs == []


def test_prose_ref_classified_presented(srf, tmp_path):
    traj = _write_traj(tmp_path, [
        {"role": "user", "content": "### testing-00035\n\nResolve."},
        {"role": "assistant", "content": "Looking at skill testing-00035 to validate Quantity."},
    ])
    _, _, _, prose_refs = srf.scan_trajectory(traj, "run_x", skillbook_ids={"testing-00035"})
    refs = [r for r in prose_refs if r.skill_id == "testing-00035"]
    assert refs and all(r.classification == "presented" for r in refs)


def test_prose_ref_classified_skillbook_when_not_injected(srf, tmp_path):
    """The debugging-00190 case: referenced in prose but not injected -> 'skillbook'."""
    traj = _write_traj(tmp_path, [
        {"role": "user", "content": "### testing-00035\n\nResolve."},  # only this injected
        {"role": "assistant", "content": "Looking at skill debugging-00190 for guidance."},
    ])
    _, _, _, prose_refs = srf.scan_trajectory(
        traj, "run_x", skillbook_ids={"debugging-00190", "testing-00035"})
    refs = [r for r in prose_refs if r.skill_id == "debugging-00190"]
    assert refs, "referenced-but-not-injected skill must NOT be missed"
    assert all(r.classification == "skillbook" for r in refs)


def test_prose_ref_classified_lookalike_for_non_skill_section(srf, tmp_path):
    """A token whose prefix isn't a skill section (e.g. bpo-43882) -> 'lookalike'."""
    traj = _write_traj(tmp_path, [
        {"role": "user", "content": "### testing-00035\n\nResolve."},
        {"role": "assistant", "content": "The bpo-43882 fix changes URL splitting."},
    ])
    _, _, _, prose_refs = srf.scan_trajectory(traj, "run_x", skillbook_ids={"testing-00035"})
    refs = [r for r in prose_refs if r.skill_id == "bpo-43882"]
    assert refs and all(r.classification == "lookalike" for r in refs)


def test_prose_ref_buckets_non_skill_lookalikes(srf, tmp_path):
    """Look-alikes (bpo-43882, well-11909) are captured as 'lookalike', never missed."""
    traj = _write_traj(tmp_path, [
        {"role": "user", "content": "### testing-00035\n\nResolve."},
        {"role": "assistant", "content": "The bpo-43882 fix and as well-11909 change behavior."},
    ])
    _, _, _, prose_refs = srf.scan_trajectory(
        traj, "run_x", skillbook_ids={"testing-00035", "debugging-00001"})
    classified = {r.skill_id: r.classification for r in prose_refs}
    assert classified.get("bpo-43882") == "lookalike"  # 'bpo' is not a skill section
    assert classified.get("well-11909") == "lookalike"  # 'well' is not a skill section


def test_prose_ref_keeps_miscited_real_section_as_unknown(srf, tmp_path):
    """debugging-00190 (valid section, no such skill) is kept and flagged unknown."""
    traj = _write_traj(tmp_path, [
        {"role": "user", "content": "### testing-00035\n\nResolve."},
        {"role": "assistant", "content": "Looking at skill debugging-00190 for guidance."},
    ])
    _, _, _, prose_refs = srf.scan_trajectory(
        traj, "run_x", skillbook_ids={"testing-00035", "debugging-00001"})
    refs = [r for r in prose_refs if r.skill_id == "debugging-00190"]
    assert refs and all(r.classification == "unknown" for r in refs)


def test_instance_ids_not_matched_as_skill_refs(srf, tmp_path):
    """Instance IDs like django__django-12345 must not be captured as skill refs."""
    traj = _write_traj(tmp_path, [
        {"role": "user", "content": "### testing-00035\n\nResolve."},
        {"role": "assistant", "content": "The issue django__django-12345 needs a fix in foo-12345."},
    ])
    _, _, _, prose_refs = srf.scan_trajectory(traj, "run_x", skillbook_ids=set())
    ids = {r.skill_id for r in prose_refs}
    assert "django-12345" not in ids  # negative-lookbehind blocks embedded instance ID


def test_prose_disabled(srf, tmp_path):
    traj = _write_traj(tmp_path, [
        {"role": "user", "content": "### testing-00001\n\nDo thing."},
        {"role": "assistant", "content": "Checking the skills list; using testing-00001 now."},
    ])
    hits, injected, prose, prose_refs = srf.scan_trajectory(traj, "run_x", include_prose=False)
    assert prose == []
    assert prose_refs == []


def test_prose_regex_word_boundary():
    rx = __import__("re").compile(r"\bskills?(?:book)?\b", __import__("re").IGNORECASE)
    assert rx.findall("skill skills skillbook Skill list")
    assert rx.search("be skillful") is None
    assert rx.search("my skillset") is None


def test_skill_id_token_regex_shapes():
    rx = __import__("re").compile(r"(?<![A-Za-z_])([a-z]+(?:_[a-z]+)*-[0-9]{5})",
                                 __import__("re").IGNORECASE)
    assert rx.search("see `debugging-00190`").group(1) == "debugging-00190"
    assert rx.search("query_handling-00199 ok").group(1) == "query_handling-00199"
    assert rx.search("regression_prevention-00146").group(1) == "regression_prevention-00146"
    # embedded instance-id substring is blocked by the lookbehind
    assert rx.search("django__django-12345") is None


def test_load_skillbook_ids(srf, tmp_path):
    run = tmp_path / "run_x" / "princeton-nlp__SWE-bench_Verified" / "skillbooks"
    run.mkdir(parents=True)
    (run / "final_skillbook.json").write_text(json.dumps({
        "skills": [
            {"id": "debugging-00001", "content": "a"},
            {"id": "testing-00035", "content": "b"},
        ]}))
    ids = srf.load_skillbook_ids(tmp_path / "run_x")
    assert ids == {"debugging-00001", "testing-00035"}
