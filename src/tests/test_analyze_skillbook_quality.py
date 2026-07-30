"""Unit tests for scripts/analyze_skillbook_quality.py pure functions.

Covers two bugs found while re-verifying the per-instance citation analysis:

1. ``_extract_presented_skills`` dropped skills whose SECTION name contains a
   hyphen (e.g. ``bug-fixing-00001``, ``code-analysis-00002``) because the
   heading regex ``[a-z_]+-\\d+`` excludes ``-``. This undercounts presented
   skills and citations for any learn run that emits hyphenated sections.

2. ``_compute_refs_per_instance`` paired a skillbook FILE ``iter_N`` with the
   trajectory FILE ``iter_N``. That correspondence is not a clean same-N mapping
   (attempt 1 has no skillbook; the book shown in trajectory ``iter_N`` is the
   one learned after the previous attempt, and resume / early-resolve make the
   file numbering drift). The robust fix reads the skills out of each
   trajectory's OWN prompt, so each attempt's citations are checked against
   exactly what the agent was shown.
"""

import importlib.util
import json
from pathlib import Path

# The analysis script lives in scripts/ (not a package), so load it by path.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_skillbook_quality.py"
_spec = importlib.util.spec_from_file_location("analyze_skillbook_quality", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

extract_presented_skills = _mod._extract_presented_skills
compute_refs_per_instance = _mod._compute_refs_per_instance


def _traj_with_book(skill_ids, assistant_text, *, intro="Use them to guide your approach."):
    """Build a trajectory whose user prompt shows the given skills and whose
    assistant message is ``assistant_text``."""
    sections = "\n\n".join(f"### {sid}\n\nbody of {sid}" for sid in skill_ids)
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Learned Strategies (Skillbook)\n\n{intro}\n\n"
                    f"{sections}\n\nCRITICAL REMINDER: be careful"
                ),
            },
            {"role": "assistant", "content": assistant_text},
        ]
    }


# --- Bug 1: hyphenated section names ---------------------------------------


def test_extract_presented_skills_handles_hyphenated_sections():
    traj = _traj_with_book(
        ["bug-fixing-00001", "code-analysis-00002", "robustness-00003"],
        "working",
    )
    skills = extract_presented_skills(traj)
    ids = [s["id"] for s in skills]
    assert set(ids) == {"bug-fixing-00001", "code-analysis-00002", "robustness-00003"}
    # section is everything before the last hyphen-digits
    assert skills[0]["section"] == "bug-fixing"


def test_extract_presented_skills_still_handles_underscore_sections():
    traj = _traj_with_book(
        ["array_operations-00001", "debugging-00002"],
        "working",
    )
    ids = [s["id"] for s in extract_presented_skills(traj)]
    assert set(ids) == {"array_operations-00001", "debugging-00002"}


# --- Bug 2: per-instance refs read each trajectory's own prompt ------------


def test_per_instance_refs_count_only_book_showing_attempts():
    """Attempt with no skillbook section must not be counted; the book-showing
    attempt that cites a skill must be."""
    trajectories = {
        "inst-1": {
            0: {"messages": [  # attempt 1: no skillbook shown
                {"role": "user", "content": "just the task, no strategies"},
                {"role": "assistant", "content": "let me look"},
            ]},
            1: _traj_with_book(  # attempt 2: book shown, cites the skill
                ["bug-fixing-00001"],
                "I'll apply [skill-bug-fixing-00001] now",
            ),
        }
    }
    res = compute_refs_per_instance({"per_instance_sbs": {}}, trajectories)
    assert res["summary"]["traj_iters"] == 1          # only the book-showing attempt
    assert res["summary"]["cite_traj_iters"] == 1     # that attempt cited a skill


def test_per_instance_refs_detect_citation_against_prompt_skills():
    """Citation detection must use the IDs presented in THAT trajectory's prompt."""
    traj = _traj_with_book(
        ["bug-fixing-00001", "code-analysis-00002"],
        "Following [skill-bug-fixing-00001] I will patch the file",
    )
    trajectories = {"inst-1": {1: traj}}
    res = compute_refs_per_instance({"per_instance_sbs": {}}, trajectories)
    refs = res["skill_refs"]
    assert refs["bug-fixing-00001"]["explicit"] == 1
    assert refs["code-analysis-00002"]["explicit"] == 0


def test_per_instance_refs_ignore_unrelated_file_iter_numbering():
    """Even if skillbook files were numbered off from trajectories, citation
    attribution must follow each trajectory's own prompt. A trajectory showing
    skill A must not be credited for a citation of skill B that it never saw."""
    traj = _traj_with_book(["alpha-00001"], "using [skill-alpha-00001] here")
    trajectories = {"inst-1": {7: traj}}  # arbitrary iter number
    res = compute_refs_per_instance({"per_instance_sbs": {7: {}}}, trajectories)
    assert res["summary"]["cite_traj_iters"] == 1
    assert "alpha-00001" in res["skill_refs"]


# --- Bug 3: eval-on-train phases (train_eval / train_eval_baseline) ignored --
#
# load_trajectories / load_results only recognized the standard split phases
# {train, val_baseline, val}. eval-on-train runs write trajectories under
# train_eval/ (with skillbook) and train_eval_baseline/ (empty). Those names
# were not in known_phases, so the split-branch was skipped and the flat-branch
# treated them as instance dirs, found no iter_*.json directly inside (they are
# nested one level deeper), and dropped them. Result: trajectories = {} ->
# traj_iters = 0 -> Cite/Prose/Any Trajs all rendered as "-".

_BENCH = "princeton-nlp__SWE-bench_Verified"


def _make_eval_on_train_run(tmp_path, *, book_traj, baseline_traj,
                            book_resolved=True, baseline_resolved=False):
    """Build a minimal eval-on-train run dir on disk and return its root."""
    for phase, traj in [("train_eval", book_traj),
                        ("train_eval_baseline", baseline_traj)]:
        d = tmp_path / _BENCH / "trajectories" / phase / "inst-1"
        d.mkdir(parents=True)
        (d / "iter_0.json").write_text(json.dumps(traj))
    for phase, resolved in [("train_eval", book_resolved),
                            ("train_eval_baseline", baseline_resolved)]:
        d = tmp_path / _BENCH / "results" / phase / "inst-1"
        d.mkdir(parents=True)
        (d / "iter_0.json").write_text(json.dumps({"resolved": resolved}))
    return tmp_path


def test_load_trajectories_recognizes_eval_on_train_phases(tmp_path):
    run_dir = _make_eval_on_train_run(
        tmp_path,
        book_traj=_traj_with_book(["bug-fixing-00001"], "Using [skill-bug-fixing-00001]"),
        baseline_traj={"messages": [
            {"role": "user", "content": "just the task"},
            {"role": "assistant", "content": "working"},
        ]},
    )
    trajs = _mod.load_trajectories(run_dir)
    # phase-prefixed keys expose both phases
    assert "train_eval/inst-1" in trajs
    assert "train_eval_baseline/inst-1" in trajs
    # unprefixed key resolves to the skillbook phase (train_eval), not baseline
    assert "inst-1" in trajs
    assert 0 in trajs["inst-1"]


def test_presented_refs_compute_for_eval_on_train_run(tmp_path):
    """The Cite/Any Trajs columns go blank precisely because traj_iters==0;
    with the phases loaded, the book-showing attempt must be counted and cited."""
    run_dir = _make_eval_on_train_run(
        tmp_path,
        book_traj=_traj_with_book(["bug-fixing-00001"], "Using [skill-bug-fixing-00001]"),
        baseline_traj={"messages": [
            {"role": "user", "content": "just the task"},
            {"role": "assistant", "content": "working"},
        ]},
    )
    run = {"trajectories": _mod.load_trajectories(run_dir), "run_dir": run_dir}
    res = _mod._compute_presented_skill_refs(run)
    assert res["summary"]["traj_iters"] == 1
    assert res["summary"]["cite_traj_iters"] == 1


def test_load_results_recognizes_eval_on_train_phases(tmp_path):
    run_dir = _make_eval_on_train_run(
        tmp_path,
        book_traj=_traj_with_book(["bug-fixing-00001"], "go"),
        baseline_traj={"messages": [
            {"role": "user", "content": "task"}, {"role": "assistant", "content": "go"}]},
    )
    results = _mod.load_results(run_dir)
    assert "train_eval/inst-1" in results
    assert "train_eval_baseline/inst-1" in results
    assert results.get("inst-1", {}).get(0) is True   # train_eval resolved


def test_paired_resolve_pairs_eval_on_train_phases(tmp_path):
    run_dir = _make_eval_on_train_run(
        tmp_path,
        book_traj=_traj_with_book(["bug-fixing-00001"], "go"),
        baseline_traj={"messages": [
            {"role": "user", "content": "task"}, {"role": "assistant", "content": "go"}]},
        book_resolved=True, baseline_resolved=False,
    )
    sp = _mod._paired_resolve(_mod.load_results(run_dir))
    assert sp["n"] == 1
    assert sp["n10"] == 1   # resolved only WITH skillbook
    assert sp["n01"] == 0
