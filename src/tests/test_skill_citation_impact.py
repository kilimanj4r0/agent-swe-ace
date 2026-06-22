"""Tests for scripts/analyze_skill_citation_impact.py (stdlib-only script)."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # project root
SCRIPT = ROOT / "scripts" / "analyze_skill_citation_impact.py"


@pytest.fixture(scope="module")
def sci():
    """Load the standalone script module by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("analyze_skill_citation_impact", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_skill_citation_impact"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_loads(sci):
    """Module imports cleanly and exposes a guarded main()."""
    assert hasattr(sci, "main")
    assert sci.SCRIPT_OK is True


def test_parse_presented_skill_ids_extracts_block_headers(sci):
    msgs = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": (
            "<pr_description>do thing</pr_description>\n\n"
            "## Learned Strategies (Skillbook)\n\n"
            "Use them to guide your approach:\n\n"
            "### django_fixing-00001\n\nCreate a patch.\n\n"
            "### code_modification-00005\n\nEdit the function.\n"
        )},
    ]
    assert sci.parse_presented_skill_ids(msgs) == {"django_fixing-00001", "code_modification-00005"}


def test_parse_presented_skill_ids_empty_when_no_block(sci):
    assert sci.parse_presented_skill_ids([{"role": "user", "content": "no skills here"}]) == set()


def test_parse_presented_skill_ids_stops_at_next_h2(sci):
    msgs = [{"role": "user", "content": (
        "## Learned Strategies (Skillbook)\n\n### a-00001\n\nx\n\n"
        "## Other Section\n\n### should-not-match-00099\n\ny\n"
    )}]
    assert sci.parse_presented_skill_ids(msgs) == {"a-00001"}


def test_parse_presented_skill_ids_ignores_non_user(sci):
    msgs = [
        {"role": "assistant", "content": "## Learned Strategies (Skillbook)\n\n### a-00001\n"},
        {"role": "user", "content": "hello"},
    ]
    assert sci.parse_presented_skill_ids(msgs) == set()


@pytest.mark.parametrize("token,presented,expected", [
    ("[skill-django_fixing-00002]", {"django_fixing-00002"}, ("clean", "django_fixing-00002")),
    ("[skill-id: code_modification-00004]", {"code_modification-00004"}, ("clean", "code_modification-00004")),
    ("[skill-id code_modification-00004]", {"code_modification-00004"}, ("clean", "code_modification-00004")),
    ("[skill-00001]", {"django_fixing-00001"}, ("unattributable", "[skill-00001]")),
    ("[skill-django_fixing-00002]", {"other-00002"}, ("unattributable", "[skill-django_fixing-00002]")),
    ("[skill-id: code-modification-00004]", {"code_modification-00004"}, ("unattributable", "[skill-id: code-modification-00004]")),
    ("[skill-00006]", set(), ("unattributable", "[skill-00006]")),
    ("[skill: django_fixing-00002]", {"django_fixing-00002"}, ("clean", "django_fixing-00002")),
    ("[skill code_modification-00005]", {"code_modification-00005"}, ("clean", "code_modification-00005")),
])
def test_classify_citation(sci, token, presented, expected):
    assert sci.classify_citation(token, presented) == expected


def test_extract_citations_separates_clean_and_unattrib(sci):
    msgs = [
        {"role": "assistant", "content": "I will use [skill-django_fixing-00002] and [skill-00001]."},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "Also [skill-django_fixing-00002] again, [skill-id: code_modification-00005]."},
    ]
    presented = {"django_fixing-00002", "code_modification-00005"}
    counts, unattrib = sci.extract_citations(msgs, presented)
    assert counts == {"django_fixing-00002": 2, "code_modification-00005": 1}
    assert unattrib == 1  # [skill-00001]


def test_extract_citations_empty(sci):
    counts, unattrib = sci.extract_citations([{"role": "assistant", "content": "no cites"}], {"a-00001"})
    assert counts == {}
    assert unattrib == 0


def test_paired_verdict_pass1_and_any_k(sci):
    # val: fail@0, pass@1 ; baseline: pass@0, fail@1
    val = [False, True, False]
    bl = [True, False, False]
    v = sci.paired_verdict(val, bl)
    assert v["pass1"] == "LOST"        # iter0: val F, bl T
    assert v["any_k"] == "STABLE_PASS" # both resolve somewhere


def test_paired_verdict_gained(sci):
    v = sci.paired_verdict([False, True], [False, False])
    assert v["pass1"] == "STABLE_FAIL"
    assert v["any_k"] == "GAINED"


def test_paired_verdict_empty_attempts(sci):
    v = sci.paired_verdict([], [])
    assert v["pass1"] == "STABLE_FAIL"
    assert v["any_k"] == "STABLE_FAIL"


def test_mcnemar_no_discordant_is_one(sci):
    assert sci.mcnemar_pvalue(0, 0) == 1.0


def test_mcnemar_symmetric_high_p(sci):
    # balanced discordants -> large p
    p = sci.mcnemar_pvalue(10, 10)
    assert p > 0.9


def test_mcnemar_skewed_small_p(sci):
    # large asymmetry, n>=25 -> chi-square path, very small p
    p = sci.mcnemar_pvalue(30, 2)
    assert p < 1e-4


def test_mcnemar_exact_small_n(sci):
    # n<25 -> exact binomial path; 8 vs 0 should be small
    p = sci.mcnemar_pvalue(8, 0)
    assert 0.0 < p < 0.02


def _make_run(tmp_path, with_baseline=True):
    bench = tmp_path / "princeton-nlp__SWE-bench_Verified"
    (bench / "trajectories" / "val" / "instA").mkdir(parents=True)
    (bench / "trajectories" / "val" / "instB").mkdir(parents=True)
    if with_baseline:
        (bench / "trajectories" / "val_baseline" / "instA").mkdir(parents=True)
    return tmp_path, bench


def test_find_benchmark_dir_resolves_subdir(sci, tmp_path):
    _, bench = _make_run(tmp_path)
    assert sci.find_benchmark_dir(tmp_path) == bench


def test_find_benchmark_dir_falls_back_to_run(sci, tmp_path):
    # no benchmark subdir -> run dir itself
    (tmp_path / "trajectories").mkdir()
    assert sci.find_benchmark_dir(tmp_path) == tmp_path


def test_discover_instances(sci, tmp_path):
    _, bench = _make_run(tmp_path)
    assert sci.discover_instances(bench, "val") == ["instA", "instB"]
    assert sci.discover_instances(bench, "val_baseline") == ["instA"]


def test_iter_ids_parses(sci, tmp_path):
    _, bench = _make_run(tmp_path)
    d = bench / "trajectories" / "val" / "instA"
    (d / "iter_0.json").write_text("{}")
    (d / "iter_10.json").write_text("{}")
    (d / "notiter.json").write_text("{}")
    assert sci._iter_ids(bench, "val", "instA") == [0, 10]


def test_load_resolved_and_trajectory(sci, tmp_path):
    import json
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"resolved": True}))
    assert sci.load_resolved(p) is True
    tp = tmp_path / "t.json"
    tp.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
    assert sci.load_trajectory(tp) == [{"role": "user", "content": "hi"}]


def _write_traj(path, presented_ids, cite_tokens):
    """Write a trajectory with a skillbook block (presented_ids) and citations."""
    block = ""
    if presented_ids:
        block = "\n\n## Learned Strategies (Skillbook)\n\n" + "".join(f"### {s}\n\nx\n" for s in presented_ids)
    cite = " ".join(cite_tokens)
    msgs = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "<pr_description>x</pr_description>" + block},
        {"role": "assistant", "content": f"THOUGHT: {cite}\n```bash\ntrue\n```"},
    ]
    path.write_text(__import__("json").dumps({"messages": msgs}))


def _write_result(path, resolved):
    path.write_text(__import__("json").dumps({"resolved": resolved}))


def test_analyze_run_synthetic(sci, tmp_path):
    bench = tmp_path / "princeton-nlp__SWE-bench_Verified"
    # instA: val cites django_fixing-00001, val resolves@0; baseline does not
    for ph in ("val", "val_baseline"):
        (bench / "trajectories" / ph / "instA").mkdir(parents=True)
        (bench / "results" / ph / "instA").mkdir(parents=True)
    _write_traj(bench / "trajectories/val/instA/iter_0.json", ["django_fixing-00001"], ["[skill-django_fixing-00001]"])
    _write_traj(bench / "trajectories/val/instA/iter_1.json", ["django_fixing-00001"], [])
    _write_result(bench / "results/val/instA/iter_0.json", True)
    _write_result(bench / "results/val/instA/iter_1.json", False)
    _write_traj(bench / "trajectories/val_baseline/instA/iter_0.json", [], [])
    _write_traj(bench / "trajectories/val_baseline/instA/iter_1.json", [], [])
    _write_result(bench / "results/val_baseline/instA/iter_0.json", False)
    _write_result(bench / "results/val_baseline/instA/iter_1.json", False)

    res = sci.analyze_run(tmp_path)
    assert res["instance_count"] == 1
    assert res["has_baseline"] is True
    assert res["verdict_counts"]["pass1"] == {"GAINED": 1, "LOST": 0, "STABLE_PASS": 0, "STABLE_FAIL": 0}
    assert res["verdict_counts"]["any_k"] == {"GAINED": 1, "LOST": 0, "STABLE_PASS": 0, "STABLE_FAIL": 0}
    sk = next(s for s in res["skills"] if s["skill_id"] == "django_fixing-00001")
    assert sk["citations"] == 1
    assert sk["cited_trajectories"] == 1
    assert sk["presented_trajectories"] == 2
    assert sk["citing_instances"] == 1
    assert sk["resolve_rate_when_cited"] == 1.0
    assert sk["resolve_rate_when_presented_not_cited"] == 0.0  # iter_1 presented-not-cited, unresolved
    assert sk["attrib_any_k"]["GAINED"] == 1
    assert res["unattributable"] == 0
    assert res["mcnemar_pass1"]["gained"] == 1
    assert res["verdict_counts"]["avg"] == {"GAINED": 1, "LOST": 0, "STABLE_PASS": 0, "STABLE_FAIL": 1}
    assert res["net_delta"]["avg"] == 1
    assert res["total_clean_citations"] == 1
    assert res["citations_by_verdict"]["any_k"]["GAINED"] == 1


def test_analyze_run_no_baseline_counts_only(sci, tmp_path):
    bench = tmp_path / "princeton-nlp__SWE-bench_Verified"
    (bench / "trajectories" / "val" / "instA").mkdir(parents=True)
    (bench / "results" / "val" / "instA").mkdir(parents=True)
    _write_traj(bench / "trajectories/val/instA/iter_0.json", ["django_fixing-00001"], ["[skill-django_fixing-00001]"])
    _write_result(bench / "results/val/instA/iter_0.json", True)
    res = sci.analyze_run(tmp_path)
    assert res["has_baseline"] is False
    assert res["instance_count"] == 1
    assert res["verdict_counts"]["pass1"] == {"GAINED": 0, "LOST": 0, "STABLE_PASS": 0, "STABLE_FAIL": 0}
    assert res["mcnemar_pass1"]["p_value"] == 1.0
    sk = res["skills"][0]
    assert sk["skill_id"] == "django_fixing-00001"
    assert sk["citations"] == 1
    assert sk["resolve_rate_when_cited"] == 1.0


def test_render_markdown_contains_sections(sci):
    res = {
        "run_dir": "/tmp/runX", "has_baseline": True, "instance_count": 5,
        "val_only_instances": [],
        "verdict_counts": {"pass1": {"GAINED": 2, "LOST": 1, "STABLE_PASS": 1, "STABLE_FAIL": 1},
                           "any_k": {"GAINED": 3, "LOST": 0, "STABLE_PASS": 1, "STABLE_FAIL": 1},
                           "avg": {"GAINED": 9, "LOST": 3, "STABLE_PASS": 4, "STABLE_FAIL": 8}},
        "net_delta": {"pass1": 1, "any_k": 3, "avg": 6},
        "mcnemar_pass1": {"gained": 2, "lost": 1, "p_value": 0.123},
        "mcnemar_avg": {"gained": 9, "lost": 3, "p_value": 0.089},
        "total_clean_citations": 9,
        "citations_by_verdict": {"any_k": {"GAINED": 4, "LOST": 1, "STABLE_PASS": 2, "STABLE_FAIL": 2},
                                 "pass1": {"GAINED": 3, "LOST": 0, "STABLE_PASS": 2, "STABLE_FAIL": 4}},
        "unattributable": 7,
        "skills": [{"skill_id": "a-00001", "citations": 9, "citing_instances": 3,
                    "presented_trajectories": 10, "cited_trajectories": 5,
                    "citation_rate": 0.5, "resolve_rate_when_cited": 0.4,
                    "resolve_rate_when_presented_not_cited": 0.3,
                    "attrib_pass1": {"GAINED": 2, "LOST": 0, "STABLE_PASS": 1, "STABLE_FAIL": 0},
                    "attrib_any_k": {"GAINED": 2, "LOST": 0, "STABLE_PASS": 1, "STABLE_FAIL": 0}}],
        "instances": [],
    }
    md = sci.render_markdown(res)
    assert "# Skill Citation Impact" in md
    assert "runX" in md
    assert "GAINED" in md and "LOST" in md
    assert "avg (per-att)" in md
    assert "McNemar" in md
    assert "Direct citations" in md
    assert "a-00001" in md
    assert "7" in md  # unattributable


def test_to_json_roundtrips(sci):
    res = {"run_dir": "/x", "has_baseline": False, "instance_count": 0,
           "val_only_instances": [], "verdict_counts": {"pass1": {}, "any_k": {}},
           "net_delta": {"pass1": 0, "any_k": 0},
           "mcnemar_pass1": {"gained": 0, "lost": 0, "p_value": 1.0},
           "unattributable": 0, "skills": [], "instances": []}
    s = sci.to_json(res)
    assert __import__("json").loads(s)["run_dir"] == "/x"


def test_main_writes_outputs_into_run_dir(sci, tmp_path):
    # minimal synthetic run: one val instance (resolves) + matching baseline (does not)
    bench = tmp_path / "princeton-nlp__SWE-bench_Verified"
    for ph in ("val", "val_baseline"):
        (bench / "trajectories" / ph / "instA").mkdir(parents=True)
        (bench / "results" / ph / "instA").mkdir(parents=True)
    _write_traj(bench / "trajectories/val/instA/iter_0.json", [], [])
    _write_result(bench / "results/val/instA/iter_0.json", True)
    _write_traj(bench / "trajectories/val_baseline/instA/iter_0.json", [], [])
    _write_result(bench / "results/val_baseline/instA/iter_0.json", False)

    rc = sci.main([str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "citations.md").is_file()
    assert (tmp_path / "citations.json").is_file()
