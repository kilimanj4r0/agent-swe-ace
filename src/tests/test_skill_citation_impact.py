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
