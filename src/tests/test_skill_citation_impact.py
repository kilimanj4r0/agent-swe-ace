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
