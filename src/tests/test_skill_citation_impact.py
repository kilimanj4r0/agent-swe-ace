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
