#!/usr/bin/env python3
"""Count direct [skill-id] citations and attribute per-skill impact.

For two-phase split runs (val/ + val_baseline/), counts citations of injected
skills in val trajectories and attributes per-skill impact via the
val-vs-val_baseline paired counterfactual. Stdlib-only; run with python3.

Usage:
    python3 scripts/analyze_skill_citation_impact.py <run_dir> [<run_dir> ...]
        [--bench auto] [--md <path>] [--json <path>] [--min-citations N]
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_OK = True  # load smoke flag

# --- citation token & skill-injection patterns ---
TOKEN_RE = re.compile(r"\[skill[-:\s][^\]]*\]", re.IGNORECASE)
SKILLBOOK_MARKER = "## Learned Strategies (Skillbook)"
SKILL_HEADER_RE = re.compile(r"^### ([a-zA-Z_][\w]*-\d+)\s*$", re.MULTILINE)

VERDICTS = ("GAINED", "LOST", "STABLE_PASS", "STABLE_FAIL")


def parse_presented_skill_ids(messages):
    """Return the set of skill IDs presented in the skillbook injection block.

    Scans user messages for the '## Learned Strategies (Skillbook)' block and
    collects '### <id>' headers within it (stops at the next h2 section).
    """
    ids = set()
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        idx = content.find(SKILLBOOK_MARKER)
        if idx < 0:
            continue
        block = content[idx:]
        nxt = block.find("\n## ", len(SKILLBOOK_MARKER))
        if nxt > 0:
            block = block[:nxt]
        ids.update(SKILL_HEADER_RE.findall(block))
    return ids


def _parse_skill_token_id(inner):
    """Extract the skill id from a token's inner text.

    Recognizes 'skill-id: X', 'skill-id X', 'skill-X'. Returns None if no id.
    """
    inner = inner.strip()
    m = re.match(r"skill-id\s*[:\s]\s*(.+)$", inner, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.match(r"skill-(.+)$", inner, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def classify_citation(token, presented):
    """Classify a citation token as clean-mapped or unattributable.

    Returns ('clean', skill_id) if the token maps to an id in `presented`,
    else ('unattributable', token).
    """
    if not (token.startswith("[") and token.endswith("]")):
        return ("unattributable", token)
    sid = _parse_skill_token_id(token[1:-1])
    if sid is not None and sid in presented:
        return ("clean", sid)
    return ("unattributable", token)


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Run directory(ies) to analyze.")
    parser.add_argument("--bench", default="auto", help="Benchmark subdir (default: auto).")
    parser.add_argument("--md", default=None, help="Markdown output path (default: <run>/citations.md).")
    parser.add_argument("--json", dest="json_out", default=None, help="JSON output path (default: <run>/citations.json).")
    parser.add_argument("--min-citations", type=int, default=0, help="Drop skills with fewer citations (default: 0).")
    args = parser.parse_args(argv)
    # wired in Task 10
    return 0


if __name__ == "__main__":
    sys_exit = __import__("sys").exit
    sys_exit(main())
