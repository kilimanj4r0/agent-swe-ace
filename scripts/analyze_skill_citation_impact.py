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


def extract_citations(assistant_messages, presented):
    """Extract and classify citations from messages.

    Returns (counts, unattributable): counts maps skill_id -> clean citation
    count; unattributable is the count of tokens not mappable to a presented id.
    """
    counts = defaultdict(int)
    unattributable = 0
    for m in assistant_messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        for tok in TOKEN_RE.findall(content):
            kind, val = classify_citation(tok, presented)
            if kind == "clean":
                counts[val] += 1
            else:
                unattributable += 1
    return dict(counts), unattributable


def _outcome(skill_resolved, base_resolved):
    if skill_resolved and not base_resolved:
        return "GAINED"
    if base_resolved and not skill_resolved:
        return "LOST"
    if skill_resolved and base_resolved:
        return "STABLE_PASS"
    return "STABLE_FAIL"


def paired_verdict(val_resolved, bl_resolved):
    """Classify a paired instance outcome at pass@1 and any-of-K.

    val_resolved, bl_resolved: list[bool] per-attempt resolution for the val
    (skillbook) and val_baseline (no skillbook) passes.
    Returns {'pass1': verdict, 'any_k': verdict}.
    """
    p1_v = bool(val_resolved[0]) if val_resolved else False
    p1_b = bool(bl_resolved[0]) if bl_resolved else False
    return {
        "pass1": _outcome(p1_v, p1_b),
        "any_k": _outcome(any(val_resolved), any(bl_resolved)),
    }


def mcnemar_pvalue(gained, lost):
    """Two-sided McNemar p-value for the paired skillbook effect.

    gained = # instances resolved WITH skillbook but not baseline (b).
    lost   = # instances resolved in baseline but not with skillbook (c).
    Exact binomial when discordant count < 25; else chi-square with continuity
    correction (df=1) via the chi2(1) == Z^2 relation -> math.erfc.
    """
    n = gained + lost
    if n == 0:
        return 1.0
    if n < 25:
        k = min(gained, lost)
        tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
        return min(1.0, 2.0 * tail)
    stat = (abs(gained - lost) - 1) ** 2 / n
    return math.erfc(math.sqrt(stat / 2.0))


def find_benchmark_dir(run_dir):
    """Locate the benchmark-scoped subdir, else return run_dir."""
    run_dir = Path(run_dir)
    for child in run_dir.iterdir():
        if child.is_dir() and child.name.startswith("princeton-nlp__SWE-bench"):
            return child
    return run_dir


def discover_instances(bench_dir, phase):
    """Sorted list of instance dirs for a phase under trajectories/."""
    d = Path(bench_dir) / "trajectories" / phase
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def _iter_ids(bench_dir, phase, inst):
    """Sorted iter numbers present for an instance's phase."""
    d = Path(bench_dir) / "trajectories" / phase / inst
    if not d.is_dir():
        return []
    ids = []
    for p in d.glob("iter_*.json"):
        m = re.search(r"iter_(\d+)\.json$", p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def load_trajectory(path):
    """Return the messages list from a trajectory JSON file."""
    with open(path) as f:
        return json.load(f).get("messages", [])


def load_resolved(path):
    """Return the resolved bool from a result JSON file."""
    with open(path) as f:
        return bool(json.load(f).get("resolved", False))


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
