#!/usr/bin/env python3
"""Build the prose-skill-reference phrasing dictionary for a set of runs.

The agent (qwen3next in particular) often refers to the skillbook by general
words — "Based on the skill", "the skillbook suggests", "the skill description" —
instead of citing a skill ID. This script scans ASSISTANT messages (the injected
skillbook lives in user messages, so we never count it), strips code blocks, and
characterizes HOW the model refers to skills, emitting a JSON dictionary.

"strategy" is deliberately NOT treated as a skill synonym: in these trajectories
it is dominated by the sklearn `strategy=` parameter (KBinsDiscretizer),
"strategic fix", and generic "different strategy" — see data/skill_prose_phrasings.json
notes. Only skill/skills/skillbook count.

Usage:
    uv run python scripts/extract_prose_phrasings.py 'data/*qwen3next*' \
        --out data/skill_prose_phrasings.json
    uv run python scripts/extract_prose_phrasings.py data/run_A data/run_B --out dict.json
"""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The core detector: a genuine prose reference to the skillbook. Word-bounded so
# it excludes skillful/skillset; "skill_id"/"skill-id" don't match either (_ and
# - : the \b after 'skill' fails against a following word char). Code is stripped
# first, so module paths / inline code never fire it.
PROSE_SKILL_RE = re.compile(r"\bskills?\b|\bskillbooks?\b", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")

# Phrase families — used only to CHARACTERIZE the "how", not for detection.
# Family 1: "Based on … skill/skillbook …" lead-in (the dominant citation form).
BASED_ON_RE = re.compile(
    r"based on (?:the\s+)?(?:skills?|skillbook\w*)(?:\s+[a-z'-]+){0,4}", re.IGNORECASE
)
# Family 2: "the skill(s)/skillbook <connector>" — skillbook as subject/object.
SUBJECT_RE = re.compile(
    r"\b(?:the|this|a|these|those)\s+"
    r"(skillbook\w*|skills?)\s+"
    r"(reference\w*\s*(?:to\s+)?|entry\s+|description[s]?\s+|says?\s+|"
    r"mention(?:s|ed)?\s+|suggests?\s+|recommend(?:s|ed)?\s+|"
    r"notes?\s+|indicates?\s+|states?\s+|advises?\s+|"
    r"relat\w*\s+to\s+|above\s+|provid(?:ed|es)\s+)",
    re.IGNORECASE,
)
# Family 3: a skill ID cited inside a skill-noun phrase (model also cites by ID).
ID_IN_PROSE_RE = re.compile(
    r"(skill\w*\s+(?:reference\s+)?(?:to\s+)?)\s*"
    r"([a-z][a-z_]*-[0-9]{3,5})\b",
    re.IGNORECASE,
)
# Lead-in verbs that attribute to the skillbook (for the "other citations" bucket).
CITATION_LEADINS = [
    "according to", "as per", "per the", "following", "as suggested",
    "as recommended", "as noted", "as described", "as outlined", "as mentioned",
    "guided by", "inspired by", "referencing", "refer to",
]


def _strip_code(text: str) -> str:
    return _INLINE_CODE_RE.sub(" ", _CODE_BLOCK_RE.sub(" ", text))


def _assistant_texts(path: Path):
    """Yield each assistant message content (code-stripped) from a trajectory."""
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    for m in d.get("messages", []):
        if m.get("role") != "assistant":
            continue
        c = m.get("content", "")
        if isinstance(c, str) and c:
            yield _strip_code(c)


def _candidate_files(runs: list[str]) -> list[Path]:
    """Use rg to narrow to trajectory files whose assistant text mentions skill."""
    # rg flags: -l list files, --no-ignore (data/ is outside git tracking norms),
    # match the bare word, restrict to iter_*.json.
    files: list[str] = []
    for run in runs:
        res = subprocess.run(
            ["rg", "-l", "-i", "--no-ignore", r"\bskill",
             run, "-g", "iter_*.json"],
            capture_output=True, text=True,
        )
        files.extend(f for f in res.stdout.split() if "trajectories" in f)
    # Dedup preserving order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(Path(f))
    return out


def build_dictionary(runs: list[str]) -> dict:
    files = _candidate_files(runs)
    if not files:
        print("No candidate trajectory files found.", file=sys.stderr)
        return {}

    # Core engagement counts
    total_trajs = 0
    prose_trajs = 0
    prose_msgs = 0
    noun_forms = Counter()
    # Family accumulators
    based = Counter()
    based_ex = defaultdict(list)
    subject = Counter()
    subject_ex = defaultdict(list)
    id_cites = Counter()
    id_ex = defaultdict(list)
    leadin_hits = Counter()

    for f in files:
        one_traj_has_prose = False
        for text in _assistant_texts(f):
            if not PROSE_SKILL_RE.search(text):
                continue
            one_traj_has_prose = True
            prose_msgs += 1
            for mm in PROSE_SKILL_RE.finditer(text):
                noun_forms[mm.group(0).lower()] += 1

            low = text.lower()
            for mm in BASED_ON_RE.finditer(text):
                key = re.sub(r"\s+", " ", mm.group(0).strip().lower())
                based[key] += 1
                if len(based_ex[key]) < 2:
                    ctx = text[max(0, mm.start() - 10):mm.end() + 60].strip()
                    based_ex[key].append(ctx[:160])
            for mm in SUBJECT_RE.finditer(text):
                key = re.sub(r"\s+", " ", mm.group(0).strip().lower())
                subject[key] += 1
                if len(subject_ex[key]) < 2:
                    ctx = text[max(0, mm.start() - 15):mm.end() + 50].strip()
                    subject_ex[key].append(ctx[:160])
            for mm in ID_IN_PROSE_RE.finditer(text):
                sid = mm.group(2).lower()
                if "__" in sid:
                    continue
                id_cites[sid] += 1
                if len(id_ex[sid]) < 1:
                    ctx = text[max(0, mm.start() - 25):mm.end() + 25].strip()
                    id_ex[sid] = ctx[:150]
            for lead in CITATION_LEADINS:
                if lead in low:
                    leadin_hits[lead] += 1

        total_trajs += 1
        if one_traj_has_prose:
            prose_trajs += 1

    def _top(counter, examples, n):
        return [
            {"phrase": k, "count": c, "examples": examples.get(k, [])[:2]}
            for k, c in counter.most_common(n)
        ]

    return {
        "description": (
            "How the agent refers to the skillbook in prose (ASSISTANT messages "
            "only, code stripped). qwen3next engages the skillbook almost "
            "exclusively this way; ID citations are rarer."
        ),
        "source_runs": runs,
        "detector": r"\bskills?\b|\bskillbooks?\b  (case-insensitive; code stripped)",
        "excluded": (
            "'strategy' is NOT a skill synonym here — dominated by sklearn "
            "`strategy=` (KBinsDiscretizer), 'strategic fix', and generic "
            "'different strategy'. The injected skillbook text (user messages) "
            "is never counted."
        ),
        "totals": {
            "candidate_trajectories": total_trajs,
            "trajectories_with_prose_ref": prose_trajs,
            "pct_trajectories_with_prose_ref": (
                round(prose_trajs / total_trajs * 100, 2) if total_trajs else 0
            ),
            "assistant_messages_with_prose_ref": prose_msgs,
        },
        "noun_surface_forms": dict(noun_forms.most_common()),
        "citation_leadins_used": dict(leadin_hits.most_common()),
        "families": {
            "based_on": {
                "label": "Based on … skill/skillbook … (dominant lead-in)",
                "entries": _top(based, based_ex, 40),
            },
            "skill_as_subject": {
                "label": "the skill(s)/skillbook <connector> (skillbook as subject)",
                "entries": _top(subject, subject_ex, 40),
            },
            "skill_id_in_prose": {
                "label": "skill ID cited inside a skill-noun phrase",
                "entries": [
                    {"skill_id": k, "count": c, "example": id_ex.get(k, "")}
                    for k, c in id_cites.most_common(30)
                ],
            },
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="Run dirs or globs (e.g. 'data/*qwen3next*')")
    ap.add_argument("--out", default="data/skill_prose_phrasings.json",
                    help="Output JSON path (default: data/skill_prose_phrasings.json)")
    args = ap.parse_args()

    # Expand shell globs ourselves (argparse won't if quoted)
    import glob as _glob
    expanded = []
    for r in args.runs:
        matches = sorted(_glob.glob(r))
        expanded.extend(matches if matches else [r])
    # Keep only existing dirs / drop misses
    runs = [r for r in expanded if Path(r).exists()]

    dictionary = build_dictionary(runs)
    if not dictionary:
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dictionary, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}", file=sys.stderr)
    t = dictionary["totals"]
    print(f"  candidate trajectories: {t['candidate_trajectories']}", file=sys.stderr)
    print(f"  with prose skill-ref  : {t['trajectories_with_prose_ref']} "
          f"({t['pct_trajectories_with_prose_ref']}%)", file=sys.stderr)
    print(f"  'Based on…' variants   : {len(dictionary['families']['based_on']['entries'])}",
          file=sys.stderr)
    print(f"  subject-verb variants : {len(dictionary['families']['skill_as_subject']['entries'])}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
