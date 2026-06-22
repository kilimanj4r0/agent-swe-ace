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


def _new_skill():
    return {
        "citations": 0,
        "cited_trajectories": 0,
        "citing_instances": set(),
        "presented_trajectories": 0,
        "resolved_when_cited": 0,
        "attrib_pass1": dict.fromkeys(VERDICTS, 0),
        "attrib_any_k": dict.fromkeys(VERDICTS, 0),
    }


def analyze_run(run_dir, min_citations=0):
    """Analyze one run dir; return a JSON-serializable results dict.

    Counts clean-mapped citations, per-skill resolution lift, and the paired
    GAINED/LOST/STABLE verdict (pass@1 + any-of-K) with a McNemar p-value.
    Degrades to counts-only when val_baseline is absent.
    """
    run_dir = Path(run_dir)
    bench = find_benchmark_dir(run_dir)
    val_insts = set(discover_instances(bench, "val"))
    bl_insts = set(discover_instances(bench, "val_baseline"))
    has_baseline = bool(bl_insts)
    paired_insts = sorted(val_insts & bl_insts) if has_baseline else sorted(val_insts)
    val_only = sorted(val_insts - bl_insts) if has_baseline else []

    per_skill = defaultdict(_new_skill)
    unattributable = 0
    inst_rows = []
    verdict_counts = {"pass1": Counter(), "any_k": Counter()}

    for inst in paired_insts:
        val_res = []
        cited_sids_inst = set()
        for it in _iter_ids(bench, "val", inst):
            msgs = load_trajectory(bench / "trajectories" / "val" / inst / f"iter_{it}.json")
            presented = parse_presented_skill_ids(msgs)
            counts, unattrib = extract_citations(msgs, presented)
            unattributable += unattrib
            try:
                resolved = load_resolved(bench / "results" / "val" / inst / f"iter_{it}.json")
            except FileNotFoundError:
                resolved = False
            val_res.append(resolved)
            for sid in presented:
                per_skill[sid]["presented_trajectories"] += 1
            for sid, c in counts.items():
                per_skill[sid]["citations"] += c
                per_skill[sid]["cited_trajectories"] += 1
                if resolved:
                    per_skill[sid]["resolved_when_cited"] += 1
                cited_sids_inst.add(sid)
        for sid in cited_sids_inst:
            per_skill[sid]["citing_instances"].add(inst)

        bl_res = []
        if has_baseline:
            for it in _iter_ids(bench, "val_baseline", inst):
                try:
                    bl_res.append(load_resolved(bench / "results" / "val_baseline" / inst / f"iter_{it}.json"))
                except FileNotFoundError:
                    bl_res.append(False)

        row = {"instance": inst, "val_resolved_attempts": val_res, "bl_resolved_attempts": bl_res}
        if has_baseline:
            pv = paired_verdict(val_res, bl_res)
            verdict_counts["pass1"][pv["pass1"]] += 1
            verdict_counts["any_k"][pv["any_k"]] += 1
            row["verdict"] = pv
            for sid in cited_sids_inst:
                per_skill[sid]["attrib_pass1"][pv["pass1"]] += 1
                per_skill[sid]["attrib_any_k"][pv["any_k"]] += 1
        else:
            row["verdict"] = None
        inst_rows.append(row)

    skills = []
    for sid, s in per_skill.items():
        if s["citations"] < min_citations:
            continue
        cited_traj = s["cited_trajectories"]
        pres = s["presented_trajectories"]
        skills.append({
            "skill_id": sid,
            "citations": s["citations"],
            "citing_instances": len(s["citing_instances"]),
            "presented_trajectories": pres,
            "cited_trajectories": cited_traj,
            "citation_rate": (cited_traj / pres) if pres else 0.0,
            "resolve_rate_when_cited": (s["resolved_when_cited"] / cited_traj) if cited_traj else 0.0,
            "attrib_pass1": dict(s["attrib_pass1"]),
            "attrib_any_k": dict(s["attrib_any_k"]),
        })
    skills.sort(key=lambda r: r["citations"], reverse=True)

    g1, l1 = verdict_counts["pass1"]["GAINED"], verdict_counts["pass1"]["LOST"]
    gk, lk = verdict_counts["any_k"]["GAINED"], verdict_counts["any_k"]["LOST"]
    return {
        "run_dir": str(run_dir),
        "has_baseline": has_baseline,
        "instance_count": len(paired_insts),
        "val_only_instances": val_only,
        "verdict_counts": {
            "pass1": {k: verdict_counts["pass1"][k] for k in VERDICTS},
            "any_k": {k: verdict_counts["any_k"][k] for k in VERDICTS},
        },
        "net_delta": {"pass1": g1 - l1, "any_k": gk - lk},
        "mcnemar_pass1": {"gained": g1, "lost": l1, "p_value": mcnemar_pvalue(g1, l1)},
        "unattributable": unattributable,
        "skills": skills,
        "instances": inst_rows,
    }


def _pct(x):
    return f"{100.0 * x:.1f}%"


def render_markdown(res):
    """Render a results dict as a human-readable Markdown report."""
    lines = ["# Skill Citation Impact", ""]
    lines.append(f"**Run:** `{res['run_dir']}`")
    lines.append(f"**Instances (paired):** {res['instance_count']}  "
                 f"**Baseline:** {'yes' if res['has_baseline'] else 'no (counts-only mode)'}")
    if res["val_only_instances"]:
        lines.append(f"**Val-only (no baseline):** {len(res['val_only_instances'])} excluded from verdicts")
    lines.append("")

    vc = res["verdict_counts"]
    if res["has_baseline"]:
        lines.append("## Paired outcome (val vs val_baseline)")
        lines.append("")
        lines.append("| Verdict | pass@1 | any-of-K |")
        lines.append("|---|---:|---:|")
        for k in VERDICTS:
            lines.append(f"| {k} | {vc['pass1'].get(k, 0)} | {vc['any_k'].get(k, 0)} |")
        nd = res["net_delta"]
        lines.append(f"| **net Δ** | **{nd['pass1']:+d}** | **{nd['any_k']:+d}** |")
        lines.append("")
        mc = res["mcnemar_pass1"]
        lines.append(f"**McNemar (pass@1):** gained={mc['gained']} lost={mc['lost']} "
                     f"p={mc['p_value']:.4g}")
        lines.append("")

    lines.append("## Per-skill citations")
    lines.append("")
    lines.append("| skill_id | citations | citing_inst | presented_traj | cited_traj | cite_rate | resolve\\|cited | GAINED(any_k) | LOST(any_k) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in res["skills"]:
        lines.append("| {sid} | {c} | {ci} | {pt} | {ct} | {cr} | {rr} | {g} | {l} |".format(
            sid=s["skill_id"], c=s["citations"], ci=s["citing_instances"],
            pt=s["presented_trajectories"], ct=s["cited_trajectories"],
            cr=_pct(s["citation_rate"]), rr=_pct(s["resolve_rate_when_cited"]),
            g=s["attrib_any_k"].get("GAINED", 0), l=s["attrib_any_k"].get("LOST", 0)))
    lines.append("")
    lines.append(f"**Unattributable citations (namespace-mismatched):** {res['unattributable']}")
    lines.append("")
    return "\n".join(lines)


def to_json(res):
    """Serialize a results dict to a JSON string."""
    return json.dumps(res, indent=2)


def main(argv=None):
    """CLI entry point. Analyze each run dir and write citations.{md,json}."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Run directory(ies) to analyze.")
    parser.add_argument("--bench", default="auto", help="Benchmark subdir (default: auto).")
    parser.add_argument("--md", default=None, help="Markdown output path (default: <run>/citations.md).")
    parser.add_argument("--json", dest="json_out", default=None, help="JSON output path (default: <run>/citations.json).")
    parser.add_argument("--min-citations", type=int, default=0, help="Drop skills with fewer citations (default: 0).")
    args = parser.parse_args(argv)

    for run_dir in args.run_dirs:
        res = analyze_run(run_dir, min_citations=args.min_citations)
        md_path = Path(args.md) if args.md else Path(run_dir) / "citations.md"
        json_path = Path(args.json_out) if args.json_out else Path(run_dir) / "citations.json"
        md_path.write_text(render_markdown(res))
        json_path.write_text(to_json(res))
        nd = res["net_delta"]
        print(f"{run_dir}: {res['instance_count']} instances, "
              f"Δ pass@1={nd['pass1']:+d} any_k={nd['any_k']:+d}, "
              f"{len(res['skills'])} cited skills, {res['unattributable']} unattributable -> "
              f"{md_path}")
    return 0


if __name__ == "__main__":
    sys_exit = __import__("sys").exit
    sys_exit(main())
