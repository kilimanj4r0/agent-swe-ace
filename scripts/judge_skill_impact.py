#!/usr/bin/env python3
"""LLM-judge for the *impact* of skillbook skills on agent behavior.

Question this script answers, per skill:
    Did presenting this skill IMPROVE, WORSEN, or stay IRRELEVANT to the agent's
    behavior and outcome — relative to the same agent solving the same task WITHOUT
    the skillbook?

Why this is tractable here (paired counterfactual):
    A split-mode run already contains, for every val instance, two trajectories on
    the IDENTICAL task with the same iterations:
        trajectories/val/<id>/iter_*.json          -> agent WITH the skillbook
        trajectories/val_baseline/<id>/iter_*.json -> agent WITHOUT any skillbook
    The skillbook is injected as a "## Learned Strategies (Skillbook)" block in the
    second user message, listing each skill as "### <section>-<id>" + content.
    So we can show the judge BOTH logs and ask it to attribute behavioral
    differences to specific skills — not blindly guess which agent saw the tips.

Three layers:
    L1 (no LLM)   objective outcome per instance from results/: GAINED / LOST /
                  STABLE_PASS / STABLE_FAIL (skillbook resolved vs baseline resolved).
    L2 (GLM)      per instance, judge each presented skill on influence /
                  applicability / effect, grounded in the contrast between the two
                  logs and the objective outcome label.
    L3 (no LLM)   aggregate per skill_id across instances -> BENEFICIAL / HARMFUL /
                  NEUTRAL / NOISE.

Note: run with system `python3` (stdlib only); `uv run` may fail on platform deps.

Usage:
    python3 scripts/judge_skill_impact.py <run_dir>
    python3 scripts/judge_skill_impact.py <run_dir> --baseline data/some_no_skillbook_run
    python3 scripts/judge_skill_impact.py <run_dir> --instances astropy__astropy-13236,django__django-10914
    python3 scripts/judge_skill_impact.py <run_dir> --limit 20 --workers 6
    uv run python scripts/judge_skill_impact.py <run_dir> --json out.json --jsonl cache.jsonl
    uv run python scripts/judge_skill_impact.py --analyze cache.jsonl   # re-aggregate from cache
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ZAI_BASE = "https://api.z.ai/api/coding/paas/v4"
DEFAULT_MODEL = "glm-5.2"
MAX_LOG_CHARS = 45_000  # per log; two logs go to the judge, so keep each modest

INFLUENCE = ("followed", "partially_followed", "ignored", "contradicted")
APPLICABILITY = ("relevant", "tangential", "irrelevant")
EFFECT = ("improved", "neutral", "harmful")

JUDGE_SYSTEM = """You are a meticulous evaluator of AI coding-agent behavior.

You are given, for ONE software issue:
1. SKILLS: learned strategies (id + text) that were injected into the prompt of ONE agent.
2. LOG_WITH_SKILLS: that agent's own messages (THOUGHT + bash commands).
3. LOG_BASELINE: a DIFFERENT run of an agent solving the SAME issue that NEVER saw any skills.
4. OUTCOME: an objective label comparing whether each run resolved the issue
   (GAINED = baseline failed but skills run passed; LOST = baseline passed but skills run
   failed; STABLE_PASS = both passed; STABLE_FAIL = both failed).

The baseline is the counterfactual: it shows what the agent does WITHOUT the skills. Use the
contrast between the two logs (not topical overlap) to attribute behavioral differences.

For EACH skill, decide three things — be strict, default to the weaker option when unsure:

influence  — did LOG_WITH_SKILLS act on this skill?
  "followed"            : clearly used the skill's specific advice/fact (e.g. jumped straight to a
                          location/approach the skill names, without discovering it first).
  "partially_followed"  : behavior consistent with the skill beyond mere topic, but a competent
                          agent might do it unprompted.
  "ignored"             : skill present but no sign it influenced behavior (topical overlap only,
                          or the agent did its own thing).
  "contradicted"        : the agent did the opposite of what the skill advises.

applicability — is the skill relevant to THIS specific issue?
  "relevant" / "tangential" / "irrelevant".

effect — what did the skill itself (its ADVICE QUALITY) do to THIS run vs the baseline?
  CRITICAL: effect rates the SKILL, not the agent's discipline. Only blame the skill for an
  outcome the agent reached BY ACTING ON IT.
  "improved"  : the agent followed the skill and it made the run more direct/correct/minimal, or
                plausibly caused a GAINED. (requires influence followed/partially_followed)
  "harmful"   : the agent followed the skill and THE SKILL'S ADVICE sent it down a wrong path /
                wasted steps / plausibly caused a LOST. (requires influence followed/partially_followed)
                Do NOT mark harmful when the agent IGNORED or CONTRADICTED good advice and then
                failed — that is a discipline failure, not a bad skill; use "neutral" + would_have_helped.
  "neutral"   : no meaningful behavioral effect from the skill (ignored, topical-only, or
                contradicted-good-advice).

would_have_helped — boolean: the skill is sound, relevant advice that the agent IGNORED or
  CONTRADICTED, and following it would plausibly have improved the run. (This is the
  adherence/discipline axis, separate from skill quality.)

For "followed"/"contradicted" and for any non-neutral effect or would_have_helped=true you MUST
give a short verbatim quote from LOG_WITH_SKILLS as evidence.

Then give an instance-level net_effect of the WHOLE skillbook: "helped" | "hurt" | "no_effect",
consistent with OUTCOME and your per-skill judgments, plus a one-sentence rationale.

Return ONLY a JSON object, no markdown:
{"skills": [{"id": "<skill id>", "influence": "...", "applicability": "...",
             "effect": "...", "would_have_helped": false, "evidence": "<quote or empty>"}, ...],
 "net_effect": "helped|hurt|no_effect",
 "rationale": "<one sentence>"}"""


# ---------------------------------------------------------------------------
# Loading / parsing
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _find_benchmark_dir(run_dir: Path):
    for sub in run_dir.iterdir():
        if sub.is_dir() and "__" in sub.name:
            return sub
    return None


def _section_of(skill_id: str) -> str:
    """code_modification-00001 -> code_modification."""
    m = re.match(r"(.+)-\d+$", skill_id)
    return m.group(1) if m else skill_id


def extract_agent_log(traj: dict) -> str:
    """Agent-visible reasoning: assistant THOUGHT + command messages."""
    return "\n".join(
        m.get("content", "")
        for m in traj.get("messages", [])
        if m.get("role") == "assistant" and isinstance(m.get("content"), str)
    )


def extract_presented_skills(traj: dict) -> list[dict]:
    """Parse the injected '## Learned Strategies (Skillbook)' block."""
    for msg in traj.get("messages", []):
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        start = content.find("Learned Strategies")
        if start < 0:
            continue
        end = content.find("CRITICAL REMINDER", start)
        if end < 0:
            end = len(content)
        block = content[start:end]
        skills = []
        for part in re.split(r"### ", block)[1:]:
            head, _, body = part.partition("\n")
            head = head.strip()
            if re.match(r"[a-z_][a-z_-]*-\d+$", head):
                skills.append({"id": head, "section": _section_of(head),
                               "content": body.strip()})
        return skills
    return []


def load_run_meta(run_dir: Path) -> dict:
    """Pull the experiment knobs from config.json for the report header."""
    cfg = _load_json(run_dir / "config.json")
    if not cfg:
        return {}
    e = cfg.get("experiment", {})
    sb = e.get("skillbook", {})
    ret = sb.get("retrieval", {})
    llm = cfg.get("llm", {})

    def _model(d):
        m = d.get("model", "?")
        return m.split("/")[-1] if "/" in m else m

    retr = ret.get("enabled", False)
    return {
        "name": e.get("name", run_dir.name),
        "agent_model": _model(llm.get("agent", {})),
        "ace_model": _model(llm.get("ace", {})),
        "learn": "swe" if sb.get("custom_swe_learn") else "default",
        "mode": sb.get("mode", "?"),
        "retrieval": f"top_k={ret.get('top_k')}" if retr else "off (full inject)",
        "val_pass_k": e.get("val_pass_k", 1),
        "split": Path(e.get("split", {}).get("manifest", "")).name or "—",
    }


def content_map_from_run(run_dir: Path, it: int) -> dict:
    """id -> skill content, parsed from the injected blocks of val trajectories.

    This is the exact text the agent was shown, so it matches the judged skills.
    """
    bench = _find_benchmark_dir(run_dir)
    if bench is None:
        return {}
    val_dir = bench / "trajectories" / "val"
    if not val_dir.is_dir():
        return {}
    cmap = {}
    for inst_dir in val_dir.iterdir():
        if not inst_dir.is_dir():
            continue
        traj = _load_json(inst_dir / f"iter_{it}.json")
        if traj is None:
            continue
        for s in extract_presented_skills(traj):
            cmap.setdefault(s["id"], s["content"])
    return cmap


def _read_resolved(results_dir: Path, phase: str, inst: str, it: int):
    f = results_dir / phase / inst / f"iter_{it}.json"
    d = _load_json(f)
    if d is None:
        return None
    return bool(d.get("resolved", False))


def _find_external_baseline_traj(run_dir: Path, inst: str, it: int):
    """Locate a trajectory for `inst` in an external baseline run.

    Prefers val_baseline, then val, then train.
    """
    bench = _find_benchmark_dir(run_dir)
    if bench is None:
        return None
    tdir = bench / "trajectories"
    for phase in ("val_baseline", "val", "train"):
        f = tdir / phase / inst / f"iter_{it}.json"
        if f.exists():
            return _load_json(f), bench / "results", phase
    return None


def classify_outcome(skill_resolved, base_resolved):
    if skill_resolved is None or base_resolved is None:
        return "UNKNOWN"
    if skill_resolved and not base_resolved:
        return "GAINED"
    if base_resolved and not skill_resolved:
        return "LOST"
    return "STABLE_PASS" if skill_resolved else "STABLE_FAIL"


def load_records(run_dir: Path, ext_baseline: Path | None, it: int) -> list[dict]:
    """Build per-instance records: skills + both logs + objective outcome."""
    bench = _find_benchmark_dir(run_dir)
    if bench is None:
        sys.exit(f"no benchmark dir under {run_dir}")
    traj_dir = bench / "trajectories"
    results_dir = bench / "results"
    val_dir = traj_dir / "val"
    if not val_dir.is_dir():
        sys.exit(f"no trajectories/val under {bench} (need split-mode run)")

    records = []
    for inst_dir in sorted(p for p in val_dir.iterdir() if p.is_dir()):
        inst = inst_dir.name
        vfile = inst_dir / f"iter_{it}.json"
        vtraj = _load_json(vfile)
        if vtraj is None:
            continue
        skills = extract_presented_skills(vtraj)
        if not skills:
            continue  # no skillbook injected -> nothing to attribute

        skill_resolved = _read_resolved(results_dir, "val", inst, it)

        # Counterfactual baseline log + resolved flag.
        if ext_baseline is not None:
            found = _find_external_baseline_traj(ext_baseline, inst, it)
            if not found:
                continue
            btraj, b_results_dir, b_phase = found
            base_resolved = _read_resolved(b_results_dir, b_phase, inst, it)
            base_src = f"{ext_baseline.name}:{b_phase}"
        else:
            btraj = _load_json(traj_dir / "val_baseline" / inst / f"iter_{it}.json")
            if btraj is None:
                continue
            base_resolved = _read_resolved(results_dir, "val_baseline", inst, it)
            base_src = "in-run:val_baseline"

        records.append({
            "instance": inst,
            "skills": skills,
            "skill_log": extract_agent_log(vtraj),
            "base_log": extract_agent_log(btraj),
            "skill_resolved": skill_resolved,
            "base_resolved": base_resolved,
            "outcome": classify_outcome(skill_resolved, base_resolved),
            "baseline_source": base_src,
        })
    return records


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

def _truncate(text: str) -> str:
    if len(text) <= MAX_LOG_CHARS:
        return text
    head = text[: int(MAX_LOG_CHARS * 0.6)]
    tail = text[-int(MAX_LOG_CHARS * 0.4):]
    return head + "\n\n[... LOG TRUNCATED ...]\n\n" + tail


def _resolved_str(v):
    return {True: "PASS", False: "FAIL", None: "unknown"}[v]


def call_judge(api_key: str, model: str, rec: dict) -> dict:
    skills_txt = "\n".join(
        f"[{s['id']}] ({s['section']}) {s['content']}" for s in rec["skills"]
    )
    user = (
        f"OUTCOME: {rec['outcome']} "
        f"(with-skills={_resolved_str(rec['skill_resolved'])}, "
        f"baseline={_resolved_str(rec['base_resolved'])})\n\n"
        f"SKILLS:\n{skills_txt}\n\n"
        f"LOG_WITH_SKILLS:\n{_truncate(rec['skill_log'])}\n\n"
        f"LOG_BASELINE:\n{_truncate(rec['base_log'])}\n\n"
        "Judge each skill per the rubric. Return ONLY the JSON object."
    )
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 2500,
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        f"{ZAI_BASE}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    last_err = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                raise ValueError(f"no JSON in response: {content[:200]}")
            parsed = json.loads(m.group(0))
            if "skills" not in parsed:
                raise ValueError("missing 'skills'")
            return parsed
        except urllib.error.HTTPError as e:
            last_err = e
            # rate limit: back off hard and exponentially; other HTTP errors: short retry
            wait = min(60, 10 * 2 ** attempt) if e.code == 429 else 5 * (attempt + 1)
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"judge failed after retries: {last_err}")


def _api_key(env_name: str = "ZAI_API_KEY", env_file: str = ".env") -> str:
    key = os.environ.get(env_name)
    if not key and env_file and Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{env_name}="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit(f"{env_name} not found (env var or {env_file})")
    return key


# ---------------------------------------------------------------------------
# L3 aggregation
# ---------------------------------------------------------------------------

def aggregate(judgements: list[dict]) -> dict:
    """judgements: list of {instance, outcome, skills:[{id,section,influence,...}], ...}."""
    per_skill = defaultdict(lambda: {
        "section": "", "content": "", "seen": 0, "relevant": 0, "followed": 0,
        "ignored": 0, "contradicted": 0, "helped": 0, "hurt": 0, "neutral": 0,
        "irrelevant": 0, "would_have_helped": 0,
    })
    outcomes = defaultdict(int)
    net = defaultdict(int)

    for j in judgements:
        outcomes[j.get("outcome", "UNKNOWN")] += 1
        net[j.get("net_effect", "no_effect")] += 1
        for s in j.get("skills", []):
            sid = s.get("id")
            if not sid:
                continue
            agg = per_skill[sid]
            agg["section"] = s.get("section") or _section_of(sid)
            if s.get("content") and not agg["content"]:
                agg["content"] = s["content"]
            agg["seen"] += 1
            appl = s.get("applicability")
            if appl == "irrelevant":
                agg["irrelevant"] += 1
            else:
                agg["relevant"] += 1
            infl = s.get("influence")
            if infl in ("followed", "partially_followed"):
                agg["followed"] += 1
            elif infl == "ignored":
                agg["ignored"] += 1
            elif infl == "contradicted":
                agg["contradicted"] += 1
            eff = s.get("effect")
            # quality axis: harmful/improved only count when the agent acted on the skill
            acted = infl in ("followed", "partially_followed")
            if eff == "improved" and acted:
                agg["helped"] += 1
            elif eff == "harmful" and acted:
                agg["hurt"] += 1
            else:
                agg["neutral"] += 1
            if s.get("would_have_helped"):
                agg["would_have_helped"] += 1

    for sid, a in per_skill.items():
        seen = a["seen"]
        rel = a["relevant"]
        a["follow_pct"] = a["followed"] / seen if seen else 0.0
        a["adherence_pct"] = a["followed"] / rel if rel else 0.0  # follow rate when relevant
        a["irrelevant_pct"] = a["irrelevant"] / seen if seen else 0.0
        a["net_score"] = a["helped"] - a["hurt"]
        a["verdict"] = _verdict(a)

    # corpus-level rollups (the "diagnosis" numbers)
    tot_seen = sum(a["seen"] for a in per_skill.values())
    tot_rel = sum(a["relevant"] for a in per_skill.values())
    tot_foll = sum(a["followed"] for a in per_skill.values())
    tot_irrel = sum(a["irrelevant"] for a in per_skill.values())
    tot_whh = sum(a["would_have_helped"] for a in per_skill.values())
    verdicts = defaultdict(int)
    for a in per_skill.values():
        verdicts[a["verdict"]] += 1

    rollup = {
        "n_skills": len(per_skill),
        "skill_slots": tot_seen,
        "adherence_overall": (tot_foll / tot_rel) if tot_rel else 0.0,
        "irrelevant_share": (tot_irrel / tot_seen) if tot_seen else 0.0,
        "would_have_helped_total": tot_whh,
        "verdicts": dict(verdicts),
    }
    return {"per_skill": dict(per_skill), "outcomes": dict(outcomes),
            "net_effect": dict(net), "n_instances": len(judgements),
            "rollup": rollup}


def _verdict(a: dict) -> str:
    """Skill-QUALITY verdict (orthogonal to adherence below)."""
    seen = a["seen"]
    if seen == 0:
        return "UNKNOWN"
    # bad advice the agent acted on and it backfired
    if a["hurt"] >= 2 and a["hurt"] > a["helped"]:
        return "HARMFUL"
    # good advice that paid off when followed
    if a["helped"] >= 2 and a["helped"] > a["hurt"]:
        return "BENEFICIAL"
    # sound, relevant advice the agent keeps ignoring/contradicting -> adherence problem
    if a["would_have_helped"] >= 2 and a["adherence_pct"] < 0.34:
        return "UNDERUSED"
    # almost never applicable to the tasks it was injected into -> targeting/noise problem
    if a["irrelevant_pct"] >= 0.7:
        return "NOISE"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_legend():
    print("LEGEND")
    print("  What is compared: per val instance, the SAME agent solves the SAME task WITH the")
    print("  learned skillbook vs WITHOUT it (paired baseline). A GLM judge reads both logs.")
    print("  Outcomes (objective, from test results — resolved with-skills vs baseline):")
    print("    GAINED=baseline FAIL→skills PASS   LOST=baseline PASS→skills FAIL")
    print("    STABLE_PASS=both pass   STABLE_FAIL=both fail")
    print("  Per-skill columns:")
    print("    seen   = instances where this skill was injected into the prompt")
    print("    rel    = of those, how many the judge found RELEVANT to the task")
    print("    adh%   = adherence: of relevant cases, how often the agent actually FOLLOWED it")
    print("    help   = times the agent followed it and it IMPROVED the run")
    print("    hurt   = times the agent followed it and the ADVICE made the run worse")
    print("    whh    = would-have-helped: sound, relevant advice the agent IGNORED/contradicted")
    print("    irrel  = times the skill was irrelevant to the task")
    print("  VERDICT (skill quality | adherence):")
    print("    BENEFICIAL = helped when followed     HARMFUL = bad advice acted on, backfired")
    print("    NOISE      = almost never applicable   UNDERUSED = good advice the agent ignores")
    print("    NEUTRAL    = no clear signal")
    print()


def print_per_instance(judgements: list[dict], content_map: dict | None = None,
                       mode: str = "changed", content_chars: int = 80):
    """Per-instance view: for each interesting instance, which skills mattered and how.

    Uses only data already in the cache — no extra API calls.
    mode: "changed" = GAINED/LOST only; "net" = any net_effect != no_effect; "all".
    """
    content_map = content_map or {}

    def interesting(j):
        if mode == "all":
            return True
        if j.get("outcome") in ("GAINED", "LOST"):
            return True
        if mode == "net":
            return j.get("net_effect", "no_effect") != "no_effect"
        return False

    order = {"GAINED": 0, "LOST": 1, "STABLE_PASS": 2, "STABLE_FAIL": 3, "UNKNOWN": 4}
    rows = sorted((j for j in judgements if interesting(j)),
                  key=lambda j: (order.get(j.get("outcome"), 9), j.get("instance", "")))
    if not rows:
        return

    marker = {"GAINED": "▲", "LOST": "▼"}
    print(f"PER-INSTANCE  ({mode}: {len(rows)} of {len(judgements)} instances)")
    for j in rows:
        oc = j.get("outcome", "?")
        print(f"\n{marker.get(oc, '■')} {j['instance']}   {oc}   net={j.get('net_effect','?')}")
        if j.get("rationale"):
            print(f"    {j['rationale']}")
        mattered = []
        for s in j.get("skills", []):
            infl, eff = s.get("influence"), s.get("effect")
            if (infl in ("followed", "partially_followed", "contradicted")
                    or eff in ("improved", "harmful") or s.get("would_have_helped")):
                mattered.append(s)
        if not mattered:
            print("    (no skill had a clear influence)")
            continue
        # most decisive first
        rank = {"harmful": 0, "improved": 1, "neutral": 2}
        mattered.sort(key=lambda s: (rank.get(s.get("effect"), 3),
                                     not s.get("would_have_helped")))
        for s in mattered:
            tag = s.get("effect") if s.get("effect") in ("improved", "harmful") else s.get("influence")
            if s.get("would_have_helped"):
                tag = f"{tag}/would-help"
            sid = s.get("id", "?")
            line = f"    [{tag}] {sid}"
            txt = " ".join((content_map.get(sid) or s.get("content") or "").split())
            if txt and content_chars:
                if len(txt) > content_chars:
                    txt = txt[:content_chars - 1].rstrip() + "…"
                line += f"  — {txt}"
            print(line)
            ev = " ".join((s.get("evidence") or "").split())
            if ev:
                if len(ev) > 120:
                    ev = ev[:119].rstrip() + "…"
                print(f"        ↳ {ev}")
    print()


def print_report(run_dir: Path, model: str, it: int, base_src: str,
                 agg: dict, judgements: list[dict],
                 content_map: dict | None = None, content_chars: int = 0,
                 run_meta: dict | None = None, per_instance: str | None = None):
    ps = agg["per_skill"]
    oc = agg["outcomes"]
    n = agg["n_instances"]
    content_map = content_map or {}

    print()
    print(f"SKILL IMPACT — {run_dir.name}")
    if run_meta:
        m = run_meta
        print(f"  RUN: {m['name']}")
        print(f"  agent={m['agent_model']}  ace/reflector={m['ace_model']}  "
              f"learn={m['learn']}  skillbook={m['mode']}")
        print(f"  retrieval={m['retrieval']}  val_pass_k={m['val_pass_k']}  split={m['split']}")
    print(f"  judge={model} | n={n} instances | iter={it} | baseline: {base_src}")
    print()

    gained, lost = oc.get("GAINED", 0), oc.get("LOST", 0)
    print("Outcomes:  "
          f"GAINED {gained}  LOST {lost}  "
          f"STABLE_PASS {oc.get('STABLE_PASS', 0)}  STABLE_FAIL {oc.get('STABLE_FAIL', 0)}"
          + (f"  UNKNOWN {oc['UNKNOWN']}" if oc.get("UNKNOWN") else ""))
    ne = agg["net_effect"]
    print("Net skillbook (judge):  "
          f"helped {ne.get('helped', 0)}  hurt {ne.get('hurt', 0)}  "
          f"no_effect {ne.get('no_effect', 0)}")
    print()

    _print_legend()

    rows = sorted(ps.items(), key=lambda kv: (kv[1]["net_score"], kv[1]["seen"]),
                  reverse=True)
    # adher% = follow rate among RELEVANT instances; whh = would-have-helped count
    print(f"{'skill_id':<26}{'section':<17}{'seen':>5}{'rel':>4}{'adh%':>6}"
          f"{'help':>5}{'hurt':>5}{'whh':>5}{'irrel':>6}  VERDICT")
    print("-" * 95)
    for sid, a in rows:
        print(f"{sid:<26}{a['section'][:16]:<17}{a['seen']:>5}{a['relevant']:>4}"
              f"{a['adherence_pct']*100:>5.0f}%{a['helped']:>5}{a['hurt']:>5}"
              f"{a['would_have_helped']:>5}{a['irrelevant']:>6}  {a['verdict']}")
        if content_chars:
            txt = " ".join((content_map.get(sid) or a.get("content") or "").split())
            if txt:
                if len(txt) > content_chars:
                    txt = txt[:content_chars - 1].rstrip() + "…"
                print(f"      ↳ {txt}")

    def names(v):
        return sorted(s for s, a in ps.items() if a["verdict"] == v)
    benef, harmful = names("BENEFICIAL"), names("HARMFUL")
    underused, noise = names("UNDERUSED"), names("NOISE")
    print()
    print("Quality axis:")
    if benef:
        print(f"  + Beneficial ({len(benef)}): {', '.join(benef)}")
    if harmful:
        print(f"  ! Harmful — bad advice acted on ({len(harmful)}): {', '.join(harmful)}")
    if noise:
        print(f"  ~ Noise — rarely applicable ({len(noise)}): {', '.join(noise)}")
    print("Adherence axis:")
    if underused:
        print(f"  > Underused — sound advice the agent ignores ({len(underused)}): "
              f"{', '.join(underused)}")
    else:
        print("  > Underused: none")
    print()
    _print_diagnosis(agg)

    if per_instance:
        print()
        print_per_instance(judgements, content_map=content_map, mode=per_instance,
                           content_chars=content_chars or 80)


def _print_diagnosis(agg: dict):
    """Auto-generated verdict — bakes the manual read-out into the script."""
    r = agg["rollup"]
    v = r["verdicts"]
    adh = r["adherence_overall"]
    irr = r["irrelevant_share"]
    harmful_n = v.get("HARMFUL", 0)
    benef_n = v.get("BENEFICIAL", 0)
    noise_n = v.get("NOISE", 0)
    underused_n = v.get("UNDERUSED", 0)

    print("DIAGNOSIS")
    print(f"  adherence (followed/relevant): {adh*100:.0f}%   "
          f"irrelevant share of slots: {irr*100:.0f}%   "
          f"would-have-helped: {r['would_have_helped_total']}")
    print(f"  skill verdicts: BENEFICIAL {benef_n}  HARMFUL {harmful_n}  "
          f"UNDERUSED {underused_n}  NOISE {noise_n}  "
          f"NEUTRAL {v.get('NEUTRAL', 0)}  (of {r['n_skills']} skills)")

    findings = []
    if harmful_n >= 2 and harmful_n >= benef_n:
        findings.append(f"HARMFUL: {harmful_n} skills give bad advice the agent acts on — "
                        "review/remove these.")
    if irr >= 0.30 or noise_n >= max(3, 0.3 * r["n_skills"]):
        findings.append(f"TARGETING/NOISE: {irr*100:.0f}% of injected skills are irrelevant to the "
                        f"task ({noise_n} skills near-always irrelevant) — wholesale injection; "
                        "retrieval / top-k filtering would help.")
    if adh < 0.4 and underused_n >= 2:
        findings.append(f"ADHERENCE: agent follows relevant skills only {adh*100:.0f}% of the time "
                        f"({underused_n} sound skills underused) — weak agent or weak injection, "
                        "not bad skills.")
    if benef_n >= 2 and harmful_n == 0:
        findings.append(f"UPSIDE: {benef_n} skills measurably helped when followed.")
    if not findings:
        findings.append("No dominant pattern: skillbook is roughly neutral on these instances.")

    print("  conclusions:")
    for f in findings:
        print(f"    - {f}")
    print()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> dict:
    cache = {}
    if path and path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cache[rec["instance"]] = rec
            except Exception:
                continue
    return cache


def append_cache(path: Path, rec: dict):
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_judge(args):
    run_dir = Path(args.run)
    ext_baseline = Path(args.baseline) if args.baseline else None
    records = load_records(run_dir, ext_baseline, args.iter)
    print(f"instances with injected skillbook: {len(records)}", file=sys.stderr)

    if args.instances:
        wanted = {s.strip() for s in args.instances.split(",") if s.strip()}
        records = [r for r in records if r["instance"] in wanted]
        missing = wanted - {r["instance"] for r in records}
        if missing:
            print(f"  ! not found / no skillbook: {', '.join(sorted(missing))}", file=sys.stderr)
        print(f"filtered to {len(records)} requested instance(s)", file=sys.stderr)
    if args.limit:
        records = records[: args.limit]

    cache_path = Path(args.jsonl) if args.jsonl else None
    cache = {} if args.no_cache else load_cache(cache_path) if cache_path else {}
    base_src = records[0]["baseline_source"] if records else "n/a"

    api_key = _api_key(args.api_key_env, args.env_file)
    judgements = []
    todo = []
    for rec in records:
        if rec["instance"] in cache:
            judgements.append(cache[rec["instance"]])
        else:
            todo.append(rec)
    print(f"cached {len(judgements)}, to judge {len(todo)}", file=sys.stderr)

    def _do(rec):
        out = call_judge(api_key, args.model, rec)
        # enrich each skill judgement with its section + content (judge omits these)
        sect = {s["id"]: s["section"] for s in rec["skills"]}
        cont = {s["id"]: s["content"] for s in rec["skills"]}
        for s in out.get("skills", []):
            sid = s.get("id", "")
            s.setdefault("section", sect.get(sid, _section_of(sid)))
            s.setdefault("content", cont.get(sid, ""))
        return {
            "instance": rec["instance"],
            "outcome": rec["outcome"],
            "skill_resolved": rec["skill_resolved"],
            "base_resolved": rec["base_resolved"],
            "baseline_source": rec["baseline_source"],
            "skills": out.get("skills", []),
            "net_effect": out.get("net_effect", "no_effect"),
            "rationale": out.get("rationale", ""),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_do, rec): rec for rec in todo}
        done = 0
        for fut in as_completed(futs):
            rec = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ! {rec['instance']} failed: {e}", file=sys.stderr)
                continue
            judgements.append(res)
            append_cache(cache_path, res)
            done += 1
            print(f"  [{done}/{len(todo)}] {rec['instance']} "
                  f"-> {res['net_effect']} ({rec['outcome']})", file=sys.stderr)

    agg = aggregate(judgements)
    cmap = content_map_from_run(run_dir, args.iter) if args.content else None
    print_report(run_dir, args.model, args.iter, base_src, agg, judgements,
                 content_map=cmap, content_chars=args.content,
                 run_meta=load_run_meta(run_dir), per_instance=args.per_instance)

    if args.json:
        out = {
            "run": str(run_dir),
            "run_meta": load_run_meta(run_dir),
            "model": args.model,
            "iter": args.iter,
            "baseline_source": base_src,
            "aggregate": agg,
            "instances": judgements,
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}", file=sys.stderr)


def run_analyze(args):
    judgements = list(load_cache(Path(args.analyze)).values())
    if not judgements:
        sys.exit(f"no judgements in {args.analyze}")
    agg = aggregate(judgements)
    base_src = judgements[0].get("baseline_source", "n/a")
    # content comes from --run (exact injected text) if given, else from the cache itself
    cmap = content_map_from_run(Path(args.run), args.iter) if (args.content and args.run) else None
    meta = load_run_meta(Path(args.run)) if args.run else None
    print_report(Path(args.analyze), "(cached)", "-", base_src, agg, judgements,
                 content_map=cmap, content_chars=args.content, run_meta=meta,
                 per_instance=args.per_instance)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"aggregate": agg, "instances": judgements}, indent=2))
        print(f"wrote {args.json}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="run dir under study (split-mode)")
    p.add_argument("--baseline", help="external baseline run dir (default: in-run val_baseline)")
    p.add_argument("--iter", type=int, default=0, help="iteration index to judge (default 0)")
    p.add_argument("--instances", help="comma-separated instance ids to judge (e.g. for a "
                   "quick run on a couple of trajectories)")
    p.add_argument("--limit", type=int, help="judge only first N instances")
    p.add_argument("--workers", type=int, default=4, help="parallel judge calls")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"GLM model (default {DEFAULT_MODEL})")
    p.add_argument("--api-key-env", default="ZAI_API_KEY",
                   help="env var name holding the z.ai key (default ZAI_API_KEY)")
    p.add_argument("--env-file", default=".env",
                   help="path to a .env file to read the key from (default .env)")
    p.add_argument("--json", help="write full structured result to this path")
    p.add_argument("--jsonl", help="raw per-instance judge cache (resumable)")
    p.add_argument("--no-cache", action="store_true", help="ignore existing --jsonl cache")
    p.add_argument("--per-instance", nargs="?", const="changed", default=None,
                   choices=["changed", "net", "all"],
                   help="add a per-instance section showing which skills influenced each "
                   "instance (changed=GAINED/LOST only [default], net=any net effect, all)")
    p.add_argument("--content", type=int, nargs="?", const=160, default=0, metavar="CHARS",
                   help="show each skill's text under its verdict row (optional max chars, "
                   "default 160). With --analyze, pass the run dir too for exact text.")
    p.add_argument("--analyze", help="re-aggregate from an existing --jsonl cache (no API calls)")
    args = p.parse_args()

    if args.analyze:
        run_analyze(args)
    elif args.run:
        run_judge(args)
    else:
        p.error("provide a run dir, or --analyze <cache.jsonl>")


if __name__ == "__main__":
    main()
