#!/usr/bin/env python3
"""Qualitative analysis of skillbooks across experiment runs.

Descriptive analysis characterizing skillbook composition and measuring
skill utilization (by references in trajectories). Does NOT evaluate causal
contribution of skillbooks to resolution rate — resolution rate is shown only
as a run characteristic, never as a delta attributed to skillbooks.

Usage:
    uv run python scripts/analyze_skillbook_quality.py data/run_a data/run_b
    uv run python scripts/analyze_skillbook_quality.py data/                    # all run_* inside
    uv run python scripts/analyze_skillbook_quality.py data/run_a --output csv
    uv run python scripts/analyze_skillbook_quality.py data/run_a --list-dumps 5
    uv run python scripts/analyze_skillbook_quality.py data/run_a --learn-mode SWE
    uv run python scripts/analyze_skillbook_quality.py data/run_a --growth
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config.llm_catalog import get_effective_llm  # noqa: E402

# ---------------------------------------------------------------------------
# Configurable definitions
# ---------------------------------------------------------------------------

# NOTE: content-based "indirect" reference matching (n-grams of skill text
# appearing in the agent's output) was removed. It measured lexical overlap,
# not skill usage: on val_baseline trajectories that never saw the skillbook it
# fired at essentially the same rate, and what signal remained was confounded
# by the instance topic (the agent writes `CharField.__init__` because it is
# fixing that code, not because it read a skill). Only explicit ID citations —
# split into selective vs. dumped below — carry a clean signal.

# An "ID dump" is a single assistant message echoing (most of) the presented
# skill list wholesale rather than citing a skill selectively. The threshold
# is tied to how many skills were presented (= top_k under retrieval, = the
# whole skillbook otherwise), so it adapts across runs instead of using one
# magic number:
#     dump_threshold = max(DUMP_MIN_IDS, min(n_presented, DUMP_CAP_IDS))
# Rationale (validated on real runs): distinct-IDs-per-message is sharply
# bimodal — selective use is 1-4 IDs, real dumps are 30+, with an empty zone
# in between. DUMP_CAP_IDS sits inside that empty zone so large skillbooks
# (where the agent echoes a 50-150 ID chunk, not the full 748) are still
# caught, while a fraction-of-skillbook rule would miss them. For small
# presented sets the cap collapses to "cited the whole list"; DUMP_MIN_IDS
# keeps a 1-2 skill book from ever counting as a dump.
DUMP_CAP_IDS = 10
DUMP_MIN_IDS = 3


def _dump_threshold(n_presented: int) -> int:
    """Min distinct skill IDs in one message to call it an ID dump."""
    return max(DUMP_MIN_IDS, min(n_presented, DUMP_CAP_IDS))


# Subdirectory names that are split / eval-on-train PHASES rather than instance
# dirs. The standard two-phase split uses {train, val_baseline, val}; eval-on-train
# mode (experiment.eval_on_train) re-passes the TRAIN split and writes its phases
# as {train_eval_baseline (empty skillbook), train_eval (skillbook)}. Without
# these names here, the flat-layout fallback treats them as instance dirs and
# silently drops every nested trajectory/result -> Cite/Any Trajs go blank.
_PHASE_DIRS = {"train", "val_baseline", "val", "train_eval", "train_eval_baseline"}
# Order in which a phase claims the unprefixed instance key when several phases
# contain the same instance. The skillbook phase (val / train_eval) must win over
# its empty-skillbook baseline so reference analysis runs on the book-showing
# pass. The two families are mutually exclusive within a single run.
_PHASE_PRIORITY = ["val", "train_eval", "val_baseline", "train", "train_eval_baseline"]

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_TIKTOKEN_AVAILABLE = False
_tokenizer = None

try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    pass


def count_tokens(text: str) -> int:
    """Count tokens. Uses tiktoken cl100k_base if available, else chars/4."""
    if _TIKTOKEN_AVAILABLE and _tokenizer is not None:
        return len(_tokenizer.encode(text))
    return max(1, len(text) // 4)


TOKEN_METHOD = "tiktoken cl100k_base" if _TIKTOKEN_AVAILABLE else "chars/4 (rough estimate)"

# ---------------------------------------------------------------------------
# Data loading (following compare_runs.py patterns)
# ---------------------------------------------------------------------------


def _find_benchmark_dir(run_dir: Path) -> Path | None:
    """Find the benchmark subdirectory (e.g. princeton-nlp__SWE-bench_Lite)."""
    for sub in run_dir.iterdir():
        if sub.is_dir() and "__" in sub.name:
            return sub
    return None


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _detect_learn_mode(config: dict) -> str:
    """Detect SWE vs Default learn mode from config."""
    sb_cfg = config.get("experiment", {}).get("skillbook", {})
    if sb_cfg.get("custom_swe_learn", config.get("experiment", {}).get("custom_swe_learn", False)):
        return "SWE"
    return "Default"


def _detect_sb_mode(config: dict) -> str:
    """Detect skillbook mode (per_instance / per_repo / global)."""
    sb_cfg = config.get("experiment", {}).get("skillbook", {})
    return sb_cfg.get("mode", config.get("experiment", {}).get("skillbook_mode", "per_instance"))


def _detect_val_ratio(config: dict) -> float | None:
    """Detect val_ratio from config."""
    return config.get("experiment", {}).get("val_ratio", None)


def _detect_context_window(config: dict) -> int:
    """Detect context window from config."""
    return config.get("agent", {}).get("context", {}).get("context_window", 65536)


def _detect_max_attempts(config: dict) -> int:
    return config.get("experiment", {}).get("max_attempts", 1)


def _detect_retrieval(config: dict) -> dict:
    """Detect retrieval settings from config."""
    sb_cfg = config.get("experiment", {}).get("skillbook", {})
    ret = sb_cfg.get("retrieval", {})
    if ret.get("enabled", False):
        return {
            "enabled": True,
            "type": ret.get("type", "llm"),  # llm | bm25 | embedding | random
            "top_k": ret.get("top_k", 5),
            "skip_threshold": ret.get("skip_threshold", 10),
        }
    return {"enabled": False}


def _retrieval_type(run: dict) -> str:
    """Retrieval type label: 'bm25', 'llm', 'embedding', 'random', or 'off'.

    Used as a grouping/stratification key in aggregated tables — type only,
    no top_k (top_k is constant within a comparison).
    """
    retrieval = run.get("retrieval", {})
    if not retrieval.get("enabled", False):
        return "off"
    return retrieval.get("type", "llm")


def _retrieval_label(run: dict) -> str:
    """Retrieval column label: '{type} k={top_k}' when retrieval ran, else 'off'.

    Compact per-run label shown in tables (type + how many skills were presented).
    """
    retrieval = run.get("retrieval", {})
    if not retrieval.get("enabled", False):
        return "off"
    return f"{retrieval.get('type', 'llm')} k={retrieval.get('top_k', 5)}"


def _extract_skills(sb: dict) -> list[dict]:
    """Extract flat list of skill dicts from a skillbook."""
    return [
        {"id": sid, "section": s.get("section", ""), "content": s.get("content", "")}
        for sid, s in sb.get("skills", {}).items()
    ]


def _sb_token_count(sb: dict) -> int:
    """Total tokens in a skillbook (sum over all skill contents)."""
    return sum(count_tokens(s.get("content", "")) for s in sb.get("skills", {}).values())


def load_per_instance_skillbooks(run_dir: Path) -> dict[str, dict[int, dict]]:
    """Load all skillbook files for per_instance mode.

    Returns {instance_id: {iter_N: skillbook_dict, ...}, ...}.
    """
    bench_dir = _find_benchmark_dir(run_dir)
    if bench_dir is None:
        return {}
    sb_dir = bench_dir / "skillbooks"
    if not sb_dir.exists():
        return {}

    result = {}
    # Check for phase subdirs (split mode: train/val_baseline/val)
    known_phases = _PHASE_DIRS
    phase_dirs = [d for d in sb_dir.iterdir() if d.is_dir() and d.name in known_phases]
    scan_dirs = []
    if phase_dirs:
        for pd in phase_dirs:
            scan_dirs.extend(d for d in pd.iterdir() if d.is_dir())
    else:
        scan_dirs = [d for d in sb_dir.iterdir() if d.is_dir()]

    for inst_dir in sorted(scan_dirs):
        if not inst_dir.is_dir():
            continue
        instance_id = inst_dir.name
        iters = {}
        for f in sorted(inst_dir.glob("iter_*.json")):
            sb = _load_json(f)
            if sb is not None:
                iters[sb.get("iteration", 0)] = sb
        if iters:
            result[instance_id] = iters
    return result


def load_per_repo_skillbooks(run_dir: Path) -> dict[str, dict]:
    """Load per-repo final skillbooks. Returns {repo_name: skillbook_dict}."""
    bench_dir = _find_benchmark_dir(run_dir)
    if bench_dir is None:
        return {}
    pr_dir = bench_dir / "skillbooks" / "per_repo"
    if not pr_dir.exists():
        return {}
    result = {}
    for repo_dir in pr_dir.iterdir():
        if not repo_dir.is_dir():
            continue
        fp = repo_dir / "final_skillbook.json"
        sb = _load_json(fp)
        if sb is not None:
            result[repo_dir.name] = sb
    return result


def load_global_skillbook(run_dir: Path) -> dict | None:
    """Load global final_skillbook.json."""
    bench_dir = _find_benchmark_dir(run_dir)
    if bench_dir is None:
        return None
    fp = bench_dir / "skillbooks" / "final_skillbook.json"
    return _load_json(fp)


def load_train_iter_skillbooks(run_dir: Path) -> dict[int, dict]:
    """Load per-iteration train skillbooks (shared in split mode)."""
    bench_dir = _find_benchmark_dir(run_dir)
    if bench_dir is None:
        return {}
    train_dir = bench_dir / "skillbooks" / "train"
    if not train_dir.exists():
        return {}
    result = {}
    for f in sorted(train_dir.glob("iter_*.json")):
        sb = _load_json(f)
        if sb is not None:
            it = int(re.search(r"iter_(\d+)", f.name).group(1))
            result[it] = sb
    return result


def load_trajectories(run_dir: Path) -> dict[str, dict[int, dict]]:
    """Load trajectory files. Returns {instance_id: {iter_N: traj_dict}}.

    For split-mode runs (val/val_baseline/train subdirs), unprefixed keys
    point to the "val" phase (which has the skillbook). Phase-prefixed keys
    (e.g. "val_baseline/django__django-12345") give access to other phases.
    """
    bench_dir = _find_benchmark_dir(run_dir)
    if bench_dir is None:
        return {}
    traj_dir = bench_dir / "trajectories"
    if not traj_dir.exists():
        return {}

    result = {}
    known_phases = _PHASE_DIRS

    def _load_inst_dir(inst_dir: Path) -> dict[int, dict] | None:
        iters = {}
        for f in sorted(inst_dir.glob("iter_*.json")):
            traj = _load_json(f)
            if traj is not None:
                it_num = int(re.search(r"iter_(\d+)", f.name).group(1))
                iters[it_num] = traj
        return iters if iters else None

    phase_dirs = sorted(d for d in traj_dir.iterdir() if d.is_dir() and d.name in known_phases)
    if phase_dirs:
        for pd in phase_dirs:
            for inst_dir in sorted(d for d in pd.iterdir() if d.is_dir()):
                iters = _load_inst_dir(inst_dir)
                if iters:
                    # Prefixed key (always set)
                    result[f"{pd.name}/{inst_dir.name}"] = iters
        # Unprefixed keys: prefer val (has skillbook), then val_baseline, then train
        for phase_name in _PHASE_PRIORITY:
            phase_dir = traj_dir / phase_name
            if not phase_dir.exists():
                continue
            for inst_dir in phase_dir.iterdir():
                if not inst_dir.is_dir():
                    continue
                name = inst_dir.name
                if name not in result:
                    prefixed = f"{phase_name}/{name}"
                    if prefixed in result:
                        result[name] = result[prefixed]
    else:
        for inst_dir in sorted(d for d in traj_dir.iterdir() if d.is_dir()):
            iters = _load_inst_dir(inst_dir)
            if iters:
                result[inst_dir.name] = iters
    return result


def load_results(run_dir: Path) -> dict[str, dict[int, bool]]:
    """Load evaluation results. Returns {instance_id: {iter_N: resolved_bool}}.

    Same phase keying as load_trajectories: unprefixed keys point to val first,
    then val_baseline, then train — so they line up with trajectory keys.
    """
    bench_dir = _find_benchmark_dir(run_dir)
    if bench_dir is None:
        return {}
    res_dir = bench_dir / "results"
    if not res_dir.exists():
        return {}

    known_phases = _PHASE_DIRS

    def _load_inst_dir(inst_dir: Path) -> dict[int, bool] | None:
        iters = {}
        for f in sorted(inst_dir.glob("iter_*.json")):
            r = _load_json(f)
            if r is not None:
                it_num = int(re.search(r"iter_(\d+)", f.name).group(1))
                iters[it_num] = bool(r.get("resolved", False))
        return iters if iters else None

    result: dict[str, dict[int, bool]] = {}
    phase_dirs = sorted(d for d in res_dir.iterdir() if d.is_dir() and d.name in known_phases)
    if phase_dirs:
        for pd in phase_dirs:
            for inst_dir in sorted(d for d in pd.iterdir() if d.is_dir()):
                iters = _load_inst_dir(inst_dir)
                if iters:
                    result[f"{pd.name}/{inst_dir.name}"] = iters
        for phase_name in _PHASE_PRIORITY:
            phase_dir = res_dir / phase_name
            if not phase_dir.exists():
                continue
            for inst_dir in phase_dir.iterdir():
                if not inst_dir.is_dir():
                    continue
                name = inst_dir.name
                if name not in result and f"{phase_name}/{name}" in result:
                    result[name] = result[f"{phase_name}/{name}"]
    else:
        for inst_dir in sorted(d for d in res_dir.iterdir() if d.is_dir()):
            iters = _load_inst_dir(inst_dir)
            if iters:
                result[inst_dir.name] = iters
    return result


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------


def load_run(run_dir: Path) -> dict | None:
    """Load all data for a single run."""
    config = _load_json(run_dir / "config.json")
    stats = _load_json(run_dir / "statistics.json")
    if config is None:
        return None

    sb_mode = _detect_sb_mode(config)
    learn_mode = _detect_learn_mode(config)

    # Load skillbooks based on mode
    per_instance_sbs = {}
    per_repo_sbs = {}
    global_sb = None

    if sb_mode == "per_instance":
        per_instance_sbs = load_per_instance_skillbooks(run_dir)
    elif sb_mode == "per_repo":
        per_repo_sbs = load_per_repo_skillbooks(run_dir)
    elif sb_mode == "global":
        global_sb = load_global_skillbook(run_dir)

    # Fallback: load skillbooks from skillbook_source_dir if local skillbooks are empty
    has_local_sb = (
        bool(per_instance_sbs)
        or bool(per_repo_sbs)
        or global_sb is not None
        or bool(load_train_iter_skillbooks(run_dir))
    )
    if not has_local_sb:
        source_dir = config.get("experiment", {}).get("skillbook_source_dir")
        if source_dir:
            # Path is relative to CWD (project root), same as in the main codebase
            source_path = Path(source_dir).resolve()
            if source_path.exists():
                print(f"  Loading skillbooks from source_dir: {source_path}", file=sys.stderr)
                if sb_mode == "per_repo":
                    per_repo_sbs = load_per_repo_skillbooks(source_path)
                elif sb_mode == "global":
                    global_sb = load_global_skillbook(source_path)
                else:
                    per_instance_sbs = load_per_instance_skillbooks(source_path)
            else:
                print(f"  Warning: skillbook_source_dir not found: {source_path}", file=sys.stderr)

    train_iter_sbs = load_train_iter_skillbooks(run_dir)

    # Load trajectories for reference analysis
    trajectories = load_trajectories(run_dir)

    resolution_rate = 0.0
    resolved_count = 0
    total_instances = 0
    if stats:
        # For split/validation-only runs, top-level resolution_rate is 0
        # (train phase has no data). Use val_skillbook_phase instead.
        resolution_rate = stats.get("resolution_rate", 0.0)
        resolved_count = stats.get("resolved_count", 0)
        total_instances = stats.get("total_instances", 0)
        if resolution_rate == 0.0:
            vsb = stats.get("val_skillbook_phase", {})
            if isinstance(vsb, dict) and vsb.get("total_instances", 0) > 0:
                resolution_rate = vsb.get("resolution_rate", 0.0)
                resolved_count = vsb.get("resolved_count", 0)
                total_instances = vsb.get("total_instances", 0)

    # Extract model names
    llm = config.get("llm", {})
    agent_section = llm.get("agent", {})
    ace_section = llm.get("ace", {})
    agent_llm = (
        get_effective_llm(agent_section).get("model", "N/A")
        if agent_section
        else "N/A"
    )
    ace_llm = (
        get_effective_llm(ace_section).get("model", "N/A")
        if ace_section
        else "N/A"
    )

    return {
        "run_dir": run_dir,
        "run_name": run_dir.name,
        "config": config,
        "statistics": stats or {},
        "sb_mode": sb_mode,
        "learn_mode": learn_mode,
        "val_ratio": _detect_val_ratio(config),
        "context_window": _detect_context_window(config),
        "max_attempts": _detect_max_attempts(config),
        "retrieval": _detect_retrieval(config),
        "resolution_rate": resolution_rate,
        "resolved_count": resolved_count,
        "total_instances": total_instances,
        "agent_llm": agent_llm,
        "ace_llm": ace_llm,
        "llm_short": _model_short(agent_llm, ace_llm),
        "per_instance_sbs": per_instance_sbs,
        "per_repo_sbs": per_repo_sbs,
        "global_sb": global_sb,
        "train_iter_sbs": train_iter_sbs,
        "trajectories": trajectories,
    }


def discover_runs(paths: list[str]) -> list[Path]:
    """Expand paths: if a path is a parent dir, find all run_* inside it."""
    run_dirs = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"Warning: path not found: {path}", file=sys.stderr)
            continue
        if path.is_dir() and not (path / "config.json").exists():
            # Parent directory — look for run_* subdirs
            found = sorted(d for d in path.iterdir() if d.is_dir() and d.name.startswith("run_"))
            if found:
                run_dirs.extend(found)
            else:
                print(f"Warning: no run_* subdirs in {path}", file=sys.stderr)
        else:
            run_dirs.append(path)
    return run_dirs


def load_runs(paths: list[str], learn_filter: str | None = None, sb_mode_filter: str | None = None) -> list[dict]:
    """Load all runs, applying optional filters."""
    run_dirs = discover_runs(paths)
    runs = []
    skipped = 0
    for rd in run_dirs:
        run = load_run(rd)
        if run is None:
            print(f"Warning: skipping (missing config.json): {rd}", file=sys.stderr)
            skipped += 1
            continue
        # Check if there are any skillbooks at all
        has_sb = (
            bool(run["per_instance_sbs"])
            or bool(run["per_repo_sbs"])
            or run["global_sb"] is not None
            or bool(run["train_iter_sbs"])
        )
        if not has_sb:
            print(f"Warning: skipping (no skillbooks found): {rd.name}", file=sys.stderr)
            skipped += 1
            continue
        if learn_filter and run["learn_mode"] != learn_filter:
            skipped += 1
            continue
        if sb_mode_filter and run["sb_mode"] != sb_mode_filter:
            skipped += 1
            continue
        runs.append(run)
    return runs, skipped


# ---------------------------------------------------------------------------
# General vs Specific classification (NOT by prefix)
# ---------------------------------------------------------------------------


def classify_skill_specificity(content: str, section: str = "") -> str:
    """Classify a skill as 'general' or 'specific'.

    General: process/methodological, no concrete repo identifiers.
    Specific: mentions concrete files, classes, functions, APIs.

    NOT classified by prefix (AVOID/VERIFIED/CONSIDER) — by content analysis.
    """
    text = (content + " " + section).lower()

    # General patterns — process-level advice
    general_phrases = [
        r"claiming (task |)completion without",
        r"claiming (success|inability|false)",
        r"creating test scripts? instead of",
        r"verifying git diff",
        r"verify(ing)? (the |)(changes|results|fix|removal|modification)",
        r"implement(ing)? verification steps",
        r"minimal,? targeted fixes?",
        r"avoid using sed commands? to modify",
        r"complex (file |)manipulations? with sed",
        r"multiple flawed attempts? to edit",
        r"multiple bash commands in one response",
        r"making (extensive |)(file |)modifications? without",
        r"attempting complex (file |)?manipulations?",
        r"sending multiple bash commands",
        r"identify (the |)correct problem area",
        r"test fixes? with the original",
        r"implement minimal,? targeted",
        r"precise sed pattern matching",
        r"surgical fix",
        r"access source files? before implementing",
        r"root cause analysis",
        r"proper validation of code changes",
        r"analyze file structure thoroughly",
        r"avoid false positives",
        r"do not claim (success|completion)",
        r"ensure no other related",
        r"confirm(ing)? (that the|execution)",
    ]
    is_general_phrase = any(re.search(p, text) for p in general_phrases)

    # Specific indicators
    has_file_path = bool(re.search(r"[a-z_]+/[a-z_]+\.(py|js|ts|rst|txt|cfg|yaml)", text))
    has_camelcase = bool(re.search(r"\b[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*\b", content))
    has_func_ref = bool(re.search(r"__\w+__|\w+\(\)|\w+\.__\w+", content))
    domain_terms = [
        r"django\.(forms|core|contrib|db|template|http|urls|views|models)",
        r"(queryset|modeladmin|formfield|template_tag|urlconf)",
        r"(permutation|singularityfunction|latex|_print_\w+|combinatorics)",
        r"(figure\.py|axes3d|mpl_toolkits|backend_|canvas)",
        r"(assertion|_compare_eq|assertion_rewriting|byte string)",
        r"(isolationforest|warm_start|onehotencoder|sklearn\.)",
        r"(manpage|man_pages|doctree_read|builder|sphinx/)",
    ]
    has_domain = any(re.search(p, text) for p in domain_terms)
    has_line_ref = bool(re.search(r"line\s?\d+|lines?\s?\d+-\d+", text))
    has_param_ref = bool(re.search(r"`[a-z_]+`|_[a-z]+_=|self\.[a-z_]+", content))

    n_specific = sum([has_file_path, has_camelcase, has_func_ref, has_domain, has_line_ref, has_param_ref])

    if is_general_phrase and n_specific <= 1:
        return "general"
    if n_specific >= 2:
        return "specific"
    if has_file_path or has_domain:
        return "specific"
    if n_specific == 1 and not is_general_phrase:
        return "specific"
    return "general"


def _extract_presented_skills(traj: dict) -> list[dict]:
    """Extract skills presented in trajectory's system/user message.

    Returns list of {"id": str, "section": str, "content": str} for each
    skill found in the skillbook section of the trajectory prompt.
    """
    import re as _re
    for msg in traj.get("messages", []):
        content = msg.get("content", "")
        start = content.find("Learned Strategies")
        if start < 0:
            continue
        end = content.find("CRITICAL REMINDER", start)
        if end < 0:
            end = len(content)
        sb_section = content[start:end]

        parts = _re.split(r"### ", sb_section)
        skills = []
        for p in parts[1:]:
            lines = p.split("\n", 1)
            heading = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            # Match skill-like headings: <section>-<digits>. Section names may
            # contain hyphens (e.g. "bug-fixing-00001", "code-analysis-00002"),
            # so the char class includes "-"; rsplit then takes the section as
            # everything before the trailing "-NNNNN".
            if _re.match(r"[a-z][a-z0-9_-]*-\d+", heading):
                section = heading.rsplit("-", 1)[0]
                skills.append({"id": heading, "section": section, "content": body})
        return skills
    return []


def compute_presented_skill_specificity(run: dict) -> dict:
    """Compute general/specific breakdown of skills actually presented to agent.

    For non-retrieval runs: all skills in the skillbook.
    For retrieval runs: skills extracted from trajectory prompts (what agent saw).

    Returns {"general": int, "specific": int, "total": int,
             "gen_pct": float, "spec_pct": float, "per_instance": list[dict]}
    """
    retrieval = run.get("retrieval", {})
    ret_enabled = retrieval.get("enabled", False)
    # Use only unprefixed trajectory keys to avoid double-counting
    trajectories = {k: v for k, v in run.get("trajectories", {}).items() if "/" not in k}
    per_instance_counts = []

    if ret_enabled:
        # Extract presented skills from each trajectory and classify
        total_gen = 0
        total_spec = 0
        for inst_id, trajs in trajectories.items():
            for it, traj in trajs.items():
                presented = _extract_presented_skills(traj)
                if not presented:
                    continue
                gen = sum(1 for s in presented if classify_skill_specificity(s["content"], s["section"]) == "general")
                spec = len(presented) - gen
                total_gen += gen
                total_spec += spec
                per_instance_counts.append({"general": gen, "specific": spec, "total": len(presented)})
        total = total_gen + total_spec
    else:
        # All skills from skillbook are presented
        all_skills = []
        if run["sb_mode"] == "per_repo":
            for sb in run["per_repo_sbs"].values():
                all_skills.extend(_extract_skills(sb))
        elif run["sb_mode"] == "global" and run["global_sb"]:
            all_skills = _extract_skills(run["global_sb"])
        elif run["sb_mode"] == "per_instance":
            for sb in _get_final_skillbooks_per_instance(run):
                all_skills.extend(_extract_skills(sb))

        total_gen = sum(1 for s in all_skills if classify_skill_specificity(s["content"], s["section"]) == "general")
        total_spec = len(all_skills) - total_gen
        total = len(all_skills)

        # For per-instance counts, each instance sees all skills
        if all_skills and trajectories:
            for inst_id, trajs in trajectories.items():
                for it in trajs:
                    per_instance_counts.append({"general": total_gen, "specific": total_spec, "total": total})

    return {
        "general": total_gen,
        "specific": total_spec,
        "total": total,
        "gen_pct": total_gen / total * 100 if total > 0 else 0,
        "spec_pct": total_spec / total * 100 if total > 0 else 0,
        "per_instance": per_instance_counts,
    }


# ---------------------------------------------------------------------------
# Reference detection
# ---------------------------------------------------------------------------


def _extract_assistant_text(traj: dict) -> str:
    """Concatenate all assistant messages from a trajectory."""
    return " ".join(_extract_assistant_messages(traj))


def _extract_assistant_messages(traj: dict) -> list[str]:
    """List of assistant message contents from a trajectory."""
    return [
        m.get("content", "") for m in traj.get("messages", []) if m.get("role") == "assistant"
    ]


# --- Prose references -----------------------------------------------------
# The agent (qwen3next especially) often refers to the skillbook by GENERAL
# WORDS — "Based on the skill", "the skillbook suggests", "the skill
# description", "according to the skillbook" — instead of citing a skill ID.
# This is a distinct engagement channel from the ID citations below: many
# trajectories reference the skillbook in prose without ever naming an ID, so
# counting only IDs understates engagement. The full phrasing dictionary (the
# "how") is in data/skill_prose_phrasings.json; here we only need the boolean
# "did the agent refer to the skillbook by words at all".
#
# Detection runs on ASSISTANT messages only (the injected skillbook text lives
# in user messages and would otherwise dominate), after stripping code blocks so
# module paths / inline code don't fire it. "strategy" is deliberately NOT
# matched: in these trajectories it is dominated by the sklearn `strategy=`
# parameter (KBinsDiscretizer), "strategic fix", and generic "different
# strategy" — not skill references.
_PROSE_SKILL_RE = re.compile(r"\bskills?\b|\bskillbooks?\b", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")


def _strip_code(text: str) -> str:
    """Remove fenced and inline code so identifiers don't trip prose detection."""
    return _INLINE_CODE_RE.sub(" ", _CODE_BLOCK_RE.sub(" ", text))


def _has_prose_skill_ref(assistant_messages: list[str]) -> bool:
    """True if any assistant message refers to the skillbook by prose words."""
    return any(
        msg and _PROSE_SKILL_RE.search(_strip_code(msg))
        for msg in assistant_messages
    )


def _count_prose_skill_ref_msgs(assistant_messages: list[str]) -> int:
    """Number of assistant messages that reference the skillbook by prose words."""
    return sum(
        1 for msg in assistant_messages
        if msg and _PROSE_SKILL_RE.search(_strip_code(msg))
    )


def _find_explicit_refs(
    assistant_messages: list[str], skill_ids: list[str]
) -> tuple[set[str], set[str], list[dict]]:
    """Find skill IDs directly mentioned in assistant messages.

    Returns (selective, dumped, dump_msgs). A message citing >= dump_threshold
    distinct skill IDs is an ID dump — the agent echoing the presented list —
    and its IDs go to `dumped` instead of `selective`. The threshold is derived
    from how many skills were presented (len(skill_ids)) via _dump_threshold,
    so it scales with top_k / skillbook size. `dump_msgs` records each such
    message for manual inspection:
    {"msg_idx": int, "n_ids": int, "ids": list[str], "snippet": str}.
    """
    threshold = _dump_threshold(len(skill_ids))
    selective: set[str] = set()
    dumped: set[str] = set()
    dump_msgs: list[dict] = []
    for idx, msg in enumerate(assistant_messages):
        found = {sid for sid in skill_ids if sid in msg}
        if not found:
            continue
        if len(found) >= threshold:
            dumped |= found
            first = min((msg.find(sid) for sid in found if sid in msg), default=0)
            start = max(0, first - 80)
            dump_msgs.append({
                "msg_idx": idx,
                "n_ids": len(found),
                "ids": sorted(found),
                "snippet": msg[start:first + 200].replace("\n", " "),
            })
        else:
            selective |= found
    return selective, dumped - selective, dump_msgs


def compute_references(run: dict) -> dict:
    """Compute reference stats for a run (cached per run_dir).

    Returns {skill_id: {"explicit": int, "dumped": int, "specificity": str/None}, ...}
    plus aggregate stats.
    """
    # Cache by run_dir so we don't recompute for each output table
    cache_key = run.get("run_dir")
    if cache_key is not None and cache_key in compute_references._cache:
        return compute_references._cache[cache_key]

    sb_mode = run["sb_mode"]
    # Use only unprefixed trajectory keys (primary phase: val for split, all for flat)
    # to avoid double-counting phase-prefixed entries
    trajectories = {k: v for k, v in run["trajectories"].items() if "/" not in k}

    # Gather all skills and their IDs per unit of analysis
    # For per_instance: skills come from the skillbook at the iteration matching the trajectory
    # For per_repo/global: all skills from the final skillbook

    if sb_mode == "per_instance":
        result = _compute_refs_per_instance(run, trajectories)
    elif sb_mode == "per_repo":
        result = _compute_refs_per_repo(run, trajectories)
    elif sb_mode == "global":
        result = _compute_refs_global(run, trajectories)
    else:
        result = {"skill_refs": {}, "summary": {}}

    if cache_key is not None:
        compute_references._cache[cache_key] = result
    return result


compute_references._cache = {}


def _compute_presented_skill_refs(run: dict) -> dict:
    """Compute reference stats for skills actually presented to the agent.

    For retrieval runs: extracts only the top_k skills shown in each trajectory
    prompt, checks citations, classifies general/specific. Returns same format
    as compute_references() so it can be used as a drop-in replacement.
    """
    cache_key = run.get("run_dir")
    if cache_key is not None and cache_key in _compute_presented_skill_refs._cache:
        return _compute_presented_skill_refs._cache[cache_key]

    trajectories = {k: v for k, v in run.get("trajectories", {}).items() if "/" not in k}

    skill_refs = {}
    total_presentations = 0
    total_explicit = 0
    total_dumped = 0
    traj_iters = 0
    cite_traj_iters = 0
    dump_traj_iters = 0
    # Prose-engagement counters (distinct from ID citations). `any_traj_iters`
    # is the deduplicated union (ID-cite OR prose-ref) so callers don't
    # double-count trajectories that cite an ID AND reference in prose.
    prose_traj_iters = 0
    any_traj_iters = 0
    both_traj_iters = 0
    prose_msgs = 0
    dump_locations = []
    traj_records = []

    for inst_id, iters in trajectories.items():
        for it, traj in iters.items():
            presented = _extract_presented_skills(traj)
            if not presented:
                continue
            assistant_msgs = _extract_assistant_messages(traj)
            if not any(assistant_msgs):
                continue
            traj_iters += 1

            presented_ids = [s["id"] for s in presented]
            explicit, dumped, dump_msgs = _find_explicit_refs(assistant_msgs, presented_ids)
            prose = _has_prose_skill_ref(assistant_msgs)
            prose_msgs += _count_prose_skill_ref_msgs(assistant_msgs)
            traj_records.append((inst_id, it, bool(explicit)))
            if explicit:
                cite_traj_iters += 1
            if prose:
                prose_traj_iters += 1
            if explicit and prose:
                both_traj_iters += 1
            if explicit or prose:
                any_traj_iters += 1
            if dumped:
                dump_traj_iters += 1
            for dm in dump_msgs:
                dump_locations.append({**dm, "instance": inst_id, "iter": it})

            for s in presented:
                sid = s["id"]
                total_presentations += 1
                spec = classify_skill_specificity(s["content"], s["section"])

                is_explicit = sid in explicit
                is_dumped = sid in dumped

                if sid not in skill_refs:
                    skill_refs[sid] = {"explicit": 0, "dumped": 0, "specificity": spec}
                if is_explicit:
                    skill_refs[sid]["explicit"] += 1
                    total_explicit += 1
                if is_dumped:
                    skill_refs[sid]["dumped"] += 1
                    total_dumped += 1

    result = {
        "skill_refs": skill_refs,
        "dump_locations": dump_locations,
        "traj_records": traj_records,
        "summary": {
            "total_skill_presentations": total_presentations,
            "explicit_refs": total_explicit,
            "dumped_refs": total_dumped,
            "any_refs": total_explicit,
            "traj_iters": traj_iters,
            "cite_traj_iters": cite_traj_iters,
            "dump_traj_iters": dump_traj_iters,
            "prose_traj_iters": prose_traj_iters,
            "any_traj_iters": any_traj_iters,
            "both_traj_iters": both_traj_iters,
            "prose_ref_msgs": prose_msgs,
        },
    }
    if cache_key is not None:
        _compute_presented_skill_refs._cache[cache_key] = result
    return result


_compute_presented_skill_refs._cache = {}


def _compute_refs_per_instance(run: dict, trajectories: dict) -> dict:
    """Reference computation for per_instance mode.

    Reads the skills actually presented out of EACH trajectory's own prompt and
    checks selective/dump citations against them. This is robust to the
    skillbook-file <-> trajectory-file iteration correspondence, which is NOT a
    clean same-N mapping: attempt 1 has no skillbook, the book shown in
    trajectory ``iter_N`` is the one learned after the previous attempt, and
    resume / early-resolve make the two file numberings drift. Using the prompt
    gives exactly what the agent saw, per attempt. (``run`` is retained in the
    signature for call-site compatibility but is no longer used.)
    """
    skill_refs = {}  # skill_id -> {"explicit": int, "dumped": int, "specificity": str}
    dump_locations = []
    traj_records = []

    total_skills = 0
    total_explicit = 0
    total_dumped = 0
    traj_iters = 0
    cite_traj_iters = 0
    dump_traj_iters = 0
    prose_traj_iters = 0
    any_traj_iters = 0
    both_traj_iters = 0
    prose_msgs = 0

    for inst_id, iters in trajectories.items():
        for it, traj in iters.items():
            presented = _extract_presented_skills(traj)
            if not presented:
                continue  # attempt had no skillbook shown (e.g. first attempt)
            assistant_msgs = _extract_assistant_messages(traj)
            if not any(assistant_msgs):
                continue

            skill_ids = [s["id"] for s in presented]
            explicit, dumped, dump_msgs = _find_explicit_refs(assistant_msgs, skill_ids)
            prose = _has_prose_skill_ref(assistant_msgs)
            prose_msgs += _count_prose_skill_ref_msgs(assistant_msgs)
            traj_records.append((inst_id, it, bool(explicit)))
            for dm in dump_msgs:
                dump_locations.append({**dm, "instance": inst_id, "iter": it})

            for s in presented:
                sid = s["id"]
                if sid not in skill_refs:
                    skill_refs[sid] = {"explicit": 0, "dumped": 0, "specificity": classify_skill_specificity(s["content"], s["section"])}
                if sid in explicit:
                    skill_refs[sid]["explicit"] += 1
                if sid in dumped:
                    skill_refs[sid]["dumped"] += 1

            total_skills += len(presented)
            total_explicit += len(explicit)
            total_dumped += len(dumped)
            traj_iters += 1
            if explicit:
                cite_traj_iters += 1
            if prose:
                prose_traj_iters += 1
            if explicit and prose:
                both_traj_iters += 1
            if explicit or prose:
                any_traj_iters += 1
            if dumped:
                dump_traj_iters += 1

    return {
        "skill_refs": skill_refs,
        "dump_locations": dump_locations,
        "traj_records": traj_records,
        "summary": {
            "total_skill_presentations": total_skills,
            "explicit_refs": total_explicit,
            "dumped_refs": total_dumped,
            "any_refs": total_explicit,
            "traj_iters": traj_iters,
            "cite_traj_iters": cite_traj_iters,
            "dump_traj_iters": dump_traj_iters,
            "prose_traj_iters": prose_traj_iters,
            "any_traj_iters": any_traj_iters,
            "both_traj_iters": both_traj_iters,
            "prose_ref_msgs": prose_msgs,
        },
    }


def _compute_refs_for_skillbook(skills: list[dict], trajectories: dict, traj_phase: str = "") -> dict:
    """Common reference computation for per_repo/global modes.

    For per_repo: match trajectories to repo by instance ID prefix.
    For global: all trajectories.
    """
    skill_refs = {}
    dump_locations = []
    traj_records = []

    total_skills_seen = 0
    total_explicit = 0
    total_dumped = 0
    traj_iters = 0
    cite_traj_iters = 0
    dump_traj_iters = 0
    prose_traj_iters = 0
    any_traj_iters = 0
    both_traj_iters = 0
    prose_msgs = 0

    # Build skill lookup
    skill_by_id = {s["id"]: s for s in skills}
    skill_ids = list(skill_by_id.keys())

    for inst_id, trajs in trajectories.items():
        # per_repo/global: skillbook is shared and present in all iters (including 0)
        for it, traj in trajs.items():
            assistant_msgs = _extract_assistant_messages(traj)
            if not any(assistant_msgs):
                continue

            explicit, dumped, dump_msgs = _find_explicit_refs(assistant_msgs, skill_ids)
            prose = _has_prose_skill_ref(assistant_msgs)
            prose_msgs += _count_prose_skill_ref_msgs(assistant_msgs)
            traj_records.append((inst_id, it, bool(explicit)))
            for dm in dump_msgs:
                dump_locations.append({**dm, "instance": inst_id, "iter": it})

            for sid in explicit | dumped:
                if sid not in skill_refs:
                    skill_refs[sid] = {"explicit": 0, "dumped": 0, "specificity": classify_skill_specificity(skill_by_id[sid]["content"], skill_by_id[sid]["section"])}
                if sid in explicit:
                    skill_refs[sid]["explicit"] += 1
                if sid in dumped:
                    skill_refs[sid]["dumped"] += 1

            total_skills_seen += len(skills)
            total_explicit += len(explicit)
            total_dumped += len(dumped)
            traj_iters += 1
            if explicit:
                cite_traj_iters += 1
            if prose:
                prose_traj_iters += 1
            if explicit and prose:
                both_traj_iters += 1
            if explicit or prose:
                any_traj_iters += 1
            if dumped:
                dump_traj_iters += 1

    # Ensure all skills are in skill_refs even if never referenced
    for s in skills:
        if s["id"] not in skill_refs:
            skill_refs[s["id"]] = {"explicit": 0, "dumped": 0, "specificity": classify_skill_specificity(s["content"], s["section"])}

    return {
        "skill_refs": skill_refs,
        "dump_locations": dump_locations,
        "traj_records": traj_records,
        "summary": {
            "total_skill_presentations": total_skills_seen,
            "explicit_refs": total_explicit,
            "dumped_refs": total_dumped,
            "any_refs": total_explicit,
            "traj_iters": traj_iters,
            "cite_traj_iters": cite_traj_iters,
            "dump_traj_iters": dump_traj_iters,
            "prose_traj_iters": prose_traj_iters,
            "any_traj_iters": any_traj_iters,
            "both_traj_iters": both_traj_iters,
            "prose_ref_msgs": prose_msgs,
        },
    }


def _compute_refs_per_repo(run: dict, trajectories: dict) -> dict:
    """Reference computation for per_repo mode."""
    per_repo = run["per_repo_sbs"]
    if not per_repo:
        return {"skill_refs": {}, "summary": {}}

    combined_refs = {}
    combined_dumps = []
    combined_records = []
    combined_summary = {
        "total_skill_presentations": 0,
        "explicit_refs": 0,
        "dumped_refs": 0,
        "any_refs": 0,
        "traj_iters": 0,
        "cite_traj_iters": 0,
        "dump_traj_iters": 0,
        "prose_traj_iters": 0,
        "any_traj_iters": 0,
        "both_traj_iters": 0,
        "prose_ref_msgs": 0,
    }

    for repo_name, sb in per_repo.items():
        skills = _extract_skills(sb)
        # Filter trajectories for this repo
        repo_prefix = repo_name.replace("__", "/")
        repo_trajs = {
            inst_id: iters
            for inst_id, iters in trajectories.items()
            if inst_id.startswith(repo_prefix.replace("/", "__"))
        }
        result = _compute_refs_for_skillbook(skills, repo_trajs)
        combined_refs.update(result["skill_refs"])
        combined_dumps.extend(result.get("dump_locations", []))
        combined_records.extend(result.get("traj_records", []))
        for k in combined_summary:
            combined_summary[k] += result["summary"].get(k, 0)

    return {"skill_refs": combined_refs, "dump_locations": combined_dumps,
            "traj_records": combined_records, "summary": combined_summary}


def _compute_refs_global(run: dict, trajectories: dict) -> dict:
    """Reference computation for global mode."""
    global_sb = run["global_sb"]
    if global_sb is None:
        return {"skill_refs": {}, "summary": {}}
    skills = _extract_skills(global_sb)
    return _compute_refs_for_skillbook(skills, trajectories)


# ---------------------------------------------------------------------------
# Table formatting (following compare_runs.py patterns)
# ---------------------------------------------------------------------------


def _print_table_rows(headers: list[str], rows: list[dict]):
    """Print a formatted ASCII table with auto-width columns."""
    col_widths = {}
    for h in headers:
        max_w = len(h)
        for row in rows:
            val = row.get(h, "")
            for line in str(val).split("\n"):
                max_w = max(max_w, len(line))
        col_widths[h] = max_w

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)
    print(header_line)
    print(sep_line)
    for row in rows:
        cell_lines = {}
        max_lines = 1
        for h in headers:
            lines = str(row.get(h, "")).split("\n")
            cell_lines[h] = lines
            max_lines = max(max_lines, len(lines))
        for i in range(max_lines):
            parts = []
            for h in headers:
                line = cell_lines[h][i] if i < len(cell_lines[h]) else ""
                parts.append(line.ljust(col_widths[h]))
            print(" | ".join(parts))


def _median_str(values: list[float | int]) -> str:
    if not values:
        return "-"
    return f"{statistics.median(values):.1f}"


def _mean_str(values: list[float | int], fmt: str = ".1f") -> str:
    if not values:
        return "-"
    return f"{statistics.mean(values):{fmt}}"


def _med_mean_str(values: list[float | int]) -> str:
    """Format as 'median (mean)'."""
    if not values:
        return "-"
    return f"{statistics.median(values):.1f} ({statistics.mean(values):.1f})"


_MODEL_ALIASES = {
    "Qwen3-Coder-30B": "qwen3coder",
    "Qwen3-Coder-Next-FP8": "qwen3coder-next",
    "glm-4.5-flash": "glm45-flash",
}


def _model_short(agent_llm: str, ace_llm: str) -> str:
    """Short model label (following compare_runs.py pattern)."""
    def _short(m):
        if m == "N/A":
            return "-"
        base = m.split("/")[-1].replace("-Instruct", "").replace("-A3B", "")
        return _MODEL_ALIASES.get(base, base)
    a, b = _short(agent_llm), _short(ace_llm)
    return a if a == b else f"{a}/{b}"


def _short_run(name: str, max_len: int = 30) -> str:
    """Shorten run dir name for display."""
    # Strip 'run_' prefix and '_completed_*' suffix
    s = re.sub(r"^run_", "", name)
    s = re.sub(r"_completed.*$", "", s)
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    return s


# ---------------------------------------------------------------------------
# Output 1: Summary tables
# ---------------------------------------------------------------------------


def _get_final_skillbooks_per_instance(run: dict) -> list[dict]:
    """Get final (last iteration) skillbook for each instance."""
    result = []
    for inst_id, iters in run["per_instance_sbs"].items():
        if not iters:
            continue
        final_iter = max(iters.keys())
        if final_iter == 0:
            continue  # iter_0 has no learned skills
        result.append(iters[final_iter])
    return result


def _compute_skillbook_stats(skillbooks: list[dict], context_window: int) -> dict:
    """Compute aggregate stats over a list of skillbooks."""
    if not skillbooks:
        return {}

    skill_counts = []
    token_counts = []
    skill_token_lengths = []

    for sb in skillbooks:
        skills = _extract_skills(sb)
        skill_counts.append(len(skills))
        total_tokens = sum(count_tokens(s["content"]) for s in skills)
        token_counts.append(total_tokens)
        for s in skills:
            skill_token_lengths.append(count_tokens(s["content"]))

    if not skill_counts:
        return {}

    median_sb_tokens = statistics.median(token_counts) if token_counts else 0
    ctx_pct = (median_sb_tokens / context_window * 100) if context_window > 0 else 0

    return {
        "n_skillbooks": len(skillbooks),
        "skill_count_med": statistics.median(skill_counts),
        "skill_count_mean": statistics.mean(skill_counts),
        "token_count_med": statistics.median(token_counts),
        "token_count_mean": statistics.mean(token_counts),
        "ctx_pct": ctx_pct,
        "skill_len_med": statistics.median(skill_token_lengths) if skill_token_lengths else 0,
        "skill_len_mean": statistics.mean(skill_token_lengths) if skill_token_lengths else 0,
        "max_iters": max(
            max(sb.get("iteration", 0) for sb in [s for s in skillbooks])
            for skillbooks in [skillbooks]
        ) if skillbooks else 0,
    }


def _baseline_delta(run: dict) -> tuple[float, float, float] | None:
    """Paired (sb_rate, baseline_rate, delta) on instances in both phases.

    Each val instance is run twice on the same task — with the learned skillbook
    (val) and with an empty one (val_baseline). Returns None for runs without a
    paired baseline (single-phase / flat runs). The delta is exact on the shared
    instance set, so it is negative when the skillbook hurts resolution.
    """
    sp = _paired_resolve(load_results(run["run_dir"]))
    if sp["n"] == 0:
        return None
    n = sp["n"]
    base_res = (sp["n11"] + sp["n01"]) / n * 100
    sb_res = (sp["n11"] + sp["n10"]) / n * 100
    return sb_res, base_res, sb_res - base_res


def _res_rate_cell(run: dict) -> str:
    """Resolve-rate cell for the summary table.

    Line 1: skillbook (val) resolve rate. When a paired val_baseline exists for
    the same instances, append two more lines — the baseline (empty-skillbook)
    rate and the skillbook delta (sb − baseline, in pp), i.e. the lift or
    regression over running with no skillbook. Runs without a baseline show the
    resolve rate alone.
    """
    bd = _baseline_delta(run)
    if bd is None:
        return f"{run['resolution_rate']*100:.1f}%"
    sb_res, base_res, delta = bd
    return f"{sb_res:.1f}%\nbase {base_res:.1f}%\nΔ{delta:+.1f}pp"


def _res_rate_compact(run: dict) -> str:
    """Compact single-line resolve-rate cell: 'bsr B% / r S% / Δ Dpp'.

    bsr = val_baseline (empty-skillbook) rate; r = skillbook (val) rate;
    Δ = r − bsr, signed, in pp. Runs without a paired baseline show just 'r S%'.
    """
    bd = _baseline_delta(run)
    if bd is None:
        return f"r{run['resolution_rate']*100:.1f}%"
    sb_res, base_res, delta = bd
    return f"bsr{base_res:.1f}% / r{sb_res:.1f}% / Δ{delta:+.1f}pp"


def print_summary_tables(runs: list[dict]):
    """Output 1: Two summary tables — per_instance vs per_repo/global."""
    # Partition runs
    pi_runs = [r for r in runs if r["sb_mode"] == "per_instance"]
    pr_runs = [r for r in runs if r["sb_mode"] == "per_repo"]
    gl_runs = [r for r in runs if r["sb_mode"] == "global"]

    if pi_runs:
        print("=== Per-instance skillbooks: summary ===")
        print()
        headers = ["Run", "LLM", "Learn", "Retrieval", "Split", "Iters", "Res Rate", "SBs", "Skills/SB", "Tokens/SB", "Ctx%", "Skill tok"]
        rows = []
        for r in pi_runs:
            final_sbs = _get_final_skillbooks_per_instance(r)
            st = _compute_skillbook_stats(final_sbs, r["context_window"])
            if not st:
                continue
            rows.append({
                "Run": _short_run(r["run_name"]),
                "LLM": r["llm_short"],
                "Learn": r["learn_mode"],
                "Retrieval": _retrieval_label(r),
                "Split": f"{r['val_ratio']:.2f}" if r["val_ratio"] else "-",
                "Iters": str(r["max_attempts"]),
                "Res Rate": _res_rate_cell(r),
                "SBs": str(st["n_skillbooks"]),
                "Skills/SB": _med_mean_str([st["skill_count_med"], st["skill_count_mean"]])
                    if st else "-",
                "Tokens/SB": _med_mean_str([st["token_count_med"], st["token_count_mean"]])
                    if st else "-",
                "Ctx%": f"{st['ctx_pct']:.1f}%" if st else "-",
                "Skill tok": _med_mean_str([st["skill_len_med"], st["skill_len_mean"]])
                    if st else "-",
            })
        # Fix Skills/SB etc — need raw values
        for r_data, r_run in zip(rows, pi_runs):
            final_sbs = _get_final_skillbooks_per_instance(r_run)
            skill_counts = [len(_extract_skills(sb)) for sb in final_sbs if _extract_skills(sb)]
            token_counts = []
            for sb in final_sbs:
                skills = _extract_skills(sb)
                if skills:
                    token_counts.append(sum(count_tokens(s["content"]) for s in skills))
            skill_tok_lens = []
            for sb in final_sbs:
                for s in _extract_skills(sb):
                    skill_tok_lens.append(count_tokens(s["content"]))

            r_data["Skills/SB"] = _med_mean_str(skill_counts) if skill_counts else "-"
            r_data["Tokens/SB"] = _med_mean_str(token_counts) if token_counts else "-"
            cw = r_run["context_window"]
            if token_counts and cw > 0:
                r_data["Ctx%"] = f"{statistics.median(token_counts)/cw*100:.1f}%"
            r_data["Skill tok"] = _med_mean_str(skill_tok_lens) if skill_tok_lens else "-"

        if rows:
            _print_table_rows(headers, rows)
        print()

    if pr_runs or gl_runs:
        print("=== Per-repo / global skillbooks: summary ===")
        print()
        headers = ["Run", "LLM", "Learn", "Retrieval", "Split", "Res Rate", "Mode", "SBs", "Skills/SB", "Tokens/SB", "Ctx%", "Skill tok"]
        rows = []
        for r in pr_runs + gl_runs:
            if r["sb_mode"] == "per_repo":
                sbs = list(r["per_repo_sbs"].values())
            else:
                sbs = [r["global_sb"]] if r["global_sb"] else []

            if not sbs:
                continue

            skill_counts = [len(_extract_skills(sb)) for sb in sbs]
            token_counts = [sum(count_tokens(s["content"]) for s in _extract_skills(sb)) for sb in sbs]
            all_skill_toks = [count_tokens(s["content"]) for sb in sbs for s in _extract_skills(sb)]

            cw = r["context_window"]
            ctx_pct = (statistics.median(token_counts) / cw * 100) if token_counts and cw > 0 else 0

            rows.append({
                "Run": _short_run(r["run_name"]),
                "LLM": r["llm_short"],
                "Learn": r["learn_mode"],
                "Retrieval": _retrieval_label(r),
                "Split": f"{r['val_ratio']:.2f}" if r["val_ratio"] else "-",
                "Res Rate": _res_rate_cell(r),
                "Mode": r["sb_mode"],
                "SBs": str(len(sbs)),
                "Skills/SB": _med_mean_str(skill_counts),
                "Tokens/SB": _med_mean_str(token_counts),
                "Ctx%": f"{ctx_pct:.1f}%",
                "Skill tok": _med_mean_str(all_skill_toks),
            })
        if rows:
            _print_table_rows(headers, rows)
        print("  (Res Rate: line 1 = skillbook (val) resolve rate; 'base' = paired "
              "val_baseline (empty-skillbook) rate; 'Δ' = skillbook − baseline, in pp)")
        print()


# ---------------------------------------------------------------------------
# Output 2: ASCII growth chart
# ---------------------------------------------------------------------------


def print_growth_chart(runs: list[dict], run_filter: list[str] | None = None):
    """Output 2: ASCII chart of skillbook growth by iteration."""
    # Only per_instance runs with multiple iterations
    pi_runs = [r for r in runs if r["sb_mode"] == "per_instance" and r["max_attempts"] > 1]
    if run_filter:
        pi_runs = [r for r in pi_runs if any(f in r["run_name"] for f in run_filter)]
    if not pi_runs:
        print("(No per_instance multi-iteration runs for growth chart)")
        return

    print("=== Skillbook growth by iteration (per_instance, avg skills per live instance) ===")
    print()

    for r in pi_runs:
        per_inst = r["per_instance_sbs"]
        if not per_inst:
            continue

        # Collect: for each iteration, count live instances and their skill counts
        iter_data = defaultdict(list)  # iter -> [skill_count, ...]
        for inst_id, iters in per_inst.items():
            for it, sb in iters.items():
                n_skills = sb.get("skill_count", 0)
                if n_skills > 0:
                    iter_data[it].append(n_skills)

        if not iter_data:
            continue

        # Build chart data
        iterations = sorted(iter_data.keys())
        avgs = []
        ns = []
        for it in iterations:
            counts = iter_data[it]
            n = len(counts)
            avg = statistics.mean(counts) if counts else 0
            avgs.append(avg)
            ns.append(n)

        # Compute deltas
        deltas = [None] + [avgs[i] - avgs[i - 1] for i in range(1, len(avgs))]

        # ASCII chart
        max_avg = max(avgs) if avgs else 1
        chart_height = 15

        ret_tag = f" [{_retrieval_label(r)}]" if r.get("retrieval", {}).get("enabled") else ""
        print(f"  {_short_run(r['run_name'], 50)} [{r['learn_mode']}]{ret_tag}")
        print()

        for row in range(chart_height, 0, -1):
            threshold = max_avg * row / chart_height
            line_parts = []
            for avg in avgs:
                if avg >= threshold:
                    line_parts.append("  ##")
                else:
                    line_parts.append("    ")
            label = f"  {threshold:>5.0f} |" + "".join(line_parts)
            print(label)

        # X-axis
        print("       +" + "-" * (len(iterations) * 4))
        x_labels = "        " + "".join(f" {it:>2} " for it in iterations)
        print(x_labels)

        # Annotations
        annot = "        "
        for i, _ in enumerate(iterations):
            annot += f"n={ns[i]:>2} "
        print(f"  {annot}")
        annot2 = "        "
        for i, it in enumerate(iterations):
            if deltas[i] is not None:
                annot2 += f"Δ{deltas[i]:>+4.1f} "
            else:
                annot2 += "      "
        print(f"  {annot2}")
        print()


# ---------------------------------------------------------------------------
# Output 3: Reference rate by learn mode × skill type
# ---------------------------------------------------------------------------


def print_reference_table(runs: list[dict]):
    """Output 3: Reference rate in cross-tab of learn_mode × retrieval × specificity."""
    print("=== Reference rate by learn mode × retrieval × skill type ===")
    print("  (Retrieval runs: stats based on top_k skills actually presented, not full skillbook)")
    print("  (Cited = explicit selective citation; Dumped counted separately and NOT included in Cited)")
    print()

    # Aggregate across all runs, grouped by (learn_mode, retrieval_type). Retrieval
    # type is a visible dimension here because it changes the citation denominator
    # (top_k presented vs the full skillbook), so mixing types in one bucket is muddy.
    # For each run, compute references and classify skills
    agg = defaultdict(lambda: {
        "general_total": 0, "specific_total": 0,
        "general_cited": 0, "specific_cited": 0,
        "general_explicit": 0, "specific_explicit": 0,
        "general_dumped": 0, "specific_dumped": 0,
    })

    for r in runs:
        ret_enabled = r.get("retrieval", {}).get("enabled", False)
        ref_data = _compute_presented_skill_refs(r) if ret_enabled else compute_references(r)
        skill_refs = ref_data.get("skill_refs", {})
        learn = r["learn_mode"]
        rtype = _retrieval_type(r)
        a = agg[(learn, rtype)]

        for sid, ref_info in skill_refs.items():
            spec = ref_info.get("specificity", "general")
            is_cited = ref_info["explicit"] > 0
            key_prefix = "general" if spec == "general" else "specific"

            a[f"{key_prefix}_total"] += 1
            if is_cited:
                a[f"{key_prefix}_cited"] += 1
            if ref_info["explicit"] > 0:
                a[f"{key_prefix}_explicit"] += 1
            if ref_info.get("dumped", 0) > 0:
                a[f"{key_prefix}_dumped"] += 1

    if not any(a for a in agg.values()):
        print("  (No reference data available)")
        return

    headers = ["Category", "Skills", "% SB", "Cited", "Ref Rate", "Explicit", "Dumped"]
    rows = []

    for learn, rtype in sorted(agg.keys()):
        a = agg[(learn, rtype)]
        total_all = a["general_total"] + a["specific_total"]

        for category, prefix in [("General", "general"), ("Specific", "specific")]:
            n = a[f"{prefix}_total"]
            if n == 0:
                continue
            pct_sb = n / total_all * 100 if total_all > 0 else 0
            cited = a[f"{prefix}_cited"]
            ref_rate = cited / n * 100 if n > 0 else 0
            rows.append({
                "Category": f"{category} ({learn}/{rtype})",
                "Skills": str(n),
                "% SB": f"{pct_sb:.1f}%",
                "Cited": str(cited),
                "Ref Rate": f"{ref_rate:.1f}%",
                "Explicit": str(a[f"{prefix}_explicit"]),
                "Dumped": str(a[f"{prefix}_dumped"]),
            })

    # Totals
    for learn, rtype in sorted(agg.keys()):
        a = agg[(learn, rtype)]
        total = a["general_total"] + a["specific_total"]
        total_cited = a["general_cited"] + a["specific_cited"]
        if total == 0:
            continue
        ref_rate = total_cited / total * 100
        rows.append({
            "Category": f"TOTAL ({learn}/{rtype})",
            "Skills": str(total),
            "% SB": "100.0%",
            "Cited": str(total_cited),
            "Ref Rate": f"{ref_rate:.1f}%",
            "Explicit": str(a["general_explicit"] + a["specific_explicit"]),
            "Dumped": str(a["general_dumped"] + a["specific_dumped"]),
        })

    _print_table_rows(headers, rows)
    print()

    # Self-check
    for learn, rtype in sorted(agg.keys()):
        a = agg[(learn, rtype)]
        total = a["general_total"] + a["specific_total"]
        gen_pct = a["general_total"] / total * 100 if total > 0 else 0
        spec_pct = a["specific_total"] / total * 100 if total > 0 else 0
        if abs(gen_pct + spec_pct - 100.0) > 0.1:
            print(f"  WARNING: self-check failed for {learn}/{rtype}: Gen({gen_pct:.1f}%) + Spec({spec_pct:.1f}%) != 100%", file=sys.stderr)
    print("  (Self-check: Gen + Spec = Total for each row — OK)")
    print()


# ---------------------------------------------------------------------------
# Output 4: Per-run reference table (retrieval-aware)
# ---------------------------------------------------------------------------


def _compute_presented_ref_rate(run: dict) -> dict:
    """Compute selective-citation rate among actually presented skills.

    For retrieval runs: skills extracted from each trajectory prompt (top_k).
    For non-retrieval: the full skillbook. A skill counts as cited only if it
    was cited selectively (explicit); dumped IDs do not count.

    Returns {"cited_presented": int, "total_presented": int, "rate": float}
    """
    ret_enabled = run.get("retrieval", {}).get("enabled", False)

    if not ret_enabled:
        ref_data = compute_references(run)
        skill_refs = ref_data.get("skill_refs", {})
        n_cited = sum(1 for r in skill_refs.values() if r["explicit"] > 0)
        n_total = len(skill_refs)
        return {
            "cited_presented": n_cited,
            "total_presented": n_total,
            "rate": n_cited / n_total * 100 if n_total > 0 else 0,
        }

    # For retrieval: per-trajectory presented skills, selective citations only
    ref_data = _compute_presented_skill_refs(run)
    summary = ref_data.get("summary", {})
    total_presented = summary.get("total_skill_presentations", 0)
    cited_presented = summary.get("any_refs", 0)

    return {
        "cited_presented": cited_presented,
        "total_presented": total_presented,
        "rate": cited_presented / total_presented * 100 if total_presented > 0 else 0,
    }


def print_per_run_reference_table(runs: list[dict]):
    """Per-run reference counts with retrieval-aware denominator."""
    print("=== Per-run references (retrieval-aware) ===")
    print()

    headers = [
        "Run", "LLM", "Learn", "Mode", "Retrieval", "Res Rate",
        "SB Skills", "Skills/i", "Gen/i", "Spec/i",
        "Explicit", "Dumped", "Cite Trajs", "Prose Trajs", "Any Trajs", "Presented RR",
    ]
    rows = []

    for r in runs:
        retrieval = r.get("retrieval", {})
        ret_enabled = retrieval.get("enabled", False)
        top_k = retrieval.get("top_k", 0)

        # For retrieval runs, references must be computed against the skills
        # actually presented in the prompt (retrieval renumbers skill IDs to
        # 00001..k, so matching against original skillbook IDs finds nothing).
        ref_data = _compute_presented_skill_refs(r) if ret_enabled else compute_references(r)
        summary = ref_data.get("summary", {})
        skill_refs = ref_data.get("skill_refs", {})

        # Total unique skills in the skillbook
        total_skills = len(skill_refs)

        explicit = summary.get("explicit_refs", 0)
        dumped = summary.get("dumped_refs", 0)
        dump_trajs = summary.get("dump_traj_iters", 0)
        cite_trajs = summary.get("cite_traj_iters", 0)
        prose_trajs = summary.get("prose_traj_iters", 0)
        any_trajs = summary.get("any_traj_iters", 0)
        both_trajs = summary.get("both_traj_iters", 0)

        # Retrieval label (type + top_k when retrieval ran, else 'off')
        ret_label = _retrieval_label(r)

        # Count trajectory iterations where skillbook was present
        n_traj_iters = summary.get("traj_iters", 0)
        if n_traj_iters == 0:
            trajectories = {k: v for k, v in r.get("trajectories", {}).items() if "/" not in k}
            n_traj_iters = sum(len(iters) for iters in trajectories.values())

        # Skills actually presented to agent per instance
        stats = r.get("statistics", {})
        ret_stats = stats.get("retrieval", {})
        total_presentations = summary.get("total_skill_presentations", 0)
        if ret_enabled:
            skills_per_inst = ret_stats.get("avg_skills_after", top_k)
        elif total_presentations > 0 and n_traj_iters > 0:
            skills_per_inst = total_presentations / n_traj_iters
        else:
            skills_per_inst = total_skills

        # General/specific breakdown of skills actually presented to agent
        spec_data = compute_presented_skill_specificity(r)
        if ret_enabled:
            per_inst_count = spec_data["per_instance"]
            n_inst = len(per_inst_count) if per_inst_count else 1
            gen_per_inst = spec_data["general"] / n_inst
            spec_per_inst = spec_data["specific"] / n_inst
        else:
            gen_per_inst = spec_data["gen_pct"] * skills_per_inst / 100
            spec_per_inst = spec_data["spec_pct"] * skills_per_inst / 100

        # Presented RR: selective-citation rate among actually presented skills
        presented_rr = _compute_presented_ref_rate(r)

        rows.append({
            "Run": _short_run(r["run_name"]),
            "LLM": r["llm_short"],
            "Learn": r["learn_mode"],
            "Mode": r["sb_mode"],
            "Retrieval": ret_label,
            "Res Rate": _res_rate_compact(r),
            "SB Skills": str(total_skills),
            "Skills/i": f"{skills_per_inst:.0f}" if skills_per_inst == int(skills_per_inst) else f"{skills_per_inst:.1f}",
            "Gen/i": f"{gen_per_inst:.1f}",
            "Spec/i": f"{spec_per_inst:.1f}",
            "Explicit": str(explicit),
            "Dumped": f"{dumped} ({dump_trajs}t)" if dumped else "0",
            "Cite Trajs": (
                f"{cite_trajs}/{n_traj_iters} ({cite_trajs / n_traj_iters * 100:.1f}%)"
                if n_traj_iters > 0 else "-"
            ),
            "Prose Trajs": (
                f"{prose_trajs}/{n_traj_iters} ({prose_trajs / n_traj_iters * 100:.1f}%)"
                if n_traj_iters > 0 else "-"
            ),
            "Any Trajs": (
                (f"{any_trajs}/{n_traj_iters} ({any_trajs / n_traj_iters * 100:.1f}%)"
                 + (f", {both_trajs} both" if both_trajs else ""))
                if n_traj_iters > 0 else "-"
            ),
            "Presented RR": f"{presented_rr['rate']:.1f}%",
        })

    if not rows:
        print("  (No runs with reference data)")
        return

    _print_table_rows(headers, rows)
    print()
    print("  (Res Rate: bsr = val_baseline (empty-skillbook) rate; r = skillbook (val) rate; "
          "Δ = r − bsr in pp;\n      runs without a paired baseline show only 'r S%')")
    print("  (Skills/i, Gen/i, Spec/i = avg per trajectory; Gen=process advice, Spec=repo-specific)")
    print("  (Explicit = selective ID citation events; Dumped = IDs from list-dump messages "
          f">=max({DUMP_MIN_IDS},min(presented,{DUMP_CAP_IDS})) IDs/message, with #trajectories)")
    print("  (Cite Trajs = trajectories with >=1 selective ID citation / total; dumps don't count)")
    print("  (Prose Trajs = trajectories where the agent referred to the skillbook by WORDS — "
          "\"skill\"/\"skillbook\"\n      e.g. \"Based on the skill\", \"the skillbook suggests\" — "
          "without necessarily citing an ID; see data/skill_prose_phrasings.json)")
    print("  (Any Trajs = Cite ∪ Prose, DEDUPLICATED: a trajectory citing an ID AND using prose "
          "is counted once;\n      \"N both\" = that overlap, so Any = Cite + Prose - both)")
    print("  (Presented RR = explicit citations / all presented skill-slots across trajectories; dumps excluded)")
    print()


# ---------------------------------------------------------------------------
# Output: Resolve rate vs citation (does using a skill associate with solving?)
# ---------------------------------------------------------------------------


def _fisher_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test p-value for a 2x2 table [[a,b],[c,d]]."""
    import math
    n = a + b + c + d
    if n == 0:
        return 1.0
    r1 = a + b
    c1 = a + c
    lg = math.lgamma

    def lhyp(k):
        return (lg(r1 + 1) + lg(c + d + 1) + lg(c1 + 1) + lg(b + d + 1)
                - lg(n + 1) - lg(k + 1) - lg(r1 - k + 1)
                - lg(c1 - k + 1) - lg(n - r1 - c1 + k + 1))

    p_obs = lhyp(a)
    total = 0.0
    for k in range(max(0, r1 + c1 - n), min(r1, c1) + 1):
        lp = lhyp(k)
        if lp <= p_obs + 1e-9:
            total += math.exp(lp)
    return min(1.0, total)


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b, c (binomial)."""
    import math
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n
    return min(1.0, 2 * tail)


def _paired_resolve(results: dict) -> dict:
    """Pair skillbook phase vs empty-skillbook baseline phase per instance.

    Standard split: val (skillbook) vs val_baseline (empty). eval-on-train:
    train_eval (skillbook) vs train_eval_baseline (empty). An instance counts as
    resolved in a phase if resolved in ANY of its attempts (pass@k semantics).
    Returns the discordant 2x2: n11 both, n10 skillbook-only (helped),
    n01 baseline-only (hurt), n00 neither.
    """
    val: dict[str, dict[int, bool]] = {}
    base: dict[str, dict[int, bool]] = {}
    for k, v in results.items():
        if "/" not in k:
            continue
        phase, inst = k.split("/", 1)
        if phase in ("val", "train_eval"):
            val[inst] = v
        elif phase in ("val_baseline", "train_eval_baseline"):
            base[inst] = v
    insts = set(val) & set(base)
    n11 = n10 = n01 = n00 = 0
    for inst in insts:
        sk = any(val[inst].values())
        bs = any(base[inst].values())
        if sk and bs:
            n11 += 1
        elif sk and not bs:
            n10 += 1
        elif bs and not sk:
            n01 += 1
        else:
            n00 += 1
    return {"n11": n11, "n10": n10, "n01": n01, "n00": n00, "n": len(insts)}


def print_val_vs_baseline(runs: list[dict]):
    """Paired resolve: same val instances WITH skillbook vs WITHOUT (baseline).

    This is the within-run experimental contrast (not correlational): each
    instance is run twice on the identical task, once with the learned
    skillbook (val) and once with an empty one (val_baseline). McNemar tests
    whether the skillbook flips outcomes, using only discordant pairs.
    """
    print("=== Resolve: same val instances WITH skillbook vs WITHOUT (paired) ===")
    print()

    headers = [
        "Run", "Agent", "Learn", "Mode", "Retrieval", "N",
        "Base res", "SB res", "Delta", "Helped", "Hurt", "McNemar p",
    ]
    rows = []
    pool = {"n11": 0, "n10": 0, "n01": 0, "n00": 0, "n": 0}

    for r in runs:
        sp = _paired_resolve(load_results(r["run_dir"]))
        if sp["n"] == 0:
            continue
        for k in pool:
            pool[k] += sp[k]
        n = sp["n"]
        base_res = (sp["n11"] + sp["n01"]) / n * 100
        sb_res = (sp["n11"] + sp["n10"]) / n * 100
        rows.append({
            "Run": _short_run(r["run_name"]),
            "Agent": _model_short(r["agent_llm"], r["agent_llm"]),
            "Learn": r["learn_mode"],
            "Mode": r["sb_mode"],
            "Retrieval": _retrieval_label(r),
            "N": str(n),
            "Base res": f"{base_res:.1f}%",
            "SB res": f"{sb_res:.1f}%",
            "Delta": f"{sb_res - base_res:+.1f}pp",
            "Helped": str(sp["n10"]),
            "Hurt": str(sp["n01"]),
            "McNemar p": f"{_mcnemar_exact(sp['n10'], sp['n01']):.3f}",
        })

    if not rows:
        print("  (No runs with paired val / val_baseline results)")
        print()
        return

    n = pool["n"]
    base_res = (pool["n11"] + pool["n01"]) / n * 100
    sb_res = (pool["n11"] + pool["n10"]) / n * 100
    rows.append({
        "Run": "POOLED (all runs)", "Agent": "-", "Learn": "-", "Mode": "-", "Retrieval": "-",
        "N": str(n),
        "Base res": f"{base_res:.1f}%",
        "SB res": f"{sb_res:.1f}%",
        "Delta": f"{sb_res - base_res:+.1f}pp",
        "Helped": str(pool["n10"]),
        "Hurt": str(pool["n01"]),
        "McNemar p": f"{_mcnemar_exact(pool['n10'], pool['n01']):.4f}",
    })

    _print_table_rows(headers, rows)
    print()
    print("  (Paired within-run: each instance run twice on the SAME task — "
          "with learned skillbook (val) vs empty (val_baseline))")
    print("  (Helped = resolved only WITH skillbook; Hurt = resolved only WITHOUT; "
          "McNemar uses only these discordant pairs)")
    print("  (resolved per phase = solved in ANY attempt; POOLED ignores run clustering)")
    print()


def _resolve_split(records: list, results: dict) -> dict:
    """Cross-tab citation × resolution over scored trajectories.

    records: list of (instance_id, iter, cited_bool).
    results: {instance_id: {iter: resolved_bool}} (unprefixed keys).
    Returns counts a=cited&resolved, b=cited&unresolved, c=notcited&resolved,
    d=notcited&unresolved, plus n_matched/n_total.
    """
    a = b = c = d = 0
    matched = 0
    for inst_id, it, cited in records:
        resolved = results.get(inst_id, {}).get(it)
        if resolved is None:
            continue
        matched += 1
        if cited:
            if resolved:
                a += 1
            else:
                b += 1
        else:
            if resolved:
                c += 1
            else:
                d += 1
    return {"a": a, "b": b, "c": c, "d": d, "matched": matched, "total": len(records)}


def _resolve_split_by_instance(records: list, results: dict) -> dict:
    """Instance-level cross-tab citation × resolution.

    Aggregates the (instance, iter) records to one row per instance over the
    skillbook-shown attempts: cited = a citation in ANY attempt, resolved =
    solved in ANY of those attempts. Removes the per-attempt dilution (and the
    per_instance iter_0 skew) by putting citation and resolution on the same
    instance population.
    """
    groups: dict[str, dict] = {}
    for inst_id, it, cited in records:
        resolved = results.get(inst_id, {}).get(it)
        if resolved is None:
            continue
        g = groups.setdefault(inst_id, {"cited": False, "resolved": False})
        g["cited"] = g["cited"] or cited
        g["resolved"] = g["resolved"] or resolved

    a = b = c = d = 0
    for g in groups.values():
        if g["cited"]:
            if g["resolved"]:
                a += 1
            else:
                b += 1
        else:
            if g["resolved"]:
                c += 1
            else:
                d += 1
    inst_in_records = len({rec[0] for rec in records})
    return {"a": a, "b": b, "c": c, "d": d, "matched": len(groups), "total": inst_in_records}


def print_resolve_vs_citation(runs: list[dict], by_instance: bool = False):
    """Resolve rate among trajectories/instances WITH a selective citation vs WITHOUT.

    Correlational only: a trajectory citing a skill is not randomly assigned —
    cited-vs-not differ in many ways besides the citation. Read as association,
    not causal effect of the skillbook.

    by_instance=False: unit is a trajectory (one agent attempt).
    by_instance=True:  unit is an instance — cited/resolved if it happened in
                       ANY skillbook-shown attempt (removes multi-attempt skew).
    """
    unit = "instances" if by_instance else "trajectories"
    print(f"=== Resolve rate: {unit} WITH selective citation vs WITHOUT ===")
    print()

    headers = [
        "Run", "Learn", "Mode", "Retrieval",
        "Cited n", "Res|cited", "NoCite n", "Res|nocite", "Delta", "Fisher p",
    ]
    rows = []
    pooled = {"a": 0, "b": 0, "c": 0, "d": 0}

    for r in runs:
        ret_enabled = r.get("retrieval", {}).get("enabled", False)
        ref_data = _compute_presented_skill_refs(r) if ret_enabled else compute_references(r)
        records = ref_data.get("traj_records", [])
        results = load_results(r["run_dir"])
        if not records or not results:
            continue
        sp = (_resolve_split_by_instance if by_instance else _resolve_split)(records, results)
        if sp["matched"] == 0:
            continue
        for k in pooled:
            pooled[k] += sp[k]

        a, b, c, d = sp["a"], sp["b"], sp["c"], sp["d"]
        n_cite, n_no = a + b, c + d
        res_cite = a / n_cite * 100 if n_cite else None
        res_no = c / n_no * 100 if n_no else None
        delta = (res_cite - res_no) if (res_cite is not None and res_no is not None) else None
        p = _fisher_2x2(a, b, c, d) if (n_cite and n_no) else None

        ret_label = _retrieval_label(r)
        rows.append({
            "Run": _short_run(r["run_name"]),
            "Learn": r["learn_mode"],
            "Mode": r["sb_mode"],
            "Retrieval": ret_label,
            "Cited n": str(n_cite),
            "Res|cited": f"{res_cite:.1f}%" if res_cite is not None else "-",
            "NoCite n": str(n_no),
            "Res|nocite": f"{res_no:.1f}%" if res_no is not None else "-",
            "Delta": f"{delta:+.1f}pp" if delta is not None else "-",
            "Fisher p": f"{p:.3f}" if p is not None else "-",
        })

    if not rows:
        print("  (No runs with both citation records and results)")
        print()
        return

    # Pooled row across all runs
    a, b, c, d = pooled["a"], pooled["b"], pooled["c"], pooled["d"]
    n_cite, n_no = a + b, c + d
    if n_cite and n_no:
        res_cite, res_no = a / n_cite * 100, c / n_no * 100
        rows.append({
            "Run": "POOLED (all runs)",
            "Learn": "-", "Mode": "-", "Retrieval": "-",
            "Cited n": str(n_cite),
            "Res|cited": f"{res_cite:.1f}%",
            "NoCite n": str(n_no),
            "Res|nocite": f"{res_no:.1f}%",
            "Delta": f"{res_cite - res_no:+.1f}pp",
            "Fisher p": f"{_fisher_2x2(a, b, c, d):.4f}",
        })

    _print_table_rows(headers, rows)
    print()
    print(f"  (CORRELATIONAL: cited vs not-cited {unit} differ in many ways; "
          "not the causal effect of the skillbook)")
    if by_instance:
        print("  (Unit = instance: cited/resolved if it happened in ANY skillbook-shown attempt)")
    else:
        print("  (Unit = trajectory = one attempt; multi-attempt runs count each attempt separately)")
    print(f"  (Res|cited = resolve rate among {unit} with >=1 selective citation; dumps don't count as citations)")
    print(f"  (POOLED merges all {unit} — ignores run clustering, so its p is anti-conservative)")
    print()


# ---------------------------------------------------------------------------
# Output 5: General vs Specific classification by repo × learn mode
# ---------------------------------------------------------------------------


def print_classification_table(runs: list[dict]):
    """Output 4: Classification by repository × learn mode × retrieval."""
    print("=== General vs Specific classification by repo × learn mode × retrieval ===")
    print()

    # Collect data from per_repo and per_instance runs
    # For per_repo: each repo is a natural unit
    # For per_instance: group by repo (extracted from instance_id)
    # Columns are keyed by (learn_mode, retrieval_type) — retrieval shapes which
    # skills the reflector ends up learning, so it is a stratification dimension.
    data = defaultdict(lambda: defaultdict(lambda: {"general": 0, "specific": 0}))

    for r in runs:
        learn = r["learn_mode"]
        rtype = _retrieval_type(r)
        key = (learn, rtype)

        if r["sb_mode"] == "per_repo":
            for repo_name, sb in r["per_repo_sbs"].items():
                skills = _extract_skills(sb)
                for s in skills:
                    cls = classify_skill_specificity(s["content"], s["section"])
                    data[key][repo_name][cls] += 1

        elif r["sb_mode"] == "per_instance":
            final_sbs = _get_final_skillbooks_per_instance(r)
            for sb in final_sbs:
                # Extract repo from skill section names or skillbook metadata
                # Instance IDs in skillbooks aren't directly available — use section names
                skills = _extract_skills(sb)
                for s in skills:
                    cls = classify_skill_specificity(s["content"], s["section"])
                    # Use "per_instance" as pseudo-repo since instances are mixed
                    data[key]["per_instance"][cls] += 1

        elif r["sb_mode"] == "global":
            if r["global_sb"]:
                skills = _extract_skills(r["global_sb"])
                for s in skills:
                    cls = classify_skill_specificity(s["content"], s["section"])
                    data[key]["global"][cls] += 1

    if not data:
        print("  (No skillbook data for classification)")
        return

    # Get all repos across (learn, retrieval) groups
    all_repos = sorted(set(repo for gd in data.values() for repo in gd.keys()))
    groups = sorted(data.keys())  # list of (learn_mode, retrieval_type)

    rows = []
    for repo in all_repos:
        row = {"Repo": repo.split("__")[-1][:20] if "__" in repo else repo[:20]}
        for learn, rtype in groups:
            col = f"{learn}/{rtype}"
            gen = data[(learn, rtype)][repo].get("general", 0)
            spec = data[(learn, rtype)][repo].get("specific", 0)
            total = gen + spec
            gen_pct = gen / total * 100 if total > 0 else 0
            spec_pct = spec / total * 100 if total > 0 else 0
            row[f"{col}_total"] = str(total)
            row[f"{col}_gen_pct"] = f"{gen_pct:.0f}%"
            row[f"{col}_spec_pct"] = f"{spec_pct:.0f}%"
        rows.append(row)

    # Total row
    total_row = {"Repo": "TOTAL"}
    for learn, rtype in groups:
        col = f"{learn}/{rtype}"
        total_gen = sum(data[(learn, rtype)][repo].get("general", 0) for repo in all_repos)
        total_spec = sum(data[(learn, rtype)][repo].get("specific", 0) for repo in all_repos)
        total_all = total_gen + total_spec
        gen_pct = total_gen / total_all * 100 if total_all > 0 else 0
        spec_pct = total_spec / total_all * 100 if total_all > 0 else 0
        total_row[f"{col}_total"] = str(total_all)
        total_row[f"{col}_gen_pct"] = f"{gen_pct:.0f}%"
        total_row[f"{col}_spec_pct"] = f"{spec_pct:.0f}%"
    rows.append(total_row)

    # Headers must match row keys (col = "{learn}/{rtype}")
    flat_headers = ["Repo"]
    for learn, rtype in groups:
        col = f"{learn}/{rtype}"
        flat_headers.extend([f"{col}_total", f"{col}_gen_pct", f"{col}_spec_pct"])

    _print_table_rows(flat_headers, rows)
    print()

    # Self-check: Gen + Spec = Total
    for learn, rtype in groups:
        for repo in all_repos:
            gen = data[(learn, rtype)][repo].get("general", 0)
            spec = data[(learn, rtype)][repo].get("specific", 0)
            total = gen + spec
            if total > 0 and abs((gen / total * 100) + (spec / total * 100) - 100.0) > 0.1:
                print(f"  WARNING: self-check failed: {learn}/{rtype}/{repo}", file=sys.stderr)
    print("  (Self-check: Gen + Spec = Total for each cell — OK)")
    print()


# ---------------------------------------------------------------------------
# Dump-message listing (for manual validation)
# ---------------------------------------------------------------------------


def print_dump_locations(runs: list[dict], max_per_run: int | None = None):
    """List every ID-dump message (run → instance → iter → snippet).

    A dump is an assistant message echoing most of the presented skill list
    (>= max(DUMP_MIN_IDS, min(presented, DUMP_CAP_IDS)) distinct IDs).
    Prints the trajectory file path so it can be opened and eyeballed directly.
    """
    print("=== ID-dump messages (manual check) ===")
    print(f"  (A dump = one assistant message citing >=max({DUMP_MIN_IDS},min(presented,{DUMP_CAP_IDS})) "
          f"distinct skill IDs — agent echoing the presented list, not selective use)")
    print()

    any_dumps = False
    for r in runs:
        ret_enabled = r.get("retrieval", {}).get("enabled", False)
        ref_data = _compute_presented_skill_refs(r) if ret_enabled else compute_references(r)
        dumps = ref_data.get("dump_locations", [])
        if not dumps:
            continue
        any_dumps = True

        # Where the trajectories live (so the user can open the files)
        bench_dir = _find_benchmark_dir(r["run_dir"])
        traj_root = bench_dir / "trajectories" if bench_dir else r["run_dir"]

        def _resolve_traj_path(inst: str, it: int) -> str:
            # Mirror load_trajectories: unprefixed keys resolve to the first
            # phase that has the instance, preferring val > val_baseline > train;
            # flat (non-split) runs keep the instance dir at the root.
            for sub in ("", "val", "val_baseline", "train"):
                cand = (traj_root / sub / inst / f"iter_{it}.json") if sub else (traj_root / inst / f"iter_{it}.json")
                if cand.exists():
                    return str(cand)
            return str(traj_root / inst / f"iter_{it}.json")

        shown = sorted(dumps, key=lambda d: (d["instance"], d["iter"], d["msg_idx"]))
        total = len(shown)
        if max_per_run is not None and total > max_per_run:
            shown = shown[:max_per_run]

        ret_tag = f" [{_retrieval_label(r)}]" if r.get("retrieval", {}).get("enabled") else ""
        print(f"  {_short_run(r['run_name'], 50)} [{r['learn_mode']}/{r['sb_mode']}]{ret_tag}  "
              f"— {total} dump message(s) across {len(set(d['instance'] for d in dumps))} instance(s)")
        for d in shown:
            traj_path = _resolve_traj_path(d["instance"], d["iter"])
            ids_preview = ", ".join(d["ids"][:6]) + (" …" if len(d["ids"]) > 6 else "")
            print(f"    • {d['instance']}  iter_{d['iter']}  msg#{d['msg_idx']}  "
                  f"({d['n_ids']} IDs)")
            print(f"        file: {traj_path}")
            print(f"        IDs : {ids_preview}")
            print(f"        text: ...{d['snippet'][:200]}...")
        if max_per_run is not None and total > max_per_run:
            print(f"    … {total - max_per_run} more (raise --list-dumps to see all)")
        print()

    if not any_dumps:
        print("  (No ID-dump messages found)")
        print()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_csv_tables(runs: list[dict], csv_path: str):
    """Write all tables to a single CSV file (one section after another)."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Summary — per_instance
        writer.writerow(["=== Per-instance summary ==="])
        writer.writerow(["Run", "Learn", "Retrieval", "Split", "Max Attempts", "Resolution Rate", "SBs",
                         "Skills/SB (med)", "Skills/SB (mean)", "Tokens/SB (med)", "Tokens/SB (mean)",
                         "Ctx%", "Skill tokens (med)", "Skill tokens (mean)"])
        for r in runs:
            if r["sb_mode"] != "per_instance":
                continue
            final_sbs = _get_final_skillbooks_per_instance(r)
            skill_counts = [len(_extract_skills(sb)) for sb in final_sbs]
            token_counts = [sum(count_tokens(s["content"]) for s in _extract_skills(sb)) for sb in final_sbs]
            skill_toks = [count_tokens(s["content"]) for sb in final_sbs for s in _extract_skills(sb)]
            cw = r["context_window"]
            ctx_pct = statistics.median(token_counts) / cw * 100 if token_counts and cw > 0 else 0
            writer.writerow([
                r["run_name"], r["learn_mode"], _retrieval_label(r), r["val_ratio"] or "-", r["max_attempts"],
                f"{r['resolution_rate']*100:.1f}%", len(final_sbs),
                f"{statistics.median(skill_counts):.1f}" if skill_counts else "-",
                f"{statistics.mean(skill_counts):.1f}" if skill_counts else "-",
                f"{statistics.median(token_counts):.0f}" if token_counts else "-",
                f"{statistics.mean(token_counts):.0f}" if token_counts else "-",
                f"{ctx_pct:.1f}%",
                f"{statistics.median(skill_toks):.0f}" if skill_toks else "-",
                f"{statistics.mean(skill_toks):.0f}" if skill_toks else "-",
            ])

        writer.writerow([])
        writer.writerow(["=== Per-repo/global summary ==="])
        writer.writerow(["Run", "Learn", "Retrieval", "Split", "Resolution Rate", "Mode", "SBs",
                         "Skills/SB (med)", "Skills/SB (mean)", "Tokens/SB (med)", "Tokens/SB (mean)",
                         "Ctx%", "Skill tokens (med)", "Skill tokens (mean)"])
        for r in runs:
            if r["sb_mode"] == "per_instance":
                continue
            sbs = list(r["per_repo_sbs"].values()) if r["sb_mode"] == "per_repo" else ([r["global_sb"]] if r["global_sb"] else [])
            if not sbs:
                continue
            skill_counts = [len(_extract_skills(sb)) for sb in sbs]
            token_counts = [sum(count_tokens(s["content"]) for s in _extract_skills(sb)) for sb in sbs]
            skill_toks = [count_tokens(s["content"]) for sb in sbs for s in _extract_skills(sb)]
            cw = r["context_window"]
            ctx_pct = statistics.median(token_counts) / cw * 100 if token_counts and cw > 0 else 0
            writer.writerow([
                r["run_name"], r["learn_mode"], _retrieval_label(r), r["val_ratio"] or "-",
                f"{r['resolution_rate']*100:.1f}%", r["sb_mode"], len(sbs),
                f"{statistics.median(skill_counts):.1f}" if skill_counts else "-",
                f"{statistics.mean(skill_counts):.1f}" if skill_counts else "-",
                f"{statistics.median(token_counts):.0f}" if token_counts else "-",
                f"{statistics.mean(token_counts):.0f}" if token_counts else "-",
                f"{ctx_pct:.1f}%",
                f"{statistics.median(skill_toks):.0f}" if skill_toks else "-",
                f"{statistics.mean(skill_toks):.0f}" if skill_toks else "-",
            ])

    print(f"CSV written to {csv_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Qualitative skillbook analysis across experiment runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("runs", nargs="+", metavar="RUN_DIR",
                        help="Run directories or parent folder (auto-discovers run_* inside)")
    parser.add_argument("--output", choices=["ascii", "csv", "both"], default="ascii",
                        help="Output format (default: ascii)")
    parser.add_argument("--csv-file", default="skillbook_analysis.csv",
                        help="CSV output path (used with --output csv|both)")
    parser.add_argument("--learn-mode", choices=["SWE", "Default"], default=None,
                        help="Filter runs by learn mode")
    parser.add_argument("--sb-mode", choices=["per_instance", "per_repo", "global"], default=None,
                        help="Filter runs by skillbook mode")
    parser.add_argument("--list-dumps", type=int, metavar="N", nargs="?", const=0, default=None,
                        help="List ID-dump messages (run/instance/iter/file) for manual check; "
                             "optional N caps messages shown per run (0/omitted = all)")
    parser.add_argument("--growth", action="store_true",
                        help="Show ASCII growth chart by iteration")
    parser.add_argument("--growth-runs", nargs="*", metavar="SUBSTR",
                        help="Filter growth chart to runs containing these substrings")
    args = parser.parse_args()

    runs, skipped = load_runs(args.runs, learn_filter=args.learn_mode, sb_mode_filter=args.sb_mode)
    if not runs:
        print("No valid runs with skillbooks found.", file=sys.stderr)
        sys.exit(1)

    # Header
    print("Skillbook Quality Analysis")
    print(f"  Tokenizer: {TOKEN_METHOD}")
    print(f"  Runs processed: {len(runs)}, skipped: {skipped}")
    print(f"  Definitions: explicit ref = skill ID cited selectively (below dump threshold); "
          f"dumped = IDs from list-dump messages (>=max({DUMP_MIN_IDS},min(presented,{DUMP_CAP_IDS})) IDs/message, agent echoing the list)")
    print("  prose ref = agent refers to the skillbook by words (skill/skillbook, e.g. "
          "\"Based on the skill\", \"the skillbook suggests\") without citing an ID; "
          "any ref = explicit ∪ prose, deduplicated (data/skill_prose_phrasings.json)")
    print("  general = process advice; specific = mentions concrete identifiers")
    print("  General/specific classification: by content analysis (NOT by AVOID/VERIFIED/CONSIDER prefix)")
    print()

    # Outputs
    print_summary_tables(runs)

    if args.growth or args.growth_runs is not None:
        print_growth_chart(runs, run_filter=args.growth_runs)

    print_reference_table(runs)
    print_per_run_reference_table(runs)
    print_val_vs_baseline(runs)
    print_resolve_vs_citation(runs, by_instance=False)
    print_resolve_vs_citation(runs, by_instance=True)
    print_classification_table(runs)

    if args.list_dumps is not None:
        print_dump_locations(runs, max_per_run=args.list_dumps or None)

    if args.output in ("csv", "both"):
        write_csv_tables(runs, args.csv_file)

    # Run legend
    print("Run legend:")
    for r in runs:
        n_sb = len(r["per_instance_sbs"]) or len(r["per_repo_sbs"]) or (1 if r["global_sb"] else 0)
        ret_tag = f" [{_retrieval_label(r)}]" if r.get("retrieval", {}).get("enabled") else ""
        print(f"  {_short_run(r['run_name'], 50)}  [{r['learn_mode']}/{r['sb_mode']}]{ret_tag}  "
              f"res={r['resolution_rate']*100:.1f}%  SBs={n_sb}")


if __name__ == "__main__":
    main()
