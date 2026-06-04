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
    uv run python scripts/analyze_skillbook_quality.py data/run_a --sample-references 5
    uv run python scripts/analyze_skillbook_quality.py data/run_a --learn-mode SWE
    uv run python scripts/analyze_skillbook_quality.py data/run_a --growth
"""

import argparse
import csv
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configurable definitions
# ---------------------------------------------------------------------------

# Stop-phrases for indirect (content) reference matching.
# These common phrases produce false positives and are excluded from matching.
INDIRECT_STOP_PHRASES = {
    "git diff",
    "source files",
    "source file",
    "test case",
    "test cases",
    "the code",
    "a fix",
    "the issue",
    "the problem",
    "make sure",
    "ensure that",
    "in the",
    "of the",
    "to the",
    "for the",
    "on the",
    "with the",
    "from the",
    "is a",
    "is an",
    "it is",
    "to be",
    "can be",
    "will be",
    "should be",
    "do not",
    "does not",
    "is not",
    "are not",
    "was not",
    "have been",
    "has been",
    "there are",
    "there is",
    "this is",
    "that is",
    "such as",
    "for example",
    "in order",
    "need to",
    "needs to",
    "try to",
    "trying to",
    "used to",
    "using the",
    "check the",
    "verify the",
    "run the",
    "look at",
    "look for",
    "file and",
    "files and",
    "code and",
    "function or",
    "method or",
}

# Minimum n-gram length (in words) for indirect matching.
INDIRECT_MIN_WORDS = 4
# Minimum character length for an n-gram to be considered (filters short noise).
INDIRECT_MIN_CHARS = 20

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
            "top_k": ret.get("top_k", 5),
            "skip_threshold": ret.get("skip_threshold", 10),
        }
    return {"enabled": False}


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
    known_phases = {"train", "val_baseline", "val"}
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
    known_phases = {"train", "val_baseline", "val"}

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
        for phase_name in ["val", "val_baseline", "train"]:
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
    agent_llm = config.get("llm", {}).get("agent", {}).get("model", "N/A")
    ace_llm = config.get("llm", {}).get("ace", {}).get("model", "N/A")

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
            # Only match skill-like headings: word_chars-NNNNN
            if _re.match(r"[a-z_]+-\d+", heading):
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
            gen_pct = total_gen / total if total > 0 else 0
            spec_pct = total_spec / total if total > 0 else 0
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
    return " ".join(
        m.get("content", "") for m in traj.get("messages", []) if m.get("role") == "assistant"
    )


def _find_explicit_refs(assistant_text: str, skill_ids: list[str]) -> set[str]:
    """Find skill IDs directly mentioned in assistant text."""
    return {sid for sid in skill_ids if sid in assistant_text}


def _build_content_ngrams(content: str) -> set[str]:
    """Build significant n-grams from skill content for indirect matching."""
    # Strip prefix (AVOID:/VERIFIED:/CONSIDER:) for matching
    clean = re.sub(r"^(AVOID|VERIFIED|CONSIDER):\s*", "", content)
    words = clean.split()
    ngrams = set()
    for i in range(max(0, len(words) - INDIRECT_MIN_WORDS + 1)):
        ngram = " ".join(words[i:i + INDIRECT_MIN_WORDS]).lower()
        if len(ngram) >= INDIRECT_MIN_CHARS and ngram not in INDIRECT_STOP_PHRASES:
            ngrams.add(ngram)
    return ngrams


def _build_ngram_index(
    skills: list[dict],
    skill_ngrams_cache: dict[str, set[str]],
) -> dict[str, str]:
    """Build inverted index: ngram → skill_id (first skill containing it)."""
    index: dict[str, str] = {}
    for s in skills:
        sid = s["id"]
        if sid not in skill_ngrams_cache:
            skill_ngrams_cache[sid] = _build_content_ngrams(s["content"])
        for ng in skill_ngrams_cache[sid]:
            if ng not in index:
                index[ng] = sid
    return index


def _find_indirect_refs(
    assistant_text: str,
    skills: list[dict],
    explicit_ids: set[str],
    skill_ngrams_cache: dict[str, set[str]] | None = None,
    ngram_index: dict[str, str] | None = None,
) -> tuple[set[str], list[dict]]:
    """Find indirect (content-based) references.

    Returns (referenced_ids, sample_matches) where sample_matches has
    {skill_id, ngram, context} for later sampling.

    Uses ngram_index (inverted index) when available for O(text × window)
    instead of O(skills × ngrams × text).
    """
    lower_text = assistant_text.lower()
    referenced = set()
    sample_matches = []

    if ngram_index is not None and skill_ngrams_cache is not None:
        # Fast path: scan text once, look up ngrams in index
        words = lower_text.split()
        matched_skills_ngrams: dict[str, str] = {}  # skill_id → first matching ngram
        for i in range(max(0, len(words) - INDIRECT_MIN_WORDS + 1)):
            ngram = " ".join(words[i:i + INDIRECT_MIN_WORDS])
            if len(ngram) < INDIRECT_MIN_CHARS:
                continue
            sid = ngram_index.get(ngram)
            if sid and sid not in explicit_ids and sid not in matched_skills_ngrams:
                matched_skills_ngrams[sid] = ngram

        # Collect sample matches with context
        for sid, ngram in matched_skills_ngrams.items():
            referenced.add(sid)
            idx = lower_text.find(ngram)
            start = max(0, idx - 40)
            end = min(len(lower_text), idx + len(ngram) + 40)
            context = assistant_text[start:end]
            sample_matches.append({
                "skill_id": sid,
                "ngram": ngram,
                "context": f"...{context}...",
            })
    else:
        # Slow fallback path
        for s in skills:
            if s["id"] in explicit_ids:
                continue
            if skill_ngrams_cache is not None and s["id"] in skill_ngrams_cache:
                ngrams = skill_ngrams_cache[s["id"]]
            else:
                ngrams = _build_content_ngrams(s["content"])
                if skill_ngrams_cache is not None:
                    skill_ngrams_cache[s["id"]] = ngrams
            for ng in ngrams:
                if ng in lower_text:
                    referenced.add(s["id"])
                    idx = lower_text.find(ng)
                    start = max(0, idx - 40)
                    end = min(len(lower_text), idx + len(ng) + 40)
                    context = assistant_text[start:end]
                    sample_matches.append({
                        "skill_id": s["id"],
                        "ngram": ng,
                        "context": f"...{context}...",
                    })
                    break

    return referenced, sample_matches


def compute_references(run: dict) -> dict:
    """Compute reference stats for a run (cached per run_dir).

    Returns {skill_id: {"explicit": bool, "indirect": bool, "specificity": str/None}, ...}
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
        result = {"skill_refs": {}, "sample_matches": [], "summary": {}}

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
    total_indirect = 0
    total_any = 0
    all_sample_matches = []

    for inst_id, iters in trajectories.items():
        for it, traj in iters.items():
            presented = _extract_presented_skills(traj)
            if not presented:
                continue
            assistant_text = _extract_assistant_text(traj)
            if not assistant_text:
                continue
            lower_text = assistant_text.lower()

            for s in presented:
                sid = s["id"]
                total_presentations += 1
                spec = classify_skill_specificity(s["content"], s["section"])

                is_explicit = sid in assistant_text
                is_indirect = False
                if not is_explicit:
                    clean = re.sub(r"^(AVOID|VERIFIED|CONSIDER):\s*", "", s["content"])
                    words = clean.lower().split()
                    for i in range(max(0, len(words) - 2)):
                        phrase = " ".join(words[i:i + 3])
                        if len(phrase) >= 15 and phrase not in INDIRECT_STOP_PHRASES:
                            if phrase in lower_text:
                                is_indirect = True
                                idx = lower_text.find(phrase)
                                start = max(0, idx - 40)
                                end = min(len(lower_text), idx + len(phrase) + 40)
                                context = assistant_text[start:end]
                                all_sample_matches.append({
                                    "skill_id": sid,
                                    "ngram": phrase,
                                    "context": f"...{context}...",
                                })
                                break

                if sid not in skill_refs:
                    skill_refs[sid] = {"explicit": 0, "indirect": 0, "specificity": spec}
                if is_explicit:
                    skill_refs[sid]["explicit"] += 1
                    total_explicit += 1
                if is_indirect:
                    skill_refs[sid]["indirect"] += 1
                    total_indirect += 1
                if is_explicit or is_indirect:
                    total_any += 1

    result = {
        "skill_refs": skill_refs,
        "sample_matches": all_sample_matches,
        "summary": {
            "total_skill_presentations": total_presentations,
            "explicit_refs": total_explicit,
            "indirect_refs": total_indirect,
            "any_refs": total_any,
        },
    }
    if cache_key is not None:
        _compute_presented_skill_refs._cache[cache_key] = result
    return result


_compute_presented_skill_refs._cache = {}


def _compute_refs_per_instance(run: dict, trajectories: dict) -> dict:
    """Reference computation for per_instance mode."""
    per_inst = run["per_instance_sbs"]
    skill_refs = {}  # skill_id -> {"explicit": int, "indirect": int}
    all_sample_matches = []
    skill_ngrams_cache: dict[str, set[str]] = {}

    total_skills = 0
    total_explicit = 0
    total_indirect = 0
    total_skills_with_any_ref = 0

    for inst_id, iters in per_inst.items():
        if inst_id not in trajectories:
            continue
        trajs = trajectories[inst_id]
        for it, sb in iters.items():
            if it == 0:
                continue  # No skillbook at iter 0
            if it not in trajs:
                continue

            skills = _extract_skills(sb)
            if not skills:
                continue

            skill_ids = [s["id"] for s in skills]
            assistant_text = _extract_assistant_text(trajs[it])

            # Build index for this skillbook (skills differ per instance/iteration)
            ngram_index = _build_ngram_index(skills, skill_ngrams_cache)

            explicit = _find_explicit_refs(assistant_text, skill_ids)
            indirect, samples = _find_indirect_refs(assistant_text, skills, explicit, skill_ngrams_cache, ngram_index)
            all_sample_matches.extend(samples)

            for s in skills:
                sid = s["id"]
                if sid not in skill_refs:
                    skill_refs[sid] = {"explicit": 0, "indirect": 0, "specificity": classify_skill_specificity(s["content"], s["section"])}
                is_explicit = sid in explicit
                is_indirect = sid in indirect
                if is_explicit:
                    skill_refs[sid]["explicit"] += 1
                if is_indirect:
                    skill_refs[sid]["indirect"] += 1

            n_skills = len(skills)
            n_any = len(explicit | indirect)
            total_skills += n_skills
            total_explicit += len(explicit)
            total_indirect += len(indirect)
            total_skills_with_any_ref += n_any

    return {
        "skill_refs": skill_refs,
        "sample_matches": all_sample_matches,
        "summary": {
            "total_skill_presentations": total_skills,
            "explicit_refs": total_explicit,
            "indirect_refs": total_indirect,
            "any_refs": total_skills_with_any_ref,
        },
    }


def _compute_refs_for_skillbook(skills: list[dict], trajectories: dict, traj_phase: str = "") -> dict:
    """Common reference computation for per_repo/global modes.

    For per_repo: match trajectories to repo by instance ID prefix.
    For global: all trajectories.
    """
    skill_refs = {}
    all_sample_matches = []

    total_skills_seen = 0
    total_explicit = 0
    total_indirect = 0
    total_any = 0

    # Build skill lookup
    skill_by_id = {s["id"]: s for s in skills}
    skill_ids = list(skill_by_id.keys())

    # Cache n-grams per skill (same skillbook for all trajectories)
    skill_ngrams_cache: dict[str, set[str]] = {}

    # Build inverted index: ngram → skill_id for fast lookup
    ngram_index = _build_ngram_index(skills, skill_ngrams_cache)

    for inst_id, trajs in trajectories.items():
        # per_repo/global: skillbook is shared and present in all iters (including 0)
        for it, traj in trajs.items():
            assistant_text = _extract_assistant_text(traj)
            if not assistant_text:
                continue

            explicit = _find_explicit_refs(assistant_text, skill_ids)
            indirect, samples = _find_indirect_refs(assistant_text, skills, explicit, skill_ngrams_cache, ngram_index)
            all_sample_matches.extend(samples)

            for sid in explicit | indirect:
                if sid not in skill_refs:
                    skill_refs[sid] = {"explicit": 0, "indirect": 0, "specificity": classify_skill_specificity(skill_by_id[sid]["content"], skill_by_id[sid]["section"])}
                if sid in explicit:
                    skill_refs[sid]["explicit"] += 1
                if sid in indirect:
                    skill_refs[sid]["indirect"] += 1

            total_skills_seen += len(skills)
            total_explicit += len(explicit)
            total_indirect += len(indirect)
            total_any += len(explicit | indirect)

    # Ensure all skills are in skill_refs even if never referenced
    for s in skills:
        if s["id"] not in skill_refs:
            skill_refs[s["id"]] = {"explicit": 0, "indirect": 0, "specificity": classify_skill_specificity(s["content"], s["section"])}

    return {
        "skill_refs": skill_refs,
        "sample_matches": all_sample_matches,
        "summary": {
            "total_skill_presentations": total_skills_seen,
            "explicit_refs": total_explicit,
            "indirect_refs": total_indirect,
            "any_refs": total_any,
        },
    }


def _compute_refs_per_repo(run: dict, trajectories: dict) -> dict:
    """Reference computation for per_repo mode."""
    per_repo = run["per_repo_sbs"]
    if not per_repo:
        return {"skill_refs": {}, "sample_matches": [], "summary": {}}

    combined_refs = {}
    combined_samples = []
    combined_summary = {
        "total_skill_presentations": 0,
        "explicit_refs": 0,
        "indirect_refs": 0,
        "any_refs": 0,
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
        combined_samples.extend(result["sample_matches"])
        for k in combined_summary:
            combined_summary[k] += result["summary"].get(k, 0)

    return {"skill_refs": combined_refs, "sample_matches": combined_samples, "summary": combined_summary}


def _compute_refs_global(run: dict, trajectories: dict) -> dict:
    """Reference computation for global mode."""
    global_sb = run["global_sb"]
    if global_sb is None:
        return {"skill_refs": {}, "sample_matches": [], "summary": {}}
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


def print_summary_tables(runs: list[dict]):
    """Output 1: Two summary tables — per_instance vs per_repo/global."""
    # Partition runs
    pi_runs = [r for r in runs if r["sb_mode"] == "per_instance"]
    pr_runs = [r for r in runs if r["sb_mode"] == "per_repo"]
    gl_runs = [r for r in runs if r["sb_mode"] == "global"]

    if pi_runs:
        print("=== Per-instance skillbooks: summary ===")
        print()
        headers = ["Run", "LLM", "Learn", "Split", "Iters", "Res Rate", "SBs", "Skills/SB", "Tokens/SB", "Ctx%", "Skill tok"]
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
                "Split": f"{r['val_ratio']:.2f}" if r["val_ratio"] else "-",
                "Iters": str(r["max_attempts"]),
                "Res Rate": f"{r['resolution_rate']*100:.1f}%",
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
        headers = ["Run", "LLM", "Learn", "Split", "Res Rate", "Mode", "SBs", "Skills/SB", "Tokens/SB", "Ctx%", "Skill tok"]
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
                "Split": f"{r['val_ratio']:.2f}" if r["val_ratio"] else "-",
                "Res Rate": f"{r['resolution_rate']*100:.1f}%",
                "Mode": r["sb_mode"],
                "SBs": str(len(sbs)),
                "Skills/SB": _med_mean_str(skill_counts),
                "Tokens/SB": _med_mean_str(token_counts),
                "Ctx%": f"{ctx_pct:.1f}%",
                "Skill tok": _med_mean_str(all_skill_toks),
            })
        if rows:
            _print_table_rows(headers, rows)
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

        max_iter = max(iter_data.keys())
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
        chart_width = max(50, len(iterations) * 8)

        print(f"  {_short_run(r['run_name'], 50)} [{r['learn_mode']}]")
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
        for i, it in enumerate(iterations):
            delta_str = f"Δ{deltas[i]:+.1f}" if deltas[i] is not None else ""
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
    """Output 3: Reference rate in cross-tab of learn_mode × specificity."""
    print("=== Reference rate by learn mode × skill type ===")
    print("  (Retrieval runs: stats based on top_k skills actually presented, not full skillbook)")
    print()

    # Aggregate across all runs, grouped by learn_mode
    # For each run, compute references and classify skills
    agg = defaultdict(lambda: {
        "general_total": 0, "specific_total": 0,
        "general_cited": 0, "specific_cited": 0,
        "general_explicit": 0, "specific_explicit": 0,
        "general_indirect": 0, "specific_indirect": 0,
    })

    for r in runs:
        ret_enabled = r.get("retrieval", {}).get("enabled", False)
        ref_data = _compute_presented_skill_refs(r) if ret_enabled else compute_references(r)
        skill_refs = ref_data.get("skill_refs", {})
        learn = r["learn_mode"]
        a = agg[learn]

        for sid, ref_info in skill_refs.items():
            spec = ref_info.get("specificity", "general")
            is_cited = ref_info["explicit"] > 0 or ref_info["indirect"] > 0
            key_prefix = "general" if spec == "general" else "specific"

            a[f"{key_prefix}_total"] += 1
            if is_cited:
                a[f"{key_prefix}_cited"] += 1
            if ref_info["explicit"] > 0:
                a[f"{key_prefix}_explicit"] += 1
            if ref_info["indirect"] > 0:
                a[f"{key_prefix}_indirect"] += 1

    if not any(a for a in agg.values()):
        print("  (No reference data available)")
        return

    headers = ["Category", "Skills", "% SB", "Cited", "Ref Rate", "Explicit", "Indirect"]
    rows = []

    for learn in sorted(agg.keys()):
        a = agg[learn]
        total_all = a["general_total"] + a["specific_total"]

        for category, prefix in [("General", "general"), ("Specific", "specific")]:
            n = a[f"{prefix}_total"]
            if n == 0:
                continue
            pct_sb = n / total_all * 100 if total_all > 0 else 0
            cited = a[f"{prefix}_cited"]
            ref_rate = cited / n * 100 if n > 0 else 0
            rows.append({
                "Category": f"{category} ({learn})",
                "Skills": str(n),
                "% SB": f"{pct_sb:.1f}%",
                "Cited": str(cited),
                "Ref Rate": f"{ref_rate:.1f}%",
                "Explicit": str(a[f"{prefix}_explicit"]),
                "Indirect": str(a[f"{prefix}_indirect"]),
            })

    # Totals
    for learn in sorted(agg.keys()):
        a = agg[learn]
        total = a["general_total"] + a["specific_total"]
        total_cited = a["general_cited"] + a["specific_cited"]
        if total == 0:
            continue
        ref_rate = total_cited / total * 100
        rows.append({
            "Category": f"TOTAL ({learn})",
            "Skills": str(total),
            "% SB": "100.0%",
            "Cited": str(total_cited),
            "Ref Rate": f"{ref_rate:.1f}%",
            "Explicit": str(a["general_explicit"] + a["specific_explicit"]),
            "Indirect": str(a["general_indirect"] + a["specific_indirect"]),
        })

    _print_table_rows(headers, rows)
    print()

    # Self-check
    for learn in sorted(agg.keys()):
        a = agg[learn]
        total = a["general_total"] + a["specific_total"]
        gen_pct = a["general_total"] / total * 100 if total > 0 else 0
        spec_pct = a["specific_total"] / total * 100 if total > 0 else 0
        if abs(gen_pct + spec_pct - 100.0) > 0.1:
            print(f"  WARNING: self-check failed for {learn}: Gen({gen_pct:.1f}%) + Spec({spec_pct:.1f}%) != 100%", file=sys.stderr)
    print("  (Self-check: Gen + Spec = Total for each row — OK)")
    print()


# ---------------------------------------------------------------------------
# Output 4: Per-run reference table (retrieval-aware)
# ---------------------------------------------------------------------------


def _compute_presented_ref_rate(run: dict) -> dict:
    """Compute ref rate among actually presented skills.

    For retrieval runs: extracts presented skills from each trajectory prompt,
    then checks if each presented skill's content appears in the assistant text.
    Uses substring matching (content phrases in assistant text) rather than
    fixed-width n-grams, which work poorly for short skill descriptions.
    For non-retrieval: uses the full skillbook reference data.

    Returns {"cited_presented": int, "total_presented": int, "rate": float}
    """
    retrieval = run.get("retrieval", {})
    ret_enabled = retrieval.get("enabled", False)
    trajectories = {k: v for k, v in run.get("trajectories", {}).items() if "/" not in k}

    if not ret_enabled:
        # For non-retrieval, use compute_references result
        ref_data = compute_references(run)
        skill_refs = ref_data.get("skill_refs", {})
        n_cited = sum(1 for r in skill_refs.values() if r["explicit"] > 0 or r["indirect"] > 0)
        n_total = len(skill_refs)
        return {
            "cited_presented": n_cited,
            "total_presented": n_total,
            "rate": n_cited / n_total * 100 if n_total > 0 else 0,
        }

    # For retrieval: check each trajectory's presented skills against assistant text
    total_presented = 0
    cited_presented = 0

    for inst_id, iters in trajectories.items():
        for it, traj in iters.items():
            presented = _extract_presented_skills(traj)
            if not presented:
                continue
            assistant_text = _extract_assistant_text(traj)
            if not assistant_text:
                continue
            lower_text = assistant_text.lower()

            for s in presented:
                total_presented += 1
                # Check explicit (skill ID in text)
                if s["id"] in assistant_text:
                    cited_presented += 1
                    continue
                # Check indirect: extract key phrases from skill content
                # Use 3-word sliding window (shorter than n-gram matching)
                content = s["content"]
                cited = False
                # First try the full content (minus prefix) as substring
                clean = re.sub(r"^(AVOID|VERIFIED|CONSIDER):\s*", "", content)
                # Try significant 3-word phrases
                words = clean.lower().split()
                for i in range(max(0, len(words) - 2)):
                    phrase = " ".join(words[i:i + 3])
                    if len(phrase) >= 15 and phrase not in INDIRECT_STOP_PHRASES:
                        if phrase in lower_text:
                            cited = True
                            break
                if cited:
                    cited_presented += 1

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
        "Run", "Learn", "Mode", "Retrieval",
        "SB Skills", "Skills/i", "Gen/i", "Spec/i",
        "Explicit", "Indirect", "Any Ref", "Refs/i",
        "Ref Rate", "Presented RR",
    ]
    rows = []

    for r in runs:
        ref_data = compute_references(r)
        summary = ref_data.get("summary", {})
        skill_refs = ref_data.get("skill_refs", {})

        retrieval = r.get("retrieval", {})
        ret_enabled = retrieval.get("enabled", False)
        top_k = retrieval.get("top_k", 0)

        # Total unique skills in the skillbook
        total_skills = len(skill_refs)

        # References (against full skillbook — may include non-presented for retrieval)
        explicit = summary.get("explicit_refs", 0)
        indirect = summary.get("indirect_refs", 0)
        any_ref = summary.get("any_refs", 0)

        # Retrieval label
        if ret_enabled:
            ret_label = f"top_k={top_k}"
        else:
            ret_label = "off"

        # Count trajectory iterations where skillbook was present (unprefixed only)
        trajectories = {k: v for k, v in r.get("trajectories", {}).items() if "/" not in k}
        n_traj_iters = sum(len(iters) for iters in trajectories.values())

        # Skills actually presented to agent per instance
        stats = r.get("statistics", {})
        ret_stats = stats.get("retrieval", {})
        total_presentations = summary.get("total_skill_presentations", 0)
        if ret_enabled:
            skills_per_inst = ret_stats.get("avg_skills_after", top_k)
        elif r["sb_mode"] == "per_repo" and total_presentations > 0 and n_traj_iters > 0:
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

        # Per-instance reference rate (against full skillbook)
        refs_per_inst = any_ref / n_traj_iters if n_traj_iters > 0 else 0
        ref_rate = refs_per_inst / skills_per_inst * 100 if skills_per_inst > 0 else 0

        # Presented RR: ref rate among actually presented skills
        presented_rr = _compute_presented_ref_rate(r)

        rows.append({
            "Run": _short_run(r["run_name"]),
            "Learn": r["learn_mode"],
            "Mode": r["sb_mode"],
            "Retrieval": ret_label,
            "SB Skills": str(total_skills),
            "Skills/i": f"{skills_per_inst:.0f}" if skills_per_inst == int(skills_per_inst) else f"{skills_per_inst:.1f}",
            "Gen/i": f"{gen_per_inst:.1f}",
            "Spec/i": f"{spec_per_inst:.1f}",
            "Explicit": str(explicit),
            "Indirect": str(indirect),
            "Any Ref": str(any_ref),
            "Refs/i": f"{refs_per_inst:.2f}",
            "Ref Rate": f"{ref_rate:.1f}%",
            "Presented RR": f"{presented_rr['rate']:.1f}%",
        })

    if not rows:
        print("  (No runs with reference data)")
        return

    _print_table_rows(headers, rows)
    print()
    print("  (Skills/i, Gen/i, Spec/i = avg per instance; Gen=process advice, Spec=repo-specific)")
    print("  (Ref Rate = refs per instance / skills per instance; Presented RR = % presented skills cited in output)")
    print()


# ---------------------------------------------------------------------------
# Output 5: General vs Specific classification by repo × learn mode
# ---------------------------------------------------------------------------


def print_classification_table(runs: list[dict]):
    """Output 4: Classification by repository × learn mode."""
    print("=== General vs Specific classification by repo × learn mode ===")
    print()

    # Collect data from per_repo and per_instance runs
    # For per_repo: each repo is a natural unit
    # For per_instance: group by repo (extracted from instance_id)
    data = defaultdict(lambda: defaultdict(lambda: {"general": 0, "specific": 0}))

    for r in runs:
        learn = r["learn_mode"]

        if r["sb_mode"] == "per_repo":
            for repo_name, sb in r["per_repo_sbs"].items():
                skills = _extract_skills(sb)
                for s in skills:
                    cls = classify_skill_specificity(s["content"], s["section"])
                    data[learn][repo_name][cls] += 1

        elif r["sb_mode"] == "per_instance":
            final_sbs = _get_final_skillbooks_per_instance(r)
            for sb in final_sbs:
                # Extract repo from skill section names or skillbook metadata
                # Instance IDs in skillbooks aren't directly available — use section names
                skills = _extract_skills(sb)
                for s in skills:
                    cls = classify_skill_specificity(s["content"], s["section"])
                    # Use "per_instance" as pseudo-repo since instances are mixed
                    data[learn]["per_instance"][cls] += 1

        elif r["sb_mode"] == "global":
            if r["global_sb"]:
                skills = _extract_skills(r["global_sb"])
                for s in skills:
                    cls = classify_skill_specificity(s["content"], s["section"])
                    data[learn]["global"][cls] += 1

    if not data:
        print("  (No skillbook data for classification)")
        return

    # Get all repos across learn modes
    all_repos = sorted(set(repo for learn_data in data.values() for repo in learn_data.keys()))
    all_learns = sorted(data.keys())

    headers = ["Repo"] + [f"{learn}" for learn in all_learns for _ in range(3)]
    # Sub-headers
    sub_headers = [""] + [s for _ in all_learns for s in ["Total", "Gen%", "Spec%"]]

    rows = []
    for repo in all_repos:
        row = {"Repo": repo.split("__")[-1][:20] if "__" in repo else repo[:20]}
        for learn in all_learns:
            gen = data[learn][repo].get("general", 0)
            spec = data[learn][repo].get("specific", 0)
            total = gen + spec
            gen_pct = gen / total * 100 if total > 0 else 0
            spec_pct = spec / total * 100 if total > 0 else 0
            row[f"{learn}_total"] = str(total)
            row[f"{learn}_gen_pct"] = f"{gen_pct:.0f}%"
            row[f"{learn}_spec_pct"] = f"{spec_pct:.0f}%"
        rows.append(row)

    # Total row
    total_row = {"Repo": "TOTAL"}
    for learn in all_learns:
        total_gen = sum(data[learn][repo].get("general", 0) for repo in all_repos)
        total_spec = sum(data[learn][repo].get("specific", 0) for repo in all_repos)
        total_all = total_gen + total_spec
        gen_pct = total_gen / total_all * 100 if total_all > 0 else 0
        spec_pct = total_spec / total_all * 100 if total_all > 0 else 0
        total_row[f"{learn}_total"] = str(total_all)
        total_row[f"{learn}_gen_pct"] = f"{gen_pct:.0f}%"
        total_row[f"{learn}_spec_pct"] = f"{spec_pct:.0f}%"
    rows.append(total_row)

    # Remap headers to match row keys
    flat_headers = ["Repo"]
    for learn in all_learns:
        flat_headers.extend([f"{learn}_total", f"{learn}_gen_pct", f"{learn}_spec_pct"])

    _print_table_rows(flat_headers, rows)
    print()

    # Self-check: Gen + Spec = Total
    for learn in all_learns:
        for repo in all_repos:
            gen = data[learn][repo].get("general", 0)
            spec = data[learn][repo].get("specific", 0)
            total = gen + spec
            if total > 0 and abs((gen / total * 100) + (spec / total * 100) - 100.0) > 0.1:
                print(f"  WARNING: self-check failed: {learn}/{repo}", file=sys.stderr)
    print("  (Self-check: Gen + Spec = Total for each cell — OK)")
    print()


# ---------------------------------------------------------------------------
# Sample references (for manual validation)
# ---------------------------------------------------------------------------


def print_sample_references(runs: list[dict], n_samples: int):
    """Print N random indirect reference matches for manual inspection."""
    print(f"=== Sample indirect references (N={n_samples}) ===")
    print()

    all_samples = []
    for r in runs:
        ref_data = compute_references(r)
        samples = ref_data.get("sample_matches", [])
        for s in samples:
            s["run"] = _short_run(r["run_name"])
            s["learn"] = r["learn_mode"]
        all_samples.extend(samples)

    if not all_samples:
        print("  (No indirect references found)")
        return

    selected = random.sample(all_samples, min(n_samples, len(all_samples)))

    for i, s in enumerate(selected, 1):
        print(f"  [{i}] {s['run']} [{s['learn']}]")
        print(f"      Skill: {s['skill_id']}")
        print(f"      N-gram: \"{s['ngram']}\"")
        print(f"      Context: {s['context']}")
        print()

    print(f"  Total indirect matches: {len(all_samples)} (showing {len(selected)})")
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
        writer.writerow(["Run", "Learn", "Split", "Max Attempts", "Resolution Rate", "SBs",
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
                r["run_name"], r["learn_mode"], r["val_ratio"] or "-", r["max_attempts"],
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
        writer.writerow(["Run", "Learn", "Split", "Resolution Rate", "Mode", "SBs",
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
                r["run_name"], r["learn_mode"], r["val_ratio"] or "-",
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
    parser.add_argument("--sample-references", type=int, metavar="N", default=None,
                        help="Print N random indirect reference matches for manual validation")
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
    print(f"Skillbook Quality Analysis")
    print(f"  Tokenizer: {TOKEN_METHOD}")
    print(f"  Runs processed: {len(runs)}, skipped: {skipped}")
    print(f"  Definitions: explicit ref = skill ID in assistant msg; "
          f"indirect ref = {INDIRECT_MIN_WORDS}-word n-gram match (min {INDIRECT_MIN_CHARS} chars); "
          f"general = process advice; specific = mentions concrete identifiers")
    print(f"  General/specific classification: by content analysis (NOT by AVOID/VERIFIED/CONSIDER prefix)")
    print()

    # Outputs
    print_summary_tables(runs)

    if args.growth or args.growth_runs is not None:
        print_growth_chart(runs, run_filter=args.growth_runs)

    print_reference_table(runs)
    print_per_run_reference_table(runs)
    print_classification_table(runs)

    if args.sample_references:
        print_sample_references(runs, args.sample_references)

    if args.output in ("csv", "both"):
        write_csv_tables(runs, args.csv_file)

    # Run legend
    print("Run legend:")
    for r in runs:
        n_sb = len(r["per_instance_sbs"]) or len(r["per_repo_sbs"]) or (1 if r["global_sb"] else 0)
        print(f"  {_short_run(r['run_name'], 50)}  [{r['learn_mode']}/{r['sb_mode']}]  "
              f"res={r['resolution_rate']*100:.1f}%  SBs={n_sb}")


if __name__ == "__main__":
    main()
