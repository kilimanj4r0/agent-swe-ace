#!/usr/bin/env python3
"""
Skillbook quality analysis for ACE-SWE experiment runs.

Analyzes final skillbook along dimensions critical for research:
  - Section distribution & concentration
  - Specificity: instance-specific solutions vs generalizable strategies
  - Distraction score: how many skills contain concrete edit instructions
  - Content quality: prefixes, length, metadata coverage
  - Cross-section redundancy (full pairwise cosine similarity)
  - Skill cross-references (explicit ID refs + implicit identifier coupling)
  - Cross-domain transfer attribution (which sections helped which repos)
  - Lost vs gained instances analysis
  - Token overhead estimation
  - Skill ID gap analysis (dedup/removal footprint)
  - Per-section quality profile

Usage:
    uv run python scripts/analyze_skillbooks.py --run-dir data/run_XXX
    uv run python scripts/analyze_skillbooks.py --run-dir data/run_XXX --skip-embeddings
    uv run python scripts/analyze_skillbooks.py --run-dir data/run_XXX -o report.md
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_run_data(run_dir: Path):
    """Load skillbook, statistics, config from a run directory.

    Looks for skillbook in benchmark-scoped subdirectory first
    (princeton-nlp__SWE-bench_Lite or Verified), falls back to flat layout.
    Also loads per-iteration train skillbooks and val trajectory directories.
    """
    stats_path = run_dir / "statistics.json"
    config_path = run_dir / "config.json"

    statistics = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    # Find skillbook — try benchmark-scoped then flat layout
    sb_dir = None
    for candidate in [
        run_dir / "princeton-nlp__SWE-bench_Lite" / "skillbooks",
        run_dir / "princeton-nlp__SWE-bench_Verified" / "skillbooks",
    ]:
        if candidate.exists():
            sb_dir = candidate
            break
    if sb_dir is None:
        sb_dir = run_dir / "skillbooks"

    final_sb_path = sb_dir / "final_skillbook.json"
    if not final_sb_path.exists():
        print(f"ERROR: no final_skillbook.json found in {sb_dir}", file=sys.stderr)
        sys.exit(1)

    skillbook = json.loads(final_sb_path.read_text())

    # Load per-iteration skillbooks from train/ subdirectory
    train_dir = sb_dir / "train"
    iter_skillbooks = {}
    if train_dir.exists():
        for f in sorted(train_dir.glob("iter_*.json")):
            iter_num = int(re.search(r"iter_(\d+)", f.name).group(1))
            iter_skillbooks[iter_num] = json.loads(f.read_text())

    # Val trajectory directories for lost/gained analysis
    traj_base = sb_dir.parent / "trajectories"
    val_traj_dir = traj_base / "val"
    vb_traj_dir = traj_base / "val_baseline"

    return {
        "skillbook": skillbook,
        "statistics": statistics,
        "config": config,
        "sb_dir": sb_dir,
        "iter_skillbooks": iter_skillbooks,
        "val_traj_dir": val_traj_dir,
        "vb_traj_dir": vb_traj_dir,
    }


def instance_to_repo(instance_id: str) -> str:
    """Convert instance ID to repo path: django__django-12184 -> django/django"""
    parts = instance_id.rsplit("-", 1)
    if "__" in parts[0]:
        return parts[0].replace("__", "/")
    return parts[0]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English code-ish text.

    This is a lower bound for content with many code identifiers (e.g.
    ClassName.method_name() tokenizes to many more tokens than 1 per word).
    """
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_overview(data: dict) -> list[str]:
    """Section 1: High-level overview.

    Reads config.json for experiment parameters and statistics.json for
    train instance counts. Reports total skills, sections, skillbook mode,
    dedup settings, and LLM models used.
    """
    sb = data["skillbook"]
    stats = data["statistics"]
    config = data["config"]
    skills = sb["skills"]

    lines = []
    lines.append("## 1. Overview\n")

    n_skills = len(skills)
    sections = set(s["section"] for s in skills.values())
    train_n = stats.get("train_phase", {}).get("total_instances", stats.get("total_instances", "?"))
    train_resolved = stats.get("train_phase", {}).get("resolved_count", stats.get("resolved_count", "?"))

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total skills | {n_skills} |")
    lines.append(f"| Unique sections | {len(sections)} |")
    lines.append(f"| Train instances | {train_n} |")
    lines.append(f"| Train resolved | {train_resolved} |")
    lines.append(f"| Skills per instance | {n_skills / max(1, int(train_n)):.1f} |")
    lines.append(f"| Skillbook mode | {config.get('experiment', {}).get('skillbook', {}).get('mode', '?')} |")
    lines.append(f"| Custom SWE learn | {config.get('experiment', {}).get('skillbook', {}).get('custom_swe_learn', '?')} |")
    lines.append(f"| Dedup enabled | {config.get('experiment', {}).get('skillbook', {}).get('deduplication', {}).get('enabled', '?')} |")
    lines.append(f"| Dedup threshold | {config.get('experiment', {}).get('skillbook', {}).get('deduplication', {}).get('similarity_threshold', '?')} |")
    lines.append(f"| Dedup within-section-only | {config.get('experiment', {}).get('skillbook', {}).get('deduplication', {}).get('within_section_only', '?')} |")
    lines.append(f"| Agent model | {config.get('llm', {}).get('agent', {}).get('model', '?')} |")
    lines.append(f"| ACE model | {config.get('llm', {}).get('ace', {}).get('model', '?')} |")
    lines.append("")

    return lines


def analyze_section_distribution(data: dict) -> list[str]:
    """Section 2: Section distribution and concentration metrics.

    Counts skills per section (category), sorted descending.
    Calculates:
      - Top-K coverage: what % of skills are in the top 5/10 sections
      - Gini coefficient: measures concentration inequality
        (0 = perfectly uniform, 1 = all skills in one section)
      - Singleton sections: sections with only 1 skill (potential over-fragmentation)
    """
    skills = list(data["skillbook"]["skills"].values())
    section_counts = Counter(s["section"] for s in skills)
    total = len(skills)

    lines = []
    lines.append("## 2. Section Distribution\n")
    lines.append(f"**{len(section_counts)} sections**, **{total} skills**\n")
    lines.append("| Section | Count | % | Cumulative % |")
    lines.append("|---------|-------|---|-------------|")

    cum = 0
    for sec, cnt in section_counts.most_common():
        pct = 100 * cnt / total
        cum += pct
        lines.append(f"| {sec} | {cnt} | {pct:.1f}% | {cum:.1f}% |")

    # Concentration: top-5, top-10, Gini
    counts_sorted = sorted(section_counts.values(), reverse=True)
    top5 = sum(counts_sorted[:5]) / total * 100
    top10 = sum(counts_sorted[:10]) / total * 100

    # Gini coefficient
    n = len(counts_sorted)
    cum_sum = sum((2 * (i + 1) - n - 1) * c for i, c in enumerate(sorted(counts_sorted)))
    gini = cum_sum / (n * sum(counts_sorted)) if sum(counts_sorted) > 0 else 0

    lines.append("")
    lines.append(f"**Concentration:** Top-5 sections cover {top5:.1f}%, top-10 cover {top10:.1f}%")
    lines.append(f"**Gini coefficient:** {gini:.3f} (0=uniform, 1=all in one section)")
    lines.append(f"**Singleton sections (1 skill):** {sum(1 for c in counts_sorted if c == 1)}")
    lines.append("")

    return lines


def analyze_specificity(data: dict) -> list[str]:
    """Section 3: Instance-specific solutions vs generalizable strategies.

    Two complementary analyses:

    3a. Category Classification
        Heuristic rules applied in priority order to classify each skill:
          - vague: multiple generic phrases + short text (platitudes)
          - instance_solution: mentions specific __init__, parameter values,
            file paths, line numbers, diff hunks — i.e. a solution to ONE task
          - fix_recipe: "Modify/Add/Change X to Y" — a concrete edit instruction
          - domain_pattern: mentions library/framework concepts (django, method, etc.)
          - general_strategy: everything else

        NOTE: domain_pattern is over-inclusive (catches any mention of "django",
        "method", "class" etc.), so 3b provides a more targeted signal.

    3b. Distraction Score
        Counts 8 concrete "edit-specific" regex signals in each skill's content.
        Each signal detects a specific kind of instance-specific instruction:
          - file_path: "sklearn/linear_model/ridge.py"
          - modify_specific: "Modify ClassName.__init__" (edit verb + named entity)
          - param_value: "= True", "= None" (specific parameter values)
          - line_number: "line 42" (exact line references)
          - call_with_arg: "method(arg=value)" (concrete function calls)
          - fix_requires: "the fix for X requires..." (fix location description)
          - bug_report: "was missing", "incorrectly" (specific bug symptoms)
          - dotted_entity: "ClassName.method" (specific code entities)

        Skills with 2+ signals are classified as potentially confusing/dangerous:
        they contain enough specific detail that the agent might try to apply
        the fix on a wrong instance. The threshold of 2+ was chosen because
        a single signal (e.g. just mentioning "django.Model") is often benign,
        but multiple signals together (file_path + line_number + dotted_entity)
        almost certainly indicate an instance-specific recipe.
    """
    skills_dict = data["skillbook"]["skills"]
    skills = list(skills_dict.values())

    lines = []
    lines.append("## 3. Specificity Analysis\n")

    # --- 3a: Heuristic category classification ---
    lines.append("### 3a. Category Classification\n")
    lines.append("Classifies each skill as instance-specific (solution to one task) or")
    lines.append("generalizable (transferable strategy). Uses text heuristics.\n")

    categories = Counter()
    examples = defaultdict(list)

    for s in skills:
        c = s["content"]
        cat = _classify_specificity(c)
        categories[cat] += 1
        if len(examples[cat]) < 3:
            examples[cat].append(f"[{s['id']}] {c[:180]}")

    lines.append("| Category | Count | % |")
    lines.append("|----------|-------|---|")
    for cat in ["instance_solution", "fix_recipe", "domain_pattern", "general_strategy", "vague"]:
        cnt = categories.get(cat, 0)
        pct = 100 * cnt / len(skills) if skills else 0
        lines.append(f"| {cat} | {cnt} | {pct:.1f}% |")
    lines.append("")

    for cat in ["instance_solution", "fix_recipe", "domain_pattern", "general_strategy", "vague"]:
        if examples[cat]:
            lines.append(f"**{cat}** examples:")
            for ex in examples[cat]:
                lines.append(f"  - {ex}")
            lines.append("")

    # --- 3b: Distraction score (instance-specific edit signals) ---
    lines.append("### 3b. Distraction Score\n")
    lines.append("Counts concrete edit-specific signals in each skill. Skills with 2+ signals")
    lines.append("are likely instance-specific recipes that could mislead the agent on a")
    lines.append("different issue.\n")

    signals_def = [
        ("file_path", r'[a-z_]+/[a-z_]+\.py', "Mentions specific file path (e.g. sklearn/linear_model/ridge.py)"),
        ("modify_specific", r'(?i)\b(modify|update|change|override|extend|rename|remove from|add to)\s+[A-Z]\w+(?:\.\w+)*',
         "Targets a specific named entity (e.g. Modify Field.__init__)"),
        ("param_value", r'=\s*(True|False|None|0o[0-7]+)',
         "Mentions specific parameter values (e.g. =False, =None)"),
        ("line_number", r'\b(line|lines)\s+\d+',
         "References specific line numbers"),
        ("call_with_arg", r'[a-z_]+\([^)]*=[^)]*\)',
         "Contains function call with keyword args"),
        ("fix_requires", r'(?i)(the fix for|the issue is|fix required|resolved by|the problem is)',
         "Describes exact fix location"),
        ("bug_report", r'(?i)(was missing|has a bug|is broken|fails when|crashes when|incorrectly)',
         "Reports a specific bug symptom"),
        ("dotted_entity", r'[A-Z][a-zA-Z]+\.[a-z_]+',
         "Contains ClassName.method references"),
    ]

    lines.append("| Signal | Description | Hits | % |")
    lines.append("|--------|-------------|------|---|")
    for name, pat, desc in signals_def:
        cnt = sum(1 for s in skills if re.search(pat, s["content"]))
        pct = 100 * cnt / len(skills) if skills else 0
        lines.append(f"| {name} | {desc} | {cnt} | {pct:.1f}% |")
    lines.append("")

    # Score each skill by counting how many different signals it triggers
    scored = []
    for sid, s in skills_dict.items():
        hits = [name for name, pat, _ in signals_def if re.search(pat, s["content"])]
        scored.append((sid, s, hits))

    sig_dist = Counter(len(hits) for _, _, hits in scored)
    lines.append("### Signal Count Distribution\n")
    lines.append("| Signals | Skills | % | Risk |")
    lines.append("|---------|--------|---|------|")
    for n in sorted(sig_dist.keys()):
        cnt = sig_dist[n]
        pct = 100 * cnt / len(skills)
        risk = "safe" if n == 0 else "low" if n == 1 else "**DANGEROUS**" if n >= 3 else "moderate"
        lines.append(f"| {n} | {cnt} | {pct:.1f}% | {risk} |")
    lines.append("")

    dangerous = [(sid, s, hits) for sid, s, hits in scored if len(hits) >= 2]
    lines.append(f"**Skills with 2+ signals (potentially confusing):** {len(dangerous)}/{len(skills)} ({100 * len(dangerous) / len(skills):.1f}%)\n")

    if dangerous:
        dangerous.sort(key=lambda x: -len(x[2]))
        lines.append("### Most Dangerous Skills (highest signal count)\n")
        lines.append("| Signals | Section | Content |")
        lines.append("|---------|---------|---------|")
        for sid, s, hits in dangerous[:25]:
            content = s["content"][:120].replace("|", "\\|")
            lines.append(f"| {', '.join(hits)} | {s['section']} | {content}... |")
        lines.append("")

    # Breakdown by section
    lines.append("### Dangerous Skills by Section\n")
    lines.append("| Section | Total | Dangerous (2+) | % |")
    lines.append("|---------|-------|----------------|---|")
    sec_total = Counter(s["section"] for s in skills)
    sec_danger = Counter(s["section"] for _, s, hits in scored if len(hits) >= 2)
    for sec, total in sec_total.most_common():
        d = sec_danger.get(sec, 0)
        if d > 0:
            lines.append(f"| {sec} | {total} | {d} | {100 * d / total:.1f}% |")
    lines.append("")

    return lines


def _classify_specificity(content: str) -> str:
    """Heuristic specificity classification.

    Applies rules in priority order (first match wins):
      1. vague — 2+ generic phrases in short text (≤120 chars)
      2. instance_solution — specific code artifacts: __init__ targets,
         parameter values, file paths, line numbers, diff hunks
      3. fix_recipe — "Modify/Add/Change X to Y" edit instructions
      4. domain_pattern — mentions library/framework concepts
      5. general_strategy — everything else
    """
    c = content
    c_lower = c.lower()

    # Very short or generic platitudes
    vague_words = ["ensure that", "be careful", "make sure", "properly",
                   "correctly handle", "pay attention", "deep understanding",
                   "consider the", "keep in mind", "important to note",
                   "should be aware", "needs to be"]
    if sum(1 for w in vague_words if w in c_lower) >= 2 and len(c) < 120:
        return "vague"

    # Instance-specific: mentions specific class.__method__, parameter values, file paths
    instance_patterns = [
        r'\w+\.__init__\s*(to|accept|take|with)',  # "RidgeClassifierCV.__init__ to accept..."
        r'=\s*(True|False|None|0x[0-9a-f]+|0o[0-7]+)',  # "store_cv_values=False"
        r'[a-z_]+/[a-z_]+\.py',  # file paths like "sklearn/linear_model/ridge.py"
        r'line\s+\d+',  # "line 42"
        r'@@\s*-\d+',  # diff hunks
    ]
    if any(re.search(p, c) for p in instance_patterns):
        return "instance_solution"

    # Fix recipe: "Modify X to do Y", "Add Z to W"
    fix_patterns = [
        r'(?i)(modify|change|update|fix|add|remove|replace|rename|refactor)\s+\w+\s+(to|by|from|in|with|for|so)',
        r'(?i)(implement|override|extend)\s+\w+\s+(to|by|for|in)',
    ]
    if any(re.search(p, c) for p in fix_patterns):
        return "fix_recipe"

    # Domain pattern: mentions specific library/framework concepts
    domain_words = [
        "django", "sympy", "matplotlib", "pytest", "sklearn", "scikit",
        "flask", "requests", "xarray", "astropy", "sphinx", "numpy",
        "orm", "queryset", "migration", "model", "view", "template",
        "widget", "form", "serializer", "matrix", "tensor", "symbol",
        "expression", "series", "function", "class", "method",
    ]
    if sum(1 for w in domain_words if w in c_lower) >= 1:
        return "domain_pattern"

    return "general_strategy"


def analyze_content_quality(data: dict) -> list[str]:
    """Section 4: Content quality dimensions.

    Reports:
      - Content length distribution (min/P25/median/P75/max/mean in chars)
        with rough token estimate (chars // 4)
      - Metadata coverage: how many skills have justification/evidence fields
        (these are filled by the UPDATE operation in SkillManager)
      - Content prefixes: AVOID/VERIFIED/CONSIDER (present only with
        custom_swe_learn=True; default ACE learn produces no prefixes)
      - Action keyword coverage: skills starting with an action verb or
        library name (proxy for how actionable the first impression is)
    """
    skills = list(data["skillbook"]["skills"].values())

    lines = []
    lines.append("## 4. Content Quality\n")

    # Length distribution
    lengths = [len(s["content"]) for s in skills]
    lines.append("### Content Length Distribution\n")
    lines.append("| Stat | Value |")
    lines.append("|------|-------|")
    lines.append(f"| Min | {min(lengths)} chars |")
    lines.append(f"| P25 | {np.percentile(lengths, 25):.0f} chars |")
    lines.append(f"| Median | {np.median(lengths):.0f} chars |")
    lines.append(f"| P75 | {np.percentile(lengths, 75):.0f} chars |")
    lines.append(f"| Max | {max(lengths)} chars |")
    lines.append(f"| Mean | {np.mean(lengths):.0f} chars ({estimate_tokens('x' * int(np.mean(lengths)))} tokens est.) |")
    lines.append("")

    # Justification / evidence coverage
    has_just = sum(1 for s in skills if s.get("justification"))
    has_evid = sum(1 for s in skills if s.get("evidence"))
    lines.append("### Metadata Coverage\n")
    lines.append("| Field | Present | % |")
    lines.append("|-------|---------|---|")
    lines.append(f"| justification | {has_just}/{len(skills)} | {100 * has_just / len(skills):.1f}% |")
    lines.append(f"| evidence | {has_evid}/{len(skills)} | {100 * has_evid / len(skills):.1f}% |")
    lines.append("")

    # Prefix analysis (AVOID/VERIFIED/CONSIDER for SWE-learn, none for default)
    prefixes = Counter()
    for s in skills:
        c = s["content"]
        if c.startswith("AVOID:"):
            prefixes["AVOID"] += 1
        elif c.startswith("VERIFIED:"):
            prefixes["VERIFIED"] += 1
        elif c.startswith("CONSIDER:"):
            prefixes["CONSIDER"] += 1
        else:
            prefixes["no_prefix"] += 1

    lines.append("### Content Prefixes\n")
    lines.append("| Prefix | Count | % |")
    lines.append("|--------|-------|---|")
    for p in ["AVOID", "VERIFIED", "CONSIDER", "no_prefix"]:
        cnt = prefixes.get(p, 0)
        pct = 100 * cnt / len(skills) if skills else 0
        lines.append(f"| {p} | {cnt} | {pct:.1f}% |")
    lines.append("")

    # Starts with action verb or not
    action_verbs = ["avoid", "use", "check", "verify", "ensure", "modify",
                    "implement", "fix", "add", "remove", "handle", "when",
                    "don't", "do not", "never", "always", "blockdiag",
                    "django", "sympy", "pytest", "sklearn", "matplotlib"]
    has_action = sum(1 for s in skills if any(s["content"].lower().startswith(v) for v in action_verbs))
    lines.append(f"**Starts with action/library keyword:** {has_action}/{len(skills)} ({100 * has_action / len(skills):.1f}%)")
    lines.append("")

    return lines


def analyze_redundancy(data: dict, embeddings: np.ndarray | None) -> list[str]:
    """Section 5: Cross-section redundancy analysis.

    Computes full pairwise cosine similarity between all skill embeddings.
    The same all-MiniLM-L6-v2 model used for dedup in the pipeline is used here.

    Key design choices:
      - Threshold matches dedup config: 0.85 cosine similarity
      - Within-section pairs: same section (dedup already handles these
        if within_section_only=True in config)
      - Cross-section pairs: different sections (NOT caught by dedup when
        within_section_only=True — these represent missed dedup opportunities)
      - Reports "potential reduction" as pairs // 2 (rough estimate of
        skills that could be removed if cross-section dedup were enabled)
    """
    skills_list = list(data["skillbook"]["skills"].values())
    total = len(skills_list)

    lines = []
    lines.append("## 5. Redundancy Analysis\n")

    if embeddings is None:
        lines.append("*Skipped (use --skip-embeddings=false or remove flag to enable).*\n")
        return lines

    # Build section -> indices mapping
    sec_to_idx = defaultdict(list)
    for i, s in enumerate(skills_list):
        sec_to_idx[s["section"]].append(i)

    # Compute full NxN cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T

    threshold = 0.85

    # Find all pairs above threshold (excluding self-similarity)
    cross_section_pairs = []
    within_section_pairs = []

    for i in range(total):
        for j in range(i + 1, total):
            if sim_matrix[i, j] >= threshold:
                sec_i = skills_list[i]["section"]
                sec_j = skills_list[j]["section"]
                pair = (i, j, float(sim_matrix[i, j]), sec_i, sec_j)
                if sec_i == sec_j:
                    within_section_pairs.append(pair)
                else:
                    cross_section_pairs.append(pair)

    lines.append(f"**Similarity threshold:** {threshold}")
    lines.append(f"**Within-section redundant pairs (≥ {threshold}):** {len(within_section_pairs)}")
    lines.append(f"**Cross-section redundant pairs (≥ {threshold}):** {len(cross_section_pairs)}")
    lines.append("")

    if cross_section_pairs:
        lines.append("### Top Cross-Section Redundant Pairs\n")
        cross_section_pairs.sort(key=lambda x: -x[2])
        lines.append("| Sim | Section A | Section B | Skill A | Skill B |")
        lines.append("|-----|-----------|-----------|---------|---------|")
        for i, j, sim, sec_a, sec_b in cross_section_pairs[:20]:
            ca = skills_list[i]["content"][:60].replace("|", "\\|")
            cb = skills_list[j]["content"][:60].replace("|", "\\|")
            lines.append(f"| {sim:.3f} | {sec_a} | {sec_b} | {ca}... | {cb}... |")
        lines.append("")

    # Per-section redundancy: count of cross-section redundant edges per section
    sec_redundancy = Counter()
    for i, j, sim, sec_a, sec_b in cross_section_pairs:
        sec_redundancy[sec_a] += 1
        sec_redundancy[sec_b] += 1

    if sec_redundancy:
        lines.append("### Sections with Most Cross-Section Redundancy\n")
        lines.append("| Section | Redundant skill-edges | Total skills | Redundancy % |")
        lines.append("|---------|----------------------|-------------|-------------|")
        for sec, cnt in sec_redundancy.most_common(15):
            total_sec = len(sec_to_idx[sec])
            pct = 100 * cnt / (total_sec * 2) if total_sec else 0  # edges, not skills
            lines.append(f"| {sec} | {cnt} | {total_sec} | {pct:.1f}% |")
        lines.append("")

    # Estimate how many skills would be removed by cross-section dedup
    skills_with_cross_dup = set()
    for i, j, sim, sec_a, sec_b in cross_section_pairs:
        skills_with_cross_dup.add(i)
        skills_with_cross_dup.add(j)
    lines.append(f"**Skills involved in cross-section redundancy:** {len(skills_with_cross_dup)} / {total}")
    lines.append(f"**Potential reduction if cross-section dedup applied:** ~{len(skills_with_cross_dup) // 2} skills removed")
    lines.append("")

    return lines


def analyze_skill_references(data: dict) -> list[str]:
    """Section 6: Cross-references between skills.

    Two levels of reference detection:

    1. Explicit references: one skill's ID (e.g. "django_migration-00001")
       appears in another skill's content, justification, or evidence fields.
       These would indicate the Reflector/SkillManager is building on existing
       skills. In practice, this is almost always 0 in default ACE learn.

    2. Implicit coupling (shared code identifiers): two skills share 3+
       lowercase identifiers (snake_case words) after filtering out common
       English stopwords and generic programming terms. This detects skills
       that talk about the same code entities (same class, function, concept)
       without explicitly referencing each other.

       The threshold of 3 shared identifiers was chosen to reduce noise from
       coincidental matches (e.g. two skills both mentioning "django" and "model").

    Reports:
      - Explicit ref count
      - Implicit coupling count (within-section vs cross-section)
      - Top coupled pairs with examples of shared identifiers
      - Topology: how many skills are "connected" vs "isolated"
    """
    sb = data["skillbook"]
    skills = sb["skills"]
    skill_ids = set(skills.keys())

    lines = []
    lines.append("## 6. Skill Cross-References\n")

    # Explicit references: one skill ID mentioned in another skill's content/justification/evidence
    explicit_refs = []
    for sid, s in skills.items():
        for field in ("content", "justification", "evidence"):
            val = s.get(field) or ""
            for other_id in skill_ids:
                if other_id != sid and other_id in val:
                    explicit_refs.append((sid, other_id, field))

    lines.append(f"**Explicit cross-references (skill ID in content/justification/evidence):** {len(explicit_refs)}")
    if explicit_refs:
        lines.append("")
        lines.append("| Source | Target | Field |")
        lines.append("|--------|--------|-------|")
        for src, tgt, field in explicit_refs[:20]:
            lines.append(f"| {src} | {tgt} | {field} |")
    lines.append("")

    # Implicit references: shared code identifiers between skills
    import re as _re
    id_pattern = _re.compile(r'[a-z_][a-z0-9_]{2,}\b')

    # Stopwords: common English words + generic programming terms that cause noise
    stopwords = {
        "the", "and", "for", "not", "are", "but", "all", "can", "has",
        "had", "was", "one", "our", "out", "use", "may", "than", "its",
        "also", "from", "when", "that", "this", "with", "will", "have",
        "been", "they", "which", "their", "about", "would", "there",
        "could", "other", "after", "should", "these", "before", "because",
        "ensure", "handle", "using", "method", "function", "class",
        "return", "value", "object", "instance", "default", "test",
        "tests", "case", "cases", "fix", "bug", "issue", "error",
        "raises", "raise", "cause", "causes", "avoid", "properly",
        "correctly", "update", "modify", "change", "instead", "without",
    }

    skill_identifiers = {}
    for sid, s in skills.items():
        ids = set(id_pattern.findall(s["content"].lower())) - stopwords
        skill_identifiers[sid] = ids

    # Count skills sharing at least 3 identifiers (threshold to reduce noise)
    shared_id_pairs = []
    sid_list = list(skill_ids)
    for i in range(len(sid_list)):
        for j in range(i + 1, len(sid_list)):
            shared = skill_identifiers[sid_list[i]] & skill_identifiers[sid_list[j]]
            if len(shared) >= 3:
                shared_id_pairs.append((sid_list[i], sid_list[j], len(shared), shared))

    lines.append(f"**Skills sharing 3+ code identifiers (implicit coupling):** {len(shared_id_pairs)}")
    lines.append("")

    if shared_id_pairs:
        shared_id_pairs.sort(key=lambda x: -x[2])
        lines.append("### Top Implicitly Coupled Skill Pairs\n")
        lines.append("| Skill A | Section A | Skill B | Section B | Shared identifiers | Examples |")
        lines.append("|---------|-----------|---------|-----------|--------------------|----------|")
        for a, b, n, shared in shared_id_pairs[:20]:
            sec_a = skills[a]["section"]
            sec_b = skills[b]["section"]
            examples = ", ".join(list(shared)[:4])
            lines.append(f"| {a} | {sec_a} | {b} | {sec_b} | {n} | {examples} |")
        lines.append("")

    # Within-section vs cross-section implicit coupling
    within = sum(1 for a, b, _, _ in shared_id_pairs if skills[a]["section"] == skills[b]["section"])
    cross = len(shared_id_pairs) - within
    lines.append(f"**Within-section implicitly coupled pairs:** {within}")
    lines.append(f"**Cross-section implicitly coupled pairs:** {cross}")
    lines.append("")

    # Topology summary
    skills_with_refs = set()
    for a, b, _, _ in shared_id_pairs:
        skills_with_refs.add(a)
        skills_with_refs.add(b)
    lines.append(f"**Skills involved in implicit coupling:** {len(skills_with_refs)} / {len(skills)}")
    lines.append(f"**Isolated skills (no coupling):** {len(skills) - len(skills_with_refs)} / {len(skills)}")
    lines.append("")

    return lines


def analyze_cross_domain(data: dict) -> list[str]:
    """Section 7: Cross-domain transfer analysis.

    Maps sections to domains (django/* -> django, sympy -> sympy, etc.)
    and analyzes whether skills from one domain help instances in another.

    For each newly_resolved instance: reports how many skills are from the
    same domain vs cross-domain (all skills are injected into every instance
    in global/per_repo mode, so the agent always sees cross-domain skills).

    Net impact: gained - lost per domain, showing which domains benefit
    from the global skillbook and which get hurt.
    """
    stats = data["statistics"]
    skills = list(data["skillbook"]["skills"].values())

    lines = []
    lines.append("## 7. Cross-Domain Transfer\n")

    summary = stats.get("summary", {})
    newly_resolved = summary.get("newly_resolved_by_skillbook", [])
    lost = summary.get("lost_by_skillbook", [])

    if not newly_resolved and not lost:
        lines.append("*No val comparison data available (single-phase run).*\n")
        return lines

    # Map section names to domains (e.g. "django_migration" -> "django")
    section_to_domain = {}
    for s in skills:
        sec = s["section"]
        if sec.startswith("django"):
            section_to_domain[sec] = "django"
        elif sec.startswith("sympy"):
            section_to_domain[sec] = "sympy"
        elif sec in ("sklearn", "scikit-learn"):
            section_to_domain[sec] = "sklearn"
        elif sec in ("matplotlib"):
            section_to_domain[sec] = "matplotlib"
        elif sec in ("pytest"):
            section_to_domain[sec] = "pytest"
        elif sec in ("flask"):
            section_to_domain[sec] = "flask"
        elif sec in ("requests"):
            section_to_domain[sec] = "requests"
        elif sec in ("xarray"):
            section_to_domain[sec] = "xarray"
        elif sec in ("astropy"):
            section_to_domain[sec] = "astropy"
        elif sec in ("sphinx"):
            section_to_domain[sec] = "sphinx"
        else:
            section_to_domain[sec] = "general"

    domain_counts = Counter(section_to_domain[s["section"]] for s in skills)
    lines.append("### Skill Distribution by Domain\n")
    lines.append("| Domain | Skills | % |")
    lines.append("|--------|--------|---|")
    for domain, cnt in domain_counts.most_common():
        lines.append(f"| {domain} | {cnt} | {100 * cnt / len(skills):.1f}% |")
    lines.append("")

    # For newly resolved: what domain is the instance, what domain are relevant skills?
    lines.append("### Newly Resolved Instances\n")
    lines.append("| Instance | Domain | Relevant skills (same domain) | Cross-domain skills |")
    lines.append("|----------|--------|------------------------------|--------------------|")
    for inst_id in newly_resolved:
        inst_domain = instance_to_repo(inst_id).split("/")[0] if "/" in instance_to_repo(inst_id) else instance_to_repo(inst_id)
        inst_domain_map = {"django": "django", "sympy": "sympy", "scikit-learn": "sklearn",
                          "matplotlib": "matplotlib", "pytest-dev": "pytest", "flask": "flask",
                          "psf": "requests", "pydata": "xarray", "astropy": "astropy",
                          "sphinx-doc": "sphinx", "mwaskom": "seaborn"}
        mapped_domain = inst_domain_map.get(inst_domain, inst_domain)
        same_domain = sum(1 for s in skills if section_to_domain.get(s["section"]) == mapped_domain)
        cross_domain = len(skills) - same_domain
        lines.append(f"| {inst_id} | {mapped_domain} | {same_domain} | {cross_domain} |")
    lines.append("")

    # For lost instances
    if lost:
        lines.append("### Lost Instances (resolved in baseline, lost with skillbook)\n")
        lines.append("| Instance | Domain |")
        lines.append("|----------|--------|")
        for inst_id in lost:
            inst_domain = instance_to_repo(inst_id).split("/")[0] if "/" in instance_to_repo(inst_id) else instance_to_repo(inst_id)
            inst_domain_map = {"django": "django", "sympy": "sympy", "scikit-learn": "sklearn",
                              "matplotlib": "matplotlib", "pytest-dev": "pytest", "flask": "flask",
                              "psf": "requests", "pydata": "xarray", "astropy": "astropy",
                              "sphinx-doc": "sphinx", "mwaskom": "seaborn"}
            lines.append(f"| {inst_id} | {inst_domain_map.get(inst_domain, inst_domain)} |")
        lines.append("")

    # Net impact by domain
    lines.append("### Net Impact Summary\n")
    gained_by_domain = Counter()
    lost_by_domain = Counter()
    inst_domain_map = {"django": "django", "sympy": "sympy", "scikit-learn": "sklearn",
                      "matplotlib": "matplotlib", "pytest-dev": "pytest", "flask": "flask",
                      "psf": "requests", "pydata": "xarray", "astropy": "astropy",
                      "sphinx-doc": "sphinx", "mwaskom": "seaborn"}
    for inst_id in newly_resolved:
        d = instance_to_repo(inst_id).split("/")[0] if "/" in instance_to_repo(inst_id) else instance_to_repo(inst_id)
        gained_by_domain[inst_domain_map.get(d, d)] += 1
    for inst_id in lost:
        d = instance_to_repo(inst_id).split("/")[0] if "/" in instance_to_repo(inst_id) else instance_to_repo(inst_id)
        lost_by_domain[inst_domain_map.get(d, d)] += 1

    lines.append("| Domain | Gained | Lost | Net |")
    lines.append("|--------|--------|------|-----|")
    all_domains = sorted(set(list(gained_by_domain.keys()) + list(lost_by_domain.keys())))
    for d in all_domains:
        gained_count = gained_by_domain.get(d, 0)
        lost_count = lost_by_domain.get(d, 0)
        lines.append(
            f"| {d} | +{gained_count} | -{lost_count} | "
            f"{'+' if gained_count - lost_count >= 0 else ''}{gained_count - lost_count} |"
        )
    total_g = sum(gained_by_domain.values())
    total_l = sum(lost_by_domain.values())
    lines.append(f"| **Total** | **+{total_g}** | **-{total_l}** | **{'+' if total_g - total_l >= 0 else ''}{total_g - total_l}** |")
    lines.append("")

    return lines


def analyze_lost_gained(data: dict) -> list[str]:
    """Section 8: Detailed lost vs gained instance comparison.

    For each newly_resolved and lost instance:
      - Shows baseline vs skillbook evaluation feedback from result JSONs
      - Compares trajectory lengths (message count, assistant message count)
        as a proxy for "distraction" — a much longer trajectory with skillbook
        suggests the agent was exploring more (could be good or bad);
        a much shorter one that fails suggests it gave up too quickly
    """
    stats = data["statistics"]
    summary = stats.get("summary", {})

    lines = []
    lines.append("## 8. Lost vs Gained Instance Details\n")

    newly = summary.get("newly_resolved_by_skillbook", [])
    lost = summary.get("lost_by_skillbook", [])

    if not newly and not lost:
        lines.append("*No val comparison data available.*\n")
        return lines

    val_results_dir = data["sb_dir"].parent / "results" / "val"
    vb_results_dir = data["sb_dir"].parent / "results" / "val_baseline"

    lines.append("### Newly Resolved (baseline failed, skillbook succeeded)\n")
    lines.append("| Instance | Baseline feedback | Skillbook feedback |")
    lines.append("|----------|-------------------|-------------------|")
    for inst_id in newly:
        vb_f = _load_result(vb_results_dir / inst_id / "iter_0.json")
        vs_f = _load_result(val_results_dir / inst_id / "iter_0.json")
        vb_fb = vb_f.get("feedback", "?")[:80].replace("|", "\\|") if vb_f else "?"
        vs_fb = vs_f.get("feedback", "?")[:80].replace("|", "\\|") if vs_f else "?"
        lines.append(f"| {inst_id} | {vb_fb} | {vs_fb} |")
    lines.append("")

    lines.append("### Lost (baseline succeeded, skillbook failed)\n")
    lines.append("| Instance | Baseline feedback | Skillbook feedback |")
    lines.append("|----------|-------------------|-------------------|")
    for inst_id in lost:
        vb_f = _load_result(vb_results_dir / inst_id / "iter_0.json")
        vs_f = _load_result(val_results_dir / inst_id / "iter_0.json")
        vb_fb = vb_f.get("feedback", "?")[:80].replace("|", "\\|") if vb_f else "?"
        vs_fb = vs_f.get("feedback", "?")[:80].replace("|", "\\|") if vs_f else "?"
        lines.append(f"| {inst_id} | {vb_fb} | {vs_fb} |")
    lines.append("")

    # Compare trajectory lengths as a distraction indicator
    lines.append("### Trajectory Length Comparison (lost vs gained)\n")
    val_traj_dir = data["val_traj_dir"]
    vb_traj_dir = data["vb_traj_dir"]

    lines.append("| Instance | Phase | Messages | Assistant msgs | Exit status |")
    lines.append("|----------|-------|----------|----------------|-------------|")
    for inst_id in newly + lost:
        label = "GAINED" if inst_id in newly else "LOST"
        for phase, tdir in [("val", val_traj_dir), ("baseline", vb_traj_dir)]:
            traj = _load_traj(tdir / inst_id / "iter_0.json")
            if traj:
                info = traj.get("info", {})
                lines.append(f"| {inst_id} | {label}/{phase} | {info.get('message_count', '?')} | {info.get('assistant_message_count', '?')} | {info.get('exit_status', '?')} |")
    lines.append("")

    return lines


def _load_result(path: Path) -> dict | None:
    """Load a result JSON file, returning None on any error."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def _load_traj(path: Path) -> dict | None:
    """Load a trajectory JSON file, returning None on any error."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def analyze_skillbook_utilization(data: dict) -> list[str]:
    """Section 9: Skillbook utilization — how many skills the agent actually uses.

    ALL skills are injected into the agent's prompt (global/per_repo mode),
    but only a fraction are actually referenced in the agent's responses.

    Detection method: for each val trajectory, scans all assistant messages
    for 4-word substrings from each skill's content. A skill is considered
    "referenced" if any 4-word window from its content appears verbatim
    in the agent's response text.

    This is a conservative (lower-bound) estimate because:
      - The agent may follow a skill's advice without quoting it verbatim
      - Paraphrased references are not detected
      - The agent may be influenced by a skill without any textual overlap

    Reports:
      - Per-instance utilization: how many of the 661 skills were referenced
      - Distribution: min/max/mean/median across all val instances
      - Most referenced skills: which skills the agent mentions most often
      - Utilization by section: which sections have the highest/lowest hit rates
      - Unused skills: skills never referenced across any val instance
    """
    sb = data["skillbook"]
    skills = sb["skills"]
    val_traj_dir = data["val_traj_dir"]

    lines = []
    lines.append("## 9. Skillbook Utilization\n")

    if not val_traj_dir.exists():
        lines.append("*No val trajectories found (single-phase run or no val dir).*\n")
        return lines

    # Pre-extract 4-word windows from each skill for matching
    skill_windows = {}
    for sid, s in skills.items():
        words = s["content"].split()
        windows = set()
        for i in range(max(1, len(words) - 3)):
            window = " ".join(words[i:i + 4]).lower()
            if len(window) > 15:  # skip very short windows (noise)
                windows.add(window)
        skill_windows[sid] = windows

    # Scan val trajectories
    inst_results = []  # (instance_id, matched_skill_ids)
    skill_ref_count = Counter()  # skill_id -> number of instances that referenced it
    total_instances = 0

    traj_files = sorted(val_traj_dir.glob("*/iter_0.json"))
    # Also check without subdirectory
    if not traj_files:
        traj_files = sorted(val_traj_dir.glob("iter_0.json"))

    for traj_file in traj_files:
        try:
            traj = json.loads(traj_file.read_text())
        except Exception:
            continue

        total_instances += 1
        instance_id = traj.get("info", {}).get("instance_id", traj_file.parent.name)

        # Concatenate all assistant messages
        assistant_msgs = [m["content"] for m in traj.get("messages", []) if m.get("role") == "assistant"]
        all_text = " ".join(assistant_msgs).lower()

        matched = []
        for sid, windows in skill_windows.items():
            if any(w in all_text for w in windows):
                matched.append(sid)
                skill_ref_count[sid] += 1

        inst_results.append((instance_id, matched))

    if not inst_results:
        lines.append("*No val trajectories found.*\n")
        return lines

    # Summary stats
    match_counts = [len(m) for _, m in inst_results]
    n_skills = len(skills)
    total_possible = n_skills * total_instances
    total_referenced = sum(match_counts)

    lines.append(f"Scanned **{total_instances}** val trajectories, **{n_skills}** skills injected into each.\n")

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total skills injected per instance | {n_skills} |")
    lines.append(f"| Skills referenced (min) | {min(match_counts)} |")
    lines.append(f"| Skills referenced (max) | {max(match_counts)} |")
    lines.append(f"| Skills referenced (mean) | {np.mean(match_counts):.1f} |")
    lines.append(f"| Skills referenced (median) | {int(np.median(match_counts))} |")
    lines.append(f"| **Utilization rate (mean)** | **{100 * np.mean(match_counts) / n_skills:.2f}%** |")
    lines.append(f"| Total injection-references | {total_referenced} / {total_possible} |")
    lines.append("")

    # Most referenced skills
    if skill_ref_count:
        lines.append("### Most Referenced Skills\n")
        lines.append("| Skill | Section | Instances referencing | % of val |")
        lines.append("|-------|---------|----------------------|----------|")
        for sid, cnt in skill_ref_count.most_common(20):
            sec = skills[sid]["section"]
            pct = 100 * cnt / total_instances
            lines.append(f"| {sid} | {sec} | {cnt} | {pct:.1f}% |")
        lines.append("")

    # Unused skills
    used_skills = set(skill_ref_count.keys())
    unused = set(skills.keys()) - used_skills
    lines.append(f"**Never referenced in any val instance:** {len(unused)} / {n_skills} ({100 * len(unused) / n_skills:.1f}%)\n")

    # Unused by section
    if unused:
        unused_by_sec = Counter(skills[sid]["section"] for sid in unused)
        total_by_sec = Counter(s["section"] for s in skills.values())
        lines.append("### Unused Skills by Section\n")
        lines.append("| Section | Total | Unused | Unused % |")
        lines.append("|---------|-------|--------|----------|")
        for sec, total in total_by_sec.most_common():
            u = unused_by_sec.get(sec, 0)
            lines.append(f"| {sec} | {total} | {u} | {100 * u / total:.1f}% |")
        lines.append("")

    # Per-instance detail (sample)
    lines.append("### Per-Instance Utilization Sample\n")
    lines.append("| Instance | Skills referenced | Referenced skill IDs |")
    lines.append("|----------|-------------------|---------------------|")
    for inst_id, matched in inst_results[:15]:
        ids_str = ", ".join(matched[:5]) + (f" +{len(matched) - 5} more" if len(matched) > 5 else "")
        lines.append(f"| {inst_id} | {len(matched)}/{n_skills} | {ids_str} |")
    lines.append("")

    return lines


def analyze_token_overhead(data: dict) -> list[str]:
    """Section 10: Token overhead of skillbook in context.

    Reconstructs the exact text that gets injected into the agent's prompt
    (skillbook section header + each skill as "### id\\n\\ncontent\\n\\n")
    and estimates its token cost.

    Token estimation: chars // 4 (rough heuristic; underestimates for content
    with many code identifiers which tokenize into multiple sub-word tokens).

    Usable context: context_window - max_tokens - 2000 (hardcoded safety buffer
    in the agent code, see CLAUDE.md: "max_input_tokens = context_window -
    max_tokens - 2000"). The skillbook is in the PROTECTED zone (first 1-2
    messages), so it is NEVER truncated — the overhead is permanent.

    Also reports per-section token cost to identify which sections consume
    the most context budget.
    """
    sb = data["skillbook"]
    skills = list(sb["skills"].values())
    config = data["config"]

    lines = []
    lines.append("## 10. Token Overhead\n")

    # Reconstruct exact text injected into prompt
    total_text = "## Learned Strategies (Skillbook)\n\nThese are strategies learned from previous attempts. Use them to guide your approach:\n\n"
    for s in skills:
        total_text += f"### {s['id']}\n\n{s['content']}\n\n"

    char_count = len(total_text)
    token_est = estimate_tokens(total_text)

    # Usable context from config (matches agent code: context_window - max_tokens - 2000)
    context_window = config.get("agent", {}).get("context", {}).get("context_window", 65536)
    max_tokens = config.get("llm", {}).get("agent", {}).get("max_tokens", 4096)
    usable = context_window - max_tokens - 2000

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Skillbook chars | {char_count:,} |")
    lines.append(f"| Skillbook tokens (est.) | {token_est:,} |")
    lines.append(f"| Context window | {context_window:,} |")
    lines.append(f"| Usable context (window - max_tokens - buffer) | {usable:,} |")
    lines.append(f"| **Skillbook overhead** | **{100 * token_est / usable:.1f}%** |")
    lines.append("")

    # Per-section overhead
    lines.append("### Per-Section Token Cost\n")
    lines.append("| Section | Skills | Tokens (est.) | % of skillbook |")
    lines.append("|---------|--------|---------------|----------------|")

    sec_sizes = Counter()
    for s in skills:
        sec_sizes[s["section"]] += estimate_tokens(s["content"]) + estimate_tokens(f"### {s['id']}\n\n")
    for sec, tokens in sec_sizes.most_common(15):
        cnt = sum(1 for s in skills if s["section"] == sec)
        pct = 100 * tokens / token_est if token_est else 0
        lines.append(f"| {sec} | {cnt} | {tokens:,} | {pct:.1f}% |")
    lines.append("")

    lines.append(f"**Interpretation:** Skillbook consumes {100 * token_est / usable:.1f}% of usable context. ")
    lines.append("This leaves less room for observation history, which triggers earlier context truncation.\n")

    return lines


def analyze_skill_gaps(data: dict) -> list[str]:
    """Section 11: Skill ID gap analysis (dedup/removal footprint).

    Skill IDs are globally sequential: {section_prefix}-{NNNNN}. The counter
    is shared across all sections (e.g. django_migration-00001, sympy-00009,
    sklearn-00092 — all from one global sequence).

    Gaps (max_id - surviving_count per section) indicate skills that were
    generated but later removed by deduplication, TAG operations, or other
    consolidation. Note: because the counter is global, the gap count per
    section also reflects skills generated for OTHER sections. The total
    across all sections gives the overall generation→survival ratio.

    Also reports skill growth across training iterations (if train/iter_N.json
    files are available), showing cumulative accumulation.
    """
    sb = data["skillbook"]
    skills = sb["skills"]

    lines = []
    lines.append("## 11. Skill ID Gap Analysis\n")
    lines.append("Gaps in skill ID sequences indicate skills removed by dedup or other operations.\n")

    # Group by section prefix, count gaps between max ID and surviving count
    section_prefixes = defaultdict(list)
    for sid in skills:
        prefix = sid.rsplit("-", 1)[0]
        num = int(sid.rsplit("-", 1)[1])
        section_prefixes[prefix].append(num)

    total_gaps = 0
    total_skills = 0
    lines.append("| Section | Surviving | Max ID | Gaps | Removed est. |")
    lines.append("|---------|-----------|--------|------|-------------|")
    for prefix in sorted(section_prefixes.keys()):
        nums = sorted(section_prefixes[prefix])
        count = len(nums)
        max_num = max(nums)
        gaps = max_num - count
        total_gaps += gaps
        total_skills += count
        if gaps > 0:
            lines.append(f"| {prefix} | {count} | {max_num} | {gaps} | ~{gaps} |")
    lines.append(f"| **TOTAL** | **{total_skills}** | | **{total_gaps}** | |")
    lines.append("")
    lines.append(f"**{total_gaps} skills removed** from {total_skills + total_gaps} total generated ")
    lines.append(f"(survival rate: {100 * total_skills / (total_skills + total_gaps):.1f}%).\n")

    # Skill growth over iterations (if train iter skillbooks available)
    iter_sbs = data["iter_skillbooks"]
    if iter_sbs:
        lines.append("### Skill Growth Over Training Iterations\n")
        lines.append("| Iteration | Skills | New skills |")
        lines.append("|-----------|--------|------------|")
        prev_ids = set()
        for it in sorted(iter_sbs.keys()):
            iter_sb = iter_sbs[it]
            curr_ids = set(iter_sb["skills"].keys())
            new = len(curr_ids - prev_ids)
            lines.append(f"| {it} | {iter_sb['skill_count']} | +{new} |")
            prev_ids = curr_ids
        lines.append("")

    return lines


def analyze_per_section_quality(data: dict) -> list[str]:
    """Section 12: Per-section quality profile.

    Combines metrics from previous analyses into a per-section summary:
      - Count and average content length
      - Justification coverage (how many skills have justification)
      - Specificity breakdown: how many are instance_solution, fix_recipe,
        domain_pattern, or general_strategy (from _classify_specificity)

    Useful for identifying sections that are predominantly instance-specific
    (high instance_solution + fix_recipe) vs genuinely strategic.
    """
    skills = list(data["skillbook"]["skills"].values())

    lines = []
    lines.append("## 12. Per-Section Quality Profile\n")

    sec_data = defaultdict(lambda: {"skills": [], "contents": []})
    for s in skills:
        sec_data[s["section"]]["skills"].append(s)
        sec_data[s["section"]]["contents"].append(s["content"])

    lines.append("| Section | Count | Avg len | Has just. | Instance-specific | Fix-recipe | Domain pattern | General |")
    lines.append("|---------|-------|---------|-----------|--------------------|------------|---------------|---------|")

    section_counts = Counter(s["section"] for s in skills)
    for sec, _ in section_counts.most_common(20):
        sec_skills = sec_data[sec]["skills"]
        cnt = len(sec_skills)
        avg_len = np.mean([len(s["content"]) for s in sec_skills])
        has_just = sum(1 for s in sec_skills if s.get("justification"))
        specs = Counter(_classify_specificity(s["content"]) for s in sec_skills)
        lines.append(
            f"| {sec} | {cnt} | {avg_len:.0f} | {has_just} | "
            f"{specs.get('instance_solution', 0)} | {specs.get('fix_recipe', 0)} | "
            f"{specs.get('domain_pattern', 0)} | {specs.get('general_strategy', 0)} |"
        )
    lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_embeddings(skills_list: list[dict]) -> np.ndarray:
    """Compute sentence embeddings for all skills using the same model
    as the pipeline dedup (all-MiniLM-L6-v2, 384-dim vectors).

    Returns a numpy array of shape (n_skills, 384).
    """
    from sentence_transformers import SentenceTransformer

    print(f"Computing embeddings for {len(skills_list)} skills...", file=sys.stderr)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    contents = [s["content"] for s in skills_list]
    embeddings = model.encode(contents, show_progress_bar=True, batch_size=128)
    return np.array(embeddings)


def main():
    parser = argparse.ArgumentParser(description="Analyze skillbooks from experiment runs")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding computation (no redundancy analysis)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Loading run data from {run_dir}...", file=sys.stderr)
    data = load_run_data(run_dir)

    # Compute embeddings if not skipped
    embeddings = None
    if not args.skip_embeddings:
        skills_list = list(data["skillbook"]["skills"].values())
        if skills_list:
            embeddings = compute_embeddings(skills_list)

    # Run all analyses
    report_lines = []
    report_lines.append(f"# Skillbook Analysis: {run_dir.name}\n")
    report_lines.append("Generated by `scripts/analyze_skillbooks.py`\n")

    report_lines.extend(analyze_overview(data))
    report_lines.extend(analyze_section_distribution(data))
    report_lines.extend(analyze_specificity(data))
    report_lines.extend(analyze_content_quality(data))
    report_lines.extend(analyze_redundancy(data, embeddings))
    report_lines.extend(analyze_skill_references(data))
    report_lines.extend(analyze_cross_domain(data))
    report_lines.extend(analyze_lost_gained(data))
    report_lines.extend(analyze_skillbook_utilization(data))
    report_lines.extend(analyze_token_overhead(data))
    report_lines.extend(analyze_skill_gaps(data))
    report_lines.extend(analyze_per_section_quality(data))

    report = "\n".join(report_lines)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
