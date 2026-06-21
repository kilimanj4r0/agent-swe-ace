# src/cli/commands.py
"""CLI entry points for ACE-SWE experiment phases."""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger

# Setup path for imports
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from config.llm import LLMConfig, create_model, create_ace_client, create_model_settings
from agents.miniswe_agent import MiniSWEAgent
from phases.predict import PredictPhase
from phases.evaluate import EvaluatePhase
from phases.learn import LearnPhase
from runners.main_loop import ExperimentLoop
from retrieval import SkillRetriever, RandomRetriever, EmbeddingRetriever, BM25Retriever
from data_io.readers import load_skillbook, load_trajectory
from data_io.writers import save_config, save_statistics, get_run_dir
from utils.logging import setup_logging
from utils.llm_observer import enable_observability

load_dotenv(_src_dir.parent / ".env")

import litellm


def _setup_console_logging(log_level: str = "INFO"):
    """Setup console-only logging (before run_dir is created)."""
    setup_logging(run_dir=None, log_level=log_level)


def apply_litellm_config(config: dict):
    """Apply LiteLLM settings from config."""
    litellm_settings = config.get("litellm", {})
    if litellm_settings.get("suppress_debug_info", False):
        litellm.suppress_debug_info = True
    level_name = litellm_settings.get("log_level", "WARNING")
    litellm.log_level = level_name
    import logging
    litellm.verbose_logger.setLevel(level_name)
    for handler in litellm.verbose_logger.handlers:
        handler.setLevel(level_name)


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _build_skill_retriever(experiment_cfg: dict):
    """Create a skill retriever from config, or None if disabled.

    Supports four retriever types via the ``type`` config field:
      - ``"llm"`` (default): Two-stage filter+rank using an LLM.
      - ``"random"``: Pick k random skills (baseline).
      - ``"embedding"``: Cosine similarity via sentence-transformers.
      - ``"bm25"``: Lexical Okapi BM25 via the bm25s package.

    Retrieval applies automatically on:
      - single-phase mode (phase=None): per_instance experiments
      - val skillbook pass (phase="val"): two-phase experiments
    Skipped on "train" and "val_baseline" phases.

    Args:
        experiment_cfg: The experiment section of the config dict.

    Returns:
        A retriever instance (SkillRetrieverBase subclass) or None.
    """
    retrieval_cfg = experiment_cfg.get("skillbook", {}).get("retrieval", {})
    if not retrieval_cfg.get("enabled", False):
        return None

    retriever_type = retrieval_cfg.get("type", "llm")

    if retriever_type == "llm":
        return _build_llm_retriever(retrieval_cfg)
    elif retriever_type == "random":
        return RandomRetriever(
            top_k=retrieval_cfg.get("top_k", 5),
            skip_threshold=retrieval_cfg.get("skip_threshold", 10),
            seed=retrieval_cfg.get("seed"),
        )
    elif retriever_type == "embedding":
        return EmbeddingRetriever(
            model_name=retrieval_cfg.get("model", "Qwen/Qwen3-Embedding-4B"),
            device=retrieval_cfg.get("device", "cuda"),
            top_k=retrieval_cfg.get("top_k", 5),
            skip_threshold=retrieval_cfg.get("skip_threshold", 10),
            include_section=retrieval_cfg.get("include_section", False),
            batch_size=retrieval_cfg.get("batch_size", 32),
            cache_dir=retrieval_cfg.get("cache_dir"),
        )
    elif retriever_type == "bm25":
        return BM25Retriever(
            top_k=retrieval_cfg.get("top_k", 5),
            skip_threshold=retrieval_cfg.get("skip_threshold", 10),
            k1=retrieval_cfg.get("k1", 1.5),
            b=retrieval_cfg.get("b", 0.75),
            include_section=retrieval_cfg.get("include_section", False),
        )
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type!r}")


def _build_llm_retriever(retrieval_cfg: dict):
    """Build an LLM-based SkillRetriever from config."""
    model = retrieval_cfg.get("model")
    if not model:
        logger.warning("[Retriever] retrieval.enabled=true but no model specified, skipping")
        return None

    api_base = retrieval_cfg.get("api_base")
    api_key = os.environ.get(retrieval_cfg.get("api_key_env", "ZAI_API_KEY"), "EMPTY")

    if not api_base:
        logger.warning("[Retriever] retrieval.enabled=true but no api_base specified, skipping")
        return None

    return SkillRetriever(
        model=model,
        api_base=api_base,
        api_key=api_key,
        top_k=retrieval_cfg.get("top_k", 5),
        skip_threshold=retrieval_cfg.get("skip_threshold", 10),
        filter_prompt=retrieval_cfg.get("filter_prompt"),
        rank_prompt=retrieval_cfg.get("rank_prompt"),
        chunk_size=retrieval_cfg.get("chunk_size", 200),
        filter_target=retrieval_cfg.get("filter_target", 100),
        temperature=retrieval_cfg.get("temperature", 0.0),
        max_tokens=retrieval_cfg.get("max_tokens", 2048),
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_instances(config: dict) -> list:
    """Load instances from SWE-bench or file."""
    # Load from dataset
    logger.info(f"Loading dataset: {config['benchmark']['dataset']}")
    dataset = load_dataset(
        config["benchmark"]["dataset"],
        split=config["benchmark"].get("split", "test"),
    )
    instances = list(dataset)

    # Limit if specified
    max_instances = config["benchmark"].get("max_instances")
    if max_instances:
        instances = instances[:max_instances]

    # Exclude instances
    exclude = config["benchmark"].get("exclude_instances", [])
    if exclude:
        exclude_set = set(exclude)
        before = len(instances)
        instances = [i for i in instances if i["instance_id"] not in exclude_set]
        logger.info(f"Excluded {before - len(instances)} instances: {before} -> {len(instances)}")

    # Filter by repos
    filter_repos = config["benchmark"].get("filter_repos")
    if filter_repos:
        filter_set = set(filter_repos)
        before = len(instances)
        instances = [i for i in instances if i.get("repo") in filter_set]
        logger.info(f"Filtered by repos {filter_repos}: {before} -> {len(instances)}")

    return instances


def _load_split_manifest(config: dict) -> dict | None:
    """Load a split manifest file if configured."""
    manifest_path = config.get("experiment", {}).get("split", {}).get("manifest")
    if not manifest_path:
        return None
    path = Path(manifest_path)
    if not path.exists():
        logger.warning(f"Split manifest not found: {path}, falling back to seed-based split")
        return None
    with open(path) as f:
        return json.load(f)


def _split_from_manifest(instances: list, manifest: dict, repo: str | None = None) -> tuple:
    """Split instances using a pre-computed manifest.

    Args:
        instances: All available instances (filtered by repo if applicable).
        manifest: Loaded manifest dict.
        repo: If set, use per_repo split for this specific repo.
              If None, use global train/val lists.

    Returns:
        (train_instances, val_instances).
    """
    if repo and repo in manifest.get("per_repo", {}):
        repo_split = manifest["per_repo"][repo]
        train_ids = set(repo_split["train"])
        val_ids = set(repo_split["val"])
    else:
        train_ids = set(manifest["train_instances"])
        val_ids = set(manifest["val_instances"])

    instance_map = {i["instance_id"]: i for i in instances}
    train_instances = [instance_map[iid] for iid in train_ids if iid in instance_map]
    val_instances = [instance_map[iid] for iid in val_ids if iid in instance_map]

    # Warn about instances not in manifest
    known_ids = train_ids | val_ids
    missing = [i["instance_id"] for i in instances if i["instance_id"] not in known_ids]
    if missing:
        logger.warning(f"{len(missing)} instances not in manifest (new dataset version?): {missing[:5]}...")

    return train_instances, val_instances


def split_instances(instances: list, config: dict, repo: str | None = None) -> tuple:
    """Split instances into train and val sets.

    Args:
        instances: List of instance dicts to split.
        config: Full config dict.
        repo: If set, split only instances for this repo (used by iterate_repos).

    Returns:
        (train_instances, val_instances). If no split config, returns (instances, []).
    """
    split_config = config.get("experiment", {}).get("split")
    if not split_config:
        return instances, []

    # Try manifest first
    manifest = _load_split_manifest(config)
    if manifest:
        ratio = split_config.get("val_ratio")
        manifest_ratio = manifest.get("val_ratio", 0.2)
        # Only warn if val_ratio is explicitly set and mismatches manifest
        if ratio is not None and abs(ratio - manifest_ratio) > 0.01:
            logger.warning(
                f"Requested val_ratio={ratio} doesn't match manifest val_ratio={manifest_ratio}, "
                f"using manifest split"
            )
        return _split_from_manifest(instances, manifest, repo=repo)

    # Fallback: seed-based split
    ratio = split_config.get("val_ratio", 0.2)
    seed = config.get("experiment", {}).get("random_seed", 42)

    rng = random.Random(seed)
    shuffled = list(instances)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * ratio))
    val_instances = shuffled[:val_count]
    train_instances = shuffled[val_count:]

    logger.info(f"Split: {len(train_instances)} train, {len(val_instances)} val (ratio={ratio}, seed={seed})")
    return train_instances, val_instances


def list_repos(config: dict):
    """List all unique repos in the dataset with instance counts.

    If split config is set, also shows train/val split details.
    """
    instances = get_instances(config)
    repo_counts = Counter(i.get("repo", "unknown") for i in instances)

    print(f"\n{'Repo':<45} {'Count':>6}")
    print("-" * 53)
    for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
        print(f"{repo:<45} {count:>6}")
    print(f"\nTotal: {len(instances)} instances across {len(repo_counts)} repos")

    # If split or filter_repos is configured, show split preview
    filter_repos = config.get("benchmark", {}).get("filter_repos")
    split_config = config.get("experiment", {}).get("split")

    if filter_repos:
        filtered = [i for i in instances if i.get("repo") in set(filter_repos)]
        print(f"\nAfter filter_repos={filter_repos}: {len(filtered)} instances")

        if split_config:
            train, val = split_instances(filtered, config)
            _print_split(train, val)
    elif split_config:
        train, val = split_instances(instances, config)
        _print_split(train, val)

    sys.exit(0)


def _print_split(train: list, val: list):
    """Print train/val split details."""
    total = len(train) + len(val)
    val_ratio = len(val) / total if total > 0 else 0
    print(f"\nTrain/Val split ({len(train)}/{len(val)}, val_ratio={val_ratio:.2f}):")
    print(f"\n  TRAIN ({len(train)} instances):")
    for inst in train:
        print(f"    - {inst['instance_id']}")
    print(f"\n  VAL ({len(val)} instances):")
    for inst in val:
        print(f"    - {inst['instance_id']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ACE-SWE Experiment: Skillbook learning with mini-swe-agent"
    )
    parser.add_argument("--config", "-c", help="Override config file (merged on top of config.yaml)")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--max-instances", "-n", type=int, help="Max instances")
    parser.add_argument("--max-attempts", "-a", type=int, help="Max attempts per instance")
    parser.add_argument(
        "--phase",
        choices=["all", "predict", "evaluate", "learn"],
        default="all",
        help="Run specific phase only",
    )
    parser.add_argument("--instance", help="Run specific instance ID")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number")
    parser.add_argument("--skillbook", help="Path to skillbook JSON")
    parser.add_argument("--trajectory", help="Path to trajectory JSON (for evaluate/learn)")
    parser.add_argument("--patch", help="Patch string (for evaluate)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--observe", action="store_true", help="Enable Opik observability")
    parser.add_argument(
        "--resume-dir",
        nargs="+",
        type=Path,
        help="Path(s) to previous run directories. Resumes from last successful "
        "iteration per instance. Fully completed instances are copied.",
    )
    parser.add_argument(
        "--custom-swe-learn",
        action="store_true",
        help="Use SWE-optimized Reflector + SkillManager (extracts anti-patterns, preserves type prefixes).",
    )
    parser.add_argument(
        "--filter-repos",
        nargs="+",
        help="Only run instances from these repos (e.g., django/django)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        help="Fraction of instances for validation (e.g., 0.2)",
    )
    parser.add_argument(
        "--list-repos",
        action="store_true",
        help="List all unique repos in dataset with instance counts and exit",
    )
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        help="Previous run dir with baseline val results (skip re-running baseline pass)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan without running anything (no LLM calls, no Docker, no files written)",
    )

    args = parser.parse_args()

    # Setup console logging (file logging added after run_dir is created)
    _setup_console_logging(args.log_level)

    # Load base config, then merge override config on top
    config = load_config("config.yaml")
    if args.config:
        override = load_config(args.config)
        config = deep_merge(config, override)

    # Apply LiteLLM settings
    apply_litellm_config(config)

    # Override config with CLI args
    if args.max_instances:
        config.setdefault("benchmark", {})["max_instances"] = args.max_instances
    if args.max_attempts:
        config.setdefault("experiment", {})["max_attempts"] = args.max_attempts
    if args.output:
        config.setdefault("output", {})["dir"] = args.output
    if args.custom_swe_learn:
        config.setdefault("experiment", {}).setdefault("skillbook", {})["custom_swe_learn"] = True
    if args.filter_repos:
        config.setdefault("benchmark", {})["filter_repos"] = args.filter_repos
    if args.val_ratio is not None:
        config.setdefault("experiment", {}).setdefault("split", {})["val_ratio"] = args.val_ratio

    # List repos and exit (must be after config loading)
    if args.list_repos:
        list_repos(config)

    # Run appropriate phase
    # Note: Observability is enabled inside run_full_experiment with run_id as project name
    if args.phase == "all":
        run_full_experiment(config, args)
    elif args.phase == "predict":
        run_predict_cmd(config, args)
    elif args.phase == "evaluate":
        run_evaluate_cmd(config, args)
    elif args.phase == "learn":
        run_learn_cmd(config, args)


def _make_agent_factory(config: dict, agent_config: LLMConfig, output_dir: Path):
    """Create a factory that produces per-worker MiniSWEAgent instances.

    Each agent gets its own LitellmModel to avoid races on n_calls/cost counters.
    """
    def factory():
        agent_model = create_model(agent_config)
        return MiniSWEAgent(
            llm_model=agent_model,
            use_docker=config.get("environment", {}).get("type") == "docker",
            step_limit=config.get("agent", {}).get("step_limit", 100),
            cost_limit=config.get("agent", {}).get("cost_limit", 5.0),
            output_dir=output_dir,
            namespace=config.get("environment", {}).get("namespace"),
            context_management=config.get("agent", {}).get("context", {}).get("enabled", True),
            context_window=config.get("agent", {}).get("context", {}).get("context_window", 65536),
            max_tokens=config.get("llm", {}).get("agent", {}).get("max_tokens", 4096),
            keep_recent_messages=config.get("agent", {}).get("context", {}).get("keep_recent_messages", 6),
            truncate_threshold=config.get("agent", {}).get("context", {}).get("truncate_threshold", 0.85),
        )
    return factory


def _run_dry_run(config: dict, args, output_dir: Path, run_name: str):
    """Print execution plan without running anything."""
    benchmark = config["benchmark"]["dataset"].replace("/", "__")
    exp = config.get("experiment", {})
    sb = exp.get("skillbook", {})
    bm = config.get("benchmark", {})
    agent_cfg = config.get("agent", {})
    llm_cfg = config.get("llm", {})

    skip_learn = exp.get("skip_learn", False)
    max_attempts = exp.get("max_attempts", 2)
    val_pass_k = exp.get("val_pass_k", 1)
    dedup = sb.get("deduplication")

    print("\n=== DRY RUN ===")

    # ── 1. Configuration ─────────────────────────────────────────────────
    print("\nConfiguration:")
    print(f"  Run name:          {run_name}")
    print(f"  Output:            {output_dir}")
    print(f"  Benchmark:         {config['benchmark']['dataset']} (split: {bm.get('split', 'test')})")
    print(f"  Skillbook mode:    {sb.get('mode', 'per_instance')}")
    print(f"  Custom SWE learn:  {sb.get('custom_swe_learn', False)}")
    print(f"  Concurrency:       {exp.get('concurrency', 1)}")
    print(f"  Max attempts:      {max_attempts}")
    print(f"  Val pass K:        {val_pass_k}")
    if skip_learn:
        print(f"  Skip learn:        True (Learn phase disabled)")
    else:
        print(f"  Force learn:       {exp.get('force_learn', True)}")
    if dedup:
        print(f"  Deduplication:     enabled (threshold={dedup.get('similarity_threshold', 'default')})")

    # ── 2. LLM ───────────────────────────────────────────────────────────
    print("\nLLM:")
    for role in ("agent", "ace"):
        role_cfg = llm_cfg.get(role, {})
        model = role_cfg.get("model", "not set")
        api_base = role_cfg.get("api_base", "default")
        max_tokens = role_cfg.get("max_tokens", "default")
        print(f"  {role}: model={model}, api_base={api_base}, max_tokens={max_tokens}")

    # ── 3. Dataset ───────────────────────────────────────────────────────
    print("\nDataset:")
    if bm.get("max_instances"):
        print(f"  Max instances:     {bm['max_instances']}")
    if bm.get("exclude_instances"):
        print(f"  Exclude:           {len(bm['exclude_instances'])} instances")
    if bm.get("filter_repos"):
        print(f"  Filter repos:      {bm['filter_repos']}")

    instances = get_instances(config)
    print(f"  Instances loaded:  {len(instances)}")

    # Instance filter
    if args.instance:
        instances = [i for i in instances if i.get("instance_id") == args.instance]
        if not instances:
            print(f"\n  ERROR: Instance not found: {args.instance}")
            return
        print(f"  Filtered to:       {args.instance}")

    # ── 4. Data Sources ──────────────────────────────────────────────────
    # Resolve baseline_run_dir (CLI > config > auto-derived from resume_dirs)
    baseline_run_dir = getattr(args, "baseline_run_dir", None)
    if not baseline_run_dir:
        config_baseline = exp.get("baseline_run_dir")
        if config_baseline:
            baseline_run_dir = Path(config_baseline)
    if baseline_run_dir:
        baseline_run_dir = Path(baseline_run_dir)
        if not baseline_run_dir.exists():
            print(f"\n  Baseline run dir:  {baseline_run_dir}  ⚠️  DOES NOT EXIST — reuse disabled")
            baseline_run_dir = None

    # Resolve resume_dirs (CLI > config)
    cli_resume_dirs = getattr(args, "resume_dir", None)
    config_resume_dirs = exp.get("resume_dirs")
    resume_dirs = cli_resume_dirs or (
        [Path(p) for p in config_resume_dirs] if config_resume_dirs else None
    )

    # Auto-derive baseline_run_dir from resume_dirs (matches execution logic)
    if not baseline_run_dir and resume_dirs:
        baseline_run_dir = resume_dirs[0]

    train_trajs_dir = exp.get("train_trajs_dir")
    skillbook_source_dir = exp.get("skillbook_source_dir")

    # Print data sources (each shown exactly once)
    if baseline_run_dir or train_trajs_dir or skillbook_source_dir or resume_dirs:
        print("\nData sources:")
        if baseline_run_dir:
            print(f"  Baseline run dir:  {baseline_run_dir}")
        if train_trajs_dir:
            print(f"  Train trajs dir:   {train_trajs_dir} (teacher distillation)")
        if skillbook_source_dir:
            print(f"  Skillbook source:  {skillbook_source_dir} (validation-only)")
        if resume_dirs:
            print(f"  Resume dirs:       {[str(d) for d in resume_dirs]}")

    # ── 5. Split ─────────────────────────────────────────────────────────
    iterate_repos_list = bm.get("iterate_repos")

    if iterate_repos_list:
        # iterate_repos mode: split per-repo
        print(f"\nIterate repos ({len(iterate_repos_list)} repos):")
        repo_groups = defaultdict(list)
        for inst in instances:
            repo_groups[inst.get("repo", "unknown")].append(inst)

        for repo in iterate_repos_list:
            repo_insts = repo_groups.get(repo, [])
            if not repo_insts:
                print(f"  {repo}: NOT FOUND in dataset")
                continue
            train, val = split_instances(repo_insts, config, repo=repo)

            extras = []
            if train_trajs_dir:
                _trajs_dir = Path(train_trajs_dir)
                if _trajs_dir.exists():
                    _covered = sum(
                        1 for i in train
                        if (_trajs_dir / i["instance_id"] / f"{i['instance_id']}.traj.json").exists()
                    )
                    extras.append(f"teacher: {_covered}/{len(train)}")
            if skillbook_source_dir:
                extras.append("validation-only")

            extra_str = f" ({', '.join(extras)})" if extras else ""
            print(f"  {repo}: {len(repo_insts)} total → {len(train)} train / {len(val)} val{extra_str}")

        # Show split source (manifest vs config)
        _manifest = _load_split_manifest(config)
        _manifest_path = exp.get("split", {}).get("manifest")
        _split_cfg = exp.get("split", {})
        if _manifest:
            print(f"  Val ratio:         {_manifest.get('val_ratio', _split_cfg.get('val_ratio', 0.2))} (from manifest)")
            if _manifest_path:
                print(f"  Manifest:          {_manifest_path}")
        elif _split_cfg.get("val_ratio"):
            print(f"  Val ratio:         {_split_cfg['val_ratio']} (from config)")

        # Per-repo train source
        if skillbook_source_dir:
            print(f"  Train:             SKIPPED (validation-only)")
        elif train_trajs_dir:
            print(f"  Train:             teacher distillation")
        elif baseline_run_dir:
            print(f"  Train:             baseline reuse from {baseline_run_dir}")

        if baseline_run_dir:
            print(f"  Val reuse:         baseline results from {baseline_run_dir}")
    else:
        # Standard split
        train_instances, val_instances = split_instances(instances, config)
        is_two_phase = len(val_instances) > 0

        print(f"\nSplit:")
        print(f"  Train:             {len(train_instances)} instances")
        print(f"  Val:               {len(val_instances)} instances")
        if is_two_phase:
            split_cfg = exp.get("split", {})
            manifest = _load_split_manifest(config)
            manifest_path = split_cfg.get("manifest")
            if manifest:
                actual_ratio = manifest.get("val_ratio", split_cfg.get("val_ratio", 0.2))
                print(f"  Val ratio:         {actual_ratio} (from manifest)")
                if manifest_path:
                    print(f"  Manifest:          {manifest_path}")
            else:
                actual_ratio = split_cfg.get("val_ratio", 0.2)
                print(f"  Val ratio:         {actual_ratio} (from config)")

        # ── 6. Resume ────────────────────────────────────────────────────
        print(f"\nResume:")
        if resume_dirs:
            from data_io.resume_scanner import scan_resume_dirs

            instance_ids = [i["instance_id"] for i in train_instances]
            resume_state = scan_resume_dirs(resume_dirs, benchmark, instance_ids, max_attempts)

            complete_ids = {iid for iid, rp in resume_state.items() if rp.is_fully_complete}
            partial_ids = {
                iid for iid, rp in resume_state.items()
                if not rp.is_fully_complete and rp.last_complete_iter >= 0
            }
            broken_ids = {
                iid for iid, rp in resume_state.items()
                if not rp.is_fully_complete and rp.last_complete_iter < 0
            }
            fresh_ids = set(instance_ids) - set(resume_state.keys())

            print(f"  Train complete:    {len(complete_ids)} (copy)")
            print(f"  Train broken:      {len(broken_ids)} (retry)")
            print(f"  Train partial:     {len(partial_ids)} (continue)")
            print(f"  Train fresh:       {len(fresh_ids)} (new)")

            if val_instances:
                val_ids = [i["instance_id"] for i in val_instances]
                val_resume = scan_resume_dirs(resume_dirs, benchmark, val_ids, max_attempts, skip_learn=True)
                val_complete = {iid for iid, rp in val_resume.items() if rp.is_fully_complete}
                val_broken = {iid for iid, rp in val_resume.items() if not rp.is_fully_complete and rp.last_complete_iter < 0}
                val_partial = {iid for iid, rp in val_resume.items() if not rp.is_fully_complete and rp.last_complete_iter >= 0}
                val_fresh = set(val_ids) - set(val_resume.keys())
                print(f"  Val complete:      {len(val_complete)} (copy)")
                print(f"  Val broken:        {len(val_broken)} (retry)")
                print(f"  Val partial:       {len(val_partial)} (continue)")
                print(f"  Val fresh:         {len(val_fresh)} (new)")
        else:
            print(f"  None — all instances processed from scratch")

        # ── 7. Baseline / Teacher coverage ───────────────────────────────
        if baseline_run_dir and not is_two_phase:
            baseline_dir = Path(baseline_run_dir)
            baseline_traj_dir = baseline_dir / benchmark / "trajectories"
            baseline_count = 0
            if baseline_traj_dir.exists():
                baseline_count = sum(
                    1 for iid in (i["instance_id"] for i in train_instances)
                    if (baseline_traj_dir / iid / "iter_0.json").exists()
                )
            print(f"\nBaseline reuse:")
            print(f"  Available:         {baseline_count}/{len(train_instances)} instances")
            print(f"  Remaining:         {len(train_instances) - baseline_count} will run from scratch")

        if baseline_run_dir and is_two_phase:
            print(f"\nBaseline reuse:")
            print(f"  Reuse from:        {baseline_run_dir} (val baseline)")

        if train_trajs_dir and is_two_phase:
            _trajs_dir = Path(train_trajs_dir)
            if _trajs_dir.exists():
                _covered = sum(
                    1 for i in train_instances
                    if (_trajs_dir / i["instance_id"] / f"{i['instance_id']}.traj.json").exists()
                )
                print(f"\nTeacher trajectory coverage:")
                print(f"  Available:         {_covered}/{len(train_instances)} train instances")
                print(f"  Skipped:           {len(train_instances) - _covered} (no teacher traj)")

        if skillbook_source_dir:
            print(f"\nValidation-only mode:")
            print(f"  Skillbook source:  {skillbook_source_dir}")
            print(f"  Training:          SKIPPED entirely")

    # ── 8. Execution Plan ────────────────────────────────────────────────
    print(f"\nExecution Plan:")
    if iterate_repos_list:
        concurrency = _resolve_iterate_repos_concurrency(exp)
        if concurrency > 1:
            workers = min(concurrency, len(iterate_repos_list))
            print(f"  Mode:              iterate_repos ({workers} repos in parallel)")
        else:
            print(f"  Mode:              iterate_repos (sequential)")
        print(f"  Per repo:          train → val baseline ({val_pass_k} attempt(s)) → val skillbook ({val_pass_k} attempt(s))")
        if skillbook_source_dir:
            print(f"  Train:             SKIPPED (validation-only)")
        elif train_trajs_dir:
            print(f"  Train:             learn-only from teacher trajectories")
        else:
            print(f"  Train:             {max_attempts} attempt(s), force_learn=True")
        print(f"  Skillbook:         {sb.get('mode', 'per_instance')} mode")
        if dedup:
            print(f"  Post-train dedup:  enabled")
    elif is_two_phase:
        print(f"  Mode:              Two-phase (train → val baseline → val skillbook)")
        print(f"  Phase 1 - Train:")
        if skillbook_source_dir:
            print(f"    SKIPPED (validation-only, skillbook loaded from {skillbook_source_dir})")
        elif train_trajs_dir:
            print(f"    Teacher distillation from {train_trajs_dir}")
            print(f"    {len(train_instances)} instances × learn-only (no predict/eval)")
        elif baseline_run_dir:
            print(f"    Baseline reuse from {baseline_run_dir}")
            print(f"    {len(train_instances)} instances × reuse predict+eval, run learn")
        else:
            print(f"    {len(train_instances)} instances × 1 attempt, force_learn=True")
        if not skillbook_source_dir:
            print(f"    Skillbook: {sb.get('mode', 'per_instance')} mode, accumulates across train")
            if dedup:
                print(f"    Post-train dedup: enabled")
        print(f"  Phase 2 - Val baseline:")
        print(f"    {len(val_instances)} instances × {val_pass_k} attempt(s), empty skillbook, frozen")
        if baseline_run_dir:
            print(f"    Reuse results from: {baseline_run_dir} (up to {val_pass_k} iterations)")
        print(f"  Phase 3 - Val skillbook:")
        print(f"    {len(val_instances)} instances × {val_pass_k} attempt(s), learned skillbook, frozen")
    else:
        print(f"  Mode:              Single-phase")
        print(f"  Instances:         {len(train_instances)} × up to {max_attempts} attempts")
        if baseline_run_dir:
            print(f"  iter_0:            reuse predict+eval from baseline, then learn + continue")
        else:
            if skip_learn:
                print(f"  Per instance:      predict → evaluate (no learning)")
            else:
                print(f"  Per instance:      predict → evaluate → (if unresolved) learn → retry")

    concurrency = exp.get("concurrency", 1)
    if concurrency > 1:
        print(f"  Concurrency:       {concurrency} (within-repo val parallel; "
              f"train sequential; evaluation serialized)")

    # ── 9. Limits ────────────────────────────────────────────────────────
    ctx = agent_cfg.get("context", {})
    agent_llm = llm_cfg.get("agent", {})
    print(f"\nLimits:")
    print(f"  Step limit:        {agent_cfg.get('step_limit', 100)}")
    print(f"  Cost limit:        ${agent_cfg.get('cost_limit', 5.0):.2f}")
    print(f"  Context window:    {ctx.get('context_window', 65536)}")
    print(f"  Max tokens:        {agent_llm.get('max_tokens', 4096)}")
    print(f"  Keep recent msgs:  {ctx.get('keep_recent_messages', 6)}")
    print(f"  Truncate thresh:   {ctx.get('truncate_threshold', 0.85)}")

    # ── 10. Observability ────────────────────────────────────────────────
    obs_cfg = config.get("observability", {})
    if args.observe or obs_cfg.get("enabled", False):
        project = obs_cfg.get("project_name", "agent-swe-ace")
        print(f"\nObservability:")
        print(f"  Project:           {project}_{output_dir.name}")

    # ── 11. Instance IDs ─────────────────────────────────────────────────
    if not iterate_repos_list:
        print(f"\nTrain instances ({len(train_instances)}):")
        for inst in train_instances[:15]:
            print(f"  - {inst['instance_id']}")
        if len(train_instances) > 15:
            print(f"  ... and {len(train_instances) - 15} more")
        if val_instances:
            print(f"\nVal instances ({len(val_instances)}):")
            for inst in val_instances[:15]:
                print(f"  - {inst['instance_id']}")
            if len(val_instances) > 15:
                print(f"  ... and {len(val_instances) - 15} more")

    print("\n=== END DRY RUN ===")

# ── iterate_repos orchestration ──────────────────────────────────────────


def _run_single_repo_experiment(
    repo: str,
    repo_instances: list,
    config: dict,
    run_dir: Path,
    run_name: str,
    agent_config: LLMConfig,
    ace_config: LLMConfig,
    agent_factory,
    evaluate_phase: EvaluatePhase,
    reflector,
    skill_manager,
    baseline_run_dir: Path | None,
    train_trajs_dir: str | None = None,
) -> dict:
    """Run a complete two-phase experiment for a single repo.

    Returns the statistics dict from the experiment run.
    """
    from ace import SkillManager, Reflector as DefaultReflector

    # Split repo instances into train/val
    train_instances, val_instances = split_instances(repo_instances, config, repo=repo)
    logger.info(f"[{repo}] Split: {len(train_instances)} train, {len(val_instances)} val")

    benchmark = config["benchmark"]["dataset"].replace("/", "__")

    # Per-repo components (each repo needs its own agent + predict + learn)
    _retriever = _build_skill_retriever(config.get("experiment", {}))
    agent = agent_factory()
    predict_phase = PredictPhase(
        agent=agent, output_dir=run_dir, run_name=run_name,
        benchmark=benchmark, model_name=agent_config.model,
        skill_retriever=_retriever,
    )
    learn_phase = LearnPhase(
        reflector=reflector,
        skill_manager=skill_manager,
        output_dir=run_dir,
        run_name=run_name,
        benchmark=benchmark,
        skillbook_mode=config.get("experiment", {}).get("skillbook", {}).get("mode", "per_instance"),
        dedup_config=config.get("experiment", {}).get("skillbook", {}).get("deduplication"),
    )

    # Resume state for this repo's instances
    max_attempts = config["experiment"].get("max_attempts", 2)
    force_learn = config["experiment"].get("force_learn", True)
    resume_state = {}
    val_resume_state = {}
    cli_resume_dirs = None
    config_resume_dirs = config.get("experiment", {}).get("resume_dirs")
    # Resume not typically used with iterate_repos but support it
    resume_dirs = [Path(p) for p in config_resume_dirs] if config_resume_dirs else None
    skip_learn = config.get("experiment", {}).get("skip_learn", False)
    if resume_dirs:
        from data_io.resume_scanner import scan_resume_dirs
        # Scan train instances
        instance_ids = [i["instance_id"] for i in train_instances]
        resume_state = scan_resume_dirs(resume_dirs, benchmark, instance_ids, max_attempts, skip_learn=skip_learn)
        complete_ids = {iid for iid, rp in resume_state.items() if rp.is_fully_complete}
        before = len(train_instances)
        train_instances = [i for i in train_instances if i["instance_id"] not in complete_ids]
        logger.info(f"[{repo}] Resume: {len(complete_ids)} complete, {len(train_instances)} to process")

        # Scan val instances for val phase resume
        val_instances_list = val_instances if val_instances else []
        if val_instances_list:
            val_ids = [i["instance_id"] for i in val_instances_list]
            val_resume_state = scan_resume_dirs(resume_dirs, benchmark, val_ids, max_attempts, skip_learn=True)
            val_complete = sum(1 for rp in val_resume_state.values() if rp.is_fully_complete)
            logger.info(f"[{repo}] Val resume: {val_complete} complete")

        # Auto-derive baseline_run_dir from resume_dirs if not already set
        if not baseline_run_dir:
            baseline_run_dir = resume_dirs[0]

    # Within-repo concurrency: drives the val passes (train is always sequential).
    repo_concurrency = config.get("experiment", {}).get("concurrency", 1)

    loop = ExperimentLoop(
        predict_phase=predict_phase,
        evaluate_phase=evaluate_phase,
        learn_phase=learn_phase,
        output_dir=run_dir,
        run_name=run_name,
        max_attempts=max_attempts,
        force_learn=force_learn,
        skip_learn=skip_learn,
        skillbook_mode=config.get("experiment", {}).get("skillbook", {}).get("mode", "per_instance"),
        resume_state=resume_state,
        benchmark=benchmark,
        concurrency=repo_concurrency,
        agent_factory=agent_factory,
        val_resume_state=val_resume_state,
    )

    # Check for val_pass_k setting
    val_pass_k = config.get("experiment", {}).get("val_pass_k", 1)

    # Check for validation-only mode
    skillbook_source_dir = config.get("experiment", {}).get("skillbook_source_dir")

    if skillbook_source_dir:
        # Validation-only mode: skip training, load skillbook from source run
        from data_io.readers import load_skillbook_for_repo
        repo_skillbook = load_skillbook_for_repo(
            Path(skillbook_source_dir), benchmark, repo
        )
        logger.info(
            f"[{repo}] Validation-only mode: loaded {len(repo_skillbook.skills())} skills "
            f"from {skillbook_source_dir}, val_pass_k={val_pass_k}"
        )

        stats = loop.run(
            [], config,  # No train instances
            val_instances=val_instances if val_instances else None,
            baseline_run_dir=baseline_run_dir,
            preloaded_skillbook=repo_skillbook,
            val_pass_k=val_pass_k,
        )

        # Copy train_phase from source run so statistics show actual training data
        source_stats_path = Path(skillbook_source_dir) / "statistics.json"
        if source_stats_path.exists():
            with open(source_stats_path) as _f:
                _source_stats = json.load(_f)
            # For iterate_repos, try per-repo stats first
            if _source_stats.get("mode") == "iterate_repos":
                _repo_stats_path = (
                    Path(skillbook_source_dir) / "statistics_per_repo"
                    / (repo.replace("/", "__") + ".json")
                )
                if _repo_stats_path.exists():
                    with open(_repo_stats_path) as _f:
                        _source_stats = json.load(_f)
            _source_train = _source_stats.get("train_phase", {})
            if _source_train.get("total_instances", 0) > 0:
                stats["train_phase"] = _source_train
    else:
        # Normal two-phase: train then validate
        stats = loop.run(
            train_instances, config,
            val_instances=val_instances if val_instances else None,
            baseline_run_dir=baseline_run_dir,
            val_pass_k=val_pass_k,
            train_trajs_dir=train_trajs_dir,
        )

    # Persist per-repo skillbook to disk (only after training)
    repo_sb = loop.repo_skillbooks.get(repo, loop.global_skillbook)
    if repo_sb and len(repo_sb.skills()) > 0:
        repo_sb_dir = run_dir / benchmark / "skillbooks" / "per_repo" / repo.replace("/", "__")
        repo_sb_dir.mkdir(parents=True, exist_ok=True)
        out_path = repo_sb_dir / "final_skillbook.json"
        skills_data = {
            "skill_count": len(repo_sb.skills()),
            "skills": {
                s.id: {
                    "id": s.id,
                    "section": s.section,
                    "content": s.content,
                    "justification": s.justification,
                    "evidence": s.evidence,
                }
                for s in repo_sb.skills()
            },
        }
        with open(out_path, "w") as f:
            json.dump(skills_data, f, indent=2)
        logger.debug(f"[{repo}] Saved per-repo skillbook ({skills_data['skill_count']} skills) to {out_path}")

    return stats


def _aggregate_iterate_stats(repo_stats: dict[str, dict], config: dict, run_dir: Path):
    """Aggregate per-repo statistics into combined statistics.json."""
    per_repo_dir = run_dir / "statistics_per_repo"
    per_repo_dir.mkdir(parents=True, exist_ok=True)

    # Write per-repo stats
    for repo, stats in repo_stats.items():
        repo_filename = repo.replace("/", "__") + ".json"
        save_statistics(statistics=stats, run_dir=per_repo_dir, filename=repo_filename)

    # Aggregate
    total_resolved = 0
    total_processed = 0
    total_instances = 0
    val_baseline_resolved = 0
    val_baseline_total = 0
    val_skillbook_resolved = 0
    val_skillbook_total = 0
    train_resolved = 0
    train_total = 0
    total_skills = 0

    for repo, stats in repo_stats.items():
        # Train phase
        tp = stats.get("train_phase", {})
        train_resolved += tp.get("resolved_count", 0)
        train_total += tp.get("total_instances", 0)
        total_skills += tp.get("total_skills_learned", 0)
        # Reused baseline
        reused = tp.get("reused_from_baseline", 0)
        train_fresh = tp.get("freshly_run", 0)

        # Val baseline
        vbp = stats.get("val_baseline_phase", {})
        val_baseline_resolved += vbp.get("resolved_count", 0)
        val_baseline_total += vbp.get("total_instances", 0)

        # Val skillbook
        vsp = stats.get("val_skillbook_phase", {})
        val_skillbook_resolved += vsp.get("resolved_count", 0)
        val_skillbook_total += vsp.get("total_instances", 0)

        # Overall
        total_resolved += stats.get("resolved_count", 0)
        total_processed += stats.get("processed_instances", 0)
        total_instances += stats.get("total_instances", 0)

    # Determine overall status
    all_completed = all(
        s.get("status") == "completed" for s in repo_stats.values()
    )

    vb_rate = val_baseline_resolved / val_baseline_total if val_baseline_total else 0.0
    vs_rate = val_skillbook_resolved / val_skillbook_total if val_skillbook_total else 0.0
    improvement = vs_rate - vb_rate

    # Per-repo improvements
    repo_improvements = {}
    for repo, stats in repo_stats.items():
        summary = stats.get("summary", {})
        imp_str = summary.get("skillbook_improvement", "N/A")
        repo_improvements[repo] = {
            "train_resolved": stats.get("train_phase", {}).get("resolved_count", 0),
            "train_total": stats.get("train_phase", {}).get("total_instances", 0),
            "val_baseline_rate": summary.get("val_baseline_resolution_rate", 0),
            "val_skillbook_rate": summary.get("val_skillbook_resolution_rate", 0),
            "improvement": imp_str,
        }

    combined = {
        "status": "completed" if all_completed else "partial",
        "mode": "iterate_repos",
        "run_name": config.get("experiment", {}).get("name", ""),
        "repos": list(repo_stats.keys()),
        "start_time": min(s.get("start_time", "") for s in repo_stats.values()),
        "end_time": max(s.get("end_time", "") for s in repo_stats.values() if s.get("end_time")),
        "total_instances": total_instances,
        "processed_instances": total_processed,
        "resolved_count": total_resolved,
        "resolution_rate": total_resolved / total_processed if total_processed else 0.0,
        "train_phase": {
            "total_instances": train_total,
            "resolved_count": train_resolved,
            "resolution_rate": train_resolved / train_total if train_total else 0.0,
            "total_skills_learned": total_skills,
        },
        "val_baseline_phase": {
            "total_instances": val_baseline_total,
            "resolved_count": val_baseline_resolved,
            "resolution_rate": vb_rate,
        },
        "val_skillbook_phase": {
            "total_instances": val_skillbook_total,
            "resolved_count": val_skillbook_resolved,
            "resolution_rate": vs_rate,
        },
        "summary": {
            "total_repos": len(repo_stats),
            "completed_repos": sum(1 for s in repo_stats.values() if s.get("status") == "completed"),
            "train_resolution_rate": train_resolved / train_total if train_total else 0.0,
            "val_baseline_resolution_rate": vb_rate,
            "val_skillbook_resolution_rate": vs_rate,
            "skillbook_improvement": f"{improvement:+.3f}",
            "skillbook_improvement_pct": f"{(improvement / vb_rate * 100) if vb_rate > 0 else 0:+.1f}%",
            "per_repo": repo_improvements,
        },
    }

    # Save per-repo skillbook statistics
    benchmark = config["benchmark"]["dataset"].replace("/", "__")
    skillbooks_dir = run_dir / benchmark / "skillbooks"
    per_repo_sb_dir = skillbooks_dir / "per_repo"
    if per_repo_sb_dir.exists():
        from utils.token_estimation import estimate_skillbook_injected_tokens

        sb_stats = {}
        for repo_dir in sorted(per_repo_sb_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            sb_file = repo_dir / "final_skillbook.json"
            if sb_file.exists():
                try:
                    with open(sb_file) as f:
                        sb_data = json.load(f)
                    repo_name = repo_dir.name
                    skills = sb_data.get("skills", {})
                    skill_list = list(skills.values())
                    skill_count = len(skill_list)

                    # Group by section
                    sections = {}
                    for s in skill_list:
                        sec = s.get("section", "unknown")
                        sections[sec] = sections.get(sec, 0) + 1

                    # Field population
                    justification_count = sum(1 for s in skill_list if s.get("justification"))
                    evidence_count = sum(1 for s in skill_list if s.get("evidence"))

                    # SWE type prefixes
                    type_prefixes = {}
                    for s in skill_list:
                        content = s.get("content", "")
                        for prefix in ("AVOID:", "VERIFIED:", "CONSIDER:"):
                            if content.startswith(prefix):
                                key = prefix.rstrip(":")
                                type_prefixes[key] = type_prefixes.get(key, 0) + 1
                                break
                    type_prefixes_pct = {
                        k: round(v / skill_count * 100, 1) for k, v in type_prefixes.items()
                    } if skill_count else {}

                    # Estimate injected token count
                    injected_tokens = estimate_skillbook_injected_tokens(skill_list)

                    # Train instances from per-repo stats
                    repo_key = repo_name.replace("__", "/")
                    train_total = repo_stats.get(repo_key, {}).get("train_phase", {}).get("total_instances", 0)

                    repo_stat = {
                        "skill_count": skill_count,
                        "train_instances": train_total,
                        "skills_per_train_instance": round(skill_count / train_total, 1) if train_total else 0,
                        "injected_tokens": injected_tokens,
                        "sections": dict(sorted(sections.items(), key=lambda x: -x[1])),
                        "field_population": {
                            "justification_count": justification_count,
                            "justification_pct": round(justification_count / skill_count * 100, 1) if skill_count else 0,
                            "evidence_count": evidence_count,
                            "evidence_pct": round(evidence_count / skill_count * 100, 1) if skill_count else 0,
                        },
                    }
                    if type_prefixes:
                        repo_stat["type_prefixes"] = dict(sorted(type_prefixes.items(), key=lambda x: -x[1]))
                        repo_stat["type_prefixes_pct"] = dict(sorted(type_prefixes_pct.items(), key=lambda x: -x[1]))
                    sb_stats[repo_name] = repo_stat
                except Exception:
                    pass
        if sb_stats:
            total_skills = sum(v["skill_count"] for v in sb_stats.values())
            total_train = sum(v["train_instances"] for v in sb_stats.values())
            total_tokens = sum(v["injected_tokens"] for v in sb_stats.values())
            sb_stats["_summary"] = {
                "total_skills": total_skills,
                "total_train_instances": total_train,
                "total_injected_tokens": total_tokens,
                "repos": len(sb_stats),
                "avg_skills_per_repo": round(total_skills / len(sb_stats), 1),
                "avg_skills_per_train_instance": round(total_skills / total_train, 1) if total_train else 0,
                "avg_injected_tokens_per_repo": round(total_tokens / len(sb_stats)),
            }
            with open(skillbooks_dir / "skillbooks_statistics.json", "w") as f:
                json.dump(sb_stats, f, indent=2)

    save_statistics(statistics=combined, run_dir=run_dir)
    return combined


def _resolve_iterate_repos_concurrency(exp_cfg: dict) -> int:
    """Between-repo concurrency for iterate_repos.

    Reads experiment.iterate_repos_concurrency. When unset, falls back to the
    legacy top-level experiment.concurrency (only if > 1) with a deprecation
    warning, so existing configs keep working until migrated.
    """
    if "iterate_repos_concurrency" in exp_cfg:
        return int(exp_cfg["iterate_repos_concurrency"])
    legacy = exp_cfg.get("concurrency", 1)
    if legacy and legacy > 1:
        logger.warning(
            "experiment.iterate_repos_concurrency is unset; falling back to "
            "experiment.concurrency=%s (deprecated). Set "
            "experiment.iterate_repos_concurrency explicitly.",
            legacy,
        )
        return int(legacy)
    return 1


def _run_iterate_repos(config: dict, args, output_dir: Path):
    """Run independent per-repo two-phase experiments for each repo in iterate_repos."""
    iterate_repos_list = config["benchmark"]["iterate_repos"]
    run_name = config["experiment"].get("name", "experiment")
    concurrency = _resolve_iterate_repos_concurrency(config["experiment"])

    # Load all instances (with exclude_instances, but no filter_repos)
    instances = _get_instances_no_filter(config)
    logger.info(f"Loaded {len(instances)} instances for iterate_repos mode")

    # Group by repo
    repo_groups = defaultdict(list)
    for inst in instances:
        repo = inst.get("repo", "unknown")
        repo_groups[repo].append(inst)

    # Validate repos
    repos_to_run = []
    for repo in iterate_repos_list:
        if repo not in repo_groups:
            logger.warning(f"Repo '{repo}' not found in dataset, skipping")
            continue
        repos_to_run.append(repo)
    logger.info(f"Running {len(repos_to_run)} repos: {repos_to_run}")

    if not repos_to_run:
        logger.error("No valid repos to run")
        return

    # Create shared components
    agent_config = LLMConfig.from_dict(config["llm"]["agent"])
    ace_config = LLMConfig.from_dict(config["llm"]["ace"])
    agent_factory = _make_agent_factory(config, agent_config, output_dir)

    benchmark = config["benchmark"]["dataset"].replace("/", "__")
    evaluate_phase = EvaluatePhase(
        use_docker=config.get("evaluation", {}).get("use_docker", True),
        timeout=config.get("evaluation", {}).get("timeout", 1800),
        rm_image=config.get("evaluation", {}).get("rm_image", True),
        output_dir=output_dir,
        run_name=run_name,
        benchmark=benchmark,
        namespace=config.get("environment", {}).get("namespace"),
    )

    # ACE reflector + skill manager (shared, stateless per call)
    ace_model = create_ace_client(ace_config.to_dict())
    ace_settings = create_model_settings(ace_config.to_dict())
    from ace import SkillManager, Reflector as DefaultReflector
    from pydantic_ai.settings import ModelSettings

    custom_swe_learn = config.get("experiment", {}).get("skillbook", {}).get("custom_swe_learn", False)
    if custom_swe_learn:
        from prompts import SWEReflector, SWESkillManager
        reflector = SWEReflector(ace_model, api_base=ace_config.api_base, api_key=ace_config.api_key, model_settings=ace_settings)
        skill_manager = SWESkillManager(ace_model, api_base=ace_config.api_base, api_key=ace_config.api_key, model_settings=ace_settings)
    else:
        if ace_config.api_base:
            os.environ["OPENAI_BASE_URL"] = ace_config.api_base
            os.environ["OPENAI_API_KEY"] = ace_config.api_key
            os.environ["OPENAI_MAX_RETRIES"] = os.getenv("ACE_LEARN_MAX_RETRIES", "50")
            default_model = f"openai:{ace_config.model}"
        else:
            default_model = ace_model
        ace_model_settings = ModelSettings(**ace_settings)
        reflector = DefaultReflector(default_model, model_settings=ace_model_settings)
        skill_manager = SkillManager(default_model, model_settings=ace_model_settings)

    # Resolve baseline_run_dir
    baseline_run_dir = getattr(args, 'baseline_run_dir', None)
    if not baseline_run_dir:
        config_baseline = config.get("experiment", {}).get("baseline_run_dir")
        if config_baseline:
            baseline_run_dir = Path(config_baseline)

    # Resolve train_trajs_dir
    train_trajs_dir = config.get("experiment", {}).get("train_trajs_dir")

    # Run per-repo experiments
    repo_stats = {}

    if concurrency > 1 and len(repos_to_run) > 1:
        effective_workers = min(concurrency, len(repos_to_run))
        logger.info(f"Running {len(repos_to_run)} repos in parallel (workers={effective_workers})")

        def _run_repo(repo):
            stats = _run_single_repo_experiment(
                repo=repo,
                repo_instances=repo_groups[repo],
                config=config,
                run_dir=output_dir,
                run_name=run_name,
                agent_config=agent_config,
                ace_config=ace_config,
                agent_factory=agent_factory,
                evaluate_phase=evaluate_phase,
                reflector=reflector,
                skill_manager=skill_manager,
                baseline_run_dir=baseline_run_dir,
                train_trajs_dir=train_trajs_dir,
            )
            return repo, stats

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_run_repo, repo): repo for repo in repos_to_run}
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    repo_name, stats = future.result()
                    repo_stats[repo_name] = stats
                    logger.info(f"[{repo_name}] completed — "
                                f"resolved {stats.get('resolved_count', '?')}/{stats.get('total_instances', '?')}")
                except Exception as e:
                    logger.error(f"[{repo}] failed: {e}")
                    repo_stats[repo] = {"status": "error", "error": str(e)}
    else:
        for i, repo in enumerate(repos_to_run):
            logger.info(f"\n{'='*60}")
            logger.info(f"Repo {i+1}/{len(repos_to_run)}: {repo} ({len(repo_groups[repo])} instances)")
            logger.info(f"{'='*60}")
            try:
                stats = _run_single_repo_experiment(
                    repo=repo,
                    repo_instances=repo_groups[repo],
                    config=config,
                    run_dir=output_dir,
                    run_name=run_name,
                    agent_config=agent_config,
                    ace_config=ace_config,
                    agent_factory=agent_factory,
                    evaluate_phase=evaluate_phase,
                    reflector=reflector,
                    skill_manager=skill_manager,
                    baseline_run_dir=baseline_run_dir,
                    train_trajs_dir=train_trajs_dir,
                )
                repo_stats[repo] = stats
                logger.info(f"[{repo}] completed — "
                            f"resolved {stats.get('resolved_count', '?')}/{stats.get('total_instances', '?')}")
            except Exception as e:
                logger.error(f"[{repo}] failed: {e}")
                repo_stats[repo] = {"status": "error", "error": str(e)}

    # Aggregate and write combined statistics
    combined = _aggregate_iterate_stats(repo_stats, config, output_dir)

    # Print summary
    summary = combined.get("summary", {})
    logger.info(f"\n{'='*60}")
    logger.info(f"iterate_repos Complete! ({summary.get('completed_repos', '?')}/{summary.get('total_repos', '?')} repos)")
    logger.info(f"Train: {combined['train_phase']['resolved_count']}/{combined['train_phase']['total_instances']} "
                f"({combined['train_phase']['resolution_rate']:.1%})")
    logger.info(f"Val baseline: {combined['val_baseline_phase']['resolution_rate']:.1%}")
    logger.info(f"Val skillbook: {combined['val_skillbook_phase']['resolution_rate']:.1%}")
    logger.info(f"Improvement: {summary.get('skillbook_improvement', 'N/A')}")
    for repo, imp in summary.get("per_repo", {}).items():
        logger.info(f"  {repo}: {imp.get('improvement', 'N/A')}")
    logger.info(f"{'='*60}")


def _get_instances_no_filter(config: dict) -> list:
    """Load instances with max_instances and exclude_instances but without filter_repos."""
    logger.info(f"Loading dataset: {config['benchmark']['dataset']}")
    dataset = load_dataset(
        config["benchmark"]["dataset"],
        split=config["benchmark"].get("split", "test"),
    )
    instances = list(dataset)

    max_instances = config["benchmark"].get("max_instances")
    if max_instances:
        instances = instances[:max_instances]

    exclude = config["benchmark"].get("exclude_instances", [])
    if exclude:
        exclude_set = set(exclude)
        instances = [i for i in instances if i["instance_id"] not in exclude_set]

    return instances


def run_full_experiment(config: dict, args):
    """Run full experiment loop."""
    # Setup run name and output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config["experiment"].get("name", "experiment")
    base_dir = Path(config.get("output", {}).get("dir", "data"))
    output_dir = get_run_dir(base_dir, timestamp)

    # Dry run: show execution plan without side effects
    if args.dry_run:
        _run_dry_run(config, args, output_dir, run_name)
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup run-specific logging to experiment.log
    log_level = config.get("output", {}).get("log_level", "INFO")
    setup_logging(run_dir=output_dir, log_level=log_level)

    logger.info(f"Run: {run_name}")
    logger.info(f"Output: {output_dir}")

    # Persist CLI-only params into config so config.json is fully reproducible
    cli_resume_dirs = getattr(args, "resume_dir", None)
    if cli_resume_dirs:
        config.setdefault("experiment", {})["resume_dirs"] = [str(p) for p in cli_resume_dirs]
    cli_baseline_dir = getattr(args, "baseline_run_dir", None)
    if cli_baseline_dir:
        config.setdefault("experiment", {})["baseline_run_dir"] = str(cli_baseline_dir)

    # Save config
    save_config(config=config, run_dir=output_dir)

    # Check for iterate_repos mode
    iterate_repos = config.get("benchmark", {}).get("iterate_repos")
    if iterate_repos:
        _run_iterate_repos(config, args, output_dir)
        return

    # Enable observability with run_id as project name (if enabled)
    # This creates a unique Opik project per run for better traceability
    observability_config = config.get("observability", {})
    if args.observe or observability_config.get("enabled", False):
        # Use the run directory name (e.g., "run_20260321_143052") as the project name
        project_base_name = observability_config.get("project_name", "agent-swe-ace")
        run_id = output_dir.name
        enable_observability(project_name=f"{project_base_name}_{run_id}")

    # Create LLM configs
    agent_config = LLMConfig.from_dict(config["llm"]["agent"])
    ace_config = LLMConfig.from_dict(config["llm"]["ace"])

    # Create agent factory (each worker gets its own agent + model)
    agent_factory = _make_agent_factory(config, agent_config, output_dir)

    # Create a single agent for sequential mode / shared predict phase
    ace_model = create_ace_client(ace_config.to_dict())
    ace_settings = create_model_settings(ace_config.to_dict())

    agent = agent_factory()

    from ace import SkillManager, Reflector as DefaultReflector
    from pydantic_ai.settings import ModelSettings

    # Check config for custom SWE learning (reflector + skill manager)
    custom_swe_learn = config.get("experiment", {}).get("skillbook", {}).get("custom_swe_learn", False)
    if custom_swe_learn:
        from prompts import SWEReflector, SWESkillManager
        reflector = SWEReflector(ace_model, api_base=ace_config.api_base, api_key=ace_config.api_key, model_settings=ace_settings)
        skill_manager = SWESkillManager(ace_model, api_base=ace_config.api_base, api_key=ace_config.api_key, model_settings=ace_settings)
        logger.info("Using SWE-optimized Reflector and SkillManager")
    else:
        # Default ACE components use PydanticAI internally.
        # For any provider with api_base, set OPENAI_BASE_URL + OPENAI_API_KEY
        # and use "openai:" prefix so PydanticAI routes through OpenAIProvider
        # (which reads these env vars). Without the prefix, ACE's resolve_model
        # prepends "litellm:" and LiteLLM fails to recognise the provider.
        if ace_config.api_base:
            os.environ["OPENAI_BASE_URL"] = ace_config.api_base
            os.environ["OPENAI_API_KEY"] = ace_config.api_key
            os.environ["OPENAI_MAX_RETRIES"] = os.getenv("ACE_LEARN_MAX_RETRIES", "50")
            default_model = f"openai:{ace_config.model}"
        else:
            default_model = ace_model
        ace_model_settings = ModelSettings(**ace_settings)
        reflector = DefaultReflector(default_model, model_settings=ace_model_settings)
        skill_manager = SkillManager(default_model, model_settings=ace_model_settings)
        logger.info("Using default ACE Reflector")

    concurrency = config["experiment"].get("concurrency", 1)

    benchmark = config["benchmark"]["dataset"].replace("/", "__")
    _retriever = _build_skill_retriever(config.get("experiment", {}))
    predict_phase = PredictPhase(agent=agent, output_dir=output_dir, run_name=run_name, benchmark=benchmark, model_name=agent_config.model, skill_retriever=_retriever)
    evaluate_phase = EvaluatePhase(
        use_docker=config.get("evaluation", {}).get("use_docker", True),
        timeout=config.get("evaluation", {}).get("timeout", 1800),
        rm_image=config.get("evaluation", {}).get("rm_image", True),
        output_dir=output_dir,
        run_name=run_name,
        benchmark=benchmark,
        namespace=config.get("environment", {}).get("namespace"),
    )
    learn_phase = LearnPhase(
        reflector=reflector,
        skill_manager=skill_manager,
        output_dir=output_dir,
        run_name=run_name,
        benchmark=benchmark,
        skillbook_mode=config.get("experiment", {}).get("skillbook", {}).get("mode", "per_instance"),
        dedup_config=config.get("experiment", {}).get("skillbook", {}).get("deduplication"),
    )

    # Get instances
    instances = get_instances(config)

    # Filter to specific instance if requested
    if args.instance:
        instances = [i for i in instances if i.get("instance_id") == args.instance]
        if not instances:
            logger.error(f"Instance not found: {args.instance}")
            sys.exit(1)

    # Split into train/val if configured
    train_instances, val_instances = split_instances(instances, config)

    # Resume from previous runs (CLI --resume-dir overrides config)
    # Note: resume only applies to train instances
    cli_resume_dirs = getattr(args, 'resume_dir', None)
    config_resume_dirs = config.get("experiment", {}).get("resume_dirs")
    resume_dirs = cli_resume_dirs or ([Path(p) for p in config_resume_dirs] if config_resume_dirs else None)
    resume_state = {}
    val_resume_state = {}
    max_attempts = config["experiment"].get("max_attempts", 2)
    force_learn = config["experiment"].get("force_learn", True)
    skip_learn = config.get("experiment", {}).get("skip_learn", False)

    if resume_dirs:
        from data_io.resume_scanner import scan_resume_dirs
        instance_ids = [i["instance_id"] for i in train_instances]
        resume_state = scan_resume_dirs(resume_dirs, benchmark, instance_ids, max_attempts, skip_learn=skip_learn)

        # Filter out fully complete instances (they get copied, not re-run)
        complete_ids = {iid for iid, rp in resume_state.items() if rp.is_fully_complete}
        before = len(train_instances)
        train_instances = [i for i in train_instances if i["instance_id"] not in complete_ids]
        logger.info(
            f"Resume: {len(complete_ids)} complete (copied), "
            f"{before - len(complete_ids) - len(train_instances)} partial (continued), "
            f"{len(train_instances)} to process"
        )

        # Scan val instances for val phase resume
        if val_instances:
            val_ids = [i["instance_id"] for i in val_instances]
            val_resume_state = scan_resume_dirs(resume_dirs, benchmark, val_ids, max_attempts, skip_learn=True)
            val_complete = sum(1 for rp in val_resume_state.values() if rp.is_fully_complete)
            logger.info(f"Val resume: {val_complete} complete")

    # Run experiment
    loop = ExperimentLoop(
        predict_phase=predict_phase,
        evaluate_phase=evaluate_phase,
        learn_phase=learn_phase,
        output_dir=output_dir,
        run_name=run_name,
        max_attempts=max_attempts,
        force_learn=force_learn,
        skip_learn=skip_learn,
        skillbook_mode=config.get("experiment", {}).get("skillbook", {}).get("mode", "per_instance"),
        resume_state=resume_state,
        benchmark=benchmark,
        concurrency=concurrency,
        agent_factory=agent_factory,
        val_resume_state=val_resume_state,
    )

    # baseline_run_dir: CLI takes priority, then config, then auto-derived from resume_dirs
    baseline_run_dir = getattr(args, 'baseline_run_dir', None)
    if not baseline_run_dir:
        config_baseline = config.get("experiment", {}).get("baseline_run_dir")
        if config_baseline:
            baseline_run_dir = Path(config_baseline)
        elif resume_dirs:
            baseline_run_dir = resume_dirs[0]

    train_trajs_dir = config.get("experiment", {}).get("train_trajs_dir")
    val_pass_k = config.get("experiment", {}).get("val_pass_k", 1)

    # Validation-only mode: skip training, load skillbook from source run
    skillbook_source_dir = config.get("experiment", {}).get("skillbook_source_dir")

    if skillbook_source_dir:
        from data_io.readers import load_skillbook
        sb_path = Path(skillbook_source_dir) / benchmark / "skillbooks" / "final_skillbook.json"
        preloaded_skillbook = load_skillbook(sb_path)
        logger.info(
            f"Validation-only mode: loaded {len(preloaded_skillbook.skills())} skills "
            f"from {sb_path}, val_pass_k={val_pass_k}"
        )
        loop.run([], config,
                 val_instances=val_instances if val_instances else None,
                 baseline_run_dir=baseline_run_dir,
                 preloaded_skillbook=preloaded_skillbook,
                 val_pass_k=val_pass_k)
    else:
        loop.run(train_instances, config,
                 val_instances=val_instances if val_instances else None,
                 baseline_run_dir=baseline_run_dir,
                 train_trajs_dir=train_trajs_dir,
                 val_pass_k=val_pass_k)


def run_predict_cmd(config: dict, args):
    """Run predict phase only."""
    if not args.instance:
        logger.error("--instance required for predict phase")
        sys.exit(1)

    output_dir = Path(config.get("output", {}).get("dir", "data"))
    run_name = config["experiment"].get("name", "experiment")

    # Load instance (lightweight, for validation)
    instances = get_instances(config)
    instance = next((i for i in instances if i["instance_id"] == args.instance), None)
    if not instance:
        logger.error(f"Instance not found: {args.instance}")
        sys.exit(1)

    if args.dry_run:
        agent_llm_cfg = config.get("llm", {}).get("agent", {})
        agent_cfg = config.get("agent", {})
        ctx = agent_cfg.get("context", {})
        print(f"\n=== DRY RUN: predict ===")
        print(f"  Run name:          {run_name}")
        print(f"  Output:            {output_dir}")
        print(f"  Instance:          {args.instance}")
        print(f"  Iteration:         {args.iteration}")
        print(f"  Skillbook:         {args.skillbook or '(empty)'}")
        print(f"  Model:             {agent_llm_cfg.get('model', 'not set')}")
        print(f"  Step limit:        {agent_cfg.get('step_limit', 100)}")
        print(f"  Cost limit:        ${agent_cfg.get('cost_limit', 5.0):.2f}")
        print(f"  Docker:            {config.get('environment', {}).get('type') == 'docker'}")
        print(f"  Context window:    {ctx.get('context_window', 65536)}")
        print(f"  Max tokens:        {agent_llm_cfg.get('max_tokens', 4096)}")
        print(f"  Keep recent msgs:  {ctx.get('keep_recent_messages', 6)}")
        print(f"  Truncate thresh:   {ctx.get('truncate_threshold', 0.85)}")
        print(f"\n=== END DRY RUN ===")
        sys.exit(0)

    # Create agent
    agent_config = LLMConfig.from_dict(config["llm"]["agent"])
    agent_model = create_model(agent_config)
    agent = MiniSWEAgent(
        llm_model=agent_model,
        use_docker=config.get("environment", {}).get("type") == "docker",
        step_limit=config.get("agent", {}).get("step_limit", 100),
        cost_limit=config.get("agent", {}).get("cost_limit", 5.0),
        output_dir=output_dir,
        namespace=config.get("environment", {}).get("namespace"),
        context_management=config.get("agent", {}).get("context", {}).get("enabled", True),
        context_window=config.get("agent", {}).get("context", {}).get("context_window", 65536),
        max_tokens=config.get("llm", {}).get("agent", {}).get("max_tokens", 4096),
        keep_recent_messages=config.get("agent", {}).get("context", {}).get("keep_recent_messages", 6),
        truncate_threshold=config.get("agent", {}).get("context", {}).get("truncate_threshold", 0.85),
    )

    # Load skillbook
    skillbook = load_skillbook(args.skillbook)

    # Run predict
    benchmark = config["benchmark"]["dataset"].replace("/", "__")
    _retriever = _build_skill_retriever(config.get("experiment", {}))
    phase = PredictPhase(agent=agent, output_dir=output_dir, run_name=run_name, benchmark=benchmark, skill_retriever=_retriever)
    result = phase.run(instance=instance, skillbook=skillbook, iteration=args.iteration)

    print(f"\nPredict result:")
    print(f"  Exit status: {result.exit_status}")
    print(f"  Patch length: {len(result.patch)} chars")
    print(f"  Trajectory: {result.trajectory_path}")


def run_evaluate_cmd(config: dict, args):
    """Run evaluate phase only."""
    if not args.instance:
        logger.error("--instance required for evaluate phase")
        sys.exit(1)

    output_dir = Path(config.get("output", {}).get("dir", "data"))
    run_name = config["experiment"].get("name", "experiment")

    # Load instance (lightweight, for validation)
    instances = get_instances(config)
    instance = next((i for i in instances if i["instance_id"] == args.instance), None)
    if not instance:
        logger.error(f"Instance not found: {args.instance}")
        sys.exit(1)

    if args.dry_run:
        eval_cfg = config.get("evaluation", {})
        if args.patch:
            patch_source = "(from --patch)"
        elif args.trajectory:
            patch_source = f"(from --trajectory: {args.trajectory})"
        else:
            patch_source = "NOT SPECIFIED"
        print(f"\n=== DRY RUN: evaluate ===")
        print(f"  Run name:          {run_name}")
        print(f"  Output:            {output_dir}")
        print(f"  Instance:          {args.instance}")
        print(f"  Iteration:         {args.iteration}")
        print(f"  Patch:             {patch_source}")
        print(f"  Docker:            {eval_cfg.get('use_docker', True)}")
        print(f"  Timeout:           {eval_cfg.get('timeout', 1800)}s")
        print(f"  RM image:          {eval_cfg.get('rm_image', True)}")
        print(f"\n=== END DRY RUN ===")
        sys.exit(0)

    # Get patch
    if args.patch:
        patch = args.patch
    elif args.trajectory:
        traj = load_trajectory(args.trajectory)
        patch = traj.get("info", {}).get("submission", "")
    else:
        logger.error("--patch or --trajectory required for evaluate phase")
        sys.exit(1)

    # Run evaluate
    benchmark = config["benchmark"]["dataset"].replace("/", "__")
    phase = EvaluatePhase(
        use_docker=config.get("evaluation", {}).get("use_docker", True),
        timeout=config.get("evaluation", {}).get("timeout", 1800),
        rm_image=config.get("evaluation", {}).get("rm_image", True),
        output_dir=output_dir,
        run_name=run_name,
        benchmark=benchmark,
        namespace=config.get("environment", {}).get("namespace"),
    )
    result = phase.run(instance=instance, patch=patch, iteration=args.iteration)

    print(f"\nEvaluate result:")
    print(f"  Resolved: {result.resolved}")
    print(f"  Feedback: {result.feedback}")
    print(f"  Result: {result.result_path}")


def run_learn_cmd(config: dict, args):
    """Run learn phase only."""
    if not args.instance or not args.trajectory:
        logger.error("--instance and --trajectory required for learn phase")
        sys.exit(1)

    output_dir = Path(config.get("output", {}).get("dir", "data"))
    run_name = config["experiment"].get("name", "experiment")

    if args.dry_run:
        ace_cfg = config.get("llm", {}).get("ace", {})
        custom_swe = config.get("experiment", {}).get("skillbook", {}).get("custom_swe_learn", False)
        print(f"\n=== DRY RUN: learn ===")
        print(f"  Run name:          {run_name}")
        print(f"  Output:            {output_dir}")
        print(f"  Instance:          {args.instance}")
        print(f"  Trajectory:        {args.trajectory}")
        print(f"  Iteration:         {args.iteration}")
        print(f"  ACE model:         {ace_cfg.get('model', 'not set')}")
        print(f"  Custom SWE:        {custom_swe}")
        print(f"  Skillbook mode:    {config.get('experiment', {}).get('skillbook', {}).get('mode', 'per_instance')}")
        print(f"\n=== END DRY RUN ===")
        sys.exit(0)

    # Load trajectory
    trajectory = load_trajectory(args.trajectory)

    # Create ACE client
    ace_config = LLMConfig.from_dict(config["llm"]["ace"])
    ace_client = create_ace_client(ace_config.to_dict())
    ace_settings = create_model_settings(ace_config.to_dict())

    from ace import SkillManager, Skillbook, Reflector as DefaultReflector
    from pydantic_ai.settings import ModelSettings

    # Check config for custom SWE learning (reflector + skill manager)
    custom_swe_learn = config.get("experiment", {}).get("skillbook", {}).get("custom_swe_learn", False)
    if custom_swe_learn:
        from prompts import SWEReflector, SWESkillManager
        reflector = SWEReflector(ace_client, api_base=ace_config.api_base, api_key=ace_config.api_key, model_settings=ace_settings)
        skill_manager = SWESkillManager(ace_client, api_base=ace_config.api_base, api_key=ace_config.api_key, model_settings=ace_settings)
        logger.info("Using SWE-optimized Reflector and SkillManager")
    else:
        # Same as run_full_experiment: use "openai:" prefix for any provider
        # with api_base so PydanticAI uses OpenAIProvider (reads env vars).
        if ace_config.api_base:
            os.environ["OPENAI_BASE_URL"] = ace_config.api_base
            os.environ["OPENAI_API_KEY"] = ace_config.api_key
            os.environ["OPENAI_MAX_RETRIES"] = os.getenv("ACE_LEARN_MAX_RETRIES", "50")
            default_model = f"openai:{ace_config.model}"
        else:
            default_model = ace_client
        ace_model_settings = ModelSettings(**ace_settings)
        reflector = DefaultReflector(default_model, model_settings=ace_model_settings)
        skill_manager = SkillManager(default_model, model_settings=ace_model_settings)
        logger.info("Using default ACE Reflector")

    # Run learn
    benchmark = config["benchmark"]["dataset"].replace("/", "__")
    phase = LearnPhase(
        reflector=reflector,
        skill_manager=skill_manager,
        output_dir=output_dir,
        run_name=run_name,
        benchmark=benchmark,
    )

    skillbook = Skillbook()
    result = phase.run(
        skillbook=skillbook,
        instance={"instance_id": args.instance},
        trajectory=trajectory,
        patch=trajectory.get("info", {}).get("submission", ""),
        iteration=args.iteration,
    )

    print(f"\nLearn result:")
    print(f"  Skills added: {result.skills_added}")
    print(f"  Skills updated: {result.skills_updated}")
    print(f"  Skillbook: {result.skillbook_path}")


if __name__ == "__main__":
    main()
