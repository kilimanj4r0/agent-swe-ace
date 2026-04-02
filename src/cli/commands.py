# src/cli/commands.py
"""CLI entry points for ACE-SWE experiment phases."""

import argparse
import json
import os
import sys
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

from config.llm import LLMConfig, create_model, create_ace_client
from agents.miniswe_agent import MiniSWEAgent
from phases.predict import PredictPhase
from phases.evaluate import EvaluatePhase
from phases.learn import LearnPhase
from runners.main_loop import ExperimentLoop
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


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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

    return instances


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
        "--baseline-dir",
        type=Path,
        help="Path to baseline run directory with existing iter_0 results. "
        "Skips predict/evaluate for iter_0, loads existing data, and continues from iter_1.",
    )
    parser.add_argument(
        "--custom-swe-learn",
        action="store_true",
        help="Use SWE-optimized Reflector + SkillManager (extracts anti-patterns, preserves type prefixes).",
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
        config.setdefault("experiment", {})["custom_swe_learn"] = True

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


def run_full_experiment(config: dict, args):
    """Run full experiment loop."""
    # Setup run name and output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config["experiment"].get("name", "experiment")
    base_dir = Path(config.get("output", {}).get("dir", "data"))
    output_dir = get_run_dir(base_dir, timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup run-specific logging to experiment.log
    log_level = config.get("output", {}).get("log_level", "INFO")
    setup_logging(run_dir=output_dir, log_level=log_level)

    logger.info(f"Run: {run_name}")
    logger.info(f"Output: {output_dir}")

    # Save config
    save_config(config=config, run_dir=output_dir)

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

    # Create components
    agent_model = create_model(agent_config)
    ace_model = create_ace_client(ace_config.to_dict())

    agent = MiniSWEAgent(
        llm_model=agent_model,
        use_docker=config.get("environment", {}).get("type") == "docker",
        step_limit=config.get("agent", {}).get("step_limit", 100),
        cost_limit=config.get("agent", {}).get("cost_limit", 5.0),
        output_dir=output_dir,
        namespace=config.get("environment", {}).get("namespace"),
        context_management=config.get("agent", {}).get("context_management", True),
        context_window=config.get("agent", {}).get("context_window", 65536),
        max_tokens=config.get("llm", {}).get("agent", {}).get("max_tokens", 4096),
        keep_recent_messages=config.get("agent", {}).get("keep_recent_messages", 6),
        truncate_threshold=config.get("agent", {}).get("truncate_threshold", 0.85),
    )

    from ace import SkillManager, Reflector as DefaultReflector

    # Check config for custom SWE learning (reflector + skill manager)
    custom_swe_learn = config.get("experiment", {}).get("custom_swe_learn", False)
    if custom_swe_learn:
        from prompts import SWEReflector, SWESkillManager
        reflector = SWEReflector(ace_model, api_base=ace_config.api_base, api_key=ace_config.api_key)
        skill_manager = SWESkillManager(ace_model, api_base=ace_config.api_base, api_key=ace_config.api_key)
        logger.info("Using SWE-optimized Reflector and SkillManager")
    else:
        # Default ACE components use PydanticAI which creates AsyncOpenAI internally.
        # For hosted_vllm, we must: (1) set OPENAI_BASE_URL so AsyncOpenAI routes
        # to local vLLM (not OpenAI's API), and (2) use the bare model name without
        # the hosted_vllm/ prefix since vLLM doesn't recognize it.
        if ace_config.provider == "hosted_vllm" and ace_config.api_base:
            os.environ["OPENAI_BASE_URL"] = ace_config.api_base
            default_model = ace_config.model
        else:
            default_model = ace_model
        reflector = DefaultReflector(default_model)
        skill_manager = SkillManager(default_model)
        logger.info("Using default ACE Reflector")

    benchmark = config["benchmark"]["dataset"].replace("/", "__")
    predict_phase = PredictPhase(agent=agent, output_dir=output_dir, run_name=run_name, benchmark=benchmark, model_name=agent_config.model)
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
        skillbook_mode=config["experiment"].get("skillbook_mode", "per_instance"),
        dedup_config=config.get("deduplication"),
    )

    # Get instances
    instances = get_instances(config)

    # Filter to specific instance if requested
    if args.instance:
        instances = [i for i in instances if i.get("instance_id") == args.instance]
        if not instances:
            logger.error(f"Instance not found: {args.instance}")
            sys.exit(1)

    # Run experiment
    baseline_dir = getattr(args, 'baseline_dir', None)
    loop = ExperimentLoop(
        predict_phase=predict_phase,
        evaluate_phase=evaluate_phase,
        learn_phase=learn_phase,
        output_dir=output_dir,
        run_name=run_name,
        max_attempts=config["experiment"].get("max_attempts", 2),
        skillbook_mode=config["experiment"].get("skillbook_mode", "per_instance"),
        baseline_dir=baseline_dir,
        benchmark=benchmark,
    )

    loop.run(instances, config)


def run_predict_cmd(config: dict, args):
    """Run predict phase only."""
    if not args.instance:
        logger.error("--instance required for predict phase")
        sys.exit(1)

    # Setup
    output_dir = Path(config.get("output", {}).get("dir", "data"))
    run_name = config["experiment"].get("name", "experiment")

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
        context_management=config.get("agent", {}).get("context_management", True),
        context_window=config.get("agent", {}).get("context_window", 65536),
        max_tokens=config.get("llm", {}).get("agent", {}).get("max_tokens", 4096),
        keep_recent_messages=config.get("agent", {}).get("keep_recent_messages", 6),
        truncate_threshold=config.get("agent", {}).get("truncate_threshold", 0.85),
    )

    # Load instance
    instances = get_instances(config)
    instance = next((i for i in instances if i["instance_id"] == args.instance), None)
    if not instance:
        logger.error(f"Instance not found: {args.instance}")
        sys.exit(1)

    # Load skillbook
    skillbook = load_skillbook(args.skillbook)

    # Run predict
    benchmark = config["benchmark"]["dataset"].replace("/", "__")
    phase = PredictPhase(agent=agent, output_dir=output_dir, run_name=run_name, benchmark=benchmark)
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

    # Load instance
    instances = get_instances(config)
    instance = next((i for i in instances if i["instance_id"] == args.instance), None)
    if not instance:
        logger.error(f"Instance not found: {args.instance}")
        sys.exit(1)

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

    # Load trajectory
    trajectory = load_trajectory(args.trajectory)

    # Create ACE client
    ace_config = LLMConfig.from_dict(config["llm"]["ace"])
    ace_client = create_ace_client(ace_config.to_dict())

    from ace import SkillManager, Skillbook, Reflector as DefaultReflector

    # Check config for custom SWE learning (reflector + skill manager)
    custom_swe_learn = config.get("experiment", {}).get("custom_swe_learn", False)
    if custom_swe_learn:
        from prompts import SWEReflector, SWESkillManager
        reflector = SWEReflector(ace_client, api_base=ace_config.api_base, api_key=ace_config.api_key)
        skill_manager = SWESkillManager(ace_client, api_base=ace_config.api_base, api_key=ace_config.api_key)
        logger.info("Using SWE-optimized Reflector and SkillManager")
    else:
        # Same fix as run_full_experiment: set OPENAI_BASE_URL for PydanticAI,
        # use bare model name without hosted_vllm/ prefix for vLLM.
        if ace_config.provider == "hosted_vllm" and ace_config.api_base:
            os.environ["OPENAI_BASE_URL"] = ace_config.api_base
            default_model = ace_config.model
        else:
            default_model = ace_client
        reflector = DefaultReflector(default_model)
        skill_manager = SkillManager(default_model)
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
