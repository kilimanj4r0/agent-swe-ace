# src/cli/commands.py
"""CLI entry points for ACE-SWE experiment phases."""

import argparse
import json
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

load_dotenv()

import litellm


def _setup_console_logging(log_level: str = "INFO"):
    """Setup console-only logging (before run_dir is created)."""
    setup_logging(run_dir=None, log_level=log_level)


def apply_litellm_config(config: dict):
    """Apply LiteLLM settings from config."""
    litellm_settings = config.get("litellm", {})
    if litellm_settings.get("suppress_debug_info", False):
        litellm.suppress_debug_info = True


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

    return instances


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ACE-SWE Experiment: Skillbook learning with mini-swe-agent"
    )
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file")
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

    args = parser.parse_args()

    # Setup console logging (file logging added after run_dir is created)
    _setup_console_logging(args.log_level)

    # Load config
    config = load_config(args.config)

    # Apply LiteLLM settings
    apply_litellm_config(config)

    # Override config with CLI args
    if args.max_instances:
        config.setdefault("benchmark", {})["max_instances"] = args.max_instances
    if args.max_attempts:
        config.setdefault("experiment", {})["max_attempts"] = args.max_attempts
    if args.output:
        config.setdefault("output", {})["dir"] = args.output

    # Enable observability if --observe flag or config.observability.enabled
    observability_config = config.get("observability", {})
    if args.observe or observability_config.get("enabled", False):
        project_name = observability_config.get("project_name", "agent-swe-ace")
        enable_observability(project_name=project_name)

    # Run appropriate phase
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

    # Create LLM configs
    agent_config = LLMConfig.from_dict(config["llm"]["agent"])
    ace_config = LLMConfig.from_dict(config["llm"]["ace"])

    # Create components
    agent_model = create_model(agent_config)
    ace_client = create_ace_client(ace_config.to_dict())

    agent = MiniSWEAgent(
        llm_model=agent_model,
        use_docker=config.get("environment", {}).get("type") == "docker",
        step_limit=config.get("agent", {}).get("step_limit", 100),
        cost_limit=config.get("agent", {}).get("cost_limit", 5.0),
        output_dir=output_dir,
    )

    from ace_next import Reflector, SkillManager

    benchmark = config["benchmark"]["dataset"].replace("/", "__")
    predict_phase = PredictPhase(agent=agent, output_dir=output_dir, run_name=run_name, benchmark=benchmark)
    evaluate_phase = EvaluatePhase(
        use_docker=config.get("evaluation", {}).get("use_docker", True),
        timeout=config.get("evaluation", {}).get("timeout", 1800),
        output_dir=output_dir,
        run_name=run_name,
        benchmark=benchmark,
    )
    learn_phase = LearnPhase(
        reflector=Reflector(ace_client),
        skill_manager=SkillManager(ace_client),
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
    loop = ExperimentLoop(
        predict_phase=predict_phase,
        evaluate_phase=evaluate_phase,
        learn_phase=learn_phase,
        output_dir=output_dir,
        run_name=run_name,
        max_attempts=config["experiment"].get("max_attempts", 2),
        skillbook_mode=config["experiment"].get("skillbook_mode", "per_instance"),
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
        output_dir=output_dir,
        run_name=run_name,
        benchmark=benchmark,
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

    from ace_next import Reflector, SkillManager, Skillbook

    # Run learn
    benchmark = config["benchmark"]["dataset"].replace("/", "__")
    phase = LearnPhase(
        reflector=Reflector(ace_client),
        skill_manager=SkillManager(ace_client),
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
