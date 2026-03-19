"""
Main Experiment Runner

Supports configuration via:
1. config.yaml file (default)
2. CLI arguments (override YAML)
3. Environment variables (.env file)

Usage:
    python run_experiment.py
    python run_experiment.py --config custom_config.yaml
    python run_experiment.py --max-instances 10 --agent-model glm-4-plus
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml
from datasets import load_dataset
from dotenv import load_dotenv

# Add src directory to path for imports (works from any directory)
_src_dir = Path(__file__).parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from config.llm import LLMConfig
from experiments.online_ace_runner import OnlineACEExperiment
from utils.llm_observer import enable_observability

# Load .env file
load_dotenv()


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "experiment.log"

    logger = logging.getLogger("experiment")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper()))
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Mapping of CLI arg names to config paths (section, key)
CLI_CONFIG_MAPPINGS = [
    ("max_instances", "benchmark", "max_instances"),
    ("max_attempts", "experiment", "max_attempts"),
    ("output", "output", "dir"),
    ("agent_provider", ("llm", "agent"), "provider"),
    ("agent_model", ("llm", "agent"), "model"),
    ("agent_api_base", ("llm", "agent"), "api_base"),
    ("ace_provider", ("llm", "ace"), "provider"),
    ("ace_model", ("llm", "ace"), "model"),
]


def merge_cli_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI arguments into config (CLI takes precedence)."""
    for arg_name, section, key in CLI_CONFIG_MAPPINGS:
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            if isinstance(section, tuple):
                # Nested section like ("llm", "agent")
                current = config
                for s in section[:-1]:
                    current = current.setdefault(s, {})
                current[section[-1]][key] = arg_value
            else:
                config.setdefault(section, {})[key] = arg_value

    # Handle --no-docker flag
    if getattr(args, 'no_docker', False):
        config.setdefault('environment', {})['type'] = 'local'
        config.setdefault('evaluation', {})['use_docker'] = False

    return config


def compute_statistics(results: List[dict]) -> dict:
    """Compute experiment statistics."""
    if not results:
        return {
            "total_instances": 0,
            "resolved": 0,
            "resolution_rate": 0,
            "avg_attempts": 0,
            "avg_attempts_resolved": 0,
            "avg_final_skills": 0,
            "resolved_instance_ids": [],
        }

    total = len(results)
    resolved_instances = [r for r in results if r["resolved"]]
    resolved = len(resolved_instances)

    total_attempts = sum(r["attempts"] for r in results)
    resolved_attempts = sum(r["attempts"] for r in resolved_instances)
    total_skills = sum(
        len(r["final_skillbook"].get("skills", {}))
        for r in results
    )

    return {
        "total_instances": total,
        "resolved": resolved,
        "resolution_rate": resolved / total if total > 0 else 0,
        "avg_attempts": total_attempts / total if total > 0 else 0,
        "avg_attempts_resolved": resolved_attempts / resolved if resolved > 0 else 0,
        "avg_final_skills": total_skills / total if total > 0 else 0,
        "resolved_instance_ids": [r["instance_id"] for r in resolved_instances],
    }


def run_experiment(
    config: Dict[str, Any],
    logger: logging.Logger,
    config_path: Optional[Path] = None
) -> tuple:
    """Run the experiment with given configuration using OnlineACE."""

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config['experiment'].get('name', 'experiment')
    output_dir_str = config['output']['dir']

    if config_path and not Path(output_dir_str).is_absolute():
        output_dir = config_path.parent / output_dir_str / f"{run_name}_{timestamp}"
    else:
        output_dir = Path(output_dir_str) / f"{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_level = config['output'].get('log_level', 'INFO')
    exp_logger = setup_logging(output_dir, log_level)

    exp_logger.info("=" * 60)
    exp_logger.info(f"Starting OnlineACE Experiment: {run_name}")
    exp_logger.info("=" * 60)

    # Save configuration
    config_save_path = output_dir / "config.json"
    safe_config = json.loads(json.dumps(config, default=str))
    with open(config_save_path, 'w') as f:
        json.dump(safe_config, f, indent=2)
    exp_logger.info(f"Configuration saved to: {config_save_path}")

    # Load dataset
    exp_logger.info(f"Loading dataset: {config['benchmark']['dataset']}")
    try:
        dataset = load_dataset(
            config['benchmark']['dataset'],
            split=config['benchmark']['split']
        )
        exp_logger.info(f"Loaded {len(dataset)} instances")
    except Exception as e:
        exp_logger.error(f"Failed to load dataset: {e}")
        raise

    # Convert dataset to list of instances
    instances = list(dataset)
    max_instances = config['benchmark'].get('max_instances')
    if max_instances:
        instances = instances[:max_instances]

    # Create LLM configs
    agent_config = LLMConfig(
        provider=config['llm']['agent'].get('provider', 'openai'),
        model=config['llm']['agent']['model'],
        api_base=config['llm']['agent'].get('api_base'),
        api_key_env=config['llm']['agent'].get('api_key_env', 'OPENAI_API_KEY'),
        temperature=config['llm']['agent'].get('temperature', 1.0),
        max_tokens=config['llm']['agent'].get('max_tokens', 4096),
    )

    ace_config = LLMConfig(
        provider=config['llm']['ace'].get('provider', 'openai'),
        model=config['llm']['ace']['model'],
        api_base=config['llm']['ace'].get('api_base'),
        api_key_env=config['llm']['ace'].get('api_key_env', 'OPENAI_API_KEY'),
        temperature=config['llm']['ace'].get('temperature', 1.0),
        max_tokens=config['llm']['ace'].get('max_tokens', 4096),
    )

    # Determine Docker settings
    use_docker = (
        config.get('environment', {}).get('type', 'docker') == 'docker'
        and config.get('evaluation', {}).get('use_docker', True)
    )
    exp_logger.info(f"Docker mode: {'enabled' if use_docker else 'disabled'}")

    # Determine observability settings
    observability_config = config.get('observability', {})
    enable_obs = observability_config.get('enabled', False)
    obs_project_name = observability_config.get('project_name', run_name)

    if enable_obs:
        enable_observability(project_name=obs_project_name)
        exp_logger.info(f"Observability enabled (Opik project: {obs_project_name})")

    # Create OnlineACE experiment
    experiment = OnlineACEExperiment(
        agent_config=agent_config,
        ace_config=ace_config,
        use_docker=use_docker,
        step_limit=config['agent'].get('step_limit', 100),
        cost_limit=config['agent'].get('cost_limit', 5.0),
        output_dir=output_dir,
        max_refinement_rounds=config['experiment'].get('max_attempts', 2) - 1,
        enable_observability=enable_obs,
    )

    # Run experiment
    exp_logger.info(f"Running OnlineACE on {len(instances)} instances...")
    results = experiment.run(instances)

    # Convert results for statistics
    # SampleResult has: sample, output (ACEStepContext), error, failed_at, cause
    # ACEStepContext has: agent_output, trace (contains ACEStepResult), etc.
    all_results = []
    for sample_result in results:
        sample = sample_result.sample
        instance_id = sample.metadata.get('instance_id', 'unknown')

        # Handle errors
        if sample_result.error:
            result = {
                "instance_id": instance_id,
                "resolved": False,
                "attempts": 1,
                "final_skillbook": {},
                "feedback": f"Error: {sample_result.error}",
                "metrics": {"resolved": 0.0, "error": 1.0},
            }
        elif sample_result.output and sample_result.output.trace:
            # Extract from trace (ACEStepResult)
            # trace is typically a list of ACEStepResult or similar
            trace = sample_result.output.trace
            # Get the last step result which should have environment_result
            if hasattr(trace, '__iter__') and not isinstance(trace, str):
                # Get last non-None environment_result from trace
                env_result = None
                for step in trace:
                    if hasattr(step, 'environment_result') and step.environment_result:
                        env_result = step.environment_result
                if env_result:
                    result = {
                        "instance_id": instance_id,
                        "resolved": env_result.metrics.get('resolved', 0) > 0.5,
                        "attempts": 1,
                        "final_skillbook": {},
                        "feedback": env_result.feedback,
                        "metrics": env_result.metrics,
                    }
                else:
                    result = {
                        "instance_id": instance_id,
                        "resolved": False,
                        "attempts": 1,
                        "final_skillbook": {},
                        "feedback": "No environment result in trace",
                        "metrics": {"resolved": 0.0},
                    }
            else:
                result = {
                    "instance_id": instance_id,
                    "resolved": False,
                    "attempts": 1,
                    "final_skillbook": {},
                    "feedback": f"Unexpected trace type: {type(trace)}",
                    "metrics": {"resolved": 0.0},
                }
        else:
            result = {
                "instance_id": instance_id,
                "resolved": False,
                "attempts": 1,
                "final_skillbook": {},
                "feedback": "No output or trace available",
                "metrics": {"resolved": 0.0},
            }

        all_results.append(result)

        # Save to jsonl
        results_path = output_dir / "results.jsonl"
        with open(results_path, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

    # Compute and save statistics
    stats = compute_statistics(all_results)
    stats_path = output_dir / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    exp_logger.info("\n" + "=" * 60)
    exp_logger.info("Experiment Complete")
    exp_logger.info("=" * 60)
    exp_logger.info(f"Total instances: {stats['total_instances']}")
    exp_logger.info(f"Resolved: {stats['resolved']} ({stats['resolution_rate']:.1%})")
    exp_logger.info(f"\nResults saved to: {output_dir}")

    return all_results, stats, output_dir


def resolve_config_path(args_config: Optional[str]) -> Path:
    """Resolve config file path from CLI args or auto-detect."""
    if args_config:
        return Path(args_config)

    script_dir = Path(__file__).parent
    candidates = [
        script_dir.parent.parent / "config.yaml",  # project root
        script_dir.parent / "config.yaml",          # src/
        Path("config.yaml"),                         # cwd
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return Path("config.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="Run mini-swe-agent with ACE skillbook learning"
    )

    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to configuration file (default: auto-detect config.yaml)"
    )
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument(
        "--max-instances", "-n",
        dest="max_instances",
        type=int,
        help="Max instances"
    )
    parser.add_argument(
        "--max-attempts", "-a",
        dest="max_attempts",
        type=int,
        help="Max attempts per instance"
    )
    parser.add_argument("--agent-provider", dest="agent_provider", choices=["openai", "vllm"])
    parser.add_argument("--agent-model", dest="agent_model", help="Agent model name")
    parser.add_argument("--agent-api-base", dest="agent_api_base", help="Agent API base URL")
    parser.add_argument("--ace-provider", dest="ace_provider", choices=["openai", "vllm"])
    parser.add_argument("--ace-model", dest="ace_model", help="ACE model name")
    parser.add_argument(
        "--no-docker",
        dest="no_docker",
        action="store_true",
        help="Disable Docker for agent and evaluation"
    )
    parser.add_argument(
        "--observe",
        dest="observe",
        action="store_true",
        help="Enable LLM observability with Opik (via ACE framework)"
    )

    args = parser.parse_args()

    config_path = resolve_config_path(args.config)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Please create a config.yaml file or specify --config path")
        sys.exit(1)

    config = load_config(str(config_path))
    config = merge_cli_config(config, args)

    # Enable observability via CLI flag
    if args.observe:
        config.setdefault('observability', {})['enabled'] = True

    logger = logging.getLogger(__name__)
    run_experiment(config, logger, config_path)


if __name__ == "__main__":
    main()
