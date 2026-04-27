"""MiniSWEAgent - Runs mini-swe-agent with skillbook injection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Any

from ace import Skillbook
from loguru import logger

from phases.predict import build_system_template, build_instance_template, build_action_observation_template
from environments.docker_env import create_docker_environment, create_local_environment
from utils.platform import get_platform_info


@dataclass
class AgentResult:
    """Result from running mini-swe-agent."""

    exit_status: str
    patch: str
    trajectory: List[dict]
    error: Optional[str] = None


class MiniSWEAgent:
    """
    Runs mini-swe-agent's DefaultAgent with skillbook injection.

    This class encapsulates the agent execution logic, handling:
    - Environment creation (Docker or local)
    - Template building with skillbook context
    - Platform info for Docker compatibility
    - Trajectory extraction
    """

    def __init__(
        self,
        llm_model,
        use_docker: bool = True,
        step_limit: int = 100,
        cost_limit: float = 5.0,
        output_dir: Optional[Path] = None,
        namespace: Optional[str] = None,
        context_management: bool = True,
        context_window: int = 65536,
        max_tokens: int = 4096,
        keep_recent_messages: int = 6,
        truncate_threshold: float = 0.85,
    ):
        """
        Args:
            llm_model: LitellmModel instance for mini-swe-agent
            use_docker: If True, use Docker environment
            step_limit: Maximum agent steps per attempt
            cost_limit: Maximum cost per attempt
            output_dir: Directory for agent-generated files (local mode)
            namespace: Optional Docker registry namespace prefix (e.g., "ghcr.io/epoch-research/")
            context_management: Enable proactive context window management
            context_window: Model's total context window in tokens
            max_tokens: Max output tokens reserved per LLM call
            keep_recent_messages: Number of recent messages to keep intact during truncation
            truncate_threshold: Fraction of max input tokens at which to start truncating
        """
        self.llm_model = llm_model
        self.use_docker = use_docker
        self.step_limit = step_limit
        self.cost_limit = cost_limit
        self.output_dir = output_dir or Path("results")
        self.namespace = namespace
        self.context_management = context_management
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.keep_recent_messages = keep_recent_messages
        self.truncate_threshold = truncate_threshold

    def run(
        self,
        problem: str,
        instance: dict,
        skillbook: Optional[Skillbook] = None,
    ) -> AgentResult:
        """
        Run mini-swe-agent on a problem.

        Args:
            problem: Problem statement to solve
            instance: SWE-bench instance dict with instance_id, repo, etc.
            skillbook: Optional skillbook for prompt injection

        Returns:
            AgentResult with exit_status, patch, trajectory, and error
        """
        try:
            from minisweagent.agents.default import DefaultAgent, AgentConfig
        except ImportError:
            return AgentResult(
                exit_status="error",
                patch="",
                trajectory=[],
                error="mini-swe-agent not installed",
            )

        try:
            instance_id = instance["instance_id"]
            logger.debug(f"Starting agent run for instance: {instance_id}")

            # Reset model counters (step_limit/cost_limit are tracked on the model object)
            # Without this, counters accumulate across iterations and cause immediate LimitsExceeded
            if hasattr(self.llm_model, 'n_calls'):
                self.llm_model.n_calls = 0
            if hasattr(self.llm_model, 'cost'):
                self.llm_model.cost = 0.0

            # Create environment
            logger.debug(f"Creating {'Docker' if self.use_docker else 'local'} environment...")
            if self.use_docker:
                env = create_docker_environment(instance, namespace=self.namespace)
            else:
                work_dir = self.output_dir / "agent_generated_files" / instance_id
                env = create_local_environment(work_dir)
            logger.debug(f"Environment created: {type(env).__name__}")

            # Build templates
            logger.debug("Building templates...")
            system_template = build_system_template()
            instance_template = build_instance_template(skillbook)
            action_observation_template = build_action_observation_template()
            logger.debug(f"Templates built, instance_template length: {len(instance_template)}")

            # Create agent config
            logger.debug("Creating agent config...")
            agent_config = AgentConfig(
                system_template=system_template,
                step_limit=self.step_limit,
                cost_limit=self.cost_limit,
                instance_template=instance_template,
                action_observation_template=action_observation_template,
            )

            # Create agent (with or without context window management)
            if self.context_management:
                from agents.context_manager import ContextAwareDefaultAgent

                max_input_tokens = self.context_window - self.max_tokens - 2000  # 2000 token safety buffer
                logger.debug(f"Creating ContextAwareDefaultAgent (max_input={max_input_tokens}, threshold={self.truncate_threshold})")
                agent = ContextAwareDefaultAgent(
                    model=self.llm_model,
                    env=env,
                    config_class=lambda: agent_config,
                    max_input_tokens=max_input_tokens,
                    keep_recent_messages=self.keep_recent_messages,
                    truncate_threshold=self.truncate_threshold,
                    max_tokens=self.max_tokens,
                )
                logger.debug("ContextAwareDefaultAgent created")
            else:
                logger.debug("Creating DefaultAgent (context management disabled)")
                agent = DefaultAgent(
                    model=self.llm_model,
                    env=env,
                    config_class=lambda: agent_config,
                )
                logger.debug("DefaultAgent created")

            # Pass platform info for Docker (DockerEnvironment doesn't provide these)
            platform_info = get_platform_info()
            logger.debug(f"Calling agent.run() with platform_info: {platform_info}")
            logger.debug(">>> FIRST LLM CALL WILL HAPPEN NOW <<<")
            exit_status, result = agent.run(problem, **platform_info)
            logger.debug(f"agent.run() completed with exit_status: {exit_status}")

            # Extract patch
            patch = ""
            if isinstance(result, dict):
                patch = result.get("submission", "") or result.get("patch", "")
            elif isinstance(result, str):
                patch = result

            # Extract trajectory
            trajectory = []
            if hasattr(agent, 'messages'):
                trajectory = agent.messages
            elif hasattr(agent, 'trajectory'):
                trajectory = agent.trajectory

            return AgentResult(
                exit_status=exit_status if isinstance(exit_status, str) else "completed",
                patch=patch,
                trajectory=trajectory,
                error=None,
            )

        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            # Extract partial trajectory even on failure (e.g. ContextWindowExceededError)
            trajectory = []
            if "agent" in dir():
                if hasattr(agent, "messages"):
                    trajectory = agent.messages
            return AgentResult(
                exit_status="error",
                patch="",
                trajectory=trajectory,
                error=str(e),
            )
