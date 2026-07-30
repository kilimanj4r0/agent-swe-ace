"""Tests for MiniSWEAgent.run() — import guard, model reset, context management branching,
patch extraction, and error handling."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import litellm

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.miniswe_agent import AgentResult, MiniSWEAgent


def test_litellm_imports_in_cold_process():
    completed = subprocess.run(
        [sys.executable, "-c", "import litellm; print('ok')"],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


class TestMiniSWEAgentImportError:
    """When minisweagent is not installed, run() should return an error result immediately."""

    @patch.dict(
        "sys.modules",
        {
            "minisweagent": None,
            "minisweagent.agents": None,
            "minisweagent.agents.default": None,
        },
    )
    def test_import_error_returns_error_result(self):
        model = Mock()
        agent = MiniSWEAgent(llm_model=model, use_docker=False)
        result = agent.run(problem="Fix the bug", instance={"instance_id": "test-1"})

        assert isinstance(result, AgentResult)
        assert result.exit_status == "error"
        assert result.patch == ""
        assert result.trajectory == []
        assert "not installed" in result.error
        assert result.error_kind == "infrastructure"


def _make_mock_agent(exit_status="submitted", run_result="patch-content", messages=None):
    """Create a mock agent with configurable run() return and messages."""
    mock = Mock()
    mock.run.return_value = (exit_status, run_result)
    mock.messages = messages if messages is not None else [{"role": "user", "content": "hello"}]
    return mock


def _install_minisweagent_mocks(mock_default_cls, mock_agent_config_cls):
    """Install a fake minisweagent.agents.default module so the local import in run() succeeds.

    Returns the fake module dict suitable for use with @patch.dict("sys.modules", ...).
    """
    fake_default = MagicMock()
    fake_default.DefaultAgent = mock_default_cls
    fake_default.AgentConfig = mock_agent_config_cls

    fake_agents = MagicMock()
    fake_agents.default = fake_default

    fake_pkg = MagicMock()
    fake_pkg.agents = fake_agents

    return {
        "minisweagent": fake_pkg,
        "minisweagent.agents": fake_agents,
        "minisweagent.agents.default": fake_default,
    }


class TestMiniSWEAgentRun:
    """Tests for the main execution path of MiniSWEAgent.run()."""

    def test_model_counter_reset(self):
        """Model n_calls and cost should be reset to 0 before each run."""
        model = Mock()
        model.n_calls = 42
        model.cost = 9.99
        mock_agent = _make_mock_agent()

        mock_default_cls = Mock(return_value=mock_agent)
        mock_config_cls = Mock(return_value="config-instance")

        miniswe_patches = _install_minisweagent_mocks(mock_default_cls, mock_config_cls)

        with patch.dict("sys.modules", miniswe_patches), \
             patch("agents.miniswe_agent.create_local_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.create_docker_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.build_system_template", return_value="sys template"), \
             patch("agents.miniswe_agent.build_instance_template", return_value="instance template"), \
             patch("agents.miniswe_agent.build_action_observation_template", return_value="action template"), \
             patch("agents.miniswe_agent.build_format_error_template", return_value="format error template"), \
             patch("agents.miniswe_agent.get_platform_info", return_value={"system": "Linux"}):

            agent = MiniSWEAgent(llm_model=model, use_docker=False, context_management=False)
            agent.run(problem="Fix", instance={"instance_id": "test-1"})

        assert model.n_calls == 0
        assert model.cost == 0.0

    def test_format_error_template_passed_to_agent_config(self):
        """AgentConfig must receive the richer format_error_template, not the bare default."""
        model = Mock()
        mock_agent = _make_mock_agent()
        mock_default_cls = Mock(return_value=mock_agent)
        mock_config_cls = Mock(return_value="config-instance")

        miniswe_patches = _install_minisweagent_mocks(mock_default_cls, mock_config_cls)

        with patch.dict("sys.modules", miniswe_patches), \
             patch("agents.miniswe_agent.create_local_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.create_docker_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.build_system_template", return_value="sys template"), \
             patch("agents.miniswe_agent.build_instance_template", return_value="instance template"), \
             patch("agents.miniswe_agent.build_action_observation_template", return_value="action template"), \
             patch("agents.miniswe_agent.build_format_error_template", return_value="format error template"), \
             patch("agents.miniswe_agent.get_platform_info", return_value={"system": "Linux"}):

            agent = MiniSWEAgent(llm_model=model, use_docker=False, context_management=False)
            agent.run(problem="Fix", instance={"instance_id": "test-1"})

        assert mock_config_cls.call_args.kwargs["format_error_template"] == "format error template"

    def test_context_management_enabled(self):
        """When context_management=True, ContextAwareDefaultAgent should be used."""
        model = Mock()
        mock_agent = _make_mock_agent()

        # DefaultAgent should NOT be called when context_management is True
        mock_default_cls = Mock(return_value=mock_agent)
        mock_ctx_agent_cls = Mock(return_value=mock_agent)
        mock_config_cls = Mock(return_value="config-instance")

        miniswe_patches = _install_minisweagent_mocks(mock_default_cls, mock_config_cls)

        # ContextAwareDefaultAgent is imported from agents.context_manager inside run()
        fake_ctx_module = MagicMock()
        fake_ctx_module.ContextAwareDefaultAgent = mock_ctx_agent_cls

        with patch.dict("sys.modules", {**miniswe_patches, "agents.context_manager": fake_ctx_module}), \
             patch("agents.miniswe_agent.create_local_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.create_docker_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.build_system_template", return_value="sys template"), \
             patch("agents.miniswe_agent.build_instance_template", return_value="instance template"), \
             patch("agents.miniswe_agent.build_action_observation_template", return_value="action template"), \
             patch("agents.miniswe_agent.build_format_error_template", return_value="format error template"), \
             patch("agents.miniswe_agent.get_platform_info", return_value={"system": "Linux"}):

            agent = MiniSWEAgent(llm_model=model, use_docker=False, context_management=True)
            result = agent.run(problem="Fix", instance={"instance_id": "test-1"})

        mock_ctx_agent_cls.assert_called_once()
        mock_default_cls.assert_not_called()
        assert isinstance(result, AgentResult)
        assert result.exit_status == "submitted"
        assert result.patch == "patch-content"

    def test_context_management_disabled(self):
        """When context_management=False, DefaultAgent should be used directly."""
        model = Mock()
        mock_agent = _make_mock_agent()
        mock_default_cls = Mock(return_value=mock_agent)
        mock_config_cls = Mock(return_value="config-instance")

        miniswe_patches = _install_minisweagent_mocks(mock_default_cls, mock_config_cls)

        with patch.dict("sys.modules", miniswe_patches), \
             patch("agents.miniswe_agent.create_local_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.create_docker_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.build_system_template", return_value="sys template"), \
             patch("agents.miniswe_agent.build_instance_template", return_value="instance template"), \
             patch("agents.miniswe_agent.build_action_observation_template", return_value="action template"), \
             patch("agents.miniswe_agent.build_format_error_template", return_value="format error template"), \
             patch("agents.miniswe_agent.get_platform_info", return_value={"system": "Linux"}):

            agent = MiniSWEAgent(llm_model=model, use_docker=False, context_management=False)
            result = agent.run(problem="Fix", instance={"instance_id": "test-1"})

        mock_default_cls.assert_called_once()
        assert isinstance(result, AgentResult)
        assert result.exit_status == "submitted"

    def test_exception_returns_partial_trajectory(self):
        """When agent.run() raises, the exception handler should return partial trajectory."""
        model = Mock()
        failing_agent = Mock()
        failing_agent.run.side_effect = RuntimeError("LLM connection lost")
        failing_agent.messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "partial"},
        ]

        mock_default_cls = Mock(return_value=failing_agent)
        mock_config_cls = Mock(return_value="config-instance")

        miniswe_patches = _install_minisweagent_mocks(mock_default_cls, mock_config_cls)

        with patch.dict("sys.modules", miniswe_patches), \
             patch("agents.miniswe_agent.create_local_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.create_docker_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.build_system_template", return_value="sys template"), \
             patch("agents.miniswe_agent.build_instance_template", return_value="instance template"), \
             patch("agents.miniswe_agent.build_action_observation_template", return_value="action template"), \
             patch("agents.miniswe_agent.build_format_error_template", return_value="format error template"), \
             patch("agents.miniswe_agent.get_platform_info", return_value={"system": "Linux"}):

            agent = MiniSWEAgent(llm_model=model, use_docker=False, context_management=False)
            result = agent.run(problem="Fix", instance={"instance_id": "test-1"})

        assert isinstance(result, AgentResult)
        assert result.exit_status == "error"
        assert result.patch == ""
        assert result.trajectory == failing_agent.messages
        assert "LLM connection lost" in result.error
        assert result.error_kind == "infrastructure"

    def test_context_window_exceeded_returns_dedicated_status(self):
        """ContextWindowExceededError should produce exit_status='ContextWindowExceeded'."""
        model = Mock()
        failing_agent = Mock()
        failing_agent.run.side_effect = litellm.ContextWindowExceededError(
            message="Context window exceeded", model="test", llm_provider="vllm"
        )
        failing_agent.messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "partial"},
        ]

        mock_default_cls = Mock(return_value=failing_agent)
        mock_config_cls = Mock(return_value="config-instance")

        miniswe_patches = _install_minisweagent_mocks(mock_default_cls, mock_config_cls)

        with patch.dict("sys.modules", miniswe_patches), \
             patch("agents.miniswe_agent.create_local_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.create_docker_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.build_system_template", return_value="sys template"), \
             patch("agents.miniswe_agent.build_instance_template", return_value="instance template"), \
             patch("agents.miniswe_agent.build_action_observation_template", return_value="action template"), \
             patch("agents.miniswe_agent.build_format_error_template", return_value="format error template"), \
             patch("agents.miniswe_agent.get_platform_info", return_value={"system": "Linux"}):

            agent = MiniSWEAgent(llm_model=model, use_docker=False, context_management=False)
            result = agent.run(problem="Fix", instance={"instance_id": "test-1"})

        assert isinstance(result, AgentResult)
        assert result.exit_status == "ContextWindowExceeded"
        assert result.patch == ""
        assert result.trajectory == failing_agent.messages
        assert "Context window exceeded" in result.error
        assert result.error_kind is None

    def test_string_result_extraction(self):
        """When agent.run() returns a string result, it should be used as the patch directly."""
        model = Mock()
        mock_agent = _make_mock_agent(exit_status="submitted", run_result="patch as plain string")

        mock_default_cls = Mock(return_value=mock_agent)
        mock_config_cls = Mock(return_value="config-instance")

        miniswe_patches = _install_minisweagent_mocks(mock_default_cls, mock_config_cls)

        with patch.dict("sys.modules", miniswe_patches), \
             patch("agents.miniswe_agent.create_local_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.create_docker_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.build_system_template", return_value="sys template"), \
             patch("agents.miniswe_agent.build_instance_template", return_value="instance template"), \
             patch("agents.miniswe_agent.build_action_observation_template", return_value="action template"), \
             patch("agents.miniswe_agent.build_format_error_template", return_value="format error template"), \
             patch("agents.miniswe_agent.get_platform_info", return_value={"system": "Linux"}):

            agent = MiniSWEAgent(llm_model=model, use_docker=False, context_management=False)
            result = agent.run(problem="Fix", instance={"instance_id": "test-1"})

        assert isinstance(result, AgentResult)
        assert result.patch == "patch as plain string"

    def test_dict_result_extraction(self):
        """When agent.run() returns a dict with 'submission', that value becomes the patch."""
        model = Mock()
        mock_agent = _make_mock_agent(
            exit_status="submitted",
            run_result={"submission": "diff content"},
        )

        mock_default_cls = Mock(return_value=mock_agent)
        mock_config_cls = Mock(return_value="config-instance")

        miniswe_patches = _install_minisweagent_mocks(mock_default_cls, mock_config_cls)

        with patch.dict("sys.modules", miniswe_patches), \
             patch("agents.miniswe_agent.create_local_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.create_docker_environment", return_value=Mock()), \
             patch("agents.miniswe_agent.build_system_template", return_value="sys template"), \
             patch("agents.miniswe_agent.build_instance_template", return_value="instance template"), \
             patch("agents.miniswe_agent.build_action_observation_template", return_value="action template"), \
             patch("agents.miniswe_agent.build_format_error_template", return_value="format error template"), \
             patch("agents.miniswe_agent.get_platform_info", return_value={"system": "Linux"}):

            agent = MiniSWEAgent(llm_model=model, use_docker=False, context_management=False)
            result = agent.run(problem="Fix", instance={"instance_id": "test-1"})

        assert isinstance(result, AgentResult)
        assert result.patch == "diff content"
