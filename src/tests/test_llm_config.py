# src/tests/test_llm_config.py
"""Tests for LLM configuration.

Tests that config.yaml settings work for both agent and ace LLMs.
Some tests make real API calls to verify connectivity.
"""

import sys
from pathlib import Path

import pytest
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import litellm

litellm.suppress_debug_info = True


@pytest.fixture
def config():
    """Load config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def agent_config(config):
    """Get agent LLM configuration."""
    return config["llm"]["agent"]


@pytest.fixture
def ace_config(config):
    """Get ACE LLM configuration."""
    return config["llm"]["ace"]


class TestConfigLoading:
    """Tests for configuration file loading."""

    def test_config_has_llm_section(self, config):
        """Test that config.yaml has llm section."""
        assert "llm" in config, "Missing 'llm' section in config.yaml"

    def test_config_has_agent_section(self, config):
        """Test that config.yaml has agent section."""
        assert "agent" in config["llm"], "Missing 'agent' section in llm config"

    def test_config_has_ace_section(self, config):
        """Test that config.yaml has ace section."""
        assert "ace" in config["llm"], "Missing 'ace' section in llm config"


class TestAgentModelCreation:
    """Tests for agent model creation from config."""

    def test_agent_llm_config_creation(self, agent_config):
        """Test that LLMConfig can be created from agent config."""
        from config.llm import LLMConfig

        llm_config = LLMConfig.from_dict(agent_config)
        assert llm_config is not None
        assert llm_config.get_model_string() is not None

    def test_agent_model_creation(self, agent_config):
        """Test that model can be created from agent config."""
        from config.llm import create_model_from_yaml

        model = create_model_from_yaml(agent_config)
        assert model is not None

    def test_agent_model_has_required_fields(self, agent_config):
        """Test that agent config has required fields."""
        assert "provider" in agent_config
        assert "model" in agent_config


class TestACEClientCreation:
    """Tests for ACE client creation from config."""

    def test_ace_llm_config_creation(self, ace_config):
        """Test that LLMConfig can be created from ACE config."""
        from config.llm import LLMConfig

        llm_config = LLMConfig.from_dict(ace_config)
        assert llm_config is not None
        assert llm_config.get_model_string() is not None

    def test_ace_client_creation(self, ace_config):
        """Test that client can be created from ACE config."""
        from config.llm import create_ace_client

        client = create_ace_client(ace_config)
        assert client is not None

    def test_ace_config_has_required_fields(self, ace_config):
        """Test that ACE config has required fields."""
        assert "provider" in ace_config
        assert "model" in ace_config


@pytest.mark.integration
class TestAgentLLMCall:
    """Tests for agent LLM with real API calls."""

    def test_agent_llm_returns_response(self, agent_config):
        """Test that agent LLM returns a non-empty response."""
        from config.llm import create_model_from_yaml

        model = create_model_from_yaml(agent_config)
        messages = [{"role": "user", "content": "Say exactly 'OK' and nothing else."}]

        response = model.query(messages)

        assert response is not None, "Agent LLM returned empty response"

    def test_agent_llm_response_has_content(self, agent_config):
        """Test that agent LLM response has content."""
        from config.llm import create_model_from_yaml

        model = create_model_from_yaml(agent_config)
        messages = [{"role": "user", "content": "Say exactly 'OK' and nothing else."}]

        response = model.query(messages)

        # Extract response content
        content = response.get("content", str(response)) if isinstance(response, dict) else str(response)
        assert len(content) > 0, "Agent LLM returned empty content"


@pytest.mark.integration
class TestACELLMCall:
    """Tests for ACE LLM with real API calls."""

    def test_ace_llm_returns_response(self, ace_config):
        """Test that ACE LLM returns a non-empty response."""
        from config.llm import create_model_from_yaml

        model = create_model_from_yaml(ace_config)
        messages = [{"role": "user", "content": "Say exactly 'OK' and nothing else."}]

        response = model.query(messages)

        assert response is not None, "ACE LLM returned empty response"

    def test_ace_llm_response_has_content(self, ace_config):
        """Test that ACE LLM response has content."""
        from config.llm import create_model_from_yaml

        model = create_model_from_yaml(ace_config)
        messages = [{"role": "user", "content": "Say exactly 'OK' and nothing else."}]

        response = model.query(messages)

        # Extract response content
        content = response.get("content", str(response)) if isinstance(response, dict) else str(response)
        assert len(content) > 0, "ACE LLM returned empty content"
