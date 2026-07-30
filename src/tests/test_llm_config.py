# src/tests/test_llm_config.py
"""Tests for LLM configuration.

Tests that config.yaml settings work for both agent and ace LLMs.
Some tests make real API calls to verify connectivity.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import litellm
import pytest
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

litellm.suppress_debug_info = True
litellm.request_timeout = 10  # Fail fast if server unreachable


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


class TestLLMConfigErrorPaths:
    """Tests for LLMConfig validation error paths."""

    def test_zai_without_api_key_raises(self):
        """Z.AI provider without api_key should raise ValueError."""
        from config.llm import LLMConfig

        env = os.environ.copy()
        env.pop("ZAI_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Z.AI API key not found"):
                LLMConfig(provider="zai")

    def test_vllm_without_api_base_raises(self):
        """hosted_vllm provider without api_base should raise ValueError."""
        from config.llm import LLMConfig

        with pytest.raises(ValueError, match="vLLM requires api_base"):
            LLMConfig(provider="hosted_vllm", api_base=None, api_key="test")

    def test_zai_with_api_key_succeeds(self):
        """Z.AI provider with explicit api_key should succeed."""
        from config.llm import LLMConfig

        cfg = LLMConfig(provider="zai", model="glm-4.5-flash", api_key="test-key")
        assert cfg.api_key == "test-key"

    def test_vllm_with_api_base_succeeds(self):
        """hosted_vllm provider with api_base and no api_key should succeed."""
        from config.llm import LLMConfig

        env = os.environ.copy()
        env.pop("HOSTED_VLLM_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig(provider="hosted_vllm", api_base="http://localhost:8000/v1")
            assert cfg.api_base == "http://localhost:8000/v1"


class TestLLMConfigHelpers:
    """Tests for LLMConfig.from_dict and get_model_string helpers."""

    def test_from_dict_zai_defaults(self):
        """from_dict with zai provider should set correct defaults."""
        from config.llm import LLMConfig

        cfg = LLMConfig.from_dict({"provider": "zai", "api_key": "test"})
        assert cfg.model == "glm-4.5-flash"
        assert cfg.api_key_env == "ZAI_API_KEY"
        assert cfg.api_base == "https://api.z.ai/api/paas/v4"

    def test_from_dict_vllm_defaults(self):
        """from_dict with hosted_vllm provider should set correct defaults."""
        from config.llm import LLMConfig

        cfg = LLMConfig.from_dict({"provider": "hosted_vllm", "api_base": "http://localhost:8000/v1"})
        assert cfg.model == "Qwen/Qwen3-Coder-30B-A3B"
        assert cfg.api_key_env == "HOSTED_VLLM_API_KEY"
        assert cfg.api_base == "http://localhost:8000/v1"

    def test_get_model_string_zai(self):
        """get_model_string for zai should return zai/<model>."""
        from config.llm import LLMConfig

        cfg = LLMConfig(provider="zai", model="glm-4.5-flash", api_key="test")
        assert cfg.get_model_string() == "zai/glm-4.5-flash"

    def test_get_model_string_vllm(self):
        """get_model_string for hosted_vllm should return hosted_vllm/<model>."""
        from config.llm import LLMConfig

        cfg = LLMConfig(provider="hosted_vllm", model="Qwen/test", api_base="http://localhost:8000/v1")
        assert cfg.get_model_string() == "hosted_vllm/Qwen/test"
