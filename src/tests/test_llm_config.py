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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from config.llm_catalog import get_effective_llm
from config.loader import load_experiment_config

load_dotenv()

litellm.suppress_debug_info = True
litellm.request_timeout = 10  # Fail fast if server unreachable


@pytest.fixture
def config():
    """Load and resolve config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    return load_experiment_config(config_path)


@pytest.fixture
def agent_config(config):
    """Get effective agent LLM configuration."""
    return get_effective_llm(config["llm"]["agent"])


@pytest.fixture
def ace_config(config):
    """Get effective ACE LLM configuration."""
    return get_effective_llm(config["llm"]["ace"])


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
                LLMConfig(
                    provider="zai",
                    api_base="https://api.z.ai/api/paas/v4",
                )

    @pytest.mark.parametrize("provider", ["zai", "hosted_vllm"])
    def test_api_base_is_required_for_every_provider(self, provider):
        """Every provider requires an explicit api_base."""
        from config.llm import LLMConfig

        with pytest.raises(ValueError, match="api_base is required"):
            LLMConfig(provider=provider, api_base=None, api_key="test")

    def test_zai_with_api_key_succeeds(self):
        """Z.AI provider with explicit api_key should succeed."""
        from config.llm import LLMConfig

        cfg = LLMConfig(
            provider="zai",
            model="glm-4.5-flash",
            api_base="https://api.z.ai/api/paas/v4",
            api_key="test-key",
        )
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

    @pytest.mark.parametrize("provider", ["zai", "hosted_vllm"])
    def test_from_dict_requires_explicit_api_base(self, provider):
        """from_dict must not hide a missing deployment endpoint."""
        from config.llm import LLMConfig

        with pytest.raises(ValueError, match="api_base is required"):
            LLMConfig.from_dict({"provider": provider, "api_key": "test"})

    def test_from_dict_preserves_extra_kwargs(self):
        """Provider-specific LiteLLM kwargs survive parsing and logging."""
        from config.llm import LLMConfig

        cfg = LLMConfig.from_dict(
            {
                "provider": "hosted_vllm",
                "model": "Qwen/test",
                "api_base": "http://localhost:8000/v1",
                "api_key_env": "HOSTED_VLLM_API_KEY",
                "temperature": 0.0,
                "max_tokens": 4096,
                "extra_kwargs": {"top_p": 0.9, "seed": 42},
            }
        )
        assert cfg.extra_kwargs == {"top_p": 0.9, "seed": 42}
        assert cfg.to_dict()["extra_kwargs"] == {"top_p": 0.9, "seed": 42}

    def test_get_model_string_zai(self):
        """get_model_string for zai should return zai/<model>."""
        from config.llm import LLMConfig

        cfg = LLMConfig(
            provider="zai",
            model="glm-4.5-flash",
            api_base="https://api.z.ai/api/paas/v4",
            api_key="test",
        )
        assert cfg.get_model_string() == "zai/glm-4.5-flash"

    def test_get_model_string_vllm(self):
        """get_model_string for hosted_vllm should return hosted_vllm/<model>."""
        from config.llm import LLMConfig

        cfg = LLMConfig(provider="hosted_vllm", model="Qwen/test", api_base="http://localhost:8000/v1")
        assert cfg.get_model_string() == "hosted_vllm/Qwen/test"

    @pytest.mark.parametrize(
        ("provider", "api_base"),
        [
            ("zai", "https://api.z.ai/api/paas/v4"),
            ("hosted_vllm", "http://localhost:8000/v1"),
        ],
    )
    def test_create_model_passes_api_base_for_every_provider(
        self,
        provider,
        api_base,
    ):
        """LiteLLM receives the selected endpoint for every provider."""
        from config.llm import LLMConfig, create_model

        cfg = LLMConfig(
            provider=provider,
            model="test-model",
            api_base=api_base,
            api_key="test-key",
            extra_kwargs={"top_p": 0.9},
        )
        with (
            patch(
                "minisweagent.models.litellm_model.LitellmModelConfig"
            ) as model_config,
            patch("minisweagent.models.litellm_model.LitellmModel"),
        ):
            create_model(cfg)

        model_kwargs = model_config.call_args.kwargs["model_kwargs"]
        assert model_kwargs["api_base"] == api_base
        assert model_kwargs["top_p"] == 0.9
