"""LLM Health Check Tests.

Verifies that agent and ace LLM endpoints from a given config are reachable
and returning valid responses. Supports both the base config.yaml and any
override config from configs/ (deep-merged on top of the base).

Usage:
    # Test default config.yaml
    pytest src/tests/test_llm_health.py -v

    # Test a specific override config
    pytest src/tests/test_llm_health.py -v --config=configs/agent-qwen3-ace-glm.yaml

    # Test multiple configs
    pytest src/tests/test_llm_health.py -v --config=configs/agent-glm-ace-glm.yaml
    pytest src/tests/test_llm_health.py -v --config=configs/agent-qwen3-ace-qwen3.yaml
"""

import sys
import time
from pathlib import Path

import pytest
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import litellm

litellm.suppress_debug_info = True


def load_merged_config(config_override_path: str | None) -> dict:
    """Load base config.yaml, optionally deep-merged with an override config."""
    project_root = Path(__file__).parent.parent.parent
    base_path = project_root / "config.yaml"

    with open(base_path) as f:
        config = yaml.safe_load(f)

    if config_override_path:
        override_path = Path(config_override_path)
        if not override_path.is_absolute():
            override_path = project_root / override_path
        if not override_path.exists():
            pytest.exit(f"Config file not found: {override_path}")
        with open(override_path) as f:
            override = yaml.safe_load(f)
        config = deep_merge(config, override)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@pytest.fixture(scope="module")
def merged_config(request):
    """Load config, optionally merged with --config override."""
    config_path = request.config.getoption("--config")
    return load_merged_config(config_path)


@pytest.fixture(scope="module")
def agent_llm_config(merged_config):
    return merged_config["llm"]["agent"]


@pytest.fixture(scope="module")
def ace_llm_config(merged_config):
    return merged_config["llm"]["ace"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROMPT = [{"role": "user", "content": "Say exactly 'OK' and nothing else."}]


def _make_llm_request(config_dict: dict) -> dict:
    """Send a minimal request via LitellmModel and return response info."""
    from config.llm import create_model_from_yaml, LLMConfig

    llm_config = LLMConfig.from_dict(config_dict)
    model = create_model_from_yaml(config_dict)

    t0 = time.time()
    response = model.query(PROMPT)
    elapsed = time.time() - t0

    content = (
        response.get("content", str(response))
        if isinstance(response, dict)
        else str(response)
    )

    return {
        "provider": llm_config.provider,
        "model": llm_config.model,
        "model_string": llm_config.get_model_string(),
        "api_base": llm_config.api_base,
        "content": content,
        "elapsed_s": round(elapsed, 2),
        "raw": response,
    }


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Verify config loads and has required LLM sections."""

    def test_config_has_llm_section(self, merged_config):
        assert "llm" in merged_config, "Config missing 'llm' section"

    def test_config_has_agent_section(self, merged_config):
        assert "agent" in merged_config["llm"], "Config missing 'llm.agent' section"

    def test_config_has_ace_section(self, merged_config):
        assert "ace" in merged_config["llm"], "Config missing 'llm.ace' section"


# ---------------------------------------------------------------------------
# LLMConfig creation tests (no API calls)
# ---------------------------------------------------------------------------


class TestLLMConfigCreation:
    """Verify LLMConfig objects can be built from config sections."""

    def test_agent_llm_config(self, agent_llm_config):
        from config.llm import LLMConfig

        cfg = LLMConfig.from_dict(agent_llm_config)
        assert cfg.get_model_string() is not None
        print(f"\n  agent: provider={cfg.provider}  model={cfg.model}  "
              f"model_string={cfg.get_model_string()}  api_base={cfg.api_base}")

    def test_ace_llm_config(self, ace_llm_config):
        from config.llm import LLMConfig

        cfg = LLMConfig.from_dict(ace_llm_config)
        assert cfg.get_model_string() is not None
        print(f"\n  ace:   provider={cfg.provider}  model={cfg.model}  "
              f"model_string={cfg.get_model_string()}  api_base={cfg.api_base}")


# ---------------------------------------------------------------------------
# Health check tests (real API calls)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAgentHealth:
    """Agent LLM endpoint health check."""

    def test_agent_reachable(self, agent_llm_config):
        """Agent LLM returns a non-empty response."""
        info = _make_llm_request(agent_llm_config)
        assert info["content"] is not None, (
            f"Agent LLM returned None  "
            f"[{info['provider']}/{info['model']} @ {info['api_base']}]"
        )
        print(f"\n  agent OK  [{info['model_string']}]  "
              f"{info['elapsed_s']}s  response={info['content'][:80]!r}")

    def test_agent_response_has_content(self, agent_llm_config):
        """Agent LLM response body is non-empty."""
        info = _make_llm_request(agent_llm_config)
        assert len(info["content"]) > 0, (
            f"Agent LLM returned empty content  "
            f"[{info['provider']}/{info['model']} @ {info['api_base']}]"
        )


@pytest.mark.integration
class TestACEHealth:
    """ACE LLM endpoint health check."""

    def test_ace_reachable(self, ace_llm_config):
        """ACE LLM returns a non-empty response."""
        info = _make_llm_request(ace_llm_config)
        assert info["content"] is not None, (
            f"ACE LLM returned None  "
            f"[{info['provider']}/{info['model']} @ {info['api_base']}]"
        )
        print(f"\n  ace   OK  [{info['model_string']}]  "
              f"{info['elapsed_s']}s  response={info['content'][:80]!r}")

    def test_ace_response_has_content(self, ace_llm_config):
        """ACE LLM response body is non-empty."""
        info = _make_llm_request(ace_llm_config)
        assert len(info["content"]) > 0, (
            f"ACE LLM returned empty content  "
            f"[{info['provider']}/{info['model']} @ {info['api_base']}]"
        )
