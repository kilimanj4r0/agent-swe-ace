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

import litellm
import pytest
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from config.llm_catalog import get_effective_llm
from config.loader import load_experiment_config

load_dotenv()

litellm.suppress_debug_info = True
litellm.request_timeout = 10  # Fail fast if server unreachable


def load_merged_config(config_override_path: str | None) -> dict:
    """Load base config plus an optional strict preset-aware override."""
    project_root = Path(__file__).parent.parent.parent
    base_path = project_root / "config.yaml"
    override_path = None
    if config_override_path:
        override_path = Path(config_override_path)
        if not override_path.is_absolute():
            override_path = project_root / override_path
        if not override_path.exists():
            pytest.exit(f"Config file not found: {override_path}")
    return load_experiment_config(base_path, override_path)


@pytest.fixture(scope="module")
def merged_config(request):
    """Load config, optionally merged with --config override."""
    config_path = request.config.getoption("--config")
    return load_merged_config(config_path)


@pytest.fixture(scope="module")
def agent_llm_config(merged_config):
    return get_effective_llm(merged_config["llm"]["agent"])


@pytest.fixture(scope="module")
def ace_llm_config(merged_config):
    return get_effective_llm(merged_config["llm"]["ace"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROMPT = [{"role": "user", "content": "Say exactly 'OK' and nothing else."}]
TOOL_NAME = "report_health"


class HealthProbe(BaseModel):
    status: str


def _raw_response(response: dict) -> dict:
    if not isinstance(response, dict):
        return {}
    return response.get("extra", {}).get("response", {})


def _make_llm_request(config_dict: dict) -> dict:
    """Send a minimal request via LitellmModel and return response info."""
    from config.llm import LLMConfig, create_model_from_yaml

    llm_config = LLMConfig.from_dict(config_dict)
    model = create_model_from_yaml(config_dict)

    t0 = time.time()
    response = model.query(PROMPT)
    elapsed = time.time() - t0

    content = ""
    if isinstance(response, dict):
        raw_content = response.get("content")
        if isinstance(raw_content, list):
            content = " ".join(
                part.get("text", str(part)) for part in raw_content
                if isinstance(part, dict)
            )
        elif raw_content is not None:
            content = str(raw_content)
    else:
        content = str(response)
    raw = _raw_response(response)

    return {
        "provider": llm_config.provider,
        "configured_model": llm_config.model,
        "model_string": llm_config.get_model_string(),
        "api_base": llm_config.api_base,
        "content": content,
        "response_model": raw.get("model"),
        "elapsed_s": round(elapsed, 2),
        "raw": response,
    }


def _make_required_tool_request(config_dict: dict) -> dict:
    """Require one named function call through the configured agent model."""
    from config.llm import LLMConfig, create_model_from_yaml

    llm_config = LLMConfig.from_dict(config_dict)
    model = create_model_from_yaml(config_dict)
    response = model.query(
        [{"role": "user", "content": f"Call {TOOL_NAME} with status OK."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": "Report endpoint health.",
                    "parameters": {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                    },
                },
            }
        ],
        tool_choice="required",
    )
    raw = _raw_response(response)
    choice = raw.get("choices", [{}])[0]
    tool_calls = choice.get("message", {}).get("tool_calls") or []
    return {
        "configured_model": llm_config.model,
        "response_model": raw.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "tool_calls": tool_calls,
    }


def _make_ace_structured_request(config_dict: dict) -> HealthProbe:
    """Exercise the same structured-output path used by ACE components."""
    from config.llm import LLMConfig, create_ace_client
    from prompts.model_utils import make_pydantic_agent

    llm_config = LLMConfig.from_dict(config_dict)
    model = create_ace_client(config_dict)
    agent = make_pydantic_agent(
        model,
        HealthProbe,
        api_base=llm_config.api_base,
        api_key=llm_config.api_key,
        max_retries=1,
        model_settings={"temperature": 0.0, "max_tokens": 256},
    )
    return agent.run_sync(
        "Return a structured health result whose status is exactly OK."
    ).output


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
@pytest.mark.parametrize("role", ["agent", "ace"])
def test_configured_role_response_contract(merged_config, role):
    """Each configured role returns content from the requested model."""
    info = _make_llm_request(get_effective_llm(merged_config["llm"][role]))

    assert info["content"].strip(), (
        f"{role} LLM returned empty content "
        f"[{info['provider']}/{info['configured_model']} @ {info['api_base']}]"
    )
    if info["response_model"]:
        assert info["configured_model"] in info["response_model"]
    print(
        f"\n  {role} OK [{info['model_string']}] "
        f"{info['elapsed_s']}s response={info['content'][:80]!r}"
    )


@pytest.mark.integration
def test_agent_required_tool_call_contract(agent_llm_config):
    """The agent endpoint must honor a required named tool call."""
    info = _make_required_tool_request(agent_llm_config)

    assert info["finish_reason"] == "tool_calls"
    assert info["tool_calls"], "Agent returned no tool calls"
    assert info["tool_calls"][0]["function"]["name"] == TOOL_NAME
    if info["response_model"]:
        assert info["configured_model"] in info["response_model"]


@pytest.mark.integration
def test_ace_structured_output_contract(ace_llm_config):
    """ACE's PydanticAI path must return validated structured output."""
    result = _make_ace_structured_request(ace_llm_config)

    assert isinstance(result, HealthProbe)
    assert result.status.strip() == "OK"
