"""
LLM Configuration Module

Loads API keys from .env file (not accessible to agents).
Supports Z.AI (Zhipu GLM) as default, and local vLLM servers.

Z.AI models use the 'zai/' prefix via LiteLLM.
See: https://docs.litellm.ai/docs/providers/zai

vLLM servers use the 'hosted_vllm/' prefix via LiteLLM.
See: https://docs.litellm.ai/docs/providers/vllm
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Any, Dict

from dotenv import load_dotenv
load_dotenv()

# Patch PydanticAI's schema transformer to inline $ref definitions.
# Z.AI models (GLM) can't handle $ref/$defs in tool schemas — they
# return referenced objects as JSON strings instead of actual objects.
# Inlining resolves all $ref into flat definitions.
try:
    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer as _OAIJT
    _oaijt_orig_init = _OAIJT.__init__
    def _oaijt_patched_init(self, schema, *, strict=None):
        _oaijt_orig_init(self, schema, strict=strict)
        self.prefer_inlined_defs = True
    _OAIJT.__init__ = _oaijt_patched_init
except ImportError:
    pass


@dataclass
class LLMConfig:
    """Unified LLM configuration."""

    provider: Literal["zai", "hosted_vllm"] = "zai"
    model: str = "glm-4.5-flash"  # Z.AI model
    api_key: Optional[str] = None
    api_key_env: str = "ZAI_API_KEY"  # Standard env var for Z.AI
    api_base: Optional[str] = None  # Required for vLLM, auto-handled for Z.AI
    temperature: float = 0.7  # Lower temperature for more deterministic code generation
    max_tokens: int = 4096
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Resolve API key from environment if not provided directly."""
        if not self.api_key:
            self.api_key = os.environ.get(self.api_key_env)

        # Z.AI requires API key, vLLM may not (local server)
        if self.provider == "zai" and not self.api_key:
            raise ValueError(
                f"Z.AI API key not found. Set {self.api_key_env} in .env file "
                "or pass api_key parameter. Get your key at: https://z.ai/"
            )

        # vLLM requires api_base
        if self.provider == "hosted_vllm" and not self.api_base:
            raise ValueError(
                "vLLM requires api_base parameter. "
                "Example: api_base='http://localhost:8000/v1'"
            )

    def to_dict(self) -> dict:
        """Convert to dictionary for logging (excludes sensitive data)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LLMConfig":
        """Create LLMConfig from a dictionary with defaults."""
        provider = config_dict.get("provider", "zai")

        # Set provider-specific defaults
        if provider == "hosted_vllm":
            defaults = {
                "model": "Qwen/Qwen3-Coder-30B-A3B",  # Popular coding model
                "api_key_env": "HOSTED_VLLM_API_KEY",  # Optional for local servers
                "api_base": "http://localhost:8000/v1",
            }
        else:  # zai
            defaults = {
                "model": "glm-4.5-flash",
                "api_key_env": "ZAI_API_KEY",
                "api_base": "https://api.z.ai/api/paas/v4",
            }

        return cls(
            provider=provider,
            model=config_dict.get("model", defaults["model"]),
            api_base=config_dict.get("api_base", defaults["api_base"]),
            api_key_env=config_dict.get("api_key_env", defaults["api_key_env"]),
            temperature=config_dict.get("temperature", 0.7),
            max_tokens=config_dict.get("max_tokens", 4096),
        )

    def get_model_string(self) -> str:
        """Get the model string for LiteLLM."""
        if self.provider == "zai":
            return f"zai/{self.model}"
        elif self.provider == "hosted_vllm":
            return f"hosted_vllm/{self.model}"
        return self.model


def create_model(config: LLMConfig):
    """
    Create a LiteLLM model instance based on configuration.

    Args:
        config: LLMConfig with provider-specific settings

    Returns:
        Configured LitellmModel instance
    """
    try:
        from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
    except ImportError:
        raise ImportError(
            "mini-swe-agent not installed. Run: "
            "pip install git+https://github.com/SWE-agent/mini-swe-agent.git@v1"
        )

    model_kwargs: Dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        **config.extra_kwargs
    }

    if config.provider == "hosted_vllm":
        model_kwargs["api_base"] = config.api_base
        if config.api_key:
            model_kwargs["api_key"] = config.api_key
        else:
            model_kwargs["api_key"] = "not-needed"
    elif config.provider == "zai":
        if config.api_key:
            model_kwargs["api_key"] = config.api_key
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

    model_config = LitellmModelConfig(
        model_name=config.get_model_string(),
        model_kwargs=model_kwargs,
        cost_tracking="ignore_errors"
    )

    return LitellmModel(config_class=lambda: model_config)


def create_model_from_yaml(config_dict: dict):
    """
    Create model from YAML configuration section.

    Args:
        config_dict: Dictionary from config.yaml llm.agent section

    Returns:
        Configured LitellmModel instance
    """
    return create_model(LLMConfig.from_dict(config_dict))


def create_ace_client(config_dict: dict):
    """
    Create ace-framework model string from YAML configuration.

    In ACE v0.9.1+, Reflector and SkillManager accept a model string
    directly (they handle LLM calls internally via PydanticAI).

    Args:
        config_dict: Dictionary from config.yaml llm.ace section

    Returns:
        Model string for ACE components (e.g. "zai/glm-5")
    """
    config = LLMConfig.from_dict(config_dict)
    return config.get_model_string()
