"""Validation and resolution for the repository LLM deployment catalog."""

from __future__ import annotations

import copy
import difflib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from yaml.constructor import ConstructorError


class LLMCatalogError(ValueError):
    """Raised when a catalog or LLM reference is invalid."""


SUPPORTED_PROVIDERS = frozenset({"zai", "hosted_vllm"})
LLM_FIELDS = frozenset(
    {
        "provider",
        "model",
        "api_base",
        "api_key_env",
        "temperature",
        "max_tokens",
        "extra_kwargs",
    }
)
REQUIRED_FIELDS = frozenset(
    {"provider", "model", "api_base", "api_key_env", "temperature", "max_tokens"}
)
PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "zai": {
        "model": "glm-4.5-flash",
        "api_base": "https://api.z.ai/api/paas/v4",
        "api_key_env": "ZAI_API_KEY",
    },
    "hosted_vllm": {
        "model": "Qwen/Qwen3-Coder-30B-A3B",
        "api_base": "http://localhost:8000/v1",
        "api_key_env": "HOSTED_VLLM_API_KEY",
    },
}

_PRESET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _error(source_path: Path, logical_path: str, message: str) -> LLMCatalogError:
    return LLMCatalogError(f"{source_path}: {logical_path}: {message}")


def _load_unique_yaml(path: Path) -> Any:
    if not path.exists():
        raise _error(path, "catalog", "file does not exist")
    try:
        return yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise _error(path, "catalog", str(exc)) from exc


def _validate_known_fields(
    values: Mapping[str, Any],
    *,
    source_path: Path,
    logical_path: str,
) -> None:
    unknown = sorted(set(values) - LLM_FIELDS)
    if "api_key" in unknown:
        raise _error(source_path, logical_path, "api_key is forbidden; use api_key_env")
    if unknown:
        raise _error(
            source_path,
            logical_path,
            f"unknown LLM field(s): {', '.join(unknown)}; "
            f"allowed: {', '.join(sorted(LLM_FIELDS))}",
        )


def _validate_llm_values(
    values: Mapping[str, Any],
    *,
    source_path: Path,
    logical_path: str,
    require_complete: bool,
) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise _error(source_path, logical_path, "LLM values must be a mapping")
    _validate_known_fields(values, source_path=source_path, logical_path=logical_path)

    if require_complete:
        missing = sorted(REQUIRED_FIELDS - set(values))
        if missing:
            raise _error(
                source_path,
                logical_path,
                f"missing required field(s): {', '.join(missing)}",
            )

    result = copy.deepcopy(dict(values))
    if require_complete:
        result.setdefault("extra_kwargs", {})

    for field in ("provider", "model", "api_base", "api_key_env"):
        if field not in result:
            continue
        if not isinstance(result[field], str) or not result[field].strip():
            raise _error(source_path, logical_path, f"{field} must be a non-empty string")

    provider = result.get("provider")
    if provider is not None and provider not in SUPPORTED_PROVIDERS:
        raise _error(
            source_path,
            logical_path,
            f"provider must be one of {sorted(SUPPORTED_PROVIDERS)}, got {provider!r}",
        )

    api_base = result.get("api_base")
    if api_base is not None:
        parsed = urlparse(api_base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise _error(
                source_path,
                logical_path,
                "api_base must be a non-empty absolute HTTP(S) URL",
            )

    api_key_env = result.get("api_key_env")
    if api_key_env is not None and not _ENV_NAME_RE.fullmatch(api_key_env):
        raise _error(
            source_path,
            logical_path,
            "api_key_env must match [A-Za-z_][A-Za-z0-9_]*",
        )

    temperature = result.get("temperature")
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0.0 <= float(temperature) <= 2.0
    ):
        raise _error(
            source_path,
            logical_path,
            "temperature must be a number in the range 0.0..2.0",
        )

    max_tokens = result.get("max_tokens")
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise _error(source_path, logical_path, "max_tokens must be a positive integer")

    extra_kwargs = result.get("extra_kwargs")
    if extra_kwargs is not None and not isinstance(extra_kwargs, Mapping):
        raise _error(source_path, logical_path, "extra_kwargs must be a mapping")

    return result


def load_llm_catalog(path: Path) -> dict[str, dict[str, Any]]:
    """Load and fully validate a version-1 LLM deployment catalog."""
    path = Path(path)
    data = _load_unique_yaml(path)
    if not isinstance(data, Mapping):
        raise _error(path, "catalog", "top level must be a mapping")

    unknown_top = sorted(set(data) - {"version", "presets"})
    if unknown_top:
        raise _error(
            path,
            "catalog",
            f"unknown catalog field(s): {', '.join(unknown_top)}",
        )

    version = data.get("version")
    if isinstance(version, bool) or version != 1 or not isinstance(version, int):
        raise _error(path, "catalog.version", "only integer schema version 1 is supported")

    presets = data.get("presets")
    if not isinstance(presets, Mapping):
        raise _error(path, "catalog.presets", "presets must be a mapping")

    result: dict[str, dict[str, Any]] = {}
    for name, raw_values in presets.items():
        if not isinstance(name, str) or not _PRESET_NAME_RE.fullmatch(name):
            raise _error(
                path,
                "catalog.presets",
                f"invalid preset name {name!r}; expected [a-z0-9][a-z0-9._-]*",
            )
        result[name] = _validate_llm_values(
            raw_values,
            source_path=path,
            logical_path=f"presets.{name}",
            require_complete=True,
        )
    return result


def deep_merge_values(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge_values(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_llm_section(
    preset: str,
    overrides: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
    *,
    source_path: Path,
    logical_path: str,
) -> dict[str, Any]:
    """Resolve one normalized preset reference into a reproducible wrapper."""
    if not isinstance(preset, str) or not preset:
        raise _error(source_path, logical_path, "preset must be a non-empty string")
    if preset not in catalog:
        suggestions = difflib.get_close_matches(preset, list(catalog), n=3, cutoff=0.4)
        suffix = f"; did you mean: {', '.join(suggestions)}" if suggestions else ""
        raise _error(source_path, logical_path, f"unknown preset {preset!r}{suffix}")
    if not isinstance(overrides, Mapping):
        raise _error(source_path, logical_path, "overrides must be a mapping")

    checked_overrides = _validate_llm_values(
        overrides,
        source_path=source_path,
        logical_path=f"{logical_path}.overrides",
        require_complete=False,
    )
    effective = copy.deepcopy(dict(catalog[preset]))
    for key, value in checked_overrides.items():
        if key == "extra_kwargs":
            effective[key] = deep_merge_values(effective.get(key, {}), value)
        else:
            effective[key] = copy.deepcopy(value)
    effective = _validate_llm_values(
        effective,
        source_path=source_path,
        logical_path=f"{logical_path}.effective",
        require_complete=True,
    )
    return {
        "preset": preset,
        "overrides": copy.deepcopy(checked_overrides),
        "effective": effective,
    }


def _normalise_legacy_effective(section: Mapping[str, Any]) -> dict[str, Any]:
    provider = section.get("provider", "zai")
    if provider not in PROVIDER_DEFAULTS:
        raise LLMCatalogError(f"legacy LLM provider is unsupported: {provider!r}")
    defaults = PROVIDER_DEFAULTS[provider]
    return {
        "provider": provider,
        "model": section.get("model", defaults["model"]),
        "api_base": section.get("api_base", defaults["api_base"]),
        "api_key_env": section.get("api_key_env", defaults["api_key_env"]),
        "temperature": section.get("temperature", 0.7),
        "max_tokens": section.get("max_tokens", 4096),
        "extra_kwargs": copy.deepcopy(section.get("extra_kwargs", {})),
    }


def get_effective_llm(section: Mapping[str, Any]) -> dict[str, Any]:
    """Return effective fields from a new wrapper or a legacy flat mapping."""
    if not isinstance(section, Mapping):
        raise LLMCatalogError("LLM section must be a mapping")
    if "effective" in section:
        effective = section["effective"]
        if not isinstance(effective, Mapping):
            raise LLMCatalogError("LLM section effective value must be a mapping")
        return copy.deepcopy(dict(effective))
    return _normalise_legacy_effective(section)
