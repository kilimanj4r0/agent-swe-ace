"""Tests for strict LLM catalog validation and resolution."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.llm_catalog import (
    LLMCatalogError,
    get_effective_llm,
    load_llm_catalog,
    resolve_llm_section,
)


def valid_values(**changes):
    values = {
        "provider": "hosted_vllm",
        "model": "Qwen/test",
        "api_base": "http://localhost:8800/v1",
        "api_key_env": "HOSTED_VLLM_API_KEY",
        "temperature": 0.0,
        "max_tokens": 4096,
        "extra_kwargs": {"seed": 1},
    }
    values.update(changes)
    return values


def write_catalog(tmp_path: Path, preset=None) -> Path:
    path = tmp_path / "llms.yaml"
    values = valid_values() if preset is None else preset
    path.write_text(
        "version: 1\n"
        "presets:\n"
        "  qwen:\n"
        + "\n".join(f"    {key}: {value!r}" for key, value in values.items())
        + "\n"
    )
    return path


def test_resolve_keeps_intent_and_builds_effective():
    result = resolve_llm_section(
        "qwen",
        {"temperature": 0.7, "extra_kwargs": {"top_p": 0.9}},
        {"qwen": valid_values()},
        source_path=Path("experiment.yaml"),
        logical_path="llm.agent",
    )
    assert result["preset"] == "qwen"
    assert result["overrides"] == {
        "temperature": 0.7,
        "extra_kwargs": {"top_p": 0.9},
    }
    assert result["effective"]["temperature"] == 0.7
    assert result["effective"]["extra_kwargs"] == {"seed": 1, "top_p": 0.9}


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("provider", "zai"),
        ("model", "glm-4.7-flash"),
        ("api_base", "https://api.z.ai/api/coding/paas/v4"),
        ("api_key_env", "ZAI_API_KEY"),
        ("temperature", 1.0),
        ("max_tokens", 8192),
        ("extra_kwargs", {"seed": 2}),
    ],
)
def test_every_supported_field_can_be_overridden(field, replacement):
    result = resolve_llm_section(
        "qwen",
        {field: replacement},
        {"qwen": valid_values()},
        source_path=Path("x.yaml"),
        logical_path="llm.agent",
    )
    assert result["effective"][field] == replacement


def test_load_catalog_adds_empty_extra_kwargs(tmp_path):
    path = write_catalog(tmp_path, {k: v for k, v in valid_values().items() if k != "extra_kwargs"})
    assert load_llm_catalog(path)["qwen"]["extra_kwargs"] == {}


def test_missing_catalog_file_has_path(tmp_path):
    path = tmp_path / "missing.yaml"
    with pytest.raises(LLMCatalogError, match=r"missing\.yaml.*does not exist"):
        load_llm_catalog(path)


@pytest.mark.parametrize("version", ["2", "true", "'1'"])
def test_only_integer_schema_version_one_is_allowed(tmp_path, version):
    path = tmp_path / "llms.yaml"
    path.write_text(f"version: {version}\npresets: {{}}\n")
    with pytest.raises(LLMCatalogError, match="version"):
        load_llm_catalog(path)


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "llms.yaml"
    path.write_text("version: 1\npresets:\n  same: {}\n  same: {}\n")
    with pytest.raises(LLMCatalogError, match="duplicate key.*same"):
        load_llm_catalog(path)


def test_unknown_top_level_key_is_rejected(tmp_path):
    path = tmp_path / "llms.yaml"
    path.write_text("version: 1\npresets: {}\nother: true\n")
    with pytest.raises(LLMCatalogError, match="unknown catalog field.*other"):
        load_llm_catalog(path)


def test_missing_presets_mapping_is_rejected(tmp_path):
    path = tmp_path / "llms.yaml"
    path.write_text("version: 1\n")
    with pytest.raises(LLMCatalogError, match="presets"):
        load_llm_catalog(path)


def test_invalid_preset_name_is_rejected(tmp_path):
    path = tmp_path / "llms.yaml"
    path.write_text("version: 1\npresets:\n  Bad Name: {}\n")
    with pytest.raises(LLMCatalogError, match="preset name"):
        load_llm_catalog(path)


def test_unknown_preset_suggests_nearest_name():
    with pytest.raises(LLMCatalogError, match="qwen3-coder.*qwen3-code"):
        resolve_llm_section(
            "qwen3-coder",
            {},
            {"qwen3-code": valid_values()},
            source_path=Path("x.yaml"),
            logical_path="llm.agent",
        )


@pytest.mark.parametrize("field", ["other", "api_key"])
def test_unknown_or_secret_catalog_field_is_rejected(tmp_path, field):
    values = valid_values()
    values[field] = "secret"
    path = write_catalog(tmp_path, values)
    message = "api_key is forbidden" if field == "api_key" else "unknown LLM field"
    with pytest.raises(LLMCatalogError, match=message):
        load_llm_catalog(path)


@pytest.mark.parametrize("field", ["provider", "model", "api_base", "api_key_env"])
def test_required_string_field_must_be_non_empty(tmp_path, field):
    path = write_catalog(tmp_path, valid_values(**{field: ""}))
    with pytest.raises(LLMCatalogError, match=field):
        load_llm_catalog(path)


def test_provider_is_restricted(tmp_path):
    path = write_catalog(tmp_path, valid_values(provider="openai"))
    with pytest.raises(LLMCatalogError, match="provider"):
        load_llm_catalog(path)


@pytest.mark.parametrize("api_base", ["localhost:8800/v1", "ftp://host/v1", "http:///v1"])
def test_api_base_must_be_absolute_http_url(tmp_path, api_base):
    path = write_catalog(tmp_path, valid_values(api_base=api_base))
    with pytest.raises(LLMCatalogError, match="api_base"):
        load_llm_catalog(path)


@pytest.mark.parametrize("env_name", ["1KEY", "HAS-DASH", ""])
def test_api_key_env_must_be_an_environment_name(tmp_path, env_name):
    path = write_catalog(tmp_path, valid_values(api_key_env=env_name))
    with pytest.raises(LLMCatalogError, match="api_key_env"):
        load_llm_catalog(path)


@pytest.mark.parametrize("temperature", [-0.1, 2.1, True, "0.7"])
def test_temperature_range_and_type_are_validated(tmp_path, temperature):
    path = write_catalog(tmp_path, valid_values(temperature=temperature))
    with pytest.raises(LLMCatalogError, match="temperature"):
        load_llm_catalog(path)


@pytest.mark.parametrize("max_tokens", [0, -1, 1.5, True, "4096"])
def test_max_tokens_must_be_a_positive_integer(tmp_path, max_tokens):
    path = write_catalog(tmp_path, valid_values(max_tokens=max_tokens))
    with pytest.raises(LLMCatalogError, match="max_tokens"):
        load_llm_catalog(path)


def test_extra_kwargs_must_be_a_mapping(tmp_path):
    path = write_catalog(tmp_path, valid_values(extra_kwargs=[]))
    with pytest.raises(LLMCatalogError, match="extra_kwargs"):
        load_llm_catalog(path)


@pytest.mark.parametrize("field", ["other", "api_key"])
def test_unknown_or_secret_override_is_rejected(field):
    message = "api_key is forbidden" if field == "api_key" else "unknown LLM field"
    with pytest.raises(LLMCatalogError, match=message):
        resolve_llm_section(
            "qwen",
            {field: "value"},
            {"qwen": valid_values()},
            source_path=Path("override.yaml"),
            logical_path="llm.agent",
        )


def test_effective_values_are_revalidated_after_overrides():
    with pytest.raises(LLMCatalogError, match="api_base"):
        resolve_llm_section(
            "qwen",
            {"provider": "zai", "api_base": "not-a-url"},
            {"qwen": valid_values()},
            source_path=Path("override.yaml"),
            logical_path="llm.agent",
        )


def test_resolution_does_not_mutate_inputs():
    catalog = {"qwen": valid_values()}
    overrides = {"extra_kwargs": {"top_p": 0.9}}
    original_catalog = copy.deepcopy(catalog)
    original_overrides = copy.deepcopy(overrides)
    resolve_llm_section(
        "qwen",
        overrides,
        catalog,
        source_path=Path("x.yaml"),
        logical_path="llm.agent",
    )
    assert catalog == original_catalog
    assert overrides == original_overrides


def test_get_effective_llm_reads_new_wrapper_by_copy():
    wrapper = {"preset": "qwen", "overrides": {}, "effective": valid_values()}
    effective = get_effective_llm(wrapper)
    effective["model"] = "changed"
    assert wrapper["effective"]["model"] == "Qwen/test"


def test_legacy_flat_mapping_gets_provider_defaults_without_guessing_preset():
    legacy = {
        "provider": "zai",
        "model": "glm-4.5-flash",
        "api_key_env": "ZAI_API_KEY",
    }
    effective = get_effective_llm(legacy)
    assert effective["api_base"] == "https://api.z.ai/api/paas/v4"
    assert effective["temperature"] == 0.7
    assert "preset" not in effective


def test_get_effective_llm_rejects_non_mapping():
    with pytest.raises(LLMCatalogError, match="mapping"):
        get_effective_llm("qwen")
