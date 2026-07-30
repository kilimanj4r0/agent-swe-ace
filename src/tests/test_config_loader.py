"""Tests for layered experiment configuration loading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.llm_catalog import LLMCatalogError, get_effective_llm
from config.loader import (
    load_experiment_config,
    merge_llm_reference,
)


def preset_values(model: str, api_base: str) -> dict:
    return {
        "provider": "hosted_vllm",
        "model": model,
        "api_base": api_base,
        "api_key_env": "HOSTED_VLLM_API_KEY",
        "temperature": 0.0,
        "max_tokens": 4096,
    }


def write_catalog(directory: Path) -> Path:
    path = directory / "llms.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "presets": {
                    "p": preset_values("model-p", "http://localhost:8000/v1"),
                    "q": preset_values("model-q", "http://localhost:8800/v1"),
                },
            },
            sort_keys=False,
        )
    )
    return path


def base_config() -> dict:
    return {
        "experiment": {
            "skillbook": {
                "retrieval": {
                    "enabled": False,
                    "type": "llm",
                    "llm": "p",
                    "top_k": 5,
                }
            }
        },
        "llm": {"agent": "p", "ace": "p"},
    }


def write_config(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


@pytest.mark.parametrize(
    ("previous", "incoming", "expected"),
    [
        (None, "p", {"preset": "p", "overrides": {}}),
        (
            {"preset": "p", "overrides": {"temperature": 0.7}},
            "q",
            {"preset": "q", "overrides": {}},
        ),
        (
            {"preset": "p", "overrides": {"temperature": 0.7}},
            {"preset": "q", "overrides": {"max_tokens": 8192}},
            {"preset": "q", "overrides": {"max_tokens": 8192}},
        ),
        (
            {"preset": "p", "overrides": {"extra_kwargs": {"seed": 1}}},
            {"overrides": {"extra_kwargs": {"top_p": 0.9}}},
            {
                "preset": "p",
                "overrides": {"extra_kwargs": {"seed": 1, "top_p": 0.9}},
            },
        ),
    ],
)
def test_merge_llm_reference(previous, incoming, expected):
    assert (
        merge_llm_reference(
            previous,
            incoming,
            source_path=Path("override.yaml"),
            logical_path="llm.agent",
        )
        == expected
    )


def test_override_only_without_inherited_preset_is_rejected():
    with pytest.raises(LLMCatalogError, match="override.yaml.*llm.agent.*inherited preset"):
        merge_llm_reference(
            None,
            {"overrides": {"temperature": 0.7}},
            source_path=Path("override.yaml"),
            logical_path="llm.agent",
        )


@pytest.mark.parametrize(
    "incoming",
    [
        {"provider": "hosted_vllm", "model": "inline"},
        {"preset": "p", "temperature": 0.7},
        {"preset": "p", "overrides": []},
    ],
)
def test_invalid_reference_shapes_are_rejected(incoming):
    with pytest.raises(LLMCatalogError, match="override.yaml.*llm.agent"):
        merge_llm_reference(
            None,
            incoming,
            source_path=Path("override.yaml"),
            logical_path="llm.agent",
        )


def test_load_resolves_short_forms_independently(tmp_path):
    catalog = write_catalog(tmp_path)
    base = base_config()
    base["llm"]["ace"] = "q"
    base_path = write_config(tmp_path / "config.yaml", base)
    result = load_experiment_config(base_path, catalog_path=catalog)
    assert result["llm"]["agent"]["preset"] == "p"
    assert get_effective_llm(result["llm"]["agent"])["model"] == "model-p"
    assert result["llm"]["ace"]["preset"] == "q"
    assert get_effective_llm(result["llm"]["ace"])["model"] == "model-q"
    assert result["experiment"]["skillbook"]["retrieval"]["llm"]["preset"] == "p"


def test_override_only_layer_patches_inherited_preset(tmp_path):
    catalog = write_catalog(tmp_path)
    base_path = write_config(tmp_path / "config.yaml", base_config())
    override_path = write_config(
        tmp_path / "override.yaml",
        {"llm": {"agent": {"overrides": {"temperature": 0.7}}}},
    )
    result = load_experiment_config(base_path, override_path, catalog_path=catalog)
    assert result["llm"]["agent"]["preset"] == "p"
    assert result["llm"]["agent"]["overrides"] == {"temperature": 0.7}
    assert get_effective_llm(result["llm"]["agent"])["temperature"] == 0.7


def test_new_preset_clears_previous_overrides(tmp_path):
    catalog = write_catalog(tmp_path)
    base = base_config()
    base["llm"]["agent"] = {"preset": "p", "overrides": {"temperature": 0.7}}
    base_path = write_config(tmp_path / "config.yaml", base)
    override_path = write_config(tmp_path / "override.yaml", {"llm": {"agent": "q"}})
    result = load_experiment_config(base_path, override_path, catalog_path=catalog)
    assert result["llm"]["agent"]["preset"] == "q"
    assert result["llm"]["agent"]["overrides"] == {}
    assert get_effective_llm(result["llm"]["agent"])["temperature"] == 0.0


def test_inline_deployment_is_rejected_with_source_and_path(tmp_path):
    catalog = write_catalog(tmp_path)
    base = base_config()
    base["llm"]["agent"] = preset_values("inline", "http://localhost:8000/v1")
    base_path = write_config(tmp_path / "config.yaml", base)
    with pytest.raises(LLMCatalogError, match=r"config\.yaml.*llm\.agent"):
        load_experiment_config(base_path, catalog_path=catalog)


@pytest.mark.parametrize(
    "legacy_field",
    ["model", "api_base", "api_key_env", "temperature", "max_tokens"],
)
def test_flat_generative_retrieval_fields_are_rejected(tmp_path, legacy_field):
    catalog = write_catalog(tmp_path)
    base = base_config()
    base["experiment"]["skillbook"]["retrieval"][legacy_field] = (
        "legacy" if legacy_field not in {"temperature", "max_tokens"} else 1
    )
    base_path = write_config(tmp_path / "config.yaml", base)
    with pytest.raises(
        LLMCatalogError,
        match=rf"experiment\.skillbook\.retrieval\.{legacy_field}",
    ):
        load_experiment_config(base_path, catalog_path=catalog)


def test_embedding_model_remains_at_retrieval_level(tmp_path):
    catalog = write_catalog(tmp_path)
    base = base_config()
    retrieval = base["experiment"]["skillbook"]["retrieval"]
    retrieval["enabled"] = True
    retrieval["type"] = "embedding"
    retrieval["model"] = "Qwen/Qwen3-Embedding-4B"
    base_path = write_config(tmp_path / "config.yaml", base)
    result = load_experiment_config(base_path, catalog_path=catalog)
    resolved = result["experiment"]["skillbook"]["retrieval"]
    assert resolved["model"] == "Qwen/Qwen3-Embedding-4B"
    assert resolved["llm"]["preset"] == "p"


def test_enabled_llm_retrieval_requires_llm_reference(tmp_path):
    catalog = write_catalog(tmp_path)
    base = base_config()
    retrieval = base["experiment"]["skillbook"]["retrieval"]
    retrieval["enabled"] = True
    del retrieval["llm"]
    base_path = write_config(tmp_path / "config.yaml", base)
    with pytest.raises(LLMCatalogError, match=r"retrieval\.llm.*required"):
        load_experiment_config(base_path, catalog_path=catalog)


def test_catalog_defaults_next_to_base_config(tmp_path):
    write_catalog(tmp_path)
    base_path = write_config(tmp_path / "config.yaml", base_config())
    result = load_experiment_config(base_path)
    assert result["llm"]["agent"]["preset"] == "p"


def test_missing_override_file_has_clear_error(tmp_path):
    catalog = write_catalog(tmp_path)
    base_path = write_config(tmp_path / "config.yaml", base_config())
    missing = tmp_path / "missing.yaml"
    with pytest.raises(LLMCatalogError, match=r"missing\.yaml.*does not exist"):
        load_experiment_config(base_path, missing, catalog_path=catalog)
