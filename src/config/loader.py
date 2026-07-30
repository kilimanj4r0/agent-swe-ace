"""Layered experiment configuration loading with strict LLM preset semantics."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .llm_catalog import (
    LLMCatalogError,
    deep_merge_values,
    load_llm_catalog,
    resolve_llm_section,
)

_LLM_PATHS = (
    ("llm", "agent"),
    ("llm", "ace"),
    ("experiment", "skillbook", "retrieval", "llm"),
)
_REFERENCE_KEYS = frozenset({"preset", "overrides"})
_FLAT_RETRIEVAL_FIELDS = frozenset(
    {"api_base", "api_key_env", "temperature", "max_tokens"}
)


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _raise(source_path: Path, logical_path: str, message: str) -> None:
    raise LLMCatalogError(f"{source_path}: {logical_path}: {message}")


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Deep merge ordinary mappings without mutating either input."""
    return deep_merge_values(base, override)


def merge_llm_reference(
    previous: object | None,
    incoming: object,
    *,
    source_path: Path,
    logical_path: str,
) -> dict[str, Any]:
    """Merge one raw LLM reference according to the preset reset/patch rules."""
    if isinstance(incoming, str):
        if not incoming:
            _raise(source_path, logical_path, "preset must be a non-empty string")
        return {"preset": incoming, "overrides": {}}

    if not isinstance(incoming, Mapping):
        _raise(
            source_path,
            logical_path,
            "LLM reference must be a preset string or a mapping",
        )

    unknown = sorted(set(incoming) - _REFERENCE_KEYS)
    if unknown:
        _raise(
            source_path,
            logical_path,
            f"mapping accepts only preset and overrides; unexpected: {', '.join(unknown)}",
        )

    overrides = incoming.get("overrides", {})
    if not isinstance(overrides, Mapping):
        _raise(source_path, logical_path, "overrides must be a mapping")

    if "preset" in incoming:
        preset = incoming["preset"]
        if not isinstance(preset, str) or not preset:
            _raise(source_path, logical_path, "preset must be a non-empty string")
        return {
            "preset": preset,
            "overrides": copy.deepcopy(dict(overrides)),
        }

    if "overrides" not in incoming:
        _raise(
            source_path,
            logical_path,
            "mapping must contain preset or overrides",
        )
    if not isinstance(previous, Mapping) or not previous.get("preset"):
        _raise(
            source_path,
            logical_path,
            "override-only form requires an inherited preset",
        )
    previous_overrides = previous.get("overrides", {})
    if not isinstance(previous_overrides, Mapping):
        _raise(source_path, logical_path, "inherited overrides must be a mapping")
    return {
        "preset": previous["preset"],
        "overrides": deep_merge_values(previous_overrides, overrides),
    }


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        _raise(path, "config", "file does not exist")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        _raise(path, "config", f"invalid YAML: {exc}")
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        _raise(path, "config", "top level must be a mapping")
    return copy.deepcopy(dict(data))


def _get_path(mapping: Mapping[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = mapping
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _remove_path(mapping: dict[str, Any], path: tuple[str, ...]) -> None:
    current: Any = mapping
    for part in path[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(path[-1], None)


def _set_path(mapping: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = mapping
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[path[-1]] = value


def _field_source(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    path: tuple[str, ...],
    *,
    base_path: Path,
    override_path: Path | None,
) -> Path:
    if override_path is not None and _get_path(override, path)[0]:
        return override_path
    return base_path


def _validate_flat_retrieval_fields(
    merged: Mapping[str, Any],
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    *,
    base_path: Path,
    override_path: Path | None,
) -> None:
    retrieval_path = ("experiment", "skillbook", "retrieval")
    present, retrieval = _get_path(merged, retrieval_path)
    if not present or not isinstance(retrieval, Mapping):
        return

    forbidden = set(_FLAT_RETRIEVAL_FIELDS)
    if retrieval.get("type", "llm") == "llm":
        forbidden.add("model")
    for field in sorted(forbidden):
        if field not in retrieval:
            continue
        full_path = (*retrieval_path, field)
        source = _field_source(
            base,
            override,
            full_path,
            base_path=base_path,
            override_path=override_path,
        )
        _raise(
            source,
            _format_path(full_path),
            "flat generative LLM fields are forbidden; use retrieval.llm",
        )


def load_experiment_config(
    base_path: Path,
    override_path: Path | None = None,
    *,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """Load base plus optional override and resolve all LLM preset references."""
    base_path = Path(base_path)
    override_path = Path(override_path) if override_path is not None else None
    base = _read_mapping(base_path)
    override = _read_mapping(override_path) if override_path is not None else {}

    base_without_refs = copy.deepcopy(base)
    override_without_refs = copy.deepcopy(override)
    references: dict[tuple[str, ...], tuple[dict[str, Any], Path]] = {}

    for path in _LLM_PATHS:
        logical_path = _format_path(path)
        current: dict[str, Any] | None = None
        source = base_path

        base_has, base_value = _get_path(base, path)
        if base_has:
            current = merge_llm_reference(
                None,
                base_value,
                source_path=base_path,
                logical_path=logical_path,
            )
            _remove_path(base_without_refs, path)

        override_has, override_value = _get_path(override, path)
        if override_has:
            current = merge_llm_reference(
                current,
                override_value,
                source_path=override_path or base_path,
                logical_path=logical_path,
            )
            source = override_path or base_path
            _remove_path(override_without_refs, path)

        if current is not None:
            references[path] = (current, source)

    merged = deep_merge(base_without_refs, override_without_refs)
    for path, (reference, _) in references.items():
        _set_path(merged, path, reference)

    _validate_flat_retrieval_fields(
        merged,
        base,
        override,
        base_path=base_path,
        override_path=override_path,
    )

    retrieval_path = ("experiment", "skillbook", "retrieval")
    has_retrieval, retrieval = _get_path(merged, retrieval_path)
    retrieval_llm_path = (*retrieval_path, "llm")
    if (
        has_retrieval
        and isinstance(retrieval, Mapping)
        and retrieval.get("enabled", False)
        and retrieval.get("type", "llm") == "llm"
        and retrieval_llm_path not in references
    ):
        _raise(
            override_path or base_path,
            _format_path(retrieval_llm_path),
            "LLM reference is required when LLM retrieval is enabled",
        )

    actual_catalog_path = Path(catalog_path) if catalog_path else base_path.parent / "llms.yaml"
    catalog = load_llm_catalog(actual_catalog_path)
    for path, (reference, source) in references.items():
        resolved = resolve_llm_section(
            reference["preset"],
            reference["overrides"],
            catalog,
            source_path=source,
            logical_path=_format_path(path),
        )
        _set_path(merged, path, resolved)

    return merged
