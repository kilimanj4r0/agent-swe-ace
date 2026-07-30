#!/usr/bin/env python3
"""Build and compare semantic inventories for the LLM preset migration."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


LLM_FIELDS = (
    "provider",
    "model",
    "api_base",
    "api_key_env",
    "temperature",
    "max_tokens",
    "extra_kwargs",
)

PROVIDER_DEFAULTS = {
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

_GENERATIVE_RETRIEVAL_FIELDS = {
    "api_base",
    "api_key_env",
    "temperature",
    "max_tokens",
}
_COMMENT_FIELD_RE = re.compile(
    r"^\s*#\s*(provider|model|api_base|api_key_env|temperature|max_tokens):\s*(.*?)\s*$"
)


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep merge without mutating either input."""
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def legacy_effective(section: Mapping[str, Any]) -> dict[str, Any]:
    provider = section.get("provider", "zai")
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unsupported legacy provider: {provider!r}")
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


def legacy_retrieval_effective(retrieval: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "provider": "hosted_vllm",
        "model": retrieval.get("model"),
        "api_base": retrieval.get("api_base"),
        "api_key_env": retrieval.get("api_key_env", "ZAI_API_KEY"),
        "temperature": retrieval.get("temperature", 0.0),
        "max_tokens": retrieval.get("max_tokens", 2048),
        "extra_kwargs": {},
    }


def _normalise_retrieval(retrieval: Mapping[str, Any]) -> dict[str, Any]:
    retriever_type = retrieval.get("type", "llm")
    ignored = set(_GENERATIVE_RETRIEVAL_FIELDS)
    ignored.add("llm")
    if retriever_type == "llm":
        ignored.add("model")
    return {
        key: copy.deepcopy(value)
        for key, value in sorted(retrieval.items())
        if key not in ignored
    }


def _snapshot_effective(config: Mapping[str, Any]) -> dict[str, Any]:
    llm = config.get("llm", {})
    retrieval = (
        config.get("experiment", {})
        .get("skillbook", {})
        .get("retrieval", {})
    )
    retriever_type = retrieval.get("type", "llm")
    return {
        "llm.agent": legacy_effective(llm.get("agent", {})),
        "llm.ace": legacy_effective(llm.get("ace", {})),
        "retrieval": _normalise_retrieval(retrieval),
        "retrieval.llm": (
            legacy_retrieval_effective(retrieval)
            if retriever_type == "llm"
            else None
        ),
    }


def _config_paths(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "configs").rglob("*.yaml"))


def _group_name(relative_path: str) -> str:
    path = Path(relative_path)
    if relative_path == "configs/test.yaml":
        return "configs/test.yaml"
    return "/".join(path.parts[:2])


def _retrieval_bucket(config: Mapping[str, Any]) -> str:
    retrieval = (
        config.get("experiment", {})
        .get("skillbook", {})
        .get("retrieval", {})
    )
    retriever_type = retrieval.get("type", "llm")
    if not retrieval.get("enabled", False):
        return f"disabled_{retriever_type}"
    return f"enabled_{retriever_type}"


def _parse_commented_value(raw_value: str) -> Any:
    value_without_comment = raw_value.split("  #", 1)[0].strip()
    try:
        return yaml.safe_load(value_without_comment)
    except yaml.YAMLError:
        return value_without_comment


def _comment_tokens(path: Path) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        match = _COMMENT_FIELD_RE.match(line)
        if not match:
            continue
        tokens.append(
            {
                "line": line_number,
                "field": match.group(1),
                "value": _parse_commented_value(match.group(2)),
            }
        )
    return tokens


def build_legacy_snapshot(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    base_path = repo_root / "config.yaml"
    base = _read_yaml(base_path)
    override_paths = _config_paths(repo_root)
    if len(override_paths) != 68:
        raise ValueError(f"Expected 68 config overrides, found {len(override_paths)}")

    effective_by_config: dict[str, Any] = {
        "config.yaml": _snapshot_effective(base)
    }
    group_counts: dict[str, int] = {}
    retrieval_counts: dict[str, int] = {}
    mixed: list[str] = []
    comments: dict[str, list[dict[str, Any]]] = {
        "config.yaml": _comment_tokens(base_path)
    }

    relative_paths: list[str] = []
    for path in override_paths:
        relative = path.relative_to(repo_root).as_posix()
        relative_paths.append(relative)
        merged = deep_merge(base, _read_yaml(path))
        effective = _snapshot_effective(merged)
        effective_by_config[relative] = effective
        comments[relative] = _comment_tokens(path)

        group = _group_name(relative)
        group_counts[group] = group_counts.get(group, 0) + 1

        bucket = _retrieval_bucket(merged)
        retrieval_counts[bucket] = retrieval_counts.get(bucket, 0) + 1

        agent_identity = (
            effective["llm.agent"]["provider"],
            effective["llm.agent"]["model"],
            effective["llm.agent"]["api_base"],
        )
        ace_identity = (
            effective["llm.ace"]["provider"],
            effective["llm.ace"]["model"],
            effective["llm.ace"]["api_base"],
        )
        if agent_identity != ace_identity:
            mixed.append(relative)

    return {
        "schema_version": 1,
        "base_config": "config.yaml",
        "override_configs": relative_paths,
        "group_counts": dict(sorted(group_counts.items())),
        "retrieval_counts": dict(sorted(retrieval_counts.items())),
        "mixed_agent_ace_configs": sorted(mixed),
        "commented_raw_fields": dict(sorted(comments.items())),
        "effective_by_config": dict(sorted(effective_by_config.items())),
    }


def write_snapshot(snapshot: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot-legacy")
    snapshot.add_argument("--repo-root", type=Path, default=Path("."))
    snapshot.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "snapshot-legacy":
        write_snapshot(build_legacy_snapshot(args.repo_root), args.output)


if __name__ == "__main__":
    main()
