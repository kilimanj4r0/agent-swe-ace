#!/usr/bin/env python3
"""Build and compare semantic inventories for the LLM preset migration."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from config.llm_catalog import get_effective_llm, load_llm_catalog  # noqa: E402
from config.loader import load_experiment_config  # noqa: E402


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
_COMMENT_ALIAS_RE = re.compile(
    r"^\s*#\s*(?:agent|ace|llm|preset):\s*"
    r"([a-z0-9][a-z0-9._-]*)\s*(?:#.*)?$"
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
    if retriever_type != "embedding":
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


def _snapshot_current_effective(config: Mapping[str, Any]) -> dict[str, Any]:
    llm = config.get("llm", {})
    retrieval = (
        config.get("experiment", {})
        .get("skillbook", {})
        .get("retrieval", {})
    )
    retriever_type = retrieval.get("type", "llm")
    return {
        "llm.agent": get_effective_llm(llm["agent"]),
        "llm.ace": get_effective_llm(llm["ace"]),
        "retrieval": _normalise_retrieval(retrieval),
        "retrieval.llm": (
            get_effective_llm(retrieval["llm"])
            if retriever_type == "llm"
            else None
        ),
    }


def _config_paths(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "configs").rglob("*.yaml"))


def _matches_include(
    relative_path: str,
    include_prefixes: tuple[str, ...],
) -> bool:
    if not include_prefixes:
        return True
    for raw_prefix in include_prefixes:
        prefix = raw_prefix.replace("\\", "/")
        if prefix.endswith("/"):
            if relative_path.startswith(prefix):
                return True
        elif relative_path == prefix:
            return True
    return False


def _selected_paths(
    repo_root: Path,
    include_prefixes: tuple[str, ...],
) -> list[Path]:
    return [
        path
        for path in _config_paths(repo_root)
        if _matches_include(path.relative_to(repo_root).as_posix(), include_prefixes)
    ]


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


def build_current_snapshot(
    repo_root: Path,
    include_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a normalized snapshot through the strict preset-aware loader."""
    repo_root = repo_root.resolve()
    base_path = repo_root / "config.yaml"
    override_paths = _selected_paths(repo_root, include_prefixes)
    include_base = not include_prefixes or _matches_include(
        "config.yaml",
        include_prefixes,
    )

    effective_by_config: dict[str, Any] = {}
    group_counts: dict[str, int] = {}
    retrieval_counts: dict[str, int] = {}
    mixed: list[str] = []
    relative_paths: list[str] = []

    if include_base:
        base = load_experiment_config(base_path)
        effective_by_config["config.yaml"] = _snapshot_current_effective(base)

    for path in override_paths:
        relative = path.relative_to(repo_root).as_posix()
        relative_paths.append(relative)
        merged = load_experiment_config(base_path, path)
        effective = _snapshot_current_effective(merged)
        effective_by_config[relative] = effective

        group = _group_name(relative)
        group_counts[group] = group_counts.get(group, 0) + 1
        bucket = _retrieval_bucket(merged)
        retrieval_counts[bucket] = retrieval_counts.get(bucket, 0) + 1

        agent_identity = tuple(
            effective["llm.agent"][field]
            for field in ("provider", "model", "api_base")
        )
        ace_identity = tuple(
            effective["llm.ace"][field]
            for field in ("provider", "model", "api_base")
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
        "effective_by_config": dict(sorted(effective_by_config.items())),
    }


def _value_diffs(expected: Any, actual: Any, path: str = "") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        diffs: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                diffs.append(f"{child}: unexpected {actual[key]!r}")
            elif key not in actual:
                diffs.append(f"{child}: missing; expected {expected[key]!r}")
            else:
                diffs.extend(_value_diffs(expected[key], actual[key], child))
        return diffs
    if expected != actual:
        return [f"{path}: expected {expected!r}, got {actual!r}"]
    return []


def compare_current_to_golden(
    repo_root: Path,
    golden_path: Path,
    include_prefixes: tuple[str, ...] = (),
) -> None:
    """Assert that migrated configs retain the golden effective semantics."""
    repo_root = repo_root.resolve()
    golden = json.loads(Path(golden_path).read_text())
    current = build_current_snapshot(repo_root, include_prefixes)
    expected_all = golden["effective_by_config"]
    expected = {
        path: _normalise_golden_for_comparison(value)
        for path, value in expected_all.items()
        if _matches_include(path, include_prefixes)
    }
    actual = current["effective_by_config"]
    diffs = _value_diffs(expected, actual, "effective_by_config")
    if diffs:
        limit = 100
        rendered = "\n".join(f"- {diff}" for diff in diffs[:limit])
        if len(diffs) > limit:
            rendered += f"\n- ... {len(diffs) - limit} more differences"
        raise AssertionError(
            "Current LLM config semantics differ from the legacy golden:\n"
            f"{rendered}"
        )


def _normalise_golden_for_comparison(
    effective: Mapping[str, Any],
) -> dict[str, Any]:
    """Drop legacy generative leakage from non-embedding retrieval settings."""
    result = copy.deepcopy(dict(effective))
    retrieval = result.get("retrieval", {})
    if retrieval.get("type", "llm") != "embedding":
        retrieval.pop("model", None)
    return result


def _commented_override_field(lines: list[str], index: int) -> bool:
    match = re.match(r"^\s*#(?P<body>.*)$", lines[index])
    if not match:
        return False
    body = match.group("body")
    current_indent = len(body) - len(body.lstrip())
    for previous in reversed(lines[:index]):
        if not previous.strip():
            continue
        previous_match = re.match(r"^\s*#(?P<body>.*)$", previous)
        if not previous_match:
            break
        previous_body = previous_match.group("body")
        stripped = previous_body.strip()
        if not stripped:
            continue
        previous_indent = len(previous_body) - len(previous_body.lstrip())
        if previous_indent >= current_indent:
            continue
        if stripped.startswith("overrides:"):
            return True
        if re.match(r"(?:agent|ace|llm|preset):", stripped):
            return False
    return False


def _validate_file_comments(path: Path, catalog_names: set[str]) -> list[str]:
    errors: list[str] = []
    lines = path.read_text().splitlines()
    for line_number, line in enumerate(lines, start=1):
        alias_match = _COMMENT_ALIAS_RE.match(line)
        if alias_match and alias_match.group(1) not in catalog_names:
            errors.append(
                f"{path}:{line_number}: unknown commented preset "
                f"{alias_match.group(1)!r}"
            )

        raw_match = _COMMENT_FIELD_RE.match(line)
        if not raw_match:
            continue
        field, raw_value = raw_match.groups()
        value = _parse_commented_value(raw_value)
        if (
            field == "model"
            and isinstance(value, str)
            and "Embedding" in value
        ):
            continue
        if _commented_override_field(lines, line_number - 1):
            continue
        errors.append(
            f"{path}:{line_number}: commented raw LLM field {field!r}; "
            "use a catalog preset alias"
        )
    return errors


def validate_commented_aliases(
    repo_root: Path,
    include_prefixes: tuple[str, ...] = (),
) -> None:
    """Validate active strict configs plus every commented preset example."""
    repo_root = repo_root.resolve()
    catalog_names = set(load_llm_catalog(repo_root / "llms.yaml"))
    paths = _selected_paths(repo_root, include_prefixes)
    include_base = not include_prefixes or _matches_include(
        "config.yaml",
        include_prefixes,
    )
    errors: list[str] = []

    if include_base:
        load_experiment_config(repo_root / "config.yaml")
        errors.extend(
            _validate_file_comments(repo_root / "config.yaml", catalog_names)
        )
    for path in paths:
        load_experiment_config(repo_root / "config.yaml", path)
        errors.extend(_validate_file_comments(path, catalog_names))

    if errors:
        raise AssertionError("Invalid LLM config comments:\n- " + "\n- ".join(errors))


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
    compare = subparsers.add_parser("compare-current")
    compare.add_argument("--repo-root", type=Path, default=Path("."))
    compare.add_argument("--golden", type=Path, required=True)
    compare.add_argument("--include", action="append", default=[])
    comments = subparsers.add_parser("validate-comments")
    comments.add_argument("--repo-root", type=Path, default=Path("."))
    comments.add_argument("--include", action="append", default=[])
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "snapshot-legacy":
        write_snapshot(build_legacy_snapshot(args.repo_root), args.output)
    elif args.command == "compare-current":
        compare_current_to_golden(
            args.repo_root,
            args.golden,
            tuple(args.include),
        )
        count = len(build_current_snapshot(
            args.repo_root,
            tuple(args.include),
        )["effective_by_config"])
        print(f"Compared {count} configuration(s): semantic match")
    elif args.command == "validate-comments":
        validate_commented_aliases(args.repo_root, tuple(args.include))
        print("Commented LLM aliases are valid")


if __name__ == "__main__":
    main()
