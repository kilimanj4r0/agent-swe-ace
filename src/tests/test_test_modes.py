"""Synthetic tests for the end-to-end smoke-test verifier."""

import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def test_modes():
    script = Path(__file__).parents[2] / "scripts" / "test_modes.py"
    spec = importlib.util.spec_from_file_location("test_modes_script", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_dir(tmp_path, statistics):
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    (run_dir / "statistics.json").write_text(json.dumps(statistics))
    return run_dir


def _write_trajectory(run_dir, info, messages=None):
    trajectory = (
        run_dir
        / "princeton-nlp__SWE-bench_Verified"
        / "trajectories"
        / "django__django-16527"
        / "iter_0.json"
    )
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(
        json.dumps({"info": info, "messages": messages or []})
    )


def test_exact_and_nested_statistics_checks(test_modes, tmp_path):
    run_dir = _run_dir(
        tmp_path,
        {
            "total_instances": 1,
            "retrieval": {"type": "BM25Retriever", "calls": 2},
        },
    )

    assert test_modes.verify(
        run_dir, {"stats_values": {"total_instances": 1}}
    ).verdict is test_modes.Verdict.PASS
    assert test_modes.verify(
        run_dir, {"stats_values": {"total_instances": 0}}
    ).verdict is test_modes.Verdict.FAIL
    assert test_modes.verify(
        run_dir, {"stats_nested": {"retrieval.type": "BM25Retriever"}}
    ).verdict is test_modes.Verdict.PASS
    assert test_modes.verify(
        run_dir, {"stats_nested": {"retrieval.type": "RandomRetriever"}}
    ).verdict is test_modes.Verdict.FAIL
    assert test_modes.verify(
        run_dir, {"stats_nested_gte": {"retrieval.calls": 2}}
    ).verdict is test_modes.Verdict.PASS
    assert test_modes.verify(
        run_dir, {"stats_nested_gte": {"retrieval.calls": 3}}
    ).verdict is test_modes.Verdict.FAIL


def test_exact_comparison_is_type_sensitive(test_modes, tmp_path):
    run_dir = _run_dir(tmp_path, {"enabled": True})

    assert test_modes.verify(
        run_dir, {"stats_values": {"enabled": 1}}
    ).verdict is test_modes.Verdict.FAIL
    assert test_modes.verify(
        run_dir, {"stats_values": {"enabled": True}}
    ).verdict is test_modes.Verdict.PASS


def test_missing_statistics_is_failure(test_modes, tmp_path):
    run_dir = tmp_path / "run_missing"
    run_dir.mkdir()

    result = test_modes.verify(
        run_dir, {"stats_nested": {"retrieval.type": "BM25Retriever"}}
    )

    assert result.verdict is test_modes.Verdict.FAIL
    assert any("statistics.json" in detail for detail in result.details)


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (
            {
                "exit_status": "error",
                "error_kind": "infrastructure",
                "error": "Docker image is missing",
            },
            "BLOCKED",
        ),
        (
            {"exit_status": "error", "error": "unexpected parser failure"},
            "FAIL",
        ),
        ({"exit_status": "LimitsExceeded"}, "PASS"),
    ],
)
def test_trajectory_exit_status_verdict(test_modes, tmp_path, info, expected):
    run_dir = _run_dir(tmp_path, {"total_instances": 1})
    _write_trajectory(run_dir, info)

    result = test_modes.verify(
        run_dir, {"stats_values": {"total_instances": 1}}
    )

    assert result.verdict is getattr(test_modes.Verdict, expected)


def test_find_fresh_run_rejects_preexisting_directory(test_modes, tmp_path):
    old = tmp_path / "run_old"
    old.mkdir()
    started_at = time.time()

    assert test_modes.find_fresh_run(tmp_path, {old}, started_at) is None

    new = tmp_path / "run_new"
    new.mkdir()
    os.utime(new, (started_at + 1, started_at + 1))
    assert test_modes.find_fresh_run(tmp_path, {old}, started_at) == new


def test_docker_image_preflight_uses_exact_namespaced_image(test_modes):
    calls = []

    def run_command(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = test_modes.check_docker_images(
        ["django__django-16527"],
        "ghcr.io/epoch-research/",
        run_command=run_command,
    )

    assert result.verdict is test_modes.Verdict.PASS
    assert calls == [[
        "docker",
        "image",
        "inspect",
        "ghcr.io/epoch-research/sweb.eval.x86_64.django__django-16527:latest",
    ]]


def test_missing_docker_image_is_blocked(test_modes):
    def run_command(command, **kwargs):
        return SimpleNamespace(
            returncode=1, stdout="", stderr="No such image"
        )

    result = test_modes.check_docker_images(
        ["django__django-16527"],
        None,
        run_command=run_command,
    )

    assert result.verdict is test_modes.Verdict.BLOCKED
    assert "sweb.eval.x86_64.django__django-16527:latest" in result.details[0]


def test_run_config_always_enables_strict_mode(test_modes, tmp_path, monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(test_modes.subprocess, "run", fake_run)
    test_modes.run_config(tmp_path / "config.yaml")

    assert "--strict" in captured["command"]


def test_configured_instances_are_resolved_before_preflight(test_modes):
    dataset = [
        {"instance_id": "keep", "repo": "org/repo"},
        {"instance_id": "excluded", "repo": "org/repo"},
        {"instance_id": "other", "repo": "other/repo"},
    ]
    config = {
        "benchmark": {
            "dataset": "org/benchmark",
            "split": "test",
            "max_instances": 3,
            "exclude_instances": ["excluded"],
            "filter_repos": ["org/repo"],
        }
    }

    instance_ids = test_modes.resolve_instance_ids(
        config,
        dataset_loader=lambda *args, **kwargs: dataset,
    )

    assert instance_ids == ["keep"]


def test_timeout_cleans_temporary_config(test_modes, tmp_path, monkeypatch):
    project_root = tmp_path
    configs_dir = project_root / "configs" / "test"
    configs_dir.mkdir(parents=True)
    (project_root / "config.yaml").write_text(
        "environment:\n  type: local\n"
    )
    source_config = configs_dir / "01_basic.yaml"
    source_config.write_text("experiment:\n  name: timeout-test\n")

    monkeypatch.setattr(test_modes, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(test_modes, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(test_modes, "DATA_DIR", project_root / "_data")

    def timeout(*args, **kwargs):
        raise test_modes.subprocess.TimeoutExpired(["command"], 12)

    monkeypatch.setattr(test_modes, "run_config", timeout)
    verdict, run_dir = test_modes.SmokeTest(total=1)._run(
        1, "01_basic.yaml"
    )

    assert verdict is test_modes.Verdict.FAIL
    assert run_dir is None
    assert list(configs_dir.iterdir()) == [source_config]
