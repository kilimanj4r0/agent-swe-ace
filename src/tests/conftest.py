"""Shared test configuration and fixtures."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--config",
        default=None,
        help="Path to override config YAML (relative to project root). "
        "If omitted, uses base config.yaml only.",
    )


# --- Shared test data factories ---


@pytest.fixture
def make_instance():
    """Factory fixture: create a valid SWE-bench instance dict with all production fields."""
    def _make(
        instance_id="test__repo-123",
        repo="test/repo",
        problem_statement="Fix the bug in the module",
        base_commit="abc123def",
        version="1.0",
        **overrides,
    ):
        data = {
            "instance_id": instance_id,
            "repo": repo,
            "problem_statement": problem_statement,
            "base_commit": base_commit,
            "version": version,
        }
        data.update(overrides)
        return data
    return _make


@pytest.fixture
def make_trajectory():
    """Factory fixture: create a trajectory dict with full production shape."""
    def _make(
        exit_status="submitted",
        submission="diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
        messages=None,
        iteration=0,
        instance_id="test__repo-123",
        model="test-model",
    ):
        return {
            "info": {
                "exit_status": exit_status,
                "submission": submission,
                "iteration": iteration,
                "instance_id": instance_id,
                "model": model,
                "message_count": len(messages) if messages else 0,
                "assistant_message_count": sum(
                    1 for m in (messages or []) if m.get("role") == "assistant"
                ),
            },
            "messages": messages or [],
        }
    return _make


@pytest.fixture
def make_result():
    """Factory fixture: create an evaluation result dict with full production shape."""
    def _make(
        resolved=False,
        feedback="Tests failed",
        patch_length=500,
        instance_id="test__repo-123",
        iteration=0,
    ):
        return {
            "resolved": resolved,
            "feedback": feedback,
            "metrics": {"resolved": 1.0 if resolved else 0.0, "patch_length": patch_length},
            "instance_id": instance_id,
            "iteration": iteration,
        }
    return _make


@pytest.fixture
def sample_zai_config():
    """Minimal zai provider config dict for unit testing (no file I/O needed)."""
    return {
        "provider": "zai",
        "model": "glm-4.5-flash",
        "api_key": "test-key-for-unit-tests",
        "api_base": "https://api.z.ai/api/paas/v4",
        "temperature": 0.7,
        "max_tokens": 4096,
    }


@pytest.fixture
def sample_vllm_config():
    """Minimal hosted_vllm provider config dict for unit testing."""
    return {
        "provider": "hosted_vllm",
        "model": "Qwen/Qwen3-Coder-30B-A3B",
        "api_base": "http://localhost:8000/v1",
        "temperature": 0.7,
        "max_tokens": 4096,
    }
