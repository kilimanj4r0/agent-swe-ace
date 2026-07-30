"""LLM Observability with Opik (via LiteLLM).

This module provides LLM call observability using Opik
via LiteLLM's built-in OpikLogger integration.

Usage:
    from utils.llm_observer import enable_observability

    # Call this BEFORE any LLM calls
    enable_observability(project_name="my-experiment")

    # Get project URL for linking
    url = get_project_url()

Note:
    ACE v0.9.1 does not ship Opik integration (only Logfire).
    This module uses LiteLLM's built-in OpikLogger instead.

    Set OPIK_URL_OVERRIDE and OPIK_API_KEY environment variables
    to configure the Opik server. For local Opik:
        OPIK_API_KEY=local
        OPIK_URL_OVERRIDE=http://localhost:5173/api
"""

import os
from typing import Optional

import litellm
from loguru import logger

# Track observability state
_enabled = False
_project_name = "agent-swe-ace"
_opik_base_url: Optional[str] = None


def enable_observability(
    project_name: str = "agent-swe-ace",
) -> None:
    """
    Enable LLM observability with Opik via LiteLLM's built-in OpikLogger.

    Registers OpikLogger as a LiteLLM callback to trace all LLM calls.

    Args:
        project_name: Project name shown in Opik UI.
            Default: agent-swe-ace

    Example:
        >>> from utils.llm_observer import enable_observability
        >>> enable_observability(project_name="my-experiment")

    Note:
        Set OPIK_API_KEY environment variable if using Opik cloud.
        For local Opik, it runs at http://localhost:5173 by default.
    """
    global _enabled, _project_name, _opik_base_url

    if _enabled:
        logger.warning("Observability already enabled")
        return

    try:
        from litellm.integrations.opik.opik import OpikLogger

        opik_logger = OpikLogger(project_name=project_name)
        litellm.callbacks = [opik_logger]

        _enabled = True
        _project_name = project_name

        # Extract base URL from OPIK_URL_OVERRIDE (removes /api suffix)
        opik_url_override = os.environ.get("OPIK_URL_OVERRIDE", "")
        if opik_url_override:
            _opik_base_url = opik_url_override.rstrip("/api").rstrip("/")
        else:
            _opik_base_url = "http://localhost:5173"

        print("LLM Observability enabled (Opik)")
        print(f"   Project: {project_name}")
        print(f"   URL: {get_project_url()}")

    except ImportError as e:
        logger.warning(f"Opik/LiteLLM integration not available: {e}")
        print("Install with: pip install opik litellm")
        print(f"   Error: {e}")
        _enabled = False
    except Exception as e:
        logger.error(f"Failed to enable observability: {e}")
        print(f"Failed to enable observability: {e}")
        _enabled = False


def disable_observability() -> None:
    """
    Disable LLM observability.

    Resets the observability state and removes the Opik callback.
    """
    global _enabled, _project_name, _opik_base_url

    litellm.callbacks = []
    _enabled = False
    _project_name = "agent-swe-ace"
    _opik_base_url = None
    print("LLM Observability disabled")


def is_enabled() -> bool:
    """Check if observability is enabled."""
    return _enabled


def get_project_name() -> str:
    """Get the current project name."""
    return _project_name


def get_project_url() -> Optional[str]:
    """
    Get the URL to the Opik project in the UI.

    Returns:
        URL string like "http://localhost:5173/projects/my-project"
        or None if observability is not enabled.
    """
    if not _enabled or not _opik_base_url:
        return None

    return f"{_opik_base_url}/projects/{_project_name}"


def get_base_url() -> Optional[str]:
    """Get the Opik base URL (without /projects suffix)."""
    return _opik_base_url
