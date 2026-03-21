"""LLM Observability with Opik (via ACE Framework).

This module provides LLM call observability using Opik
via ACE framework's built-in observability support.

Usage:
    from utils.llm_observer import enable_observability, get_opik_step

    # Call this BEFORE any LLM calls
    enable_observability(project_name="my-experiment")

    # Get OpikStep for ACE pipeline
    opik_step = get_opik_step()

    # Get project URL for linking
    url = get_project_url()

    # Use in ACE.from_roles():
    # runner = ACE.from_roles(..., extra_steps=[opik_step])
"""

import os
from typing import Optional, Any, Dict

from loguru import logger

# Track observability state
_enabled = False
_project_name = "agent-swe-ace"
_opik_base_url: Optional[str] = None


def enable_observability(
    project_name: str = "agent-swe-ace",
) -> None:
    """
    Enable LLM observability with Opik.

    Uses ACE framework's register_opik_litellm_callback for per-LLM-call
    cost tracking and tracing.

    Args:
        project_name: Project name shown in Opik UI.
            Default: agent-swe-ace

    Example:
        >>> from utils.llm_observer import enable_observability
        >>> enable_observability(project_name="my-experiment")
        🔍 LLM Observability enabled (Opik)
           Project: my-experiment

    Note:
        Set OPIK_API_KEY environment variable if using Opik cloud.
        For local Opik, it runs at http://localhost:5173 by default.
    """
    global _enabled, _project_name, _opik_base_url

    if _enabled:
        logger.warning("Observability already enabled")
        return

    try:
        from ace_next import register_opik_litellm_callback

        register_opik_litellm_callback(project_name=project_name)
        _enabled = True
        _project_name = project_name

        # Extract base URL from OPIK_URL_OVERRIDE (removes /api suffix)
        opik_url_override = os.environ.get("OPIK_URL_OVERRIDE", "")
        if opik_url_override:
            # Remove /api suffix to get the UI base URL
            _opik_base_url = opik_url_override.rstrip("/api").rstrip("/")
        else:
            _opik_base_url = "http://localhost:5173"

        print("🔍 LLM Observability enabled (Opik)")
        print(f"   Project: {project_name}")
        print(f"   URL: {get_project_url()}")

    except ImportError as e:
        logger.warning(f"ACE observability dependencies not installed: {e}")
        print("⚠️  Install with: pip install ace-framework[observability]")
        print(f"   Error: {e}")
        _enabled = False
    except Exception as e:
        logger.error(f"Failed to enable observability: {e}")
        print(f"⚠️  Failed to enable observability: {e}")
        _enabled = False


def disable_observability() -> None:
    """
    Disable LLM observability.

    Resets the observability state.
    """
    global _enabled, _project_name, _opik_base_url

    _enabled = False
    _project_name = "agent-swe-ace"
    _opik_base_url = None
    print("🔍 LLM Observability disabled")


def is_enabled() -> bool:
    """Check if observability is enabled."""
    return _enabled


def get_opik_step() -> Optional[Any]:
    """
    Get OpikStep for ACE pipeline extra_steps.

    Returns None if observability is not enabled or OpikStep is not available.

    Example:
        >>> from utils.llm_observer import enable_observability, get_opik_step
        >>> from ace_next import ACE
        >>>
        >>> enable_observability(project_name="my-experiment")
        >>> opik_step = get_opik_step()
        >>>
        >>> runner = ACE.from_roles(
        ...     agent=agent,
        ...     reflector=reflector,
        ...     skill_manager=skill_manager,
        ...     environment=environment,
        ...     extra_steps=[opik_step] if opik_step else [],
        ... )
    """
    if not _enabled:
        return None

    try:
        from ace_next import OpikStep

        return OpikStep(project_name=_project_name)
    except ImportError as e:
        logger.warning(f"OpikStep not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to create OpikStep: {e}")
        return None


def get_project_name() -> str:
    """Get the current project name."""
    return _project_name


def get_project_url() -> Optional[str]:
    """
    Get the URL to the Opik project in the UI.

    Returns:
        URL string like "http://localhost:5173/projects/my-project"
        or None if observability is not enabled.

    Example:
        >>> enable_observability(project_name="run_20260321_143052")
        >>> url = get_project_url()
        >>> print(url)
        http://localhost:5173/projects/run_20260321_143052
    """
    if not _enabled or not _opik_base_url:
        return None

    # Opik project URL format: {base_url}/projects/{project_name}
    return f"{_opik_base_url}/projects/{_project_name}"


def get_base_url() -> Optional[str]:
    """Get the Opik base URL (without /projects suffix)."""
    return _opik_base_url
