"""Platform detection utilities for template variables."""

import platform
from typing import Any, Dict


def get_platform_info() -> Dict[str, Any]:
    """
    Get platform information for template variables.

    DockerEnvironment doesn't provide these by default, but
    mini-swe-agent's instance template requires them.
    """
    return platform.uname()._asdict()
