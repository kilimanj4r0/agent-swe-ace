"""Simple patch validation without Docker harness."""

import logging
import re
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


def validate_patch_simple(instance: dict, patch: str) -> bool:
    """
    Simple validation without full Docker harness.

    Checks:
    1. Patch is non-empty
    2. Patch appears to be valid diff format

    Args:
        instance: SWE-bench instance dict (unused but kept for API consistency)
        patch: The patch string to validate

    Returns:
        True if patch passes basic validation
    """
    if not patch or not patch.strip():
        return False

    # Check for valid diff format
    has_diff_header = patch.startswith("diff --git")
    has_hunk_markers = "--- " in patch and "+++ " in patch

    if not has_diff_header and not has_hunk_markers:
        logger.warning("Patch does not appear to be in diff format")
        return False

    return True


def check_syntax(patch: str) -> Dict[str, Union[bool, List[str]]]:
    """
    Check if the patch introduces syntax errors in Python code.

    Returns:
        Dict with 'valid' boolean and 'errors' list
    """
    errors: List[str] = []

    # Check for syntax issues in the patch hunks
    hunk_pattern = r'@@.*@@\n((?:[+-].*\n)+)'
    hunks = re.findall(hunk_pattern, patch)

    for hunk in hunks:
        lines = hunk.split('\n')
        added_lines = [
            line[1:].strip()
            for line in lines
            if line.startswith('+') and not line.startswith('+++')
        ]

        code = '\n'.join(added_lines)
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            errors.append(f"Syntax error in patch: {e}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }
