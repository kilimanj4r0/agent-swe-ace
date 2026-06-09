# src/retrieval/base.py
"""Shared base mixin and helpers for skill retrievers."""

from ace.core.skillbook import Skill


class SkillRetrieverBase:
    """Mixin providing common config summary logic for skill retrievers.

    Subclasses must set: top_k, skip_threshold, model.
    Subclasses must implement: retrieve(skillbook, instance) -> list[Skill].
    """

    top_k: int
    skip_threshold: int
    model: str

    def get_config_summary(self) -> dict:
        """Return retriever config parameters for statistics logging."""
        return {
            "type": self.__class__.__name__,
            "model": self.model,
            "top_k": self.top_k,
            "skip_threshold": self.skip_threshold,
        }


def extract_issue_info(instance: dict) -> tuple[str, str, str]:
    """Extract repo, title, body from a SWE-bench instance dict.

    Args:
        instance: SWE-bench instance dict with 'repo' and 'problem_statement' keys.

    Returns:
        (repo, title, body) tuple. repo uses '__' separator (e.g. 'django__django').
    """
    repo = instance.get("repo", "unknown").replace("/", "__")
    problem = instance.get("problem_statement", "")
    title = problem.split("\n", 1)[0]
    body = problem.split("\n", 1)[1] if "\n" in problem else ""
    return repo, title, body
